import os
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

class AdvancedHybridRAG:
    """
    A production-grade RAG pipeline implementing:
    1. Hybrid Search (Dense FAISS + Sparse BM25)
    2. Contextual Document Compression (Cross-Encoder Re-ranking)
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        # Use OpenAI for embeddings and Generation
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Build pipeline components
        self.docs = self._load_and_split_documents()
        self.retriever = self._build_hybrid_retriever()
        self.rag_chain = self._build_rag_chain()

    def _load_and_split_documents(self) -> List[Document]:
        """Loads text and splits it into semantic chunks."""
        print("[INFO] Loading and splitting documents...")
        loader = TextLoader(self.data_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, 
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )
        return text_splitter.split_documents(documents)

    def _build_hybrid_retriever(self):
        """Constructs an ensemble retriever with a cross-encoder re-ranker."""
        print("[INFO] Building FAISS Vector Store and BM25 Retriever...")
        
        # 1. Sparse Retriever (BM25 for exact keyword matches)
        bm25_retriever = BM25Retriever.from_documents(self.docs)
        bm25_retriever.k = 3

        # 2. Dense Retriever (FAISS for semantic similarity)
        vectorstore = FAISS.from_documents(self.docs, self.embeddings)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 3. Ensemble Retriever (Hybrid Search)
        # Weights: 40% Keyword, 60% Semantic
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6]
        )

        # 4. Cross-Encoder Re-ranker (Contextual Compression)
        print("[INFO] Initializing Cross-Encoder Re-ranker...")
        # High accuracy, lightweight local model for re-ranking
        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
        compressor = CrossEncoderReranker(model=model, top_n=2)
        
        # Combine Ensemble Retrieval with Re-ranking compression
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble_retriever
        )
        
        return compression_retriever

    def _build_rag_chain(self):
        """Builds the final LCEL (LangChain Expression Language) chain."""
        prompt_template = """You are an expert AI assistant. Answer the question strictly based on the provided context. 
If you don't know the answer based on the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # LCEL Pipeline: Format Docs -> Prompt -> LLM -> String Output
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def query(self, question: str) -> str:
        """Executes a query against the RAG pipeline."""
        print(f"\n[QUERY] {question}")
        print("[INFO] Generating response...")
        return self.rag_chain.invoke(question)

if __name__ == "__main__":
    # Ensure the OPENAI_API_KEY is available in the environment to run this
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment. The script requires it to run.")
    else:
        # Define corpus path
        corpus_path = os.path.join(os.path.dirname(__file__), "data", "sample_corpus.txt")
        
        rag_system = AdvancedHybridRAG(data_path=corpus_path)
        
        # Example Query
        answer = rag_system.query("What represents the state-of-the-art in document retrieval systems?")
        print(f"\n[ANSWER]\n{answer}\n")
        
        answer2 = rag_system.query("Which re-ranker is computationally expensive?")
        print(f"\n[ANSWER]\n{answer2}\n")
