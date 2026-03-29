# Advanced Generative AI & LangChain Architecture

This repository contains production-grade implementations of advanced Large Language Model (LLM) patterns using **LangChain** and **LangGraph**. It serves as a comprehensive portfolio demonstrating deep expertise in Generative AI architectures, moving beyond simple API wrappers to orchestrate complex, resilient, and highly capable AI systems.

## 🚀 Key Architectures Implemented

1. **Hybrid Retrieval-Augmented Generation (RAG)**
   - Combines dense Vector Search (e.g., FAISS + OpenAI Embeddings) with sparse Keyword Search (BM25) to maximize retrieval precision.
   - Utilizes Cross-Encoder Re-ranking to contextually compress and sort the retrieved documents before passing them to the LLM.

2. **Multi-Agent Workflows (LangGraph)**
   - Implements a deterministic, graph-based architecture using LangGraph.
   - Features a Supervisor node delegating tasks between a specific **Researcher Agent** (equipped with search tools) and a **Synthesizer Agent**, demonstrating the ability to handle complex, multi-step reasoning tasks reliably.

3. **Structured Information Extraction**
   - Demonstrates how to strictly enforce schema-compliant JSON outputs from messy, unstructured text using OpenAI Tool Calling and Pydantic validators.
   - Useful for downstream data pipelines, entity extraction, and automated categorization.

4. **Asynchronous Agents with Memory**
   - Showcases non-blocking execution mechanisms (asyncio) for high-performance agentic loops.
   - Integrates advanced conversational memory limits (`ConversationBufferWindowMemory`) to prevent context window bloat while maintaining conversational state.

## 🛠️ Prerequisites & Setup

1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone <your-repo-url>
   cd langchain-advanced-architectures
   ```

2. **Create a virtual environment (Recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Copy the example environment file and insert your API keys:
   ```bash
   cp .env.example .env
   ```
   *Required Keys:*
   - `OPENAI_API_KEY` (For core LLM and Embeddings)
   - `TAVILY_API_KEY` or generic search API keys (Optional, for Agent search tools)

## 📁 Repository Structure
```
├── src/
│   ├── advanced_rag/      # Hybrid Search and Contextual Re-ranking
│   ├── multi_agent/       # LangGraph Supervisor and Sub-Agents
│   ├── extraction/        # Pydantic Structural Validation
│   ├── async_agent/       # Async Execution with Memory
│   └── utils/             # Telemetry, Logging, and configuration
├── requirements.txt
├── .env.example
└── README.md
```

## 🧠 Why This Architecture?
While standard "Chains" are sufficient for simple QA, enterprise applications require robustness. 
- **Graph-based workflows** over linear chains allow cyclic, conditional loops (e.g., self-reflection and error-correction).
- **Hybrid Search** solves the limitation of pure semantic similarity (which often struggles with exact keyword matches like names or serial numbers).
- **Structured Outputs** act as the essential bridge between probabilistic LLMs and deterministic backend databases.

---
*Built to demonstrate proficiency in advanced AI Engineering.*
