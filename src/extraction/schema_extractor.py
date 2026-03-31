import os
import json
from enum import Enum
from typing import List, Optional
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# ==========================================
# 1. Define the Pydantic Schema
# ==========================================
# Enums restrict the LLM to specific categorical choices
class Sentiments(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class FeatureIssue(BaseModel):
    """Represents a specific feature or issue mentioned in the feedback."""
    component: str = Field(description="The specific part of the product mentioned (e.g., UI, Database, API)")
    description: str = Field(description="Summary of what the user says about this component")
    is_bug: bool = Field(description="True if the user is reporting a software bug or defect, False otherwise")

class CustomerFeedbackExtract(BaseModel):
    """The master schema for parsing customer reviews."""
    customer_name: Optional[str] = Field(description="The name of the customer, if provided")
    company_name: Optional[str] = Field(description="The company the customer works for, if provided")
    overall_sentiment: Sentiments = Field(description="The general emotional tone of the review")
    key_features_mentioned: List[FeatureIssue] = Field(
        description="A list of specific product components mentioned and what was said about them",
        default_factory=list
    )
    churn_risk: bool = Field(description="True if the customer explicitly mentions leaving, cancelling, or switching to a competitor")

# ==========================================
# 2. Build the Extractor Pipeline
# ==========================================
class SchemaExtractor:
    """
    Advanced Information Extraction system.
    Demonstrates forcing an LLM to output highly structured JSON using function calling,
    bridging the gap between unstructured NLP and deterministic backend databases.
    """
    
    def __init__(self):
        # We need a model that strongly supports function calling / structured outputs.
        # gpt-4o or gpt-4o-mini are ideal.
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Bind the Pydantic schema to the model
        # This forces the LLM to return a CustomerFeedbackExtract object
        self.structured_llm = self.llm.with_structured_output(CustomerFeedbackExtract)
        
        # Build the extraction prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert data pipeline assistant. Your task is to extract highly normalized, strictly typed data from unstructured text.\n"
                       "If information is missing, use null/None. Pay close attention to boolean flags and categorical enums."),
            ("human", "Extract information from the following customer feedback:\n\n{feedback}")
        ])
        
        # LCEL Chain
        self.chain = self.prompt | self.structured_llm

    def extract(self, text: str) -> CustomerFeedbackExtract:
        """Runs the extraction chain on the provided raw text."""
        print("[INFO] Passing unstructured text to LLM for Schema Binding...")
        result = self.chain.invoke({"feedback": text})
        return result

# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment. The script requires it to run.")
    else:
        # Sample Unstructured Data (Messy Customer Review)
        sample_feedback = '''
        From: john.doe@acmecorp.com
        Message: Hey team. We've been using your platform for 6 months here at Acme Corp. 
        Generally it's okay, but the API rate limiting is a major problem for us. Our ingestion jobs fail every night. 
        Also, the new dashboard UI looks great and is much faster than the old one. 
        However, because of the API issues, my boss is telling me to evaluate CompetitorX by next month if this isn't fixed.
        '''
        
        print("\n[INPUT TEXT]")
        print(sample_feedback.strip())
        print("\n---")
        
        # Run Extraction
        extractor = SchemaExtractor()
        extracted_data = extractor.extract(sample_feedback)
        
        print("\n[EXTRACTED STRUCTURED DATA] (Pydantic Object)")
        print(repr(extracted_data))
        
        print("\n[JSON SERIALIZATION FOR DATABASE]")
        # Pydantic v2 serialization 
        print(json.dumps(extracted_data.model_dump(), indent=2))
        print("\n======================\n")
