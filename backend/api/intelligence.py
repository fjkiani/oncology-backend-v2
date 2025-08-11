from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from tavily import TavilyClient
import os
from typing import Optional, List
import json
from backend.core.llm_utils import get_llm_text_response # Import the LLM utility

router = APIRouter()

# --- Configuration ---
# Your Tavily API key should be set as an environment variable.
# For local development, you can create a .env file and load it.
# Example .env file:
# TAVILY_API_KEY="your_actual_key_here"
#
# In a real deployment, this would be managed as a secret.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("dotenv not installed, skipping. Make sure TAVILY_API_KEY is set manually.")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query: str

class TavilyResult(BaseModel):
    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[str] = None

class SearchResponse(BaseModel):
    answer: Optional[str] = None
    results: list[TavilyResult]

class SynthesisRequest(BaseModel):
    content: str

class BiologicalEntity(BaseModel):
    name: str = Field(..., description="The name of the biological entity, e.g., 'VEGFR-2', 'Neovastat'.")
    type: str = Field(..., description="The type of entity, e.g., 'Gene', 'Protein', 'Compound'.")
    description: str = Field(..., description="A brief description of the entity's role in the context of the text.")

class SynthesizedIntelligence(BaseModel):
    summary: str = Field(..., description="A high-level summary of the provided text.")
    entities: List[BiologicalEntity] = Field(..., description="A list of identified biological entities.")
    mechanisms: List[str] = Field(..., description="Key mechanisms of action described in the text.")
    conclusions: List[str] = Field(..., description="Primary conclusions drawn from the text.")

# --- API Endpoints ---

@router.post("/search", response_model=SearchResponse)
async def perform_search(search_request: SearchRequest):
    """
    API endpoint to perform an intelligent web search using the Tavily API.
    This provides a synthesized answer and a list of cited sources.
    """
    if not TAVILY_API_KEY:
        return {"answer": "TAVILY_API_KEY is not configured on the server.", "results": []}

    try:
        response = tavily_client.search(
            query=search_request.query,
            search_depth="advanced", # Use "advanced" for more comprehensive results
            include_answer=True,    # Request a synthesized answer
            max_results=5           # Limit to the top 5 most relevant sources
        )
        return response
    except Exception as e:
        # Log the exception for debugging
        print(f"Error calling Tavily API: {e}")
        return {"answer": "An error occurred while communicating with the intelligence service.", "results": []}

@router.post("/synthesize", response_model=SynthesizedIntelligence)
async def perform_synthesis(synthesis_request: SynthesisRequest):
    """
    Receives raw text content and uses an LLM to perform Semantic Deconstruction,
    extracting structured data.
    """
    # Define the specialized prompt for the ZSIS protocol
    prompt = f"""
    As a specialist intelligence analyst, your task is to perform a Semantic Deconstruction of the following research text.
    Analyze the text and extract the key information into a structured JSON object.

    The JSON object must conform to the following schema:
    {{
        "summary": "A concise, high-level summary of the entire text.",
        "entities": [
            {{
                "name": "Name of the biological entity (e.g., gene, protein, compound)",
                "type": "Type of entity (e.g., Gene, Protein, Compound)",
                "description": "A brief, one-sentence description of the entity's role in this context."
            }}
        ],
        "mechanisms": [
            "A list of key mechanisms of action described.",
            "Another mechanism."
        ],
        "conclusions": [
            "A list of primary conclusions or findings.",
            "Another conclusion."
        ]
    }}

    Here is the text to analyze:
    ---
    {synthesis_request.content}
    ---

    Provide ONLY the JSON object as your response.
    """

    # Get the raw string response from the LLM
    llm_response_str = await get_llm_text_response(prompt)

    try:
        # Clean the response to ensure it's valid JSON
        # The model sometimes wraps the JSON in ```json ... ```
        if llm_response_str.strip().startswith("```json"):
            llm_response_str = llm_response_str.strip()[7:-3]
        
        # Parse the string into a Python dictionary
        parsed_response = json.loads(llm_response_str)
        
        # Validate the dictionary against our Pydantic model
        return SynthesizedIntelligence(**parsed_response)
        
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error parsing LLM response as JSON: {e}")
        print(f"Raw LLM Response was: {llm_response_str}")
        raise HTTPException(status_code=500, detail="Failed to parse intelligence data from AI model.") 