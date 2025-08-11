import modal
from pathlib import Path

# Create Modal app
app = modal.App("oncology-backend")

# Define the image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "fastapi",
    "requests", 
    "python-dotenv",
    "httpx",
    "pandas",
    "google-generativeai",
    "astrapy",
    "uvicorn",
    "langchain-google-genai",
    "langchain",
    "langchain-community",
    "web3",
    "biopython",
    "firecrawl",
    "sentence-transformers"
])

# Create the FastAPI app
@app.function(image=image)
def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import sys
    import os
    
    # Add the current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Import and create the FastAPI app
    from main import app as fastapi_app
    
    # Add CORS middleware
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return fastapi_app

# Deploy as web endpoint
@app.asgi_app(image=image)
def fastapi_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import sys
    import os
    
    # Add the current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Import and create the FastAPI app
    from main import app as fastapi_app
    
    # Add CORS middleware
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return fastapi_app

if __name__ == "__main__":
    app.serve() 