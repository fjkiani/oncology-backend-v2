import modal
import subprocess
import sys
from pathlib import Path

# Create Modal app
app = modal.App("oncology-backend-v2")

# Define the image with all dependencies and copy the entire backend
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
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
    .add_local_dir(".", "/app")  # Copy entire backend directory
    .workdir("/app")  # Set working directory
)

# Deploy FastAPI as a web server
@app.function(image=image, timeout=300)
@modal.web_server(8000, startup_timeout=120)
def fastapi_server():
    import os
    import sys
    
    # Change to app directory
    os.chdir("/app")
    sys.path.insert(0, "/app")
    
    # Start the FastAPI server using uvicorn with correct binding
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--workers", "1"
    ]
    subprocess.Popen(cmd) 