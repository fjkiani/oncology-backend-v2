"""
Vercel-compatible entry point for the Oncology Backend
This file adapts the main FastAPI app for Vercel's serverless environment
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import the FastAPI app from main.py
from main import app

# Vercel expects the app to be named 'app'
# The main.py file already creates the FastAPI app instance
# so we just need to make sure it's accessible here

# For Vercel, we need to ensure the app is properly configured
if not hasattr(app, 'add_middleware'):
    # Fallback: create a basic FastAPI app if import fails
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(title="Oncology Backend", version="1.0.0")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {"message": "Oncology Backend API", "status": "running"}

# Export the app for Vercel
# Vercel will look for a variable named 'app' in this file 