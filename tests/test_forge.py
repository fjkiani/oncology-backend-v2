import sys
import os
from fastapi.testclient import TestClient

# Add the parent directory ('oncology-backend') to the Python path
# This allows us to import the 'backend' module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app # Import the main FastAPI app from the root level

client = TestClient(app)

def test_generate_therapeutic_protein():
    """
    Tests the /api/forge/generate_therapeutic_protein endpoint.
    """
    payload = {
        "target_protein": "MMP-9"
    }
    response = client.post("/api/forge/generate_therapeutic_protein", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "new_protein_name" in data
    assert "new_protein_sequence" in data
    assert data["new_protein_name"] == "CS-MMP-9i-001" 
 
 
 
 
 
 
 
 