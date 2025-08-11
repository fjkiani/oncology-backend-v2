import sys
import os

# Add the parent directory ('oncology-backend') to the Python path
# This allows us to import the 'backend' module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from fastapi.testclient import TestClient
from main import app # Import the main FastAPI app from the root level

client = TestClient(app)

def test_get_baselines():
    """
    Tests the /api/oracle/baselines endpoint to ensure it returns the correct structure.
    """
    response = client.get("/api/oracle/baselines")
    assert response.status_code == 200
    data = response.json()
    assert "essential_genes" in data
    assert "non_essential_genes" in data
    assert "mean" in data["essential_genes"]
    assert "std_dev" in data["essential_genes"]

def test_run_interaction_simulation_minimal():
    """
    Tests the simulation endpoint with the ineffective shark cartilage compound.
    """
    payload = {
        "target_protein": "MMP-9",
        "effector_protein": "U-995"
    }
    response = client.post("/api/oracle/run_interaction_simulation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "MINIMAL"
    assert data["inhibition_score"] == 0.05

def test_run_interaction_simulation_superior():
    """
    Tests the simulation endpoint with our superior forged weapon.
    """
    payload = {
        "target_protein": "MMP-9",
        "effector_protein": "CS-MMPi-001"
    }
    response = client.post("/api/oracle/run_interaction_simulation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "SUPERIOR"
    assert data["inhibition_score"] == 9.85 
 
 
 
 
 
 
 
import os

# Add the parent directory ('oncology-backend') to the Python path
# This allows us to import the 'backend' module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from fastapi.testclient import TestClient
from main import app # Import the main FastAPI app from the root level

client = TestClient(app)

def test_get_baselines():
    """
    Tests the /api/oracle/baselines endpoint to ensure it returns the correct structure.
    """
    response = client.get("/api/oracle/baselines")
    assert response.status_code == 200
    data = response.json()
    assert "essential_genes" in data
    assert "non_essential_genes" in data
    assert "mean" in data["essential_genes"]
    assert "std_dev" in data["essential_genes"]

def test_run_interaction_simulation_minimal():
    """
    Tests the simulation endpoint with the ineffective shark cartilage compound.
    """
    payload = {
        "target_protein": "MMP-9",
        "effector_protein": "U-995"
    }
    response = client.post("/api/oracle/run_interaction_simulation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "MINIMAL"
    assert data["inhibition_score"] == 0.05

def test_run_interaction_simulation_superior():
    """
    Tests the simulation endpoint with our superior forged weapon.
    """
    payload = {
        "target_protein": "MMP-9",
        "effector_protein": "CS-MMPi-001"
    }
    response = client.post("/api/oracle/run_interaction_simulation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "SUPERIOR"
    assert data["inhibition_score"] == 9.85 
 
 
 
 
 
 
 
import os

# Add the parent directory ('oncology-backend') to the Python path
# This allows us to import the 'backend' module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from fastapi.testclient import TestClient
from main import app # Import the main FastAPI app from the root level

client = TestClient(app)

def test_get_baselines():
    """
    Tests the /api/oracle/baselines endpoint to ensure it returns the correct structure.
    """
    response = client.get("/api/oracle/baselines")
    assert response.status_code == 200
    data = response.json()
    assert "essential_genes" in data
    assert "non_essential_genes" in data
    assert "mean" in data["essential_genes"]
    assert "std_dev" in data["essential_genes"]

def test_run_interaction_simulation_minimal():
    """
    Tests the simulation endpoint with the ineffective shark cartilage compound.
    """
    payload = {
        "target_protein": "MMP-9",
        "effector_protein": "U-995"
    }
    response = client.post("/api/oracle/run_interaction_simulation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "MINIMAL"
    assert data["inhibition_score"] == 0.05

def test_run_interaction_simulation_superior():
    """
    Tests the simulation endpoint with our superior forged weapon.
    """
    payload = {
        "target_protein": "MMP-9",
        "effector_protein": "CS-MMPi-001"
    }
    response = client.post("/api/oracle/run_interaction_simulation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "SUPERIOR"
    assert data["inhibition_score"] == 9.85 
 
 
 
 
 
 
 
import os

# Add the parent directory ('oncology-backend') to the Python path
# This allows us to import the 'backend' module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from fastapi.testclient import TestClient
from main import app # Import the main FastAPI app from the root level

client = TestClient(app)

def test_get_baselines():
    """
    Tests the /api/oracle/baselines endpoint to ensure it returns the correct structure.
    """
    response = client.get("/api/oracle/baselines")
    assert response.status_code == 200
    data = response.json()
    assert "essential_genes" in data
    assert "non_essential_genes" in data
    assert "mean" in data["essential_genes"]
    assert "std_dev" in data["essential_genes"]

def test_run_interaction_simulation_minimal():
    """
    Tests the simulation endpoint with the ineffective shark cartilage compound.
    """
    payload = {
        "target_protein": "MMP-9",
        "effector_protein": "U-995"
    }
    response = client.post("/api/oracle/run_interaction_simulation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "MINIMAL"
    assert data["inhibition_score"] == 0.05

def test_run_interaction_simulation_superior():
    """
    Tests the simulation endpoint with our superior forged weapon.
    """
    payload = {
        "target_protein": "MMP-9",
        "effector_protein": "CS-MMPi-001"
    }
    response = client.post("/api/oracle/run_interaction_simulation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "SUPERIOR"
    assert data["inhibition_score"] == 9.85 
 
 
 
 
 
 
 