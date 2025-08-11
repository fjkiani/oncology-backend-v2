from fastapi import APIRouter
from pydantic import BaseModel, Field
from ..clients.zeta_oracle_client import ZetaOracleClient
import random # Import random for baseline simulation

router = APIRouter()
client = ZetaOracleClient()

# --- Pydantic Models ---
class ZetaScoreRequest(BaseModel):
    baseline_sequence: str = Field(..., description="The long DNA sequence representing the baseline state (e.g., wild-type).")
    perturbed_sequence: str = Field(..., description="The long DNA sequence representing the altered state (e.g., with a gene knockout).")

class ZetaScoreResponse(BaseModel):
    zeta_score: float = Field(..., description="The calculated change in log-likelihood. A large negative score implies high impact.")
    confidence: float = Field(..., description="The model's confidence in the zeta_score, from 0.0 to 1.0.")
    verdict: str = Field(..., description="A qualitative verdict based on the score.")
    commentary: str = Field(..., description="AI-generated commentary on the result.")

    # --- Pydantic Models for Interaction Simulation ---
class InteractionRequest(BaseModel):
    target_protein: str
    effector_protein: str

class InteractionResponse(BaseModel):
    inhibition_score: float = Field(..., description="The calculated inhibition score, where lower is better.")
    verdict: str = Field(..., description="The final verdict based on the score.")

# --- Pydantic Models for Baseline Data ---

class BaselineData(BaseModel):
    name: str
    description: str
    mean_zeta_score: float
    quartiles: list[float] # [min, 25th, 50th, 75th, max]
    mean: float
    std_dev: float

class BaselineResponse(BaseModel):
    essential_genes: BaselineData
    non_essential_genes: BaselineData

# --- API Endpoints ---

@router.get("/baselines", response_model=BaselineResponse)
async def get_baselines():
    """
    Provides pre-calculated baseline Zeta Score distributions for canonical
    essential and non-essential genes. This is used to contextualize the
    score of a novel target.
    """
    # In a real system, these values would be derived from running the Zeta Oracle
    # over a large set of known genes from databases like DEG and caching the results.
    # Here, we simulate realistic distributions.
    return {
        "essential_genes": {
            "name": "Known Essential Genes",
            "description": "Represents targets critical for cell survival.",
            "mean_zeta_score": -4.85,
            "quartiles": [-9.5, -6.2, -4.5, -2.1, -1.1],
            "mean": -4.85,  # Added required field for tests
            "std_dev": 2.1  # Added required field for tests
        },
        "non_essential_genes": {
            "name": "Known Non-Essential Genes",
            "description": "Represents targets with high functional redundancy.",
            "mean_zeta_score": -0.05,
            "quartiles": [-0.8, -0.25, -0.04, 0.15, 0.5],
            "mean": -0.05,  # Added required field for tests
            "std_dev": 0.3  # Added required field for tests
        }
    }

@router.post("/run_interaction_simulation", response_model=InteractionResponse)
async def run_interaction_simulation(request: InteractionRequest):
    """
    Mocks a complex protein-protein interaction simulation for the Gauntlet.
    """
    if "CS-MMPi" in request.effector_protein:
        score, verdict = 9.85, "SUPERIOR"
    elif request.effector_protein == "U-995":
        score, verdict = 0.05, "MINIMAL"
    else:
        score, verdict = 1.25, "MODERATE"
    return InteractionResponse(inhibition_score=score, verdict=verdict)


@router.post("/calculate_zeta_score", response_model=ZetaScoreResponse)
async def calculate_zeta_score(request: ZetaScoreRequest):
    """
    This endpoint now acts as a proxy to the live ZetaOracleClient,
    which invokes the remote, production-grade Evo2 model.
    """
    # Call the client with just the sequences, as that's what its method signature expects
    return client.calculate_zeta_score(
        baseline_sequence=request.baseline_sequence,
        perturbed_sequence=request.perturbed_sequence
    ) 