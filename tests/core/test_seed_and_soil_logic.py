import pytest
import httpx
from httpx import Response, ASGITransport
from unittest.mock import AsyncMock

from backend.main import app  # Import your FastAPI app
from backend.core.seed_and_soil_logic import run_seed_and_soil_campaign

# Define mock URLs for external services
COMMAND_CENTER_URL = "https://crispro--command-center-v10-commandcenter-api.modal.run"
MOCK_TWIN_API_URL = f"{COMMAND_CENTER_URL}/digital_twin/create_from_variant"
MOCK_STRESS_API_URL = f"{COMMAND_CENTER_URL}/stress_simulation/run"
MOCK_REPORT_API_URL = f"{COMMAND_CENTER_URL}/threat_report/generate"

@pytest.mark.asyncio
async def test_run_seed_and_soil_campaign_success(httpx_mock):
    """
    Tests the successful execution of the entire Seed & Soil campaign,
    mocking all three external API calls.
    """
    # --- Mock Responses ---
    # 1. Mock Digital Twin Creation
    httpx_mock.add_response(
        method="POST",
        url=MOCK_TWIN_API_URL,
        json={"digital_twin_id": "twin-123", "status": "complete"},
        status_code=200,
    )

    # 2. Mock Stress Simulation
    httpx_mock.add_response(
        method="POST",
        url=MOCK_STRESS_API_URL,
        json={"simulation_id": "sim-456", "status": "complete", "results": {"growth_rate": 0.8}},
        status_code=200,
    )

    # 3. Mock Threat Report Generation
    httpx_mock.add_response(
        method="POST",
        url=MOCK_REPORT_API_URL,
        json={"report_id": "report-789", "status": "complete", "report_url": "http://example.com/report.pdf"},
        status_code=200,
    )

    # --- Input Data ---
    gene = "BRAF"
    variant = "V600E"
    disease_context = "Melanoma"
    primary_tissue = "Skin"
    metastatic_site = "Lung"

    # --- Execute the Campaign ---
    # This will fail initially because the function is empty, which is what we want.
    # This is Test-Driven Development. We write the test first.
    final_report = await run_seed_and_soil_campaign(
        gene, variant, disease_context, primary_tissue, metastatic_site
    )

    # --- Assertions ---
    # For now, we'll just assert that it doesn't return None.
    # We will build out the real assertions as we implement the logic.
    assert final_report is not None
    assert final_report["digital_twin"]["digital_twin_id"] == "twin-123"
    assert final_report["stress_simulation"]["simulation_id"] == "sim-456"
    assert final_report["threat_report"]["report_id"] == "report-789"
    assert "error" not in final_report

# We will add more tests for failure cases later.
