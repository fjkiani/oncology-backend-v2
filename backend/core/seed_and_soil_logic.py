import httpx
import asyncio
import os
from typing import Dict, Any

# --- Constants ---
COMMAND_CENTER_URL = os.environ.get("COMMAND_CENTER_URL", "https://crispro--command-center-v11-path-lock-web-app-dev.modal.run")
ASSESS_THREAT_ENDPOINT = f"{COMMAND_CENTER_URL}/workflow/assess_threat"

# --- Mock Implementations for Seed & Soil stages ---
# These functions will eventually contain real logic or calls to other services.
# For now, they return mock data to match the Streamlit app's behavior.

async def _run_gene_essentiality_prediction(tissue: str) -> Dict[str, Any]:
    """Mocks the creation of a digital twin for baseline tissue vulnerability."""
    print(f"🧬 [MOCK] Creating digital twin of healthy {tissue} tissue...")
    await asyncio.sleep(2)  # Simulate processing time
    # This mock data is based on the structure observed in the Streamlit app.
    return {
        "essentiality_profile": [
            {"target_gene": "POLR2A", "essentiality_score": -0.98},
            {"target_gene": "RPL8", "essentiality_score": -0.95},
            {"target_gene": "MYC", "essentiality_score": -0.92},
            # ... add more mock genes if needed
        ],
        "status": "complete"
    }

async def _simulate_environmental_stress(baseline_profile: Dict[str, Any], environmental_factor: str) -> Dict[str, Any]:
    """Mocks the simulation of a pathogenic mutation's effect on the tissue."""
    print(f"⚡ [MOCK] Simulating invasion of '{environmental_factor}'...")
    await asyncio.sleep(3)
    # Mock a "stressed" profile, showing slightly different scores
    return {
        "essentiality_profile": [
            {"target_gene": "POLR2A", "essentiality_score": -0.99}, # More essential
            {"target_gene": "HIF1A", "essentiality_score": -0.97}, # Newly essential
            {"target_gene": "MYC", "essentiality_score": -0.91}, # Less essential
        ],
        "status": "complete"
    }

async def _run_threat_report_generation(baseline_profile: Dict[str, Any], stressed_profile: Dict[str, Any], environmental_factor: str) -> Dict[str, Any]:
    """Mocks the final analysis to find synthetic lethal vulnerabilities."""
    print("📊 [MOCK] Analyzing seed-soil compatibility and identifying vulnerabilities...")
    await asyncio.sleep(2)
    return {
        "environmental_factor": environmental_factor,
        "synthetic_lethalities_identified": 2,
        "synthetic_lethalities": [
            {"gene": "HIF1A", "vulnerability_increase": 0.85, "rationale": "Becomes critical under hypoxic stress induced by the mutation."},
            {"gene": "PARP1", "vulnerability_increase": 0.72, "rationale": "Exploits DNA repair dependency created by the mutation."}
        ],
        "status": "complete"
    }


async def run_seed_and_soil_campaign(gene: str, variant: str, disease_context: str, primary_tissue: str, metastatic_site: str) -> Dict[str, Any]:
    """
    Orchestrates the entire Seed & Soil campaign:
    1. Assesses the initial threat via the live CommandCenter.
    2. (Mock) Creates a digital twin of the metastatic site.
    3. (Mock) Runs a stress simulation on the twin.
    4. (Mock) Generates a final threat report.
    """
    final_report = {
        "campaign_inputs": {
            "gene": gene, "variant": variant, "disease_context": disease_context,
            "primary_tissue": primary_tissue, "metastatic_site": metastatic_site
        },
        "stages": {}
    }

    # --- Stage 1: Initial Threat Assessment (LIVE CALL) ---
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            print(f"⚔️ [LIVE] Assessing initial threat for {gene} {variant} via V11 CommandCenter...")
            payload = {"gene_symbol": gene, "protein_change": variant}
            response = await client.post(ASSESS_THREAT_ENDPOINT, json=payload)
            response.raise_for_status()
            threat_assessment_result = response.json()
            final_report["stages"]["initial_threat_assessment"] = threat_assessment_result
    except httpx.RequestError as e:
        error_detail = f"Failed to connect to CommandCenter: {e}"
        print(f"❌ {error_detail}")
        return {"error": "Initial threat assessment failed", "details": error_detail}
    except httpx.HTTPStatusError as e:
        error_detail = f"CommandCenter returned an error: {e.response.status_code} - {e.response.text}"
        print(f"❌ {error_detail}")
        return {"error": "Initial threat assessment failed", "details": error_detail}

    # --- Stage 2: Digital Twin Creation (MOCK) ---
    baseline_profile = await _run_gene_essentiality_prediction(metastatic_site)
    final_report["stages"]["digital_twin_creation"] = baseline_profile

    # --- Stage 3: Invasion Simulation (MOCK) ---
    environmental_factor = f"{gene} {variant}"
    stressed_profile = await _simulate_environmental_stress(baseline_profile, environmental_factor)
    final_report["stages"]["invasion_simulation"] = stressed_profile

    # --- Stage 4: Threat Report Generation (MOCK) ---
    threat_report = await _run_threat_report_generation(baseline_profile, stressed_profile, environmental_factor)
    final_report["stages"]["threat_report"] = threat_report

    return final_report
