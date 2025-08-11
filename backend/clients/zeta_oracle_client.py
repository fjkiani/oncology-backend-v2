import os
import random
from dotenv import load_dotenv
import httpx

# The live URL for our deployed Zeta Oracle service.
ZETA_ORACLE_URL = os.getenv("ZETA_ORACLE_URL", "https://crispro--zeta-oracle-v2-zetaoracle-api.modal.run")

class ZetaOracleClient:
    """
    A client for interacting with the deployed ZetaOracle Modal service via HTTP.
    This is the modern, robust, doctrine-compliant implementation.
    """
    def __init__(self):
        load_dotenv()
        self.tier = os.getenv("ORACLE_TIER", "production")

    def _invoke_oracle(self, action: str, params: dict) -> dict:
        """Helper to make a direct HTTP call to the oracle's invoke endpoint."""
        payload = {"action": action, "params": params}
        url = f"{ZETA_ORACLE_URL}/invoke"
        print(f"CLIENT: Invoking Zeta Oracle via HTTP POST to {url} with action: {action}")
        try:
            # Increased timeout to handle Modal cold starts for the H100 GPU.
            with httpx.Client(timeout=300.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            print(f"CLIENT ERROR: HTTP Status Error calling Zeta Oracle: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            print(f"CLIENT ERROR: Failed to invoke Zeta Oracle: {e}")
            raise

    def calculate_zeta_score(self, baseline_sequence: str, perturbed_sequence: str) -> dict:
        """
        Invokes the remote Zeta Oracle to calculate the log-likelihood difference.
        This has been upgraded to handle the new confidence score.
        """
        try:
            result = self._invoke_oracle(
                action="score",
                params={"reference_sequence": baseline_sequence, "alternate_sequence": perturbed_sequence}
            )
            print(f"CLIENT: Received score from Zeta Oracle: {result.get('zeta_score')}")
            
            zeta_score = result.get('zeta_score', 0)
            confidence = result.get('confidence', 0.0) # NEW: Get confidence score
            
            return {
                "zeta_score": zeta_score,
                "confidence": confidence, # NEW: Pass confidence through
                "verdict": result.get('interpretation', "ERROR"),
                "commentary": f"Live Zeta Oracle calculation complete. Status: {result.get('status', 'error')}"
            }
        except Exception as e:
            return {
                "zeta_score": 0, 
                "confidence": 0.0, # NEW: Default confidence on error
                "verdict": "ERROR", 
                "commentary": "Failed to connect to the live Zeta Oracle service."
            }

    def generate_inhibitor(self, target_gene_sequence: str, design_goal: str) -> dict:
        """
        Invokes the remote Zeta Forge to generate a superior inhibitor.
        """
        try:
            # Step 1: Generate a new sequence
            generation_result = self._invoke_oracle(
                action="generate",
                params={"prompt": f"Generate a protein sequence that strongly inhibits the function of a protein with the following sequence: {target_gene_sequence[:100]}...", "n_tokens": 120}
            )
            new_protein_sequence = generation_result.get("sequence")
            if not new_protein_sequence:
                raise Exception("Generation failed to return a sequence.")

            # Step 2: Score the new sequence against the original for impact
            score_result = self.calculate_zeta_score(
                baseline_sequence=target_gene_sequence,
                perturbed_sequence=new_protein_sequence
            )

            print(f"CLIENT: Received new protein and score from Zeta Forge: {score_result['zeta_score']}")
            return {
                "new_protein_name": "Zeta-Forged-Inhibitor-01",
                "new_protein_sequence": new_protein_sequence,
                "new_zeta_score": score_result['zeta_score'],
                "commentary": "Live generation and scoring complete via Zeta Forge."
            }
        except Exception as e:
            # FIX: Return a compliant error object to prevent ResponseValidationError
            return {
                "new_protein_name": "ERROR", 
                "new_protein_sequence": "N/A",
                "new_zeta_score": 0, 
                "commentary": "Failed to connect to the live Zeta Forge service."
            } 