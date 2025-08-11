import os
import re
import logging
from typing import Dict, Any
from Bio.Seq import Seq # Assuming Bio.Seq is available for translation if needed

logger = logging.getLogger(__name__)

# Amino Acid codes from DigitalTwinBase
AA_CODES = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q',
    'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K',
    'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W',
    'Tyr': 'Y', 'Val': 'V'
}

def get_canonical_sequence(gene_symbol: str) -> str:
    """Retrieves the canonical protein sequence from a local FASTA file."""
    # Adjust path to be relative to the project root, assuming data is in crispr-assistant-main/data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    fasta_path = os.path.join(project_root, 'data', 'reference', f"{gene_symbol}.fasta")
    
    if not os.path.exists(fasta_path):
        logger.error(f"FASTA file for gene '{gene_symbol}' not found at {fasta_path}")
        raise FileNotFoundError(f"FASTA file for gene '{gene_symbol}' not found at {fasta_path}")
    
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
    return "".join([line.strip() for line in lines if not line.startswith('>')])

def apply_hgvs_mutation(sequence: str, hgvsp: str) -> str:
    """Applies a missense mutation described in HGVS protein notation."""
    match = re.match(r"p\\.\(?(?P<ref>[A-Z][a-z]{2}|[A-Z])(?P<pos>\\d+)(?P<alt>[A-Z][a-z]{2}|[A-Z])\\)?", hgvsp)
    if not match:
        raise ValueError(f"Invalid or unsupported HGVS notation: {hgvsp}")
    
    groups = match.groupdict()
    ref_aa, pos_str, alt_aa = groups['ref'], groups['pos'], groups['alt']
    position = int(pos_str) - 1
    
    ref_aa_one = AA_CODES.get(ref_aa) if len(ref_aa) == 3 else ref_aa
    alt_aa_one = AA_CODES.get(alt_aa) if len(alt_aa) == 3 else alt_aa
    
    if not ref_aa_one or not alt_aa_one:
         raise ValueError(f"Invalid amino acid code in HGVS string: {hgvsp}")
    
    if not position < len(sequence):
        raise ValueError(f"Position {position+1} is out of bounds for sequence of length {len(sequence)}")
    
    if sequence[position] != ref_aa_one:
        raise ValueError(f"Reference AA at position {position+1} is '{sequence[position]}', but HGVS string specifies '{ref_aa_one}'.")
    
    mutated_sequence = list(sequence)
    mutated_sequence[position] = alt_aa_one
    return "".join(mutated_sequence) 