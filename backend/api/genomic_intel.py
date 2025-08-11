from fastapi import APIRouter, HTTPException, Query
import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio

router = APIRouter()

# --- Pydantic Models ---

class GeneInfo(BaseModel):
    gene_id: str
    symbol: str
    name: str
    chromosome: str
    start_pos: int
    end_pos: int
    description: str

class ClinVarVariant(BaseModel):
    id: str
    title: str
    clinical_significance: str

class GeneDetailsResponse(BaseModel):
    gene_info: GeneInfo
    sequence: str
    clinvar_variants: List[ClinVarVariant]


# --- Helper functions to call external APIs ---

async def search_gene_id(client: httpx.AsyncClient, gene_symbol: str) -> str:
    """Search NCBI for the gene ID."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "gene",
        "term": f"{gene_symbol}[Gene Name] AND Homo sapiens[Organism]",
        "retmode": "json",
    }
    response = await client.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    if "esearchresult" in data and data["esearchresult"]["idlist"]:
        return data["esearchresult"]["idlist"][0]
    raise HTTPException(status_code=404, detail=f"Gene ID for {gene_symbol} not found.")

async def fetch_gene_summary(client: httpx.AsyncClient, gene_id: str) -> Dict[str, Any]:
    """Fetch detailed summary for a gene from NCBI."""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id={gene_id}&retmode=json"
    response = await client.get(url)
    response.raise_for_status()
    data = response.json()
    if "result" in data and gene_id in data["result"]:
        return data["result"][gene_id]
    raise HTTPException(status_code=404, detail=f"Details for Gene ID {gene_id} not found.")

async def fetch_gene_sequence(client: httpx.AsyncClient, chrom: str, start: int, end: int, genome: str) -> str:
    """Fetch gene sequence from UCSC Genome Browser API."""
    # UCSC API is 0-based for start
    api_url = f"https://api.genome.ucsc.edu/getData/sequence?genome={genome};chrom={chrom};start={start-1};end={end}"
    response = await client.get(api_url)
    response.raise_for_status()
    data = response.json()
    if data.get("dna"):
        return data["dna"].upper()
    raise HTTPException(status_code=404, detail=f"Sequence for {chrom}:{start}-{end} not found in UCSC ({genome}). Error: {data.get('error')}")

async def fetch_clinvar_variants(client: httpx.AsyncClient, chrom: str, start: int, end: int, genome: str) -> List[ClinVarVariant]:
    """Fetch ClinVar variants for a given genomic range."""
    chrom_formatted = chrom.replace("chr", "")
    # Use hg38 position for searching
    position_field = "chrpos38" if genome == "hg38" else "chrpos37"
    search_term = f"{chrom_formatted}[Chromosome] AND {start}:{end}[{position_field}]"
    
    # 1. ESearch to find variant IDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "clinvar",
        "term": search_term,
        "retmode": "json",
        "retmax": "50" # Limit to 50 variants for performance
    }
    search_response = await client.get(search_url, params=search_params)
    search_response.raise_for_status()
    search_data = search_response.json()

    variant_ids = search_data.get("esearchresult", {}).get("idlist", [])
    if not variant_ids:
        return []

    # 2. ESummary to get variant details
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    summary_params = {
        "db": "clinvar",
        "id": ",".join(variant_ids),
        "retmode": "json"
    }
    summary_response = await client.get(summary_url, params=summary_params)
    summary_response.raise_for_status()
    summary_data = summary_response.json().get("result", {})

    variants = []
    for var_id in variant_ids:
        if var_id in summary_data:
            var_details = summary_data[var_id]
            variants.append(ClinVarVariant(
                id=var_id,
                title=var_details.get("title", "N/A"),
                clinical_significance=var_details.get("clinical_significance", {}).get("description", "N/A")
            ))
    return variants

# --- API Endpoint ---

@router.get("/{gene_symbol}/details", response_model=GeneDetailsResponse)
async def get_gene_details(
    gene_symbol: str,
    genome: str = Query("hg38", description="Reference genome (e.g., hg38, hg19)")
):
    """
    Provides a consolidated report for a given gene symbol, including
    details, sequence, and ClinVar variants.
    """
    print(f"--- [DEBUG] Received request for gene_symbol: {gene_symbol} ---")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            gene_id = await search_gene_id(client, gene_symbol)
            print(f"--- [DEBUG] Found Gene ID: {gene_id} ---")
            gene_summary = await fetch_gene_summary(client, gene_id)

            if not gene_summary.get("genomicinfo"):
                raise HTTPException(status_code=404, detail="Genomic location info not found in NCBI summary.")
            
            genomic_info = gene_summary["genomicinfo"][0]
            chrom = genomic_info.get("chraccver")
            start_pos = int(genomic_info.get("chrstart"))
            end_pos = int(genomic_info.get("chrstop"))
            
            # Limit sequence fetch to 10,000 bp for performance
            fetch_end = min(end_pos, start_pos + 10000)

            # Fetch sequence and ClinVar variants concurrently
            sequence_task = fetch_gene_sequence(client, chrom, start_pos, fetch_end, genome)
            clinvar_task = fetch_clinvar_variants(client, chrom, start_pos, end_pos, genome)

            sequence, clinvar_variants = await asyncio.gather(sequence_task, clinvar_task)

            gene_info = GeneInfo(
                gene_id=gene_id,
                symbol=gene_summary.get("name", gene_symbol),
                name=gene_summary.get("description"),
                chromosome=chrom,
                start_pos=start_pos,
                end_pos=end_pos,
                description=gene_summary.get("summary", "")
            )

            return GeneDetailsResponse(
                gene_info=gene_info,
                sequence=sequence,
                clinvar_variants=clinvar_variants,
            )
        except HTTPException as e:
            print(f"--- [DEBUG] HTTPException: {e.detail} ---")
            raise e
        except Exception as e:
            print(f"--- [DEBUG] Unexpected Exception: {str(e)} ---")
            raise HTTPException(status_code=500, detail=str(e)) 