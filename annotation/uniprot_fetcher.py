"""
annotation/uniprot_fetcher.py
══════════════════════════════
Fetch per-sequence annotations from UniProt REST API v2 and InterPro.

What is fetched
───────────────
  UniProt (https://rest.uniprot.org):
    - organism name + NCBI taxonomy ID
    - gene name
    - protein name (recommended name)
    - GO terms (biological process, molecular function, cellular component)
    - reviewed status (Swiss-Prot vs. TrEMBL)

  InterPro (https://www.ebi.ac.uk/interpro/api):
    - Domain boundary list [{start, end, interpro_id, name}]

Design decisions
────────────────
  1. All results are cached to output/annotation_cache.json.
     Subsequent runs with the same IDs skip API calls entirely.

  2. Rate limiting: UniProt allows ~10 req/s; we use 0.12 s delay (8/s)
     to stay safely under. InterPro is stricter; 0.5 s delay used.

  3. ID detection: we try to extract a UniProt accession from the FASTA
     header using two patterns:
       • NCBI/SwissProt format:  >sp|P12345|NAME  or >tr|P12345|...
       • Bare accession:         >P12345 or >P12345.1
       • GenBank / DDBJ / NCBI: treated as non-UniProt; organism set to
         "Unknown" and domains left empty (no InterPro lookup).

  4. Annotations for sequences without resolvable UniProt IDs are left
     as empty stubs so the rest of the pipeline can proceed unchanged.

References
──────────
UniProt REST API v2: https://rest.uniprot.org/docs/
InterPro API:        https://www.ebi.ac.uk/interpro/api/
"""

import logging
log = logging.getLogger(__name__)


import os
import re
import json
import time
import requests
from tqdm import tqdm

CACHE_PATH    = "output/annotation_cache.json"
UNIPROT_URL   = "https://rest.uniprot.org/uniprotkb/{acc}.json"
INTERPRO_URL  = "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{acc}/?format=json"

UNIPROT_DELAY  = 0.12   # seconds between UniProt requests
INTERPRO_DELAY = 0.5    # seconds between InterPro requests

# Regex patterns to extract UniProt accession from FASTA header IDs
_SP_TR      = re.compile(r"^(?:sp|tr)\|([A-Z][A-Z0-9]{5})\|", re.I)
_BARE_ACCN  = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(\.\d+)?$", re.I)


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as fh:
            return json.load(fh)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as fh:
        json.dump(cache, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# UniProt accession extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_uniprot_accession(seq_id: str) -> str | None:
    """
    Try to extract a UniProt accession from a FASTA header ID string.

    Handles:
      sp|P12345|NAME  →  P12345
      tr|Q8N123|NAME  →  Q8N123
      P12345          →  P12345
      P12345.1        →  P12345
      AAA24053        →  None  (GenBank-style — not a UniProt accession)

    Returns the accession string, or None if not detectable.
    """
    # Try sp|..|.. or tr|..|.. format
    m = _SP_TR.match(seq_id)
    if m:
        return m.group(1).upper()

    # Try bare UniProt accession (with optional .version suffix)
    m = _BARE_ACCN.match(seq_id)
    if m:
        return m.group(1).upper()

    return None


# ─────────────────────────────────────────────────────────────────────────────
# UniProt REST fetcher
# ─────────────────────────────────────────────────────────────────────────────

def _empty_annotation() -> dict:
    """Return a stub annotation dict for sequences with no UniProt data."""
    return {
        "accession":   None,
        "protein_name": "Unknown",
        "gene_name":   "Unknown",
        "organism":    "Unknown",
        "taxon_id":    None,
        "reviewed":    False,
        "go_terms":    [],
        "domains":     [],
    }


def fetch_uniprot_annotation(accession: str) -> dict:
    """
    Fetch annotation for a single UniProt accession via the REST API.

    Returns a dict with keys:
      accession, protein_name, gene_name, organism, taxon_id,
      reviewed, go_terms (list of GO acc strings), domains (list — from InterPro step).

    On any network / parse error, returns an empty annotation stub.
    """
    url = UNIPROT_URL.format(acc=accession)
    try:
        r = requests.get(url, timeout=20,
                         headers={"Accept": "application/json"})
        if r.status_code == 404:
            return _empty_annotation()
        r.raise_for_status()
        data = r.json()
    except Exception:
        return _empty_annotation()

    ann = _empty_annotation()
    ann["accession"] = accession
    ann["reviewed"]  = data.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)"

    # Organism
    org = data.get("organism", {})
    ann["organism"] = org.get("scientificName", "Unknown")
    ann["taxon_id"] = org.get("taxonId", None)

    # Gene name
    genes = data.get("genes", [])
    if genes:
        gene_names = genes[0].get("geneName", {})
        ann["gene_name"] = gene_names.get("value", "Unknown")

    # Protein name
    desc = data.get("proteinDescription", {})
    rec  = desc.get("recommendedName", {})
    full = rec.get("fullName", {})
    ann["protein_name"] = full.get("value", "Unknown")

    # GO terms
    go_terms = []
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "GO":
            go_id = ref.get("id")
            if go_id:
                go_terms.append(go_id)
    ann["go_terms"] = go_terms

    return ann


# ─────────────────────────────────────────────────────────────────────────────
# InterPro domain fetcher
# ─────────────────────────────────────────────────────────────────────────────

def fetch_interpro_domains(accession: str) -> list[dict]:
    """
    Fetch InterPro domain boundary annotations for a UniProt accession.

    Returns list of dicts:
      [{'interpro_id': 'IPR000001', 'name': 'Kringle', 'start': 10, 'end': 85}, ...]

    On any error returns empty list.
    """
    url = INTERPRO_URL.format(acc=accession)
    try:
        r = requests.get(url, timeout=30,
                         headers={"Accept": "application/json"})
        if r.status_code in (404, 204):
            return []
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    domains: list[dict] = []
    for entry in data.get("results", []):
        meta    = entry.get("metadata", {})
        ipr_id  = meta.get("accession", "")
        name    = meta.get("name", "")
        for prot in entry.get("proteins", []):
            for loc in prot.get("entry_protein_locations", []):
                for frag in loc.get("fragments", []):
                    start = frag.get("start")
                    end   = frag.get("end")
                    if start is not None and end is not None:
                        domains.append({
                            "interpro_id": ipr_id,
                            "name":        name,
                            "start":       int(start),
                            "end":         int(end),
                        })

    return domains


# ─────────────────────────────────────────────────────────────────────────────
# Batch annotation driver
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_annotations(seq_ids: list[str],
                           fetch_interpro: bool = True,
                           interpro_delay: float = INTERPRO_DELAY,
                           uniprot_delay: float  = UNIPROT_DELAY
                           ) -> dict[str, dict]:
    """
    Fetch UniProt + optional InterPro annotations for all sequence IDs.

    Workflow per sequence:
      1. Try to parse a UniProt accession from seq_id.
      2. If found and not cached: fetch UniProt annotation + InterPro domains.
      3. Cache result. Subsequent calls reuse cache without network round-trips.
      4. If no UniProt accession detected: store empty annotation stub.

    Parameters
    ----------
    seq_ids         : list of sequence IDs (from FASTA headers)
    fetch_interpro  : also fetch InterPro domain boundaries (default True)
    interpro_delay  : seconds between InterPro requests
    uniprot_delay   : seconds between UniProt requests

    Returns
    -------
    dict[seq_id -> annotation_dict]
    """
    cache = _load_cache()
    annotations: dict[str, dict] = {}

    n_fetched_uniprot   = 0
    n_fetched_interpro  = 0
    n_cached            = 0
    n_no_accession      = 0

    log.info(f"\n  Fetching annotations for {len(seq_ids):,} sequences ...")
    for seq_id in tqdm(seq_ids, desc="  Annotating", unit="seq", ncols=80):
        accession = extract_uniprot_accession(seq_id)

        if accession is None:
            ann = _empty_annotation()
            ann["_source_id"] = seq_id
            annotations[seq_id] = ann
            n_no_accession += 1
            continue

        cache_key = f"uniprot:{accession}"

        if cache_key in cache:
            ann = cache[cache_key]
            n_cached += 1
        else:
            ann = fetch_uniprot_annotation(accession)
            time.sleep(uniprot_delay)
            n_fetched_uniprot += 1

            if fetch_interpro and accession is not None:
                domains = fetch_interpro_domains(accession)
                time.sleep(interpro_delay)
                ann["domains"] = domains
                n_fetched_interpro += 1

            cache[cache_key] = ann

        ann["_source_id"] = seq_id
        annotations[seq_id] = ann

    _save_cache(cache)

    log.info("  Annotation summary:")
    log.info(f"    UniProt fetched  : {n_fetched_uniprot:>5}")
    log.info(f"    InterPro fetched : {n_fetched_interpro:>5}")
    log.info(f"    From cache       : {n_cached:>5}")
    log.info(f"    No accession     : {n_no_accession:>5}")
    log.info(f"    Saved cache      → {CACHE_PATH}")
    return annotations
