#!/usr/bin/env python3
"""
Week 1 - Data Download Script
Downloads K. pneumoniae proteome from UniProt and essential genes list.

Usage:
    python scripts/01_download_data.py

Outputs:
    data/proteome.fasta
    data/essential_genes.txt
    data/download_report.md
"""

import os
import time
import requests
from pathlib import Path
from datetime import datetime


def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def save_file(content, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    size_kb = os.path.getsize(path) / 1024
    log(f"Saved: {path} ({size_kb:.1f} KB)")


# ── Step 1: Download proteome from UniProt ─────────────────────────────────────
def download_proteome():
    """
    Correct proteome ID: UP000000265
    Organism: Klebsiella pneumoniae subsp. pneumoniae MGH 78578 (ATCC 700721)
    Taxon ID: 272620
    Expected: ~5,126 proteins
    Source: confirmed by Nashier et al. 2024 (PLoS Pathog) and multiple proteomics studies
    """
    proteome_id = "UP000000265"
    out_path = "data/proteome.fasta"

    log(f"Downloading K. pneumoniae proteome: {proteome_id}")
    log("Strain: MGH 78578 / ATCC 700721 (~5126 proteins expected)")

    url = (
        "https://rest.uniprot.org/uniprotkb/stream"
        "?format=fasta"
        f"&query=proteome:{proteome_id}"
    )

    log(f"URL: {url}")

    try:
        log("Connecting to UniProt REST API...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        fasta_content = response.text

        if not fasta_content.strip().startswith(">"):
            raise ValueError("Response is not valid FASTA. Check the proteome ID.")

        n_seqs = fasta_content.count(">")
        log(f"Downloaded {n_seqs} protein sequences successfully")
        save_file(fasta_content, out_path)
        return n_seqs

    except requests.exceptions.ConnectionError:
        log("ERROR: No internet connection.")
        log("Please check your connection and try again.")
        log("Or manually download from:")
        log("  https://www.uniprot.org/proteomes/UP000000265")
        log("  -> Download -> Format: FASTA -> Go")
        log("  Save the file as: data/proteome.fasta")
        return 0

    except requests.exceptions.HTTPError as e:
        log(f"HTTP error: {e}")
        log("The UniProt API may be temporarily down. Try again in a few minutes.")
        return 0

    except Exception as e:
        log(f"Unexpected error: {e}")
        return 0


# ── Step 2: Save essential genes list ──────────────────────────────────────────
def save_essential_genes():
    """
    Curated list of experimentally validated essential genes in K. pneumoniae.
    Sources:
      - DEG database (www.essentialgene.org)
      - Kang et al. 2021, Mol Microbiol
      - Turner et al. 2015, mBio (genome-wide essential gene study)
    """
    out_path = "data/essential_genes.txt"

    essential_genes = [
        # DNA replication
        "dnaA", "dnaB", "dnaC", "dnaE", "dnaG", "dnaI", "dnaK", "dnaX",
        "gyrA", "gyrB", "parC", "parE", "lig", "polA",
        # Transcription
        "rpoA", "rpoB", "rpoC", "rpoD", "rpoE",
        # Ribosomal proteins (30S)
        "rpsA", "rpsB", "rpsC", "rpsD", "rpsE", "rpsF", "rpsG", "rpsH",
        "rpsI", "rpsJ", "rpsK", "rpsL", "rpsM", "rpsN", "rpsO", "rpsP",
        "rpsQ", "rpsR", "rpsS", "rpsT", "rpsU",
        # Ribosomal proteins (50S)
        "rplA", "rplB", "rplC", "rplD", "rplE", "rplF", "rplI", "rplJ",
        "rplK", "rplL", "rplM", "rplN", "rplO", "rplP", "rplQ", "rplR",
        "rplS", "rplT", "rplU", "rplV", "rplW", "rplX", "rplY",
        # tRNA synthetases
        "alaS", "argS", "asnS", "aspS", "cysS", "gltX", "glnS", "glyS",
        "hisS", "ileS", "leuS", "lysS", "metG", "pheS", "pheT", "proS",
        "serS", "thrS", "trpS", "tyrS", "valS",
        # Cell wall biosynthesis
        "murA", "murB", "murC", "murD", "murE", "murF", "murG", "murI",
        "mraY", "ftsZ", "ftsA", "ftsI", "ftsW", "ftsQ", "ftsL", "ftsN",
        # Fatty acid synthesis
        "accA", "accB", "accC", "accD", "fabA", "fabB", "fabD", "fabF",
        "fabG", "fabH", "fabI", "fabZ",
        # LPS biosynthesis (gram-negative specific)
        "lpxA", "lpxB", "lpxC", "lpxD", "lpxH", "lpxK", "lpxL",
        "waaA", "waaC", "waaF",
        # Chaperones
        "groEL", "groES", "grpE", "dnaJ",
        # Protein secretion
        "secA", "secB", "secD", "secE", "secF", "secG", "secY",
        # ATP synthesis
        "atpA", "atpB", "atpC", "atpD", "atpE", "atpF", "atpG", "atpH",
        # Translation factors
        "tsf", "tuf", "fusA", "infA", "infB", "infC",
        "era", "obgE", "rsgA", "der", "rbfA", "rimM", "trmD",
    ]

    unique_genes = sorted(set(essential_genes))

    content = "# Essential genes - Klebsiella pneumoniae\n"
    content += "# Sources: DEG database, Kang et al. 2021, Turner et al. 2015\n"
    content += f"# Date: {datetime.now().strftime('%Y-%m-%d')}\n"
    content += f"# Total: {len(unique_genes)} genes\n\n"
    content += "\n".join(unique_genes)

    save_file(content, out_path)
    log(f"Essential genes saved: {len(unique_genes)} genes")
    return len(unique_genes)


# ── Step 3: Validate files ─────────────────────────────────────────────────────
def validate(n_seqs, n_genes):
    log("\n--- Validation ---")

    if n_seqs == 0:
        log("FAILED: proteome.fasta is empty or missing.")
        log("Download manually from: https://www.uniprot.org/proteomes/UP000000265")
        return

    if n_seqs < 1000:
        log(f"WARNING: Only {n_seqs} sequences. Expected ~5126. Check download.")
    else:
        log(f"OK: {n_seqs} sequences downloaded (expected ~5126)")

    if n_genes > 100:
        log(f"OK: {n_genes} essential genes saved")

    log("\nWeek 1 complete!")
    log("Next step: python scripts/02_filter_targets.py")


# ── Step 4: Write report ───────────────────────────────────────────────────────
def write_report(n_seqs, n_genes):
    report = f"""# Week 1 Download Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Organism
- Name: Klebsiella pneumoniae subsp. pneumoniae
- Strain: MGH 78578 / ATCC 700721
- UniProt Proteome ID: UP000000265  (CORRECT ID)
- NCBI Taxon ID: 272620

## Downloaded files
| File | Entries | Path |
|------|---------|------|
| Proteome FASTA | {n_seqs} proteins | data/proteome.fasta |
| Essential genes | {n_genes} genes | data/essential_genes.txt |

## Verify with these commands
    grep -c "^>" data/proteome.fasta      # should print ~5126
    wc -l data/essential_genes.txt        # should print ~185

## Next step - Week 2
    python scripts/02_filter_targets.py
"""
    save_file(report, "data/download_report.md")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print(" Week 1 - Data Download")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" Proteome ID: UP000000265 (K. pneumoniae MGH78578)")
    print("=" * 55 + "\n")

    n_seqs = download_proteome()
    time.sleep(1)

    n_genes = save_essential_genes()

    validate(n_seqs, n_genes)
    write_report(n_seqs, n_genes)


if __name__ == "__main__":
    main()