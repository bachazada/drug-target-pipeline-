#!/usr/bin/env python3
"""
Week 2 - Biological Filtering (fixed)
Handles Windows WSL paths with spaces in directory names.
"""

import os
import subprocess
import requests
import time
from pathlib import Path
from datetime import datetime
from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def save_file(content, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    log(f"Saved: {path}")


MIN_LEN = 50
MAX_LEN = 2000
BLAST_EVALUE = 1e-5
BLAST_IDENTITY = 30


# ── Step 1: Load proteome ──────────────────────────────────────────────────────
def load_proteome(fasta_path="data/proteome.fasta"):
    log(f"Loading proteome from {fasta_path}...")
    records = list(SeqIO.parse(fasta_path, "fasta"))
    log(f"Loaded {len(records)} proteins")
    return records


# ── Step 2: Length filter ──────────────────────────────────────────────────────
def filter_by_length(records):
    log(f"\nStep 1: Length filter ({MIN_LEN}-{MAX_LEN} aa)...")
    kept = [r for r in records if MIN_LEN <= len(r.seq) <= MAX_LEN]
    log(f"  Removed: {len(records) - len(kept)} proteins")
    log(f"  Kept:    {len(kept)} proteins")
    return kept


# ── Step 3: Essential gene filter ─────────────────────────────────────────────
def filter_essential(records, essential_path="data/essential_genes.txt"):
    log(f"\nStep 2: Essential gene filter...")
    with open(essential_path) as f:
        essential = set(
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        )
    log(f"  Essential gene list: {len(essential)} genes")

    essential_records = []
    for rec in records:
        gene = ""
        if "GN=" in rec.description:
            gene = rec.description.split("GN=")[1].split(" ")[0].lower()
        if gene and gene in essential:
            essential_records.append(rec)

    log(f"  Essential proteins matched: {len(essential_records)}")
    log(f"  Non-essential removed:      {len(records) - len(essential_records)}")
    return essential_records


# ── Step 4: Human homolog removal ─────────────────────────────────────────────
def remove_human_homologs(records):
    log(f"\nStep 3: Human homolog removal...")

    blast_ok = _check_blast()

    if blast_ok:
        log("  BLAST found — attempting local alignment")
        result = _blast_filter(records)
        if result is not None:
            return result
        else:
            log("  BLAST failed — switching to keyword fallback")
            return _keyword_filter(records)
    else:
        log("  BLAST not found — using keyword fallback")
        return _keyword_filter(records)


def _check_blast():
    try:
        r = subprocess.run(["blastp", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _blast_filter(records):
    """
    Runs makeblastdb + blastp.
    Uses absolute resolved paths to handle spaces in directory names (WSL/Windows).
    Returns None on any failure so caller can fall back to keyword filter.
    """
    cwd = Path.cwd().resolve()
    human_fasta = cwd / "data" / "human_proteome.fasta"
    human_db    = cwd / "data" / "human_proteome_db"
    query_path  = cwd / "data" / "query_targets.fasta"
    blast_out   = cwd / "results" / "blast_human_hits.txt"

    # Download human proteome if needed
    if not human_fasta.exists():
        log("  Downloading human proteome from UniProt...")
        try:
            url = (
                "https://rest.uniprot.org/uniprotkb/stream"
                "?format=fasta"
                "&query=proteome:UP000005640+AND+reviewed:true"
            )
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            human_fasta.write_text(response.text)
            n = response.text.count(">")
            log(f"  Downloaded {n} human proteins")
        except Exception as e:
            log(f"  Download failed: {e}")
            return None

    # Build BLAST database — use str() on all Path objects
    db_index = Path(str(human_db) + ".pin")
    if not db_index.exists():
        log("  Building BLAST database...")
        try:
            result = subprocess.run(
                [
                    "makeblastdb",
                    "-in",     str(human_fasta),
                    "-dbtype", "prot",
                    "-out",    str(human_db),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                log(f"  makeblastdb stderr: {result.stderr.strip()[:400]}")
                return None
            log("  BLAST database built successfully")
        except Exception as e:
            log(f"  makeblastdb exception: {e}")
            return None

    # Write query FASTA
    SeqIO.write(records, str(query_path), "fasta")

    # Run BLAST
    blast_out.parent.mkdir(parents=True, exist_ok=True)
    log(f"  Running BLAST ({len(records)} queries vs human proteome)...")
    log(f"  This may take 2-5 minutes...")
    try:
        result = subprocess.run(
            [
                "blastp",
                "-query",           str(query_path),
                "-db",              str(human_db),
                "-out",             str(blast_out),
                "-outfmt",          "6 qseqid sseqid pident evalue bitscore",
                "-evalue",          str(BLAST_EVALUE),
                "-num_threads",     "4",
                "-max_target_seqs", "1",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            log(f"  blastp stderr: {result.stderr.strip()[:400]}")
            return None
    except Exception as e:
        log(f"  blastp exception: {e}")
        return None

    # Parse BLAST hits
    human_similar = set()
    if blast_out.exists() and blast_out.stat().st_size > 0:
        blast_df = pd.read_csv(
            str(blast_out), sep="\t", header=None,
            names=["qid", "sid", "pident", "evalue", "bitscore"]
        )
        hits = blast_df[blast_df["pident"] >= BLAST_IDENTITY]
        human_similar = set(hits["qid"].tolist())
        log(f"  Human homologs found: {len(human_similar)}")

    filtered = [r for r in records if r.id not in human_similar]
    log(f"  Removed: {len(human_similar)} human homologs")
    log(f"  Kept:    {len(filtered)} non-human-homolog targets")
    return filtered


def _keyword_filter(records):
    """
    Keyword-based fallback — removes protein families conserved in humans.
    """
    log("  Applying keyword-based human homolog filter...")
    conserved = [
        "atp synthase", "rna polymerase", "dna polymerase",
        "elongation factor", "chaperonin", "heat shock",
        "superoxide dismutase", "thioredoxin", "glutaredoxin",
    ]
    kept, removed = [], 0
    for rec in records:
        desc = rec.description.lower()
        if any(kw in desc for kw in conserved):
            removed += 1
        else:
            kept.append(rec)
    log(f"  Removed (conserved families): {removed}")
    log(f"  Kept: {len(kept)}")
    log("  NOTE: for the full BLAST filter, run:")
    log("        conda install -c bioconda blast")
    return kept


# ── Step 5: Save results ───────────────────────────────────────────────────────
def save_results(records, out_path="results/filtered_targets.fasta"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, out_path, "fasta")
    kb = os.path.getsize(out_path) / 1024
    log(f"\nFiltered targets saved: {out_path} ({kb:.1f} KB)")
    log(f"Total candidate drug targets: {len(records)}")


# ── Step 6: Funnel plot ────────────────────────────────────────────────────────
def plot_funnel(counts, labels):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#4A90D9", "#5BAD8F", "#E05C3A", "#F5A623"]

    bars = axes[0].barh(
        range(len(counts)), counts,
        color=colors[:len(counts)], edgecolor="white", linewidth=0.8
    )
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=11)
    axes[0].set_xlabel("Number of proteins")
    axes[0].set_title("Filtering funnel", fontsize=13)
    axes[0].invert_yaxis()
    for bar, count in zip(bars, counts):
        axes[0].text(
            bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
            f"{count:,}", va="center", fontsize=10, fontweight="bold"
        )

    pcts = [c / counts[0] * 100 for c in counts]
    axes[1].plot(range(len(pcts)), pcts, "o-",
                 color="#4A90D9", linewidth=2, markersize=8)
    for i, (pct, label) in enumerate(zip(pcts, labels)):
        axes[1].annotate(
            f"{pct:.1f}%", (i, pct),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=10
        )
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    axes[1].set_ylabel("% of original proteome retained")
    axes[1].set_title("Retention at each step", fontsize=13)
    axes[1].set_ylim(0, 115)

    plt.suptitle(
        "K. pneumoniae — Drug Target Filtering",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    Path("visualizations").mkdir(exist_ok=True)
    plt.savefig("visualizations/filtering_funnel.png", dpi=150, bbox_inches="tight")
    plt.show()
    log("Funnel plot saved: visualizations/filtering_funnel.png")


# ── Step 7: Write report ───────────────────────────────────────────────────────
def write_report(counts, labels, final_records):
    gene_names = []
    for rec in final_records[:20]:
        if "GN=" in rec.description:
            gene_names.append(rec.description.split("GN=")[1].split(" ")[0])

    rows = ""
    for i, (label, count) in enumerate(zip(labels, counts)):
        removed = (counts[i - 1] - count) if i > 0 else 0
        rows += f"| {label} | {count:,} | {removed:,} |\n"

    report = f"""# Week 2 - Filtering Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Filter summary
| Step | Proteins remaining | Removed |
|------|--------------------|---------|
{rows}
## Final candidate set
- **{len(final_records)} drug target candidates**
- Retention: {len(final_records)/counts[0]*100:.1f}% of original proteome

## Top candidates (first 20 by gene name)
{', '.join(gene_names)}

## Next step
    python scripts/03_feature_engineering.py
"""
    save_file(report, "results/filtering_report.md")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print(" Week 2 - Biological Filtering")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55 + "\n")

    counts, labels = [], []

    records = load_proteome()
    counts.append(len(records))
    labels.append("Full proteome")

    records = filter_by_length(records)
    counts.append(len(records))
    labels.append("Length filter")

    records = filter_essential(records)
    counts.append(len(records))
    labels.append("Essential genes")

    records = remove_human_homologs(records)
    counts.append(len(records))
    labels.append("Human homologs removed")

    save_results(records)
    plot_funnel(counts, labels)
    write_report(counts, labels, records)

    print("\n" + "=" * 55)
    print(f" Week 2 complete!")
    print(f" Drug target candidates: {len(records)}")
    print(f" Next: python scripts/03_feature_engineering.py")
    print("=" * 55)


if __name__ == "__main__":
    main()