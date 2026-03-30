#!/usr/bin/env python3
"""
Week 2 - Biological Filtering (BLAST space-in-path fix)
Copies BLAST database to /tmp before running to avoid WSL path-with-spaces bug.
"""

import os
import shutil
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from Bio import SeqIO
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # ← add this line (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

MIN_LEN = 50
MAX_LEN = 2000
BLAST_EVALUE = 1e-5
BLAST_IDENTITY = 30

# Safe temp directory — guaranteed no spaces on any system
TMP_DIR = Path("/tmp/drug_target_blast")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def save_file(content, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    log(f"Saved: {path}")


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

    kept = []
    for rec in records:
        gene = ""
        if "GN=" in rec.description:
            gene = rec.description.split("GN=")[1].split(" ")[0].lower()
        if gene and gene in essential:
            kept.append(rec)

    log(f"  Essential proteins matched: {len(kept)}")
    log(f"  Non-essential removed:      {len(records) - len(kept)}")
    return kept


# ── Step 4: Human homolog removal ─────────────────────────────────────────────
def remove_human_homologs(records):
    log(f"\nStep 3: Human homolog removal...")

    if _check_blast():
        log("  BLAST found — using /tmp workaround for WSL path spaces")
        result = _blast_filter_tmp(records)
        if result is not None:
            return result
        log("  BLAST failed — falling back to keyword filter")

    return _keyword_filter(records)


def _check_blast():
    try:
        r = subprocess.run(["blastp", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _blast_filter_tmp(records):
    """
    Root fix for WSL 'Bacha Zada' space-in-path bug:
    All BLAST files live in /tmp/drug_target_blast/ which has no spaces.
    BLAST never sees the Windows path at all.
    """
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    human_fasta = TMP_DIR / "human_proteome.fasta"
    human_db    = TMP_DIR / "human_db"
    query_fasta = TMP_DIR / "query.fasta"
    blast_out   = TMP_DIR / "blast_out.tsv"

    # Download human proteome into /tmp
    if not human_fasta.exists():
        log("  Downloading human proteome to /tmp/ ...")
        try:
            url = (
                "https://rest.uniprot.org/uniprotkb/stream"
                "?format=fasta"
                "&query=proteome:UP000005640+AND+reviewed:true"
            )
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            human_fasta.write_text(r.text)
            log(f"  Downloaded {r.text.count('>')} human proteins")
        except Exception as e:
            log(f"  Download failed: {e}")
            return None
    else:
        log("  Human proteome already in /tmp — reusing")

    # Build BLAST database in /tmp
    if not Path(str(human_db) + ".pin").exists():
        log("  Building BLAST database in /tmp/ ...")
        r = subprocess.run(
            ["makeblastdb", "-in", str(human_fasta),
             "-dbtype", "prot", "-out", str(human_db)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            log(f"  makeblastdb failed: {r.stderr.strip()[:300]}")
            return None
        log("  Database ready")
    else:
        log("  BLAST database already built — reusing")

    # Write query FASTA to /tmp
    SeqIO.write(records, str(query_fasta), "fasta")
    log(f"  {len(records)} query proteins written to /tmp/")

    # Run blastp — all paths in /tmp, zero spaces
    log("  Running blastp... (2-5 min)")
    r = subprocess.run(
        [
            "blastp",
            "-query",           str(query_fasta),
            "-db",              str(human_db),
            "-out",             str(blast_out),
            "-outfmt",          "6 qseqid sseqid pident evalue bitscore",
            "-evalue",          str(BLAST_EVALUE),
            "-num_threads",     "4",
            "-max_target_seqs", "1",
        ],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        log(f"  blastp failed: {r.stderr.strip()[:300]}")
        return None

    # Copy result back to project results/
    project_out = Path("results/blast_human_hits.tsv")
    project_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(blast_out), str(project_out))

    # Parse hits
    human_similar = set()
    if blast_out.stat().st_size > 0:
        df = pd.read_csv(str(blast_out), sep="\t", header=None,
                         names=["qid", "sid", "pident", "evalue", "bitscore"])
        hits = df[df["pident"] >= BLAST_IDENTITY]
        human_similar = set(hits["qid"].tolist())
        log(f"  Human homologs identified: {len(human_similar)}")
    else:
        log("  No BLAST hits — all candidates are non-human homologs")

    filtered = [rec for rec in records if rec.id not in human_similar]
    log(f"  Removed: {len(human_similar)} human homologs")
    log(f"  Kept:    {len(filtered)} safe drug target candidates")
    return filtered


def _keyword_filter(records):
    log("  Applying keyword-based fallback filter...")
    conserved = [
        "atp synthase", "rna polymerase", "dna polymerase",
        "elongation factor", "chaperonin", "heat shock",
        "superoxide dismutase", "thioredoxin", "glutaredoxin",
    ]
    kept, removed = [], 0
    for rec in records:
        if any(kw in rec.description.lower() for kw in conserved):
            removed += 1
        else:
            kept.append(rec)
    log(f"  Removed (conserved families): {removed} | Kept: {len(kept)}")
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

    bars = axes[0].barh(range(len(counts)), counts,
                        color=colors[:len(counts)], edgecolor="white", linewidth=0.8)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=11)
    axes[0].set_xlabel("Number of proteins")
    axes[0].set_title("Filtering funnel", fontsize=13)
    axes[0].invert_yaxis()
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                     f"{count:,}", va="center", fontsize=10, fontweight="bold")

    pcts = [c / counts[0] * 100 for c in counts]
    axes[1].plot(range(len(pcts)), pcts, "o-", color="#4A90D9", linewidth=2, markersize=8)
    for i, (pct, label) in enumerate(zip(pcts, labels)):
        axes[1].annotate(f"{pct:.1f}%", (i, pct),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=10)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    axes[1].set_ylabel("% of original proteome retained")
    axes[1].set_title("Retention at each step", fontsize=13)
    axes[1].set_ylim(0, 115)

    plt.suptitle("K. pneumoniae — Drug Target Filtering",
                 fontsize=14, fontweight="bold")
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

## Top candidates (first 20)
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
    counts.append(len(records)); labels.append("Full proteome")

    records = filter_by_length(records)
    counts.append(len(records)); labels.append("Length filter")

    records = filter_essential(records)
    counts.append(len(records)); labels.append("Essential genes")

    records = remove_human_homologs(records)
    counts.append(len(records)); labels.append("Human homologs removed")

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