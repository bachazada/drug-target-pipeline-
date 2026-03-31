#!/usr/bin/env python3
"""
Week 5 - Structure Prediction Preparation (LOCAL SCRIPT)
==========================================================
Run this locally BEFORE opening Google Colab.

What it does:
  1. Reads your ML rankings (results/target_scores.csv)
  2. Extracts the top N candidate sequences from filtered_targets.fasta
  3. Saves individual FASTA files + a combined FASTA for ColabFold
  4. Prints a summary table so you know exactly what to submit

Usage:
    python scripts/05_prepare_structures.py

Outputs:
    results/colabfold_input/          <- upload this folder to Colab
        combined_targets.fasta        <- all top targets in one file
        individual/                   <- one .fasta per protein
            01_ftsI.fasta
            02_pheT.fasta
            ...
    results/structure_targets.csv     <- metadata for each target
    visualizations/target_summary.png <- quick overview plot
"""

import matplotlib
matplotlib.use("Agg")

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from datetime import datetime
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

TOP_N = 20          # take top 20 by ML score — ColabFold free tier handles this fine
MIN_PLDDT_CUTOFF = 70.0  # we'll filter structures below this after Colab runs


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Step 1: Load ML rankings ───────────────────────────────────────────────────
def load_rankings(scores_path="results/target_scores.csv"):
    df = pd.read_csv(scores_path)
    df = df.sort_values("druggability_score", ascending=False).head(TOP_N)
    df = df.reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    log(f"Loaded top {len(df)} targets from ML rankings")
    return df


# ── Step 2: Extract sequences from filtered FASTA ─────────────────────────────
def extract_sequences(rankings, fasta_path="results/filtered_targets.fasta"):
    log(f"Extracting sequences from {fasta_path}...")

    all_records = {rec.id: rec for rec in SeqIO.parse(fasta_path, "fasta")}
    target_ids  = set(rankings["uniprot_id"].tolist())

    matched = []
    for _, row in rankings.iterrows():
        uid = row["uniprot_id"]
        if uid in all_records:
            rec = all_records[uid]
            matched.append({
                "rank":               row["rank"],
                "uniprot_id":         uid,
                "gene":               row["gene"],
                "druggability_score": row["druggability_score"],
                "priority":           row["priority"],
                "length":             len(rec.seq),
                "sequence":           str(rec.seq),
                "record":             rec,
            })
        else:
            log(f"  WARNING: {uid} ({row['gene']}) not found in FASTA — skipping")

    log(f"Matched {len(matched)} / {len(rankings)} targets to sequences")
    return matched


# ── Step 3: Add extra metadata useful for ColabFold ───────────────────────────
def enrich_metadata(targets):
    log("Computing additional metadata...")
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    for t in targets:
        seq = t["sequence"]
        clean = "".join(aa for aa in seq.upper() if aa in amino_acids)
        try:
            pa = ProteinAnalysis(clean)
            t["molecular_weight_kda"] = round(pa.molecular_weight() / 1000, 1)
            t["isoelectric_point"]    = round(pa.isoelectric_point(), 2)
            t["instability_index"]    = round(pa.instability_index(), 1)
            t["is_stable"]            = t["instability_index"] < 40
        except Exception:
            t["molecular_weight_kda"] = round(len(clean) * 0.11, 1)
            t["isoelectric_point"]    = 7.0
            t["instability_index"]    = 35.0
            t["is_stable"]            = True

        # Estimate ColabFold runtime (roughly 1 min per 100 aa on free GPU)
        t["est_runtime_min"] = max(1, round(len(clean) / 100))

    total_time = sum(t["est_runtime_min"] for t in targets)
    log(f"Estimated total ColabFold runtime: ~{total_time} min on free Colab GPU")
    return targets


# ── Step 4: Write FASTA files ─────────────────────────────────────────────────
def write_fasta_files(targets):
    out_dir        = Path("results/colabfold_input")
    individual_dir = out_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    # Individual FASTA per protein (clean headers for ColabFold)
    for t in targets:
        fname = individual_dir / f"{t['rank']:02d}_{t['gene']}.fasta"
        header = f">{t['gene']}|{t['uniprot_id']}|score={t['druggability_score']:.3f}"
        with open(fname, "w") as f:
            f.write(f"{header}\n{t['sequence']}\n")

    # Combined FASTA (all targets, one file) — for batch submission
    combined_path = out_dir / "combined_targets.fasta"
    with open(combined_path, "w") as f:
        for t in targets:
            header = f">{t['gene']}|{t['uniprot_id']}|score={t['druggability_score']:.3f}"
            f.write(f"{header}\n{t['sequence']}\n")

    log(f"Saved {len(targets)} individual FASTA files → {individual_dir}")
    log(f"Saved combined FASTA → {combined_path}")
    return out_dir


# ── Step 5: Save metadata CSV ─────────────────────────────────────────────────
def save_metadata(targets):
    rows = []
    for t in targets:
        rows.append({
            "rank":               t["rank"],
            "gene":               t["gene"],
            "uniprot_id":         t["uniprot_id"],
            "druggability_score": t["druggability_score"],
            "priority":           t["priority"],
            "length_aa":          t["length"],
            "mw_kda":             t["molecular_weight_kda"],
            "isoelectric_point":  t["isoelectric_point"],
            "instability_index":  t["instability_index"],
            "is_stable":          t["is_stable"],
            "est_runtime_min":    t["est_runtime_min"],
            "pdb_filename":       f"{t['gene']}_unrelaxed_rank_001.pdb",  # ColabFold default name
            "plddt_status":       "pending",  # filled in after Colab run
        })

    df = pd.DataFrame(rows)
    out = "results/structure_targets.csv"
    df.to_csv(out, index=False)
    log(f"Saved metadata: {out}")
    return df


# ── Step 6: Summary visualisation ─────────────────────────────────────────────
def plot_summary(targets):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    genes   = [t["gene"] for t in targets]
    scores  = [t["druggability_score"] for t in targets]
    lengths = [t["length"] for t in targets]
    runtimes = [t["est_runtime_min"] for t in targets]

    # Priority colours
    colors = []
    for t in targets:
        if t["priority"] == "Priority 1 (high)":    colors.append("#E05C3A")
        elif t["priority"] == "Priority 2 (medium)": colors.append("#F5A623")
        else:                                         colors.append("#B0C4DE")

    # Plot 1: Druggability score
    axes[0].barh(range(len(genes)), scores, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(range(len(genes)))
    axes[0].set_yticklabels(genes, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].axvline(0.75, color="#E05C3A", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].axvline(0.50, color="#F5A623", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].set_xlabel("Druggability score")
    axes[0].set_title("ML druggability score", fontsize=12)

    # Plot 2: Protein length
    axes[1].barh(range(len(genes)), lengths, color="#4A90D9", edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(range(len(genes)))
    axes[1].set_yticklabels(genes, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].axvline(500, color="gray", linestyle="--", linewidth=1, alpha=0.5,
                    label="500 aa (slower ColabFold)")
    axes[1].set_xlabel("Protein length (aa)")
    axes[1].set_title("Sequence length", fontsize=12)
    axes[1].legend(fontsize=8)

    # Plot 3: Estimated Colab runtime
    axes[2].barh(range(len(genes)), runtimes, color="#5BAD8F", edgecolor="white", linewidth=0.5)
    axes[2].set_yticks(range(len(genes)))
    axes[2].set_yticklabels(genes, fontsize=9)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Estimated runtime (min)")
    axes[2].set_title("ColabFold est. runtime", fontsize=12)

    p1 = mpatches.Patch(color="#E05C3A", label="Priority 1")
    p2 = mpatches.Patch(color="#F5A623", label="Priority 2")
    fig.legend(handles=[p1, p2], loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Week 5 — Targets submitted to ColabFold",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("visualizations/target_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("Saved: visualizations/target_summary.png")


# ── Step 7: Print submission instructions ─────────────────────────────────────
def print_instructions(targets, out_dir):
    total_time = sum(t["est_runtime_min"] for t in targets)
    p1_count   = sum(1 for t in targets if "Priority 1" in t["priority"])

    print("\n" + "=" * 60)
    print(" COLABFOLD SUBMISSION INSTRUCTIONS")
    print("=" * 60)
    print(f"""
OPTION A — Submit one at a time (recommended for beginners):
  1. Open: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
  2. For each protein in results/colabfold_input/individual/:
     - Paste the sequence into the 'query_sequence' field
     - Set jobname to the gene name (e.g. ftsI)
     - Run all cells (Runtime → Run all)
     - Download the results ZIP when done
  3. Extract .pdb files into: results/structures/

OPTION B — Use our Colab notebook (recommended):
  1. Upload the notebook: notebooks/week5_colabfold.ipynb to Colab
  2. Upload: results/colabfold_input/combined_targets.fasta
  3. Run all cells — it processes all {len(targets)} proteins automatically

PRIORITY ORDER (run these first):
""")
    for t in targets[:p1_count]:
        print(f"  [{t['rank']:2d}] {t['gene']:<10} score={t['druggability_score']:.3f}  "
              f"length={t['length']} aa  est. ~{t['est_runtime_min']} min")

    print(f"""
AFTER COLAB:
  1. Download all .pdb files from Colab
  2. Place them in: results/structures/
  3. Run: python scripts/05b_validate_structures.py
     (checks pLDDT scores and filters low-confidence models)

Estimated total GPU time: ~{total_time} min
Free Colab GPU limit:     ~60-90 min/session
Recommendation:           Run Priority 1 targets ({p1_count} proteins) first
""")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print(" Week 5 - Structure Prediction Preparation")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55 + "\n")

    rankings = load_rankings()
    targets  = extract_sequences(rankings)
    targets  = enrich_metadata(targets)
    out_dir  = write_fasta_files(targets)
    df_meta  = save_metadata(targets)
    plot_summary(targets)
    print_instructions(targets, out_dir)

    print(f"\n Week 5 prep complete!")
    print(f" {len(targets)} FASTA files ready in: {out_dir}")
    print(f" Now open Google Colab and run the notebook.")
    print("=" * 55)


if __name__ == "__main__":
    main()
