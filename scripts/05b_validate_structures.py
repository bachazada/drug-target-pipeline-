#!/usr/bin/env python3
"""
Week 5b - Structure Validation (LOCAL — run after downloading from Colab)
==========================================================================
Run this AFTER you have downloaded week5_results.zip from Colab and
extracted the PDB files into results/structures/

What it does:
  1. Reads all .pdb files in results/structures/
  2. Extracts mean + per-residue pLDDT scores
  3. Flags low-confidence regions (pLDDT < 50 = disordered)
  4. Updates results/structure_results.csv
  5. Generates final structure quality plots

Usage:
    python scripts/05b_validate_structures.py

Outputs:
    results/structure_results.csv  (updated with local pLDDT values)
    visualizations/plddt_summary.png
    visualizations/plddt_per_residue.png
"""

import matplotlib
matplotlib.use("Agg")

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Parse PDB for pLDDT ───────────────────────────────────────────────────────
def parse_pdb_plddt(pdb_path):
    """
    In ColabFold output PDBs, the B-factor column stores the
    per-residue pLDDT confidence score (0–100).
    """
    residue_plddt = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                res_num = int(line[22:26].strip())
                bfactor = float(line[60:66].strip())
                if res_num not in residue_plddt:
                    residue_plddt[res_num] = []
                residue_plddt[res_num].append(bfactor)
            except ValueError:
                continue

    if not residue_plddt:
        return [], 0.0

    per_residue = [np.mean(vals) for vals in residue_plddt.values()]
    mean_plddt  = round(np.mean(per_residue), 2)
    return per_residue, mean_plddt


# ── Main validation ───────────────────────────────────────────────────────────
def validate_structures(structures_dir="results/structures"):
    pdb_files = list(Path(structures_dir).glob("*.pdb"))

    if not pdb_files:
        log(f"No PDB files found in {structures_dir}/")
        log("Make sure you extracted week5_results.zip and copied the .pdb files there.")
        return None

    log(f"Found {len(pdb_files)} PDB files in {structures_dir}/")

    rows = []
    per_residue_data = {}

    for pdb_path in sorted(pdb_files):
        gene = pdb_path.stem  # filename without extension
        per_residue, mean_plddt = parse_pdb_plddt(str(pdb_path))

        if not per_residue:
            log(f"  WARNING: Could not parse {pdb_path.name}")
            continue

        n_high    = sum(1 for p in per_residue if p >= 90)
        n_conf    = sum(1 for p in per_residue if p >= 70)
        n_low     = sum(1 for p in per_residue if p < 50)
        pct_conf  = round(n_conf / len(per_residue) * 100, 1)
        pct_disor = round(n_low  / len(per_residue) * 100, 1)

        status = ("very_high" if mean_plddt >= 90
                  else "high"   if mean_plddt >= 70
                  else "medium" if mean_plddt >= 50
                  else "low")

        rows.append({
            "gene":              gene,
            "pdb_file":          pdb_path.name,
            "length_aa":         len(per_residue),
            "mean_plddt":        mean_plddt,
            "pct_confident":     pct_conf,
            "pct_disordered":    pct_disor,
            "plddt_status":      status,
            "proceed_to_docking": mean_plddt >= 70,
        })
        per_residue_data[gene] = per_residue

        log(f"  {gene:<12} mean_pLDDT={mean_plddt:.1f}  "
            f"confident={pct_conf}%  disordered={pct_disor}%  [{status}]")

    df = pd.DataFrame(rows).sort_values("mean_plddt", ascending=False)

    # Save updated CSV
    out = "results/structure_results_local.csv"
    df.to_csv(out, index=False)
    log(f"\nSaved: {out}")

    n_proceed = df["proceed_to_docking"].sum()
    log(f"Structures proceeding to docking (pLDDT ≥ 70): {n_proceed} / {len(df)}")

    return df, per_residue_data


# ── Plot 1: Mean pLDDT bar chart ──────────────────────────────────────────────
def plot_plddt_summary(df):
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))

    color_map = {
        "very_high": "#4A90D9",
        "high":      "#5BAD8F",
        "medium":    "#F5A623",
        "low":       "#E05C3A",
    }
    colors = [color_map[s] for s in df["plddt_status"]]

    bars = ax.barh(range(len(df)), df["mean_plddt"],
                   color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["gene"], fontsize=10)
    ax.invert_yaxis()
    ax.axvline(70, color="#F5A623", linestyle="--", linewidth=1.5,
               label="pLDDT 70 (docking threshold)")
    ax.axvline(90, color="#4A90D9", linestyle="--", linewidth=1.5,
               label="pLDDT 90 (very high confidence)")
    ax.set_xlabel("Mean pLDDT score")
    ax.set_xlim(0, 100)
    ax.set_title("Structure confidence — ColabFold predictions",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # Add value labels
    for bar, val in zip(bars, df["mean_plddt"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=v, label=k.replace("_"," ").title())
                  for k, v in color_map.items()]
    ax.legend(handles=legend_els, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = "visualizations/plddt_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {out}")


# ── Plot 2: Per-residue pLDDT for top targets ─────────────────────────────────
def plot_per_residue(df, per_residue_data, top_n=6):
    top_genes = df.head(top_n)["gene"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, gene in enumerate(top_genes):
        if gene not in per_residue_data:
            continue
        plddt = per_residue_data[gene]
        residues = range(1, len(plddt) + 1)

        # Color by confidence band
        colors = ["#4A90D9" if p >= 90 else "#5BAD8F" if p >= 70
                  else "#F5A623" if p >= 50 else "#E05C3A"
                  for p in plddt]

        axes[i].bar(residues, plddt, color=colors, width=1.0, linewidth=0)
        axes[i].axhline(70, color="#F5A623", linestyle="--", linewidth=1, alpha=0.8)
        axes[i].axhline(90, color="#4A90D9", linestyle="--", linewidth=1, alpha=0.8)
        axes[i].set_ylim(0, 100)
        axes[i].set_xlabel("Residue position", fontsize=9)
        axes[i].set_ylabel("pLDDT", fontsize=9)
        mean_p = df[df["gene"]==gene]["mean_plddt"].values[0]
        axes[i].set_title(f"{gene}  (mean={mean_p:.1f})", fontsize=11, fontweight="500")

    plt.suptitle("Per-residue pLDDT — Top 6 drug targets",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "visualizations/plddt_per_residue.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print(" Week 5b - Structure Validation")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55 + "\n")

    Path("results/structures").mkdir(parents=True, exist_ok=True)
    Path("visualizations").mkdir(exist_ok=True)

    result = validate_structures()
    if result is None:
        return

    df, per_residue_data = result
    plot_plddt_summary(df)
    plot_per_residue(df, per_residue_data)

    print("\n" + "=" * 55)
    n = df["proceed_to_docking"].sum()
    print(f" Week 5 complete! {n} structures ready for docking.")
    print(f" Next: python scripts/06_pocket_detection.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
