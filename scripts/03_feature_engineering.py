#!/usr/bin/env python3
"""
Week 3 - Feature Engineering for ML
=====================================
Converts protein sequences into numerical features for ML druggability prediction.

Features extracted per protein:
  - Basic: length, molecular weight, isoelectric point
  - Composition: amino acid frequencies (20 features)
  - Physicochemical: hydrophobicity, charge, instability index, aromaticity
  - Structural: secondary structure fraction (helix/sheet/coil)

Usage:
    python scripts/03_feature_engineering.py

Outputs:
    data/features.csv          — feature matrix (95 proteins x ~35 features)
    data/features_scaled.csv   — scaled version ready for ML
    visualizations/feature_heatmap.png
    visualizations/feature_correlation.png
    visualizations/feature_distributions.png
"""

import matplotlib
matplotlib.use("Agg")   # no GUI needed — works in WSL

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

Path("visualizations").mkdir(exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Amino acids ────────────────────────────────────────────────────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}


# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_features(record):
    """
    Extracts all ML features from a single SeqRecord.
    Returns a dict of feature_name -> value.
    """
    seq = str(record.seq).upper()

    # Remove ambiguous characters (X, B, Z, U) — ProteinAnalysis can't handle them
    clean_seq = "".join(aa for aa in seq if aa in AMINO_ACIDS)

    if len(clean_seq) < 10:
        return None  # skip if sequence is too short after cleaning

    features = {}

    # ── Gene / ID ──────────────────────────────────────────────────────────────
    features["uniprot_id"] = record.id
    gene = ""
    if "GN=" in record.description:
        gene = record.description.split("GN=")[1].split(" ")[0]
    features["gene"] = gene

    # ── Basic features ─────────────────────────────────────────────────────────
    features["length"] = len(clean_seq)

    try:
        pa = ProteinAnalysis(clean_seq)

        features["molecular_weight"]  = round(pa.molecular_weight(), 2)
        features["isoelectric_point"] = round(pa.isoelectric_point(), 3)
        features["instability_index"] = round(pa.instability_index(), 3)
        features["aromaticity"]       = round(pa.aromaticity(), 4)
        features["gravy"]             = round(pa.gravy(), 4)  # hydrophobicity (GRAVY score)

        # Charge at pH 7.4 (physiological)
        features["charge_ph7"] = round(pa.charge_at_pH(7.4), 3)

    except Exception:
        # If BioPython fails on unusual sequences, fill with NaN
        for key in ["molecular_weight", "isoelectric_point", "instability_index",
                    "aromaticity", "gravy", "charge_ph7"]:
            features[key] = np.nan

    # ── Amino acid composition (20 features, each = fraction 0..1) ────────────
    total = len(clean_seq)
    for aa in AMINO_ACIDS:
        features[f"aa_{aa}"] = round(clean_seq.count(aa) / total, 5)

    # ── Dipeptide ratios (biologically meaningful pairs) ──────────────────────
    # These capture local sequence patterns better than single AA counts
    important_dipeptides = ["LL", "VV", "FF", "WW", "GG", "PP", "KK", "RR", "DE", "KR"]
    for dp in important_dipeptides:
        count = sum(1 for i in range(len(clean_seq)-1) if clean_seq[i:i+2] == dp)
        features[f"dp_{dp}"] = round(count / max(total - 1, 1), 5)

    # ── Physicochemical ratios ─────────────────────────────────────────────────
    # Hydrophobic residues (good for membrane/pocket binding)
    hydrophobic = sum(clean_seq.count(aa) for aa in "VILMFYW")
    features["hydrophobic_fraction"] = round(hydrophobic / total, 4)

    # Charged residues
    positive = sum(clean_seq.count(aa) for aa in "KRH")
    negative = sum(clean_seq.count(aa) for aa in "DE")
    features["positive_fraction"] = round(positive / total, 4)
    features["negative_fraction"] = round(negative / total, 4)
    features["charge_ratio"]      = round((positive - negative) / max(total, 1), 4)

    # Polar uncharged
    polar = sum(clean_seq.count(aa) for aa in "STNQ")
    features["polar_fraction"] = round(polar / total, 4)

    # Tiny residues (G, A, S) — indicate flexible loops
    tiny = sum(clean_seq.count(aa) for aa in "GAS")
    features["tiny_fraction"] = round(tiny / total, 4)

    # Cysteine count (important for disulfide bonds / active sites)
    features["cysteine_count"] = clean_seq.count("C")

    # ── Sequence complexity (Shannon entropy) ─────────────────────────────────
    aa_freqs = [clean_seq.count(aa) / total for aa in AMINO_ACIDS if clean_seq.count(aa) > 0]
    entropy = -sum(f * np.log2(f) for f in aa_freqs)
    features["sequence_entropy"] = round(entropy, 4)

    return features


# ── Main extraction loop ───────────────────────────────────────────────────────
def build_feature_matrix(fasta_path="results/filtered_targets.fasta"):
    log(f"Loading filtered targets from {fasta_path}...")
    records = list(SeqIO.parse(fasta_path, "fasta"))
    log(f"Processing {len(records)} proteins...")

    rows = []
    skipped = 0
    for i, rec in enumerate(records):
        feats = extract_features(rec)
        if feats is None:
            skipped += 1
            continue
        rows.append(feats)
        if (i + 1) % 20 == 0:
            log(f"  Processed {i+1}/{len(records)}")

    df = pd.DataFrame(rows)
    log(f"Feature matrix: {df.shape[0]} proteins x {df.shape[1]} features")
    if skipped:
        log(f"Skipped {skipped} proteins (too short after cleaning)")
    return df


# ── Save features ──────────────────────────────────────────────────────────────
def save_features(df):
    # Raw features
    out = "data/features.csv"
    df.to_csv(out, index=False)
    log(f"Saved: {out}")

    # Scaled features (for ML — drop ID cols, scale numerics)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_cols      = ["uniprot_id", "gene"]

    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(df[numeric_cols].fillna(0))
    df_scaled = pd.DataFrame(scaled_vals, columns=numeric_cols)
    df_scaled.insert(0, "gene",       df["gene"].values)
    df_scaled.insert(0, "uniprot_id", df["uniprot_id"].values)

    out_scaled = "data/features_scaled.csv"
    df_scaled.to_csv(out_scaled, index=False)
    log(f"Saved: {out_scaled}")

    return df_scaled, numeric_cols


# ── Visualisation 1: Feature distributions ────────────────────────────────────
def plot_distributions(df):
    key_features = [
        "length", "molecular_weight", "isoelectric_point",
        "instability_index", "gravy", "charge_ph7",
        "hydrophobic_fraction", "sequence_entropy"
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for i, feat in enumerate(key_features):
        col = df[feat].dropna()
        axes[i].hist(col, bins=20, color="#4A90D9", edgecolor="white", linewidth=0.5)
        axes[i].axvline(col.median(), color="#E05C3A", linestyle="--",
                        linewidth=1.5, label=f"Median: {col.median():.2f}")
        axes[i].set_title(feat.replace("_", " "), fontsize=11)
        axes[i].set_xlabel("")
        axes[i].legend(fontsize=8)

    plt.suptitle("Feature distributions — 95 drug target candidates",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "visualizations/feature_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {out}")


# ── Visualisation 2: Correlation heatmap ──────────────────────────────────────
def plot_correlation(df, numeric_cols):
    # Use only key physicochemical features for readable heatmap
    key_cols = [c for c in numeric_cols if not c.startswith("aa_") and not c.startswith("dp_")]

    corr = df[key_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Feature correlation matrix — physicochemical features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "visualizations/feature_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {out}")


# ── Visualisation 3: AA composition heatmap ───────────────────────────────────
def plot_aa_heatmap(df):
    aa_cols = [c for c in df.columns if c.startswith("aa_")]
    gene_labels = df["gene"].fillna(df["uniprot_id"]).tolist()

    aa_data = df[aa_cols].fillna(0)
    aa_data.index = gene_labels
    aa_data.columns = [c.replace("aa_", "") for c in aa_cols]

    fig, ax = plt.subplots(figsize=(16, max(8, len(gene_labels) * 0.22)))
    sns.heatmap(
        aa_data, cmap="YlOrRd",
        linewidths=0.3, ax=ax,
        cbar_kws={"label": "Amino acid fraction"}
    )
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Protein (gene name)")
    ax.set_title("Amino acid composition — all 95 drug target candidates",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "visualizations/aa_composition_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {out}")


# ── Visualisation 4: Top features ranked by variance ──────────────────────────
def plot_top_features(df, numeric_cols):
    """
    Shows which features vary the most across proteins.
    High-variance features are most useful for ML discrimination.
    """
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top20 = variances.head(20)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(range(len(top20)), top20.values,
                   color="#5BAD8F", edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f.replace("aa_","AA: ").replace("dp_","DP: ")
                        for f in top20.index], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Variance (higher = more discriminative for ML)")
    ax.set_title("Top 20 most variable features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "visualizations/top_features_variance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {out}")


# ── Write summary ──────────────────────────────────────────────────────────────
def write_summary(df, numeric_cols):
    n_features = len(numeric_cols)
    n_proteins = len(df)

    # Quick stats on key features
    stats = df[["length", "molecular_weight", "isoelectric_point",
                "instability_index", "gravy", "hydrophobic_fraction"]].describe()

    summary = f"""# Week 3 - Feature Engineering Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Feature matrix
- Proteins: {n_proteins}
- Features: {n_features} numerical + 2 ID columns
- Output: data/features.csv, data/features_scaled.csv

## Feature categories
| Category | Features | Count |
|----------|----------|-------|
| Basic | length, MW, pI, instability, GRAVY, charge | 6 |
| AA composition | frequency of each amino acid | 20 |
| Dipeptide ratios | biologically relevant pairs | 10 |
| Physicochemical | hydrophobic, charged, polar fractions | 7 |
| Complexity | Shannon entropy | 1 |

## Key statistics
{stats.round(3).to_string()}

## Visualisations saved
- visualizations/feature_distributions.png
- visualizations/feature_correlation.png
- visualizations/aa_composition_heatmap.png
- visualizations/top_features_variance.png

## Next step
    python scripts/04_ml_model.py
"""
    Path("results").mkdir(exist_ok=True)
    with open("results/feature_engineering_report.md", "w") as f:
        f.write(summary)
    log("Saved: results/feature_engineering_report.md")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print(" Week 3 - Feature Engineering")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55 + "\n")

    # Build feature matrix
    df = build_feature_matrix()

    # Save raw + scaled versions
    df_scaled, numeric_cols = save_features(df)

    # Visualisations
    log("\nGenerating visualisations...")
    plot_distributions(df)
    plot_correlation(df, numeric_cols)
    plot_aa_heatmap(df)
    plot_top_features(df, numeric_cols)

    # Summary
    write_summary(df, numeric_cols)

    # Preview
    log("\nFeature matrix preview:")
    preview_cols = ["gene", "length", "molecular_weight", "isoelectric_point",
                    "gravy", "hydrophobic_fraction", "sequence_entropy"]
    print(df[preview_cols].head(10).to_string(index=False))

    print("\n" + "=" * 55)
    print(f" Week 3 complete!")
    print(f" Features: {len(numeric_cols)} per protein")
    print(f" Proteins: {len(df)}")
    print(f" Next: python scripts/04_ml_model.py")
    print("=" * 55)


if __name__ == "__main__":
    main()