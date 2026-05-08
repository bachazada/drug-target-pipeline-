#!/usr/bin/env python3
"""
Week 8 - Pipeline Setup & Validation (clean rebuild)
All DataFrames are deduplicated on 'gene' before any merge.
"""

import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from datetime import datetime

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Safe loader: always returns a deduplicated DataFrame ─────────────────────
def load(path, cols, gene_col="gene", rename=None):
    """
    Load a CSV, optionally rename a column to 'gene',
    deduplicate on gene, and return only requested columns.
    Never raises — returns empty DataFrame on any error.
    """
    try:
        df = pd.read_csv(path)
        if rename and rename in df.columns:
            df = df.rename(columns={rename: "gene"})
        if gene_col in df.columns and gene_col != "gene":
            df = df.rename(columns={gene_col: "gene"})
        # Strip rank prefixes like "7_murA" → "murA"
        if "gene" in df.columns:
            df["gene"] = df["gene"].astype(str).str.replace(
                r"^\d+_", "", regex=True).str.strip()
        # Keep only requested columns that actually exist
        keep = ["gene"] + [c for c in cols if c in df.columns and c != "gene"]
        df = df[keep].copy()
        # Sort best-first before dedup so we keep the most informative row
        sort_col = next((c for c in ["druggability_score","mean_plddt",
                                      "best_pocket_score","best_affinity",
                                      "final_score"] if c in df.columns), None)
        if sort_col:
            ascending = sort_col == "best_affinity"  # affinity: more negative = better
            df = df.sort_values(sort_col, ascending=ascending)
        df = df.drop_duplicates("gene", keep="first").reset_index(drop=True)
        log(f"  Loaded {path} → {len(df)} rows, cols: {list(df.columns)}")
        return df
    except Exception as e:
        log(f"  WARNING: could not load {path}: {e}")
        return pd.DataFrame(columns=["gene"] + cols)


# ── Check project files ───────────────────────────────────────────────────────
def check_files():
    log("Checking project files...")
    required = {
        "Scripts": [
            "scripts/01_download_data.py", "scripts/02_filter_targets.py",
            "scripts/03_feature_engineering.py", "scripts/04_ml_model.py",
            "scripts/05_prepare_structures.py", "scripts/05b_validate_structures.py",
            "scripts/06_pocket_detection.py", "scripts/07_docking.py",
        ],
        "Results": [
            "data/proteome.fasta", "data/essential_genes.txt", "data/features.csv",
            "results/filtered_targets.fasta", "results/target_scores.csv",
            "results/structure_results_local.csv", "results/pocket_summary.csv",
            "results/final_targets_for_docking.csv", "results/docking_scores.csv",
        ],
        "Config": ["config.yaml", "environment.yml", "Snakefile"],
    }
    all_ok = True
    for cat, files in required.items():
        print(f"\n  {cat}:")
        for f in files:
            ok = Path(f).exists()
            print(f"    {'✓' if ok else '✗ MISSING'}  {f}")
            if not ok:
                all_ok = False

    pdbs = list(Path("results/structures").glob("*.pdb")) \
        if Path("results/structures").exists() else []
    print(f"\n  Structures:")
    print(f"    {'✓' if pdbs else '✗'}  results/structures/ ({len(pdbs)} PDB files)")
    return all_ok


# ── Build combined results ────────────────────────────────────────────────────
def build_combined():
    log("\nBuilding combined results table...")

    ml = load("results/target_scores.csv",
              ["druggability_score", "priority"])

    pl = load("results/structure_results_local.csv",
              ["mean_plddt", "pct_confident", "plddt_status"])

    # pocket_summary may have gene_clean instead of gene
    pk_raw = pd.read_csv("results/pocket_summary.csv") \
        if Path("results/pocket_summary.csv").exists() else pd.DataFrame()
    if not pk_raw.empty:
        if "gene_clean" in pk_raw.columns and "gene" not in pk_raw.columns:
            pk_raw = pk_raw.rename(columns={"gene_clean": "gene"})
        elif "gene_clean" in pk_raw.columns:
            pk_raw = pk_raw.drop(columns=["gene_clean"])
    pk = load.__wrapped__(pk_raw, ["best_pocket_score","n_pockets",
                                    "center_x","center_y","center_z"]) \
        if False else _dedup_df(pk_raw,
            ["best_pocket_score","n_pockets","center_x","center_y","center_z"],
            sort_col="best_pocket_score", ascending=False)

    dk = load("results/docking_scores.csv",
              ["ligand","best_affinity","ref_affinity","delta_vs_ref"])

    # Merge step by step — log shape after each merge
    combined = ml.copy()
    log(f"  After ML:      {combined.shape}")

    combined = combined.merge(pl,  on="gene", how="left")
    log(f"  After pLDDT:   {combined.shape}")

    combined = combined.merge(pk,  on="gene", how="left")
    log(f"  After pockets: {combined.shape}")

    combined = combined.merge(dk,  on="gene", how="left")
    log(f"  After docking: {combined.shape}")

    # Final composite score
    ml_n  = pd.to_numeric(combined["druggability_score"], errors="coerce").fillna(0.65)
    pl_n  = pd.to_numeric(combined.get("mean_plddt",    pd.Series()), errors="coerce").fillna(88) / 100
    pk_s  = pd.to_numeric(combined.get("best_pocket_score", pd.Series()), errors="coerce").fillna(0.5)
    pk_mx = pk_s.max()
    pk_n  = pk_s / pk_mx if pk_mx > 0 else pk_s * 0 + 0.5
    aff   = pd.to_numeric(combined.get("best_affinity", pd.Series()), errors="coerce")
    mn, mx = aff.min(), aff.max()
    aff_n = ((mn - aff) / (mn - mx + 1e-9)).fillna(0)

    combined["final_score"] = (
        ml_n  * 0.30 +
        pl_n  * 0.20 +
        pk_n  * 0.25 +
        aff_n * 0.25
    ).astype(float).round(4)

    combined = combined.sort_values("final_score", ascending=False).reset_index(drop=True)
    combined["final_rank"] = range(1, len(combined) + 1)

    combined.to_csv("results/combined_results.csv", index=False)
    log(f"  Saved: results/combined_results.csv — {len(combined)} proteins, "
        f"{len(combined.columns)} columns")
    return combined


def _dedup_df(df, cols, sort_col=None, ascending=True):
    """Helper to dedup a raw DataFrame that may not have 'gene' column yet."""
    if df.empty:
        return pd.DataFrame(columns=["gene"] + cols)
    if "gene" not in df.columns:
        return pd.DataFrame(columns=["gene"] + cols)
    df = df.copy()
    df["gene"] = df["gene"].astype(str).str.replace(r"^\d+_", "", regex=True).str.strip()
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending)
    df = df.drop_duplicates("gene", keep="first")
    keep = ["gene"] + [c for c in cols if c in df.columns]
    return df[keep].reset_index(drop=True)


# ── Summary plot ──────────────────────────────────────────────────────────────
def plot_summary(df):
    log("\nGenerating pipeline summary plot...")
    Path("visualizations").mkdir(exist_ok=True)

    show = df.dropna(subset=["druggability_score"]).head(15).copy()
    genes = show["gene"].tolist()
    n = len(genes)

    has_docking = "best_affinity" in show.columns and show["best_affinity"].notna().any()
    ncols = 4 if has_docking else 3
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, max(6, n * 0.5)))

    def hbar(ax, vals, title, colors, xlabel):
        c = colors if isinstance(colors, list) else [colors] * n
        ax.barh(range(n), vals, color=c, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(genes, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight="500")
        ax.set_xlabel(xlabel, fontsize=9)
        for i, v in enumerate(vals):
            try:
                ax.text(float(v) + abs(float(v)) * 0.01 + 0.01, i,
                        f"{float(v):.2f}", va="center", fontsize=7.5)
            except Exception:
                pass

    hbar(axes[0], show["druggability_score"].astype(float),
         "ML druggability score", "#4A90D9", "Score (0–1)")
    axes[0].axvline(0.75, color="#E05C3A", linestyle="--", linewidth=1,
                    alpha=0.7, label="Priority 1")
    axes[0].legend(fontsize=8)

    plddt = pd.to_numeric(show.get("mean_plddt", pd.Series([88]*n)),
                          errors="coerce").fillna(88).tolist()
    plddt_cols = ["#4A90D9" if v >= 90 else "#5BAD8F" if v >= 70
                  else "#F5A623" for v in plddt]
    hbar(axes[1], plddt, "Structure confidence (pLDDT)", plddt_cols, "pLDDT")
    axes[1].axvline(90, color="#4A90D9", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].axvline(70, color="orange",  linestyle="--", linewidth=1, alpha=0.5)

    pk_v = pd.to_numeric(show.get("best_pocket_score", pd.Series([30]*n)),
                         errors="coerce").fillna(30).tolist()
    hbar(axes[2], pk_v, "P2Rank pocket score", "#5BAD8F", "Pocket score")

    if has_docking:
        aff = pd.to_numeric(show["best_affinity"], errors="coerce").fillna(0).tolist()
        aff_cols = ["#E05C3A" if v <= -7 else "#F5A623" if v <= -5
                    else "#B0C4DE" for v in aff]
        hbar(axes[3], aff, "Docking affinity\n(AutoDock Vina)",
             aff_cols, "Affinity (kcal/mol)")
        axes[3].axvline(-7, color="orange", linestyle="--", linewidth=1,
                        alpha=0.6, label="-7 drug-like")
        axes[3].legend(fontsize=8)

    plt.suptitle("K. pneumoniae Drug Target Discovery — Full Pipeline Summary",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("visualizations/pipeline_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("  Saved: visualizations/pipeline_summary.png")


# ── Methods doc ───────────────────────────────────────────────────────────────
def write_methods():
    Path("docs").mkdir(exist_ok=True)
    methods = """# Methods — AI Drug Target Discovery Pipeline
*K. pneumoniae | Bacha Zada | University of Göttingen*

## Step 1 — Data collection
- Proteome: UniProt REST API, ID UP000000265 (5,126 proteins)
- Essential genes: DEG database, 154 genes

## Step 2 — Biological filtering
- Length filter: 50–2,000 aa → removed 70 proteins
- Essential gene matching: 155 proteins retained
- Human homolog removal: BLAST e-value < 1e-5, identity 30%
  vs UniProt human proteome UP000005640 (20,427 proteins)
- Final: 95 candidates

## Step 3 — Feature engineering
- 44 features: physicochemical (6), AA composition (20),
  dipeptides (10), fractions (7), Shannon entropy (1)
- Scaled with StandardScaler

## Step 4 — ML model
- Random Forest, 300 trees, balanced class weights
- 5-fold stratified CV: ROC-AUC 0.725 ± 0.019
- 14 Priority 1 targets (score ≥ 0.75)

## Step 5 — Structure prediction
- ColabFold (AlphaFold2_ptm, 3 recycles, T4 GPU)
- 14 structures, all pLDDT ≥ 87.5 (range 87.5–96.8)

## Step 6 — Pocket detection
- P2Rank v2.4 with -c alphafold configuration
- 13/13 structures processed

## Step 7 — Molecular docking
- AutoDock Vina, exhaustiveness=8, box=20Å
- Open Babel 3.1.0 for format conversion
- Ligands from PubChem (3D conformers)
- 8/10 targets docked successfully

## Composite scoring
final_score = ML(30%) + pLDDT(20%) + pocket(25%) + docking(25%)

## Known limitations
1. fosfomycin/murA: covalent inhibitor — underestimated by non-covalent docking
2. gyrB/novobiocin: box centre mismatch — novobiocin site in N-terminal domain
3. groEL, pheT: timeout/failure due to large protein size
4. rpoC: skipped — GPU RAM exceeded (1,407 aa)

## Tool versions
| Tool | Version |
|------|---------|
| Python | 3.10 |
| scikit-learn | 1.3 |
| BioPython | 1.81 |
| ColabFold | 1.5 |
| P2Rank | 2.4 |
| AutoDock Vina | 52ec525 |
| Open Babel | 3.1.0 |
| Snakemake | 7.32 |

## References
1. Mirdita M et al. ColabFold. Nat Methods. 2022.
2. Krivak R, Hoksza D. P2Rank. J Cheminform. 2018.
3. Eberhardt J et al. AutoDock Vina 1.2. J Chem Inf Model. 2021.
4. Wishart DS et al. DrugBank 5.0. Nucleic Acids Res. 2018.
5. Luo H et al. DEG 10.0. Nucleic Acids Res. 2014.
"""
    with open("docs/METHODS.md", "w") as f:
        f.write(methods)
    log("  Saved: docs/METHODS.md")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print(f" Week 8 - Pipeline Setup & Validation")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    Path("logs").mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)
    Path("visualizations").mkdir(exist_ok=True)

    all_ok = check_files()
    combined = build_combined()

    if combined is not None and not combined.empty:
        plot_summary(combined)
        write_methods()

        print("\n" + "─" * 62)
        print(" FINAL COMBINED RANKING (top 10)")
        print("─" * 62)
        show_cols = [c for c in ["final_rank", "gene", "druggability_score",
                                  "mean_plddt", "best_pocket_score",
                                  "best_affinity", "final_score"]
                     if c in combined.columns]
        print(combined[show_cols].head(10).to_string(index=False))

    print("""
╔══════════════════════════════════════════════════╗
║  SNAKEMAKE COMMANDS                              ║
╠══════════════════════════════════════════════════╣
║  snakemake --cores 4 --dryrun   (preview)        ║
║  snakemake --cores 4            (run all)        ║
║  snakemake combine_results      (results only)   ║
║  snakemake --dag | dot -Tpng > dag.png           ║
╚══════════════════════════════════════════════════╝
""")
    print("=" * 55)
    print(f" Week 8 {'complete ✓' if all_ok else 'complete (check missing files)'}")
    print(f" Next: streamlit run app.py   (Week 9)")
    print("=" * 55)


if __name__ == "__main__":
    main()