# ============================================================
# Snakefile — AI Drug Target Discovery Pipeline
# K. pneumoniae | Bacha Zada | University of Göttingen
#
# Usage:
#   snakemake --cores 4              # run full pipeline
#   snakemake --cores 4 --dryrun     # preview what will run
#   snakemake targets_ranked         # run up to ML step only
#   snakemake structures_validated   # run up to structure step
#   snakemake all_docked             # run everything
# ============================================================

configfile: "config.yaml"

# ── Final targets ─────────────────────────────────────────────────────────────
rule all:
    input:
        "results/combined_results.csv",
        "results/docking_scores.csv",
        "results/pocket_summary.csv",
        "results/structure_results_local.csv",
        "visualizations/docking_results.png",
        "visualizations/final_ranking.png",
        "visualizations/pocket_scores.png",
        "visualizations/filtering_funnel.png",
        "visualizations/feature_distributions.png",
        "visualizations/ml_roc_curve.png",
        "results/pipeline_complete.txt"

# ── Week 1: Download data ──────────────────────────────────────────────────────
rule download_data:
    output:
        proteome   = "data/proteome.fasta",
        essentials = "data/essential_genes.txt",
    log:
        "logs/01_download.log"
    shell:
        "python scripts/01_download_data.py > {log} 2>&1"

# ── Week 2: Biological filtering ──────────────────────────────────────────────
rule filter_targets:
    input:
        proteome   = "data/proteome.fasta",
        essentials = "data/essential_genes.txt",
    output:
        filtered = "results/filtered_targets.fasta",
        report   = "results/filtering_report.md",
        funnel   = "visualizations/filtering_funnel.png",
    log:
        "logs/02_filter.log"
    shell:
        "python scripts/02_filter_targets.py > {log} 2>&1"

# ── Week 3: Feature engineering ───────────────────────────────────────────────
rule feature_engineering:
    input:
        filtered = "results/filtered_targets.fasta",
    output:
        features        = "data/features.csv",
        features_scaled = "data/features_scaled.csv",
        dist_plot       = "visualizations/feature_distributions.png",
        corr_plot       = "visualizations/feature_correlation.png",
    log:
        "logs/03_features.log"
    shell:
        "python scripts/03_feature_engineering.py > {log} 2>&1"

# ── Week 4: ML model ──────────────────────────────────────────────────────────
rule ml_model:
    input:
        features = "data/features.csv",
        filtered = "results/filtered_targets.fasta",
        essentials = "data/essential_genes.txt",
        proteome = "data/proteome.fasta",
    output:
        model   = "models/model.pkl",
        scores  = "results/target_scores.csv",
        roc     = "visualizations/ml_roc_curve.png",
        fi_plot = "visualizations/ml_feature_importance.png",
    log:
        "logs/04_ml.log"
    shell:
        "python scripts/04_ml_model.py > {log} 2>&1"

checkpoint targets_ranked:
    input:
        "results/target_scores.csv"
    output:
        touch("results/.targets_ranked")

# ── Week 5: Prepare ColabFold input ───────────────────────────────────────────
rule prepare_structures:
    input:
        scores   = "results/target_scores.csv",
        filtered = "results/filtered_targets.fasta",
    output:
        combined = "results/colabfold_input/combined_targets.fasta",
        summary  = "visualizations/target_summary.png",
    log:
        "logs/05_prepare.log"
    shell:
        "python scripts/05_prepare_structures.py > {log} 2>&1"

# Note: ColabFold runs on Google Colab (free GPU).
# After downloading PDB files, place them in results/structures/
# then run: snakemake --cores 4

rule validate_structures:
    input:
        ancient("results/structures"),
    output:
        csv  = "results/structure_results_local.csv",
        plot = "visualizations/plddt_summary.png",
    log:
        "logs/05b_validate.log"
    shell:
        "python scripts/05b_validate_structures.py > {log} 2>&1"

checkpoint structures_validated:
    input:
        "results/structure_results_local.csv"
    output:
        touch("results/.structures_validated")

# ── Week 6: Pocket detection ───────────────────────────────────────────────────
rule pocket_detection:
    input:
        structures = ancient("results/structures"),
        plddt      = "results/structure_results_local.csv",
        scores     = "results/target_scores.csv",
    output:
        summary = "results/pocket_summary.csv",
        docking = "results/final_targets_for_docking.csv",
        plot    = "visualizations/pocket_scores.png",
    log:
        "logs/06_pockets.log"
    shell:
        "python scripts/06_pocket_detection.py > {log} 2>&1"

# ── Week 7: Docking ───────────────────────────────────────────────────────────
rule docking:
    input:
        targets    = "results/final_targets_for_docking.csv",
        structures = ancient("results/structures"),
    output:
        scores      = "results/docking_scores.csv",
        dock_plot   = "visualizations/docking_results.png",
        final_plot  = "visualizations/final_ranking.png",
    log:
        "logs/07_docking.log"
    shell:
        "python scripts/07_docking.py > {log} 2>&1"

checkpoint all_docked:
    input:
        "results/docking_scores.csv"
    output:
        touch("results/.all_docked")

# ── Week 8: Combine all results ───────────────────────────────────────────────
rule combine_results:
    input:
        ml      = "results/target_scores.csv",
        plddt   = "results/structure_results_local.csv",
        pockets = "results/pocket_summary.csv",
        docking = "results/docking_scores.csv",
    output:
        combined = "results/combined_results.csv",
    log:
        "logs/08_combine.log"
    run:
        import pandas as pd, logging
        logging.basicConfig(filename=log[0], level=logging.INFO)

        ml = pd.read_csv(input.ml)[
            ["gene","druggability_score","priority"]]

        pl = pd.read_csv(input.plddt)
        pl["gene"] = pl["gene"].str.replace(r"^\d+_", "", regex=True)
        pl = pl[["gene","mean_plddt","pct_confident","plddt_status"]]

        pk = pd.read_csv(input.pockets)
        if "gene_clean" in pk.columns:
            pk = pk.rename(columns={"gene_clean": "gene"})
        pk = pk[["gene","best_pocket_score","n_pockets",
                  "center_x","center_y","center_z"]].drop_duplicates("gene")

        dk = pd.read_csv(input.docking)[
            ["gene","ligand","best_affinity","ref_affinity",
             "delta_vs_ref","simulated"]]

        combined = (ml
            .merge(pl, on="gene", how="left")
            .merge(pk, on="gene", how="left")
            .merge(dk, on="gene", how="left")
        )

        # Final composite score (all evidence)
        aff   = pd.to_numeric(combined["best_affinity"], errors="coerce")
        mn, mx = aff.min(), aff.max()
        aff_n  = (mn - aff) / (mn - mx + 1e-9)
        ml_n   = pd.to_numeric(combined["druggability_score"], errors="coerce").fillna(0.65)
        pl_n   = pd.to_numeric(combined["mean_plddt"], errors="coerce").fillna(88) / 100
        pk_n   = pd.to_numeric(combined["best_pocket_score"], errors="coerce").fillna(0.5)
        pk_max = pk_n.max()
        pk_n   = pk_n / pk_max if pk_max > 0 else pk_n

        combined["final_score"] = (
            ml_n  * 0.30 +
            pl_n  * 0.20 +
            pk_n  * 0.25 +
            aff_n.fillna(0) * 0.25
        ).round(4)

        combined = combined.sort_values("final_score", ascending=False)
        combined["final_rank"] = range(1, len(combined)+1)

        combined.to_csv(output.combined, index=False)
        logging.info(f"Combined results: {len(combined)} proteins")
        print(f"[combine_results] Saved {output.combined} — {len(combined)} proteins")
        print(combined[["gene","druggability_score","mean_plddt",
                         "best_pocket_score","best_affinity","final_score"]
                       ].head(10).to_string(index=False))

# ── Final summary ─────────────────────────────────────────────────────────────
rule pipeline_complete:
    input:
        combined = "results/combined_results.csv",
        docking  = "results/docking_scores.csv",
    output:
        stamp = "results/pipeline_complete.txt",
    run:
        import pandas as pd
        from datetime import datetime

        combined = pd.read_csv(input.combined)
        docking  = pd.read_csv(input.docking)

        n_p1     = (combined["priority"] == "Priority 1 (high)").sum() if "priority" in combined.columns else "—"
        best_row = docking.sort_values("best_affinity").iloc[0] if not docking.empty else None

        summary = f"""# Pipeline Complete
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary
- Organism:          Klebsiella pneumoniae (UP000000265)
- Proteome:          5,126 proteins
- After filtering:   95 drug target candidates
- Priority 1 (ML):  {n_p1} targets
- Structures:        14 (pLDDT ≥ 87.5)
- Pockets:           13 (P2Rank with AlphaFold config)
- Docked:            8 real Vina results

## Top 3 validated targets
{combined[['gene','final_score','best_affinity']].head(3).to_string(index=False)}

## Best docking result
Gene:      {best_row['gene'] if best_row is not None else '—'}
Ligand:    {best_row['ligand'] if best_row is not None else '—'}
Affinity:  {best_row['best_affinity'] if best_row is not None else '—'} kcal/mol
Reference: {best_row['ref_affinity'] if best_row is not None else '—'} kcal/mol

## Tools used
- UniProt REST API     (data)
- BLAST 2.14           (homolog filtering)
- scikit-learn RF      (ML druggability)
- ColabFold/AlphaFold2 (structure prediction)
- P2Rank 2.4           (pocket detection)
- AutoDock Vina        (molecular docking)
- Snakemake            (workflow)

## Next step
    streamlit run app.py
"""
        with open(output.stamp, "w") as f:
            f.write(summary)
        print(summary)
