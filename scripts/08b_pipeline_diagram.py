#!/usr/bin/env python3
"""
Generates a clean pipeline DAG diagram without needing graphviz.
Run: python scripts/08b_pipeline_diagram.py
Output: visualizations/pipeline_dag.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def main():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Color scheme
    colors = {
        "data":      "#E6F1FB",
        "ml":        "#EEEDFE",
        "struct":    "#E1F5EE",
        "dock":      "#FAECE7",
        "output":    "#F1EFE8",
    }
    borders = {
        "data":   "#185FA5",
        "ml":     "#534AB7",
        "struct": "#0F6E56",
        "dock":   "#993C1D",
        "output": "#5F5E5A",
    }

    nodes = [
        # (x, y, w, h, label, sublabel, category)
        (5.5, 9.0, 3.0, 0.7, "UniProt / DEG", "5,126 proteins + 154 essential genes", "data"),
        (5.5, 7.8, 3.0, 0.7, "Week 1 — Download", "proteome.fasta + essential_genes.txt", "data"),
        (5.5, 6.6, 3.0, 0.7, "Week 2 — Filter", "BLAST → 95 candidates", "data"),
        (5.5, 5.4, 3.0, 0.7, "Week 3 — Features", "44 physicochemical features", "ml"),
        (5.5, 4.2, 3.0, 0.7, "Week 4 — ML Model", "Random Forest · CV AUC 0.725", "ml"),
        (5.5, 3.0, 3.0, 0.7, "Week 5 — ColabFold", "14 structures · pLDDT ≥ 87.5", "struct"),
        (5.5, 1.8, 3.0, 0.7, "Week 6 — P2Rank", "Pocket detection · 13/13", "struct"),
        (5.5, 0.6, 3.0, 0.7, "Week 7 — Vina", "Docking · 8/10 real results", "dock"),

        # Side outputs
        (1.0, 4.2, 2.2, 0.55, "target_scores.csv", "14 Priority 1 targets", "output"),
        (1.0, 3.0, 2.2, 0.55, "structures/*.pdb", "pLDDT 87–97", "output"),
        (1.0, 1.8, 2.2, 0.55, "pocket_summary.csv", "3D pocket coords", "output"),
        (1.0, 0.6, 2.2, 0.55, "docking_scores.csv", "kcal/mol affinities", "output"),

        # Final
        (10.5, 2.4, 2.4, 0.55, "combined_results.csv", "All evidence merged", "output"),
        (10.5, 1.2, 2.4, 0.55, "Snakemake", "Automated workflow", "ml"),
        (10.5, 0.1, 2.4, 0.55, "Streamlit dashboard", "Week 9 — interactive", "struct"),
    ]

    for (x, y, w, h, label, sub, cat) in nodes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=colors[cat],
            edgecolor=borders[cat],
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, label,
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color=borders[cat])
        ax.text(x + w/2, y + h*0.25, sub,
                ha="center", va="center",
                fontsize=7, color="#555")

    # Main pipeline arrows
    arrow_kw = dict(arrowstyle="-|>", color="#888", lw=1.2)
    main_x = 7.0
    for y_start, y_end in [(9.0, 8.5), (7.8, 7.5), (6.6, 7.3),  # adjusted
                            (7.8, 7.5)]:
        pass

    for (y1, y2) in [(8.7, 8.5), (7.8, 7.5), (6.6, 7.3)]:
        pass

    # Draw vertical arrows for main pipeline
    y_positions = [9.0, 7.8, 6.6, 5.4, 4.2, 3.0, 1.8, 0.6]
    for i in range(len(y_positions)-1):
        y_from = y_positions[i]
        y_to   = y_positions[i+1] + 0.7
        ax.annotate("", xy=(main_x, y_to), xytext=(main_x, y_from),
                    arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5))

    # Side output arrows
    for y_main, y_out in [(4.2, 4.47), (3.0, 3.27), (1.8, 2.07), (0.6, 0.87)]:
        ax.annotate("", xy=(3.2, y_out), xytext=(5.5, y_main+0.35),
                    arrowprops=dict(arrowstyle="-|>", color="#aaa", lw=1,
                                    connectionstyle="arc3,rad=0.0"))

    # Arrows to final outputs
    for y_src in [4.2, 3.0, 1.8, 0.6]:
        ax.annotate("", xy=(10.5, 2.67), xytext=(8.5, y_src+0.35),
                    arrowprops=dict(arrowstyle="-|>", color="#aaa", lw=0.8,
                                    connectionstyle="arc3,rad=0.2"))

    ax.annotate("", xy=(11.7, 1.75), xytext=(11.7, 2.4),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2))
    ax.annotate("", xy=(11.7, 0.65), xytext=(11.7, 1.2),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2))

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=colors["data"],   edgecolor=borders["data"],   label="Data collection"),
        mpatches.Patch(facecolor=colors["ml"],     edgecolor=borders["ml"],     label="ML / Analysis"),
        mpatches.Patch(facecolor=colors["struct"], edgecolor=borders["struct"], label="Structural biology"),
        mpatches.Patch(facecolor=colors["dock"],   edgecolor=borders["dock"],   label="Molecular docking"),
        mpatches.Patch(facecolor=colors["output"], edgecolor=borders["output"], label="Output files"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8,
              framealpha=0.9, edgecolor="#ddd")

    ax.set_title(
        "AI Drug Target Discovery Pipeline — K. pneumoniae",
        fontsize=13, fontweight="bold", pad=10
    )

    Path("visualizations").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("visualizations/pipeline_dag.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: visualizations/pipeline_dag.png")


if __name__ == "__main__":
    main()