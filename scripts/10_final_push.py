#!/usr/bin/env python3
"""
Week 10 — Final GitHub Push + Deployment
Run: python scripts/10_final_push.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run(cmd):
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip():
        print(f"    {r.stdout.strip()[:200]}")
    if r.returncode != 0 and r.stderr.strip():
        print(f"    ! {r.stderr.strip()[:200]}")
    return r.returncode == 0


def write_gitignore():
    content = """# Python
__pycache__/
*.pyc
.eggs/

# Jupyter
.ipynb_checkpoints/

# Large data
data/proteome.fasta
data/human_proteome.fasta
data/human_proteome_db.*
data/query_targets.fasta
data/training_positives.csv
data/training_negatives.csv

# Large structure files
results/structures/*.pdb
results/colabfold_input/combined_targets.fasta

# Docking intermediates
results/docking/*.pdbqt
results/blast_human_hits.*

# Models
models/*.pkl
models/*.joblib

# Tools
tools/
/tmp/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Snakemake
.snakemake/
logs/

# Keep empty dirs
!data/.gitkeep
!results/.gitkeep
!models/.gitkeep
"""
    Path(".gitignore").write_text(content)
    print("[✓] .gitignore updated")


def write_license():
    content = f"""MIT License

Copyright (c) {datetime.now().year} Bacha Zada

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""
    Path("LICENSE").write_text(content)
    print("[✓] LICENSE created")


def main():
    print("=" * 55)
    print(" Week 10 - Final GitHub Push")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55 + "\n")

    write_gitignore()
    write_license()

    print("\n[Git] Staging files...")
    for f in [
        "README.md", "app.py", "Snakefile", "config.yaml",
        "environment.yml", "requirements.txt", ".gitignore", "LICENSE",
        "scripts/", "docs/",
        "results/combined_results.csv",
        "results/target_scores.csv",
        "results/docking_scores.csv",
        "results/pocket_summary.csv",
        "results/structure_results_local.csv",
        "results/pipeline_complete.txt",
        "data/features.csv",
        "data/essential_genes.txt",
    ]:
        run(f"git add {f}")

    run("git add visualizations/*.png")

    print("\n[Git] Committing...")
    run('git commit -m "Week 10: complete AI drug target discovery pipeline - '
        'RF AUC 0.725, ColabFold, P2Rank, AutoDock Vina, Streamlit dashboard"')

    print("\n[Git] Pushing...")
    if not run("git push origin main"):
        run("git push origin master")

    print("""
╔══════════════════════════════════════════════════════╗
║  DEPLOY TO STREAMLIT CLOUD                           ║
╠══════════════════════════════════════════════════════╣
║  1. https://share.streamlit.io                       ║
║  2. Sign in with GitHub                              ║
║  3. New app → repo: bachazada/drug-target-pipeline   ║
║  4. Main file: app.py  →  Deploy                     ║
║                                                      ║
║  Add to CV:                                          ║
║  github.com/bachazada/drug-target-pipeline           ║
╚══════════════════════════════════════════════════════╝
""")
    print("=" * 55)
    print(" Week 10 complete!")
    print("=" * 55)


if __name__ == "__main__":
    main()
