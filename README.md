# AI-Assisted Drug Target Discovery Pipeline

**Author:** Bacha Zada — M.Sc. Computational Biology & Bioinformatics, University of Göttingen  
**Target organism:** *Klebsiella pneumoniae* (ATCC 43816 / MGH 78578)  
**Goal:** Rank and validate drug targets using ML + structural biology + molecular docking

---

## Pipeline overview

```
UniProt proteome → Biological filtering → ML druggability scoring
→ Structure prediction (ColabFold) → Pocket detection (P2Rank)
→ Docking (AutoDock Vina) → Ranked targets → Streamlit dashboard
```

## Project structure

```
drug_target_pipeline/
├── data/
│   ├── proteome.fasta          # K. pneumoniae full proteome
│   ├── essential_genes.txt     # DEG-validated essential genes
│   └── filtered_targets.fasta  # After biological filtering (Week 2)
├── results/
│   ├── structures/             # ColabFold .pdb outputs (Week 5)
│   ├── pockets/                # P2Rank binding sites (Week 6)
│   └── docking/                # AutoDock Vina scores (Week 7)
├── models/                     # Trained ML model (Week 4)
├── notebooks/                  # Jupyter analysis notebooks
├── scripts/                    # Python scripts for each step
├── visualizations/             # Plots and figures
├── Snakefile                   # Full pipeline (Week 8)
├── config.yaml                 # All parameters in one place
├── environment.yml             # Reproducible conda environment
└── README.md
```

## Quick start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate drug_target_pipeline

# 2. Download data
python scripts/01_download_data.py

# 3. Run full pipeline (after Week 8)
snakemake --cores 4
```

## Data sources

| Data | Source | URL |
|------|--------|-----|
| K. pneumoniae proteome | UniProt | https://www.uniprot.org |
| Essential genes | DEG database | http://www.essentialgene.org |
| Known drug targets | DrugBank | https://go.drugbank.com |
| Protein structures | RCSB PDB | https://www.rcsb.org |
| Docking ligands | ZINC database | https://zinc.docking.org |

## Weekly progress

- [x] Week 1 — Project setup + data collection
- [ ] Week 2 — Biological filtering
- [ ] Week 3 — Feature engineering
- [ ] Week 4 — ML model
- [ ] Week 5 — Structure prediction
- [ ] Week 6 — Pocket detection
- [ ] Week 7 — Docking
- [ ] Week 8 — Snakemake pipeline
- [ ] Week 9 — Streamlit dashboard
- [ ] Week 10 — Polish + deployment
