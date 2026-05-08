# Methods — AI Drug Target Discovery Pipeline
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
