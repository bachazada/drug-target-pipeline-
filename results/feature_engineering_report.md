# Week 3 - Feature Engineering Report
Generated: 2026-03-30 18:10

## Feature matrix
- Proteins: 95
- Features: 45 numerical + 2 ID columns
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
         length  molecular_weight  isoelectric_point  instability_index   gravy  hydrophobic_fraction
count    95.000            95.000             95.000             95.000  95.000                95.000
mean    344.063         38008.661              7.071             36.161  -0.152                 0.334
std     241.942         26866.055              2.083             11.671   0.356                 0.047
min      61.000          6734.880              4.233              4.049  -1.090                 0.211
25%     154.500         17159.825              5.435             28.992  -0.340                 0.313
50%     306.000         33882.670              6.084             35.201  -0.193                 0.325
75%     431.000         47366.325              9.512             43.803   0.007                 0.344
max    1407.000        155260.440             11.151             87.245   1.267                 0.515

## Visualisations saved
- visualizations/feature_distributions.png
- visualizations/feature_correlation.png
- visualizations/aa_composition_heatmap.png
- visualizations/top_features_variance.png

## Next step
    python scripts/04_ml_model.py
