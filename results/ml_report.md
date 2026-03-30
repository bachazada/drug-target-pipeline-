# Week 4 - ML Druggability Prediction Report
Generated: 2026-03-30 23:27

## Model performance
| Metric | Score |
|--------|-------|
| CV ROC-AUC | 0.725 ± 0.019 |
| CV Avg Precision | 0.803 ± 0.023 |
| Folds | 5-fold stratified |

## Target ranking summary
- Total candidates scored: 95
- Priority 1 (score ≥ 0.75): 14 targets → proceed to ColabFold
- Model: Random Forest, 300 trees, balanced class weight

## Top 10 targets
 rank gene  druggability_score          priority  molecular_weight  isoelectric_point
    1 ftsI              0.8347 Priority 1 (high)          64001.87              9.589
    2 pheT              0.8064 Priority 1 (high)          86962.35              5.058
    3 metG              0.8019 Priority 1 (high)          76151.72              5.319
    4 fabI              0.8017 Priority 1 (high)          27910.75              5.433
    5 rpoC              0.7995 Priority 1 (high)         155260.44              6.465
    6 ftsI              0.7903 Priority 1 (high)          63366.92              6.506
    7 rpsC              0.7864 Priority 1 (high)          25838.74             10.295
    8 murA              0.7841 Priority 1 (high)          44584.74              5.613
    9 murC              0.7796 Priority 1 (high)          53491.12              5.878
   10 gyrA              0.7660 Priority 1 (high)          97031.05              4.976

## Files saved
- models/model.pkl
- results/target_scores.csv
- visualizations/ml_feature_importance.png
- visualizations/ml_roc_curve.png
- visualizations/ml_score_distribution.png

## Next step
    python scripts/05_structure_prediction.py
    (Structure prediction for top 19 candidates via ColabFold)
