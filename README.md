# Logistic Regression — Patron Subscription Renewal (Train + Score)

This project builds a **logistic regression** model in Python to predict whether a theater patron will **renew their season ticket subscription**, then applies the trained model to a separate scoring dataset to generate **predictions and probabilities**.

It is designed as a clean, reproducible **Python workflow** for classification and model interpretation, including:
- Ensuring `RenewedSubscription` is treated as a **binary (0/1)** label
- Retaining `PatronID` as an identifier (**not** a predictor)
- Range-checking scoring observations against training feature ranges (and removing out-of-range rows)
- Fitting a logistic regression model using **all predictors** (no feature removal)
- Exporting interpretability outputs (coefficients, p-values, odds ratios)
- Exporting evaluation artifacts (confusion matrix, ROC curve, AUC)
- Writing an “evidence locker” of outputs (CSVs + PNG visuals) to `output/`

---

## Folder Structure

```text
Logistic Regression/
├─ data/
│  ├─ patron_subscription_renewal_train.csv
│  └─ patron_subscription_renewal_score.csv
├─ src/
│  └─ train_score_patron_subscription_logistic_regression.py
├─ output/
│  └─ (generated files)
└─ README.md
```
How to Run
From the project root folder (Logistic Regression/), run:

python src/train_score_patron_subscription_logistic_regression.py

Outputs (Evidence Locker)
All files are written to output/.

Data validation / range filtering
- training_feature_ranges.csv
- scoring_out_of_range_summary.csv
- scoring_rows_removed_out_of_range.csv (only if rows were removed)

Model outputs (interpretability)
- logistic_regression_model_summary.txt
- model_coefficients_and_odds_ratios.csv
- predictor_pvalues.csv

Performance (training)
- classification_metrics.csv
- confusion_matrix.csv
- roc_curve_points.csv

Scoring predictions
- patron_renewal_predictions.csv
- top10_most_likely_to_renew.csv
- top10_least_likely_to_renew.csv

Visuals (PNG)
- class_balance.png
- probability_distribution_training.png
- confusion_matrix.png
- roc_curve.png
- coefficients_log_odds.png

Notes
- Predictions use a default decision threshold of 0.50 (ProbabilityRenew >= 0.50 → Yes).
- Coefficients are in log-odds. Odds ratios are computed as exp(coefficient).
- The range-check step is a conservative guardrail against scoring inputs outside the training distribution.

🌐 **PixelKraze Analytics (Portfolio):** https://pixelkraze.com/?utm_source=github&utm_medium=readme&utm_campaign=portfolio&utm_content=homepage

