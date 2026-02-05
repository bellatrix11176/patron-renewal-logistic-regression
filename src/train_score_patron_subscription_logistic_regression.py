from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

# =============================================================================
# Patron Subscription Renewal — Logistic Regression (Train + Score + Evidence Locker)
# =============================================================================

LABEL_COL = "RenewedSubscription"
ID_COL = "PatronID"
THRESHOLD = 0.50  # classification threshold

# -----------------------------
# Repo-friendly paths (anchored to project root)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Logistic Regression
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "patron_subscription_renewal_train.csv"
SCORE_PATH = DATA_DIR / "patron_subscription_renewal_score.csv"


def to_binary(series: pd.Series) -> pd.Series:
    """
    Convert common binominal label formats to 0/1.
    Accepts: 0/1, True/False, Yes/No, Y/N, Renewed/Not Renewed (case-insensitive).
    """
    s = series.copy()

    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    coerced = pd.to_numeric(s, errors="coerce")
    if coerced.notna().all():
        unique_vals = sorted(coerced.unique().tolist())
        if set(unique_vals).issubset({0, 1}):
            return coerced.astype(int)
        return (coerced == max(unique_vals)).astype(int)

    s_str = s.astype(str).str.strip().str.lower()
    positive = {"1", "true", "yes", "y", "renewed", "renew", "renewedsubscription"}
    negative = {"0", "false", "no", "n", "not renewed", "not_renewed", "did not renew", "didn't renew"}

    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s_str.isin(positive)] = 1
    out[s_str.isin(negative)] = 0

    if out.isna().any():
        vals = s_str.value_counts().index.tolist()
        if len(vals) == 2:
            out[s_str == vals[0]] = 0
            out[s_str == vals[1]] = 1

    if out.isna().any():
        bad = sorted(s_str[out.isna()].unique().tolist())
        raise ValueError(f"Unrecognized label values in {LABEL_COL}: {bad}")

    return out.astype(int)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    # -----------------------------
    # Load data
    # -----------------------------
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")
    if not SCORE_PATH.exists():
        raise FileNotFoundError(f"Scoring file not found: {SCORE_PATH}")

    train = pd.read_csv(TRAIN_PATH)
    score = pd.read_csv(SCORE_PATH)

    if LABEL_COL not in train.columns:
        raise ValueError(f"Training file is missing label column: {LABEL_COL}")
    if ID_COL not in train.columns or ID_COL not in score.columns:
        raise ValueError(f"Expected ID column '{ID_COL}' in both train and score.")

    y = to_binary(train[LABEL_COL])
    feature_cols = [c for c in train.columns if c not in [LABEL_COL, ID_COL]]

    missing_in_score = sorted(set(feature_cols) - set(score.columns))
    if missing_in_score:
        raise ValueError(f"Scoring data is missing these feature columns: {missing_in_score}")

    # Coerce features to numeric
    for col in feature_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        score[col] = pd.to_numeric(score[col], errors="coerce")

    # Build modeling frame (drop missing predictors/label)
    train_model = train.copy()
    train_model[LABEL_COL] = y
    train_model = train_model.dropna(subset=feature_cols + [LABEL_COL]).copy()

    y_model = train_model[LABEL_COL].astype(int)
    X_model = train_model[feature_cols].copy()

    # -----------------------------
    # Range-filter scoring set vs training ranges
    # -----------------------------
    ranges = train_model[feature_cols].agg(["min", "max"]).T.rename(columns={"min": "train_min", "max": "train_max"})
    ranges.index.name = "feature"
    ranges.to_csv(OUT_DIR / "training_feature_ranges.csv")

    out_of_range_summary = []
    in_range_mask = pd.Series(True, index=score.index)

    for col in feature_cols:
        col_min = float(ranges.loc[col, "train_min"])
        col_max = float(ranges.loc[col, "train_max"])

        below = score[col] < col_min
        above = score[col] > col_max

        out_of_range_summary.append(
            {
                "feature": col,
                "train_min": col_min,
                "train_max": col_max,
                "score_below_min_count": int(below.sum()),
                "score_above_max_count": int(above.sum()),
                "score_out_of_range_count": int((below | above).sum()),
            }
        )

        in_range_mask &= score[col].between(col_min, col_max, inclusive="both") & score[col].notna()

    pd.DataFrame(out_of_range_summary).sort_values(
        by="score_out_of_range_count", ascending=False
    ).to_csv(OUT_DIR / "scoring_out_of_range_summary.csv", index=False)

    score_in_range = score[in_range_mask].copy()
    score_removed = score[~in_range_mask].copy()
    if not score_removed.empty:
        score_removed.to_csv(OUT_DIR / "scoring_rows_removed_out_of_range.csv", index=False)

    # -----------------------------
    # Logistic Regression (keep ALL predictors)
    # -----------------------------
    X = sm.add_constant(X_model, has_constant="add")
    model = sm.Logit(y_model, X).fit(disp=False)

    with open(OUT_DIR / "logistic_regression_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    params = model.params
    pvals = model.pvalues

    coef_df = pd.DataFrame(
        {
            "feature": params.index,
            "coefficient_log_odds": params.values,
            "p_value": pvals.reindex(params.index).values,
            "odds_ratio": np.exp(params.values),
        }
    )
    coef_df.to_csv(OUT_DIR / "model_coefficients_and_odds_ratios.csv", index=False)

    pvals.drop(labels=["const"], errors="ignore").sort_values().to_csv(
        OUT_DIR / "predictor_pvalues.csv", header=["p_value"]
    )

    # -----------------------------
    # Training performance (threshold, CM, ROC, AUC)
    # -----------------------------
    prob_train = model.predict(X).to_numpy()
    pred_train = (prob_train >= THRESHOLD).astype(int)

    accuracy = float(accuracy_score(y_model, pred_train))
    precision = float(precision_score(y_model, pred_train, zero_division=0))
    recall = float(recall_score(y_model, pred_train, zero_division=0))
    auc = float(roc_auc_score(y_model, prob_train))

    cm = confusion_matrix(y_model, pred_train, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    pd.DataFrame(
        {
            "metric": ["accuracy", "precision", "recall", "auc", "threshold", "n_train_used"],
            "value": [accuracy, precision, recall, auc, THRESHOLD, int(len(train_model))],
        }
    ).to_csv(OUT_DIR / "classification_metrics.csv", index=False)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_No", "Actual_Yes"],
        columns=["Pred_No", "Pred_Yes"],
    )
    cm_df.to_csv(OUT_DIR / "confusion_matrix.csv", index=True)

    fpr, tpr, roc_thresholds = roc_curve(y_model, prob_train)
    roc_points = pd.DataFrame({"threshold": roc_thresholds, "fpr": fpr, "tpr": tpr})
    roc_points.to_csv(OUT_DIR / "roc_curve_points.csv", index=False)

    # -----------------------------
    # Score the (filtered) scoring dataset
    # -----------------------------
    X_score = sm.add_constant(score_in_range[feature_cols], has_constant="add")
    valid_score_mask = X_score.notna().all(axis=1)

    score_in_range_valid = score_in_range.loc[valid_score_mask].copy()
    X_score_valid = X_score.loc[valid_score_mask]

    prob_score = model.predict(X_score_valid).to_numpy()
    pred_score = (prob_score >= THRESHOLD).astype(int)

    scored_df = pd.DataFrame(
        {
            ID_COL: score_in_range_valid[ID_COL].values,
            "PredictedRenewal": np.where(pred_score == 1, "Yes", "No"),
            "ProbabilityRenew": prob_score,
            "ProbabilityNotRenew": 1.0 - prob_score,
        }
    )
    scored_df.to_csv(OUT_DIR / "patron_renewal_predictions.csv", index=False)

    scored_df.sort_values("ProbabilityRenew", ascending=False).head(10).to_csv(
        OUT_DIR / "top10_most_likely_to_renew.csv", index=False
    )
    scored_df.sort_values("ProbabilityRenew", ascending=True).head(10).to_csv(
        OUT_DIR / "top10_least_likely_to_renew.csv", index=False
    )

    # "Most persuadable" ≈ closest to 0.50
    scored_df["DistanceFrom50"] = np.abs(scored_df["ProbabilityRenew"] - 0.50)
    persuadable_row = scored_df.sort_values("DistanceFrom50", ascending=True).iloc[0]
    persuadable_patron_id = int(persuadable_row[ID_COL])
    persuadable_prob = float(persuadable_row["ProbabilityRenew"])

    predicted_renew_count = int((pred_score == 1).sum())

    pvals_no_const = pvals.drop(labels=["const"], errors="ignore")
    poorest_predictor = str(pvals_no_const.sort_values(ascending=False).index[0])
    poorest_pvalue = float(pvals_no_const[poorest_predictor])

    # -----------------------------
    # VISUALS
    # -----------------------------
    # 1) Class balance
    plt.figure()
    classes, class_counts = np.unique(y_model.to_numpy(), return_counts=True)
    plt.bar([str(c) for c in classes], class_counts)
    plt.title("Training Class Balance (RenewedSubscription)")
    plt.xlabel("Class (0=No, 1=Yes)")
    plt.ylabel("Count")
    save_fig(OUT_DIR / "class_balance.png")

    # 2) Probability distribution (training)
    plt.figure()
    plt.hist(prob_train, bins=30)
    plt.title("Predicted Probability of Renewal (Training)")
    plt.xlabel("ProbabilityRenew")
    plt.ylabel("Count")
    save_fig(OUT_DIR / "probability_distribution_training.png")

    # 3) Confusion matrix heatmap
    plt.figure()
    plt.imshow(cm, aspect="auto")
    plt.title("Confusion Matrix (Training)")
    plt.xticks([0, 1], ["Pred_No", "Pred_Yes"])
    plt.yticks([0, 1], ["Actual_No", "Actual_Yes"])
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(int(v)), ha="center", va="center")
    save_fig(OUT_DIR / "confusion_matrix.png")

    # 4) ROC curve
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.title(f"ROC Curve (Training) — AUC={auc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    save_fig(OUT_DIR / "roc_curve.png")

    # 5) Coefficients (log-odds), excluding intercept
    coef_plot = coef_df[coef_df["feature"] != "const"].copy().sort_values("coefficient_log_odds")
    plt.figure(figsize=(10, 6))
    plt.barh(coef_plot["feature"], coef_plot["coefficient_log_odds"])
    plt.title("Logistic Regression Coefficients (Log-Odds)")
    plt.xlabel("Coefficient (log-odds)")
    plt.ylabel("Feature")
    save_fig(OUT_DIR / "coefficients_log_odds.png")

    # -----------------------------
    # Summary output
    # -----------------------------
    missing_predictor_rows = int((~valid_score_mask).sum())

    print("\n=== OUTPUT EVIDENCE LOCKER ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Training file: {TRAIN_PATH}")
    print(f"Scoring file:  {SCORE_PATH}")
    print(f"\nRemoved scoring rows (out of range): {len(score_removed)}")
    print(f"Also removed scoring rows (missing predictors): {missing_predictor_rows}")

    print("\nTraining performance (threshold=0.50):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

    print("\nScoring summary:")
    print(f"  Predicted renewals (Yes): {predicted_renew_count}")
    print(f"  Most persuadable PatronID (closest to 0.50): {persuadable_patron_id} (p={persuadable_prob:.4f})")

    print("\nModel insight:")
    print(f"  Poorest predictor (highest p-value): {poorest_predictor} (p={poorest_pvalue:.6f})")

    print("\nFiles written to output/:")
    print(" - training_feature_ranges.csv")
    print(" - scoring_out_of_range_summary.csv")
    if not score_removed.empty:
        print(" - scoring_rows_removed_out_of_range.csv")
    print(" - logistic_regression_model_summary.txt")
    print(" - model_coefficients_and_odds_ratios.csv")
    print(" - predictor_pvalues.csv")
    print(" - classification_metrics.csv")
    print(" - confusion_matrix.csv")
    print(" - roc_curve_points.csv")
    print(" - patron_renewal_predictions.csv")
    print(" - top10_most_likely_to_renew.csv")
    print(" - top10_least_likely_to_renew.csv")
    print(" - class_balance.png")
    print(" - probability_distribution_training.png")
    print(" - confusion_matrix.png")
    print(" - roc_curve.png")
    print(" - coefficients_log_odds.png")


if __name__ == "__main__":
    main()
