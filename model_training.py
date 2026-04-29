"""
Model Training
==============
Trains XGBoost on the 8 behavior features from train.csv.
Evaluates on test.csv. Saves model + feature columns.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

# ─── Config ───────────────────────────────────────────────────────────────────

FEATURES = [
    "income_regularity",
    "spending_consistency",
    "bill_payment_ratio",
    "savings_rate",
    "emi_burden_ratio",
    "upi_activity_volume",
    "rent_payment_regularity",
    "cash_withdrawal_ratio",
]
TARGET = "TARGET"

TRAIN_FILE  = "train.csv"
TEST_FILE   = "test.csv"
MODEL_FILE  = "credit_model.pkl"
COLS_FILE   = "feature_columns.pkl"
FI_PLOT     = "feature_importance.png"
CM_PLOT     = "confusion_matrix.png"

# ─── Load Data ────────────────────────────────────────────────────────────────

print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test  = test_df[FEATURES]
y_test  = test_df[TARGET]

print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"  Train default rate: {y_train.mean():.1%}")
print(f"  Test  default rate: {y_test.mean():.1%}")

# ─── Train Model ──────────────────────────────────────────────────────────────

print("\nTraining XGBoost...")

# Class imbalance weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"  Class ratio (neg/pos): {scale_pos_weight:.2f}")

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

print("  Training complete.")

# ─── Evaluate ─────────────────────────────────────────────────────────────────

print("\nEvaluating on test set...")

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred       = (y_pred_proba >= 0.5).astype(int)

auc       = roc_auc_score(y_test, y_pred_proba)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print(f"\n  {'Metric':<20} {'Value':>8}")
print(f"  {'-'*30}")
print(f"  {'AUC-ROC':<20} {auc:>8.4f}")
print(f"  {'Accuracy':<20} {accuracy:>8.4f}")
print(f"  {'Precision':<20} {precision:>8.4f}")
print(f"  {'Recall':<20} {recall:>8.4f}")
print(f"  {'F1 Score':<20} {f1:>8.4f}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

# ─── Feature Importance Plot ──────────────────────────────────────────────────

print("Saving feature importance plot...")

fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#7c3aed" if v == fi.max() else "#a78bfa" for v in fi.values]
bars = ax.barh(fi.index, fi.values, color=colors, edgecolor="white", height=0.6)
ax.set_xlabel("Feature Importance (XGBoost gain)", fontsize=11)
ax.set_title("Credit Scoring Model — Feature Importance", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, fi.max() * 1.15)

for bar, val in zip(bars, fi.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(FI_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FI_PLOT}")

# ─── Confusion Matrix Plot ────────────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
disp.plot(ax=ax, colorbar=False, cmap="Purples")
ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(CM_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {CM_PLOT}")

# ─── Save Model ───────────────────────────────────────────────────────────────

joblib.dump(model, MODEL_FILE)
joblib.dump(FEATURES, COLS_FILE)
print(f"\n  Model saved → {MODEL_FILE}")
print(f"  Feature columns saved → {COLS_FILE}")

print("\n✅ model_training.py complete.")
