"""
DUAL MODEL TRAINING SCRIPT
===========================
Trains TWO ML models for comparison:
1. XGBoost (Conservative: max_depth=3, high regularization)
2. XGBoost (Aggressive: max_depth=5, balanced approach)

Evaluates both on test set and saves the best performer.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

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

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

MODEL_1_FILE = "credit_model_v1.pkl"  # Conservative model
MODEL_2_FILE = "credit_model_v2.pkl"  # Aggressive model
FINAL_MODEL_FILE = "credit_model.pkl"  # Best model (replaces old one)
COLS_FILE = "feature_columns.pkl"
RESULTS_FILE = "training_results.txt"

print("=" * 90)
print("DUAL ML MODEL TRAINING PIPELINE")
print("=" * 90)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("STEP 1: LOAD DATA")
print("─" * 90)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print(f"\n✓ Training data: {len(X_train)} rows")
print(f"✓ Test data: {len(X_test)} rows")
print(f"✓ Features: {len(FEATURES)}")
print(f"\nClass distribution (Train):")
print(f"  - Non-default (0): {(y_train == 0).sum()} ({(y_train == 0).mean():.1%})")
print(f"  - Default (1):     {(y_train == 1).sum()} ({(y_train == 1).mean():.1%})")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: TRAIN MODEL 1 (CONSERVATIVE)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("STEP 2: TRAIN MODEL 1 - CONSERVATIVE (max_depth=3, High Regularization)")
print("─" * 90)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print(f"\nClass weight ratio: {scale_pos_weight:.2f}")
print("Hyperparameters:")
print("  n_estimators=250, max_depth=3, learning_rate=0.05")
print("  subsample=0.7, colsample_bytree=0.7, min_child_weight=8")
print("  reg_alpha=0.5, reg_lambda=2.0 (strong regularization)")

model_1 = XGBClassifier(
    n_estimators=250,
    max_depth=3,  # Shallower, more conservative
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=8,  # Prevent overfitting
    gamma=0.2,
    reg_alpha=0.5,  # Stronger L1 regularization
    reg_lambda=2.0,  # Stronger L2 regularization
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

print("\nTraining Model 1...")
model_1.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print("✓ Model 1 training complete")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: TRAIN MODEL 2 (AGGRESSIVE)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("STEP 3: TRAIN MODEL 2 - BALANCED (max_depth=5, Moderate Regularization)")
print("─" * 90)

print("\nHyperparameters:")
print("  n_estimators=300, max_depth=5, learning_rate=0.06")
print("  subsample=0.8, colsample_bytree=0.8, min_child_weight=5")
print("  reg_alpha=0.1, reg_lambda=1.0 (moderate regularization)")

model_2 = XGBClassifier(
    n_estimators=300,
    max_depth=5,  # Deeper, captures more patterns
    learning_rate=0.06,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,  # Moderate L1 regularization
    reg_lambda=1.0,  # Moderate L2 regularization
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

print("\nTraining Model 2...")
model_2.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print("✓ Model 2 training complete")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: EVALUATE BOTH MODELS
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("STEP 4: EVALUATE BOTH MODELS ON TEST SET")
print("─" * 90)

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{model_name}:")
    print(f"  AUC-ROC:      {auc:.4f}")
    print(f"  Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  F1 Score:     {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives:  {tn} | False Positives: {fp}")
    print(f"    False Negatives: {fn} | True Positives:  {tp}")
    
    return {
        'name': model_name,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
    }

metrics_1 = evaluate_model(model_1, X_test, y_test, "MODEL 1 (Conservative)")
metrics_2 = evaluate_model(model_2, X_test, y_test, "MODEL 2 (Balanced)")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: COMPARE AND SELECT BEST
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("STEP 5: MODEL COMPARISON & SELECTION")
print("─" * 90)

# Create comparison table
print("\n┌─ Comparison Table ─────────────────────────────────────────────────────────┐")
print("│ Metric        │ Model 1 (Conservative) │ Model 2 (Balanced)  │ Better       │")
print("├───────────────┼────────────────────────┼─────────────────────┼──────────────┤")

metrics_to_compare = ['auc', 'accuracy', 'precision', 'recall', 'f1']
scores = {}

for metric in metrics_to_compare:
    m1_val = metrics_1[metric]
    m2_val = metrics_2[metric]
    better = "M1 ✓" if m1_val > m2_val else ("M2 ✓" if m2_val > m1_val else "Tie")
    print(f"│ {metric:13} │ {m1_val:22.4f} │ {m2_val:19.4f} │ {better:12} │")
    scores[metric] = (m1_val, m2_val)

print("└───────────────┴────────────────────────────────────────────────────────────┘")

# Select best model based on weighted score
# Weight: AUC (40%), Recall (40%), Precision (20%)
score_1 = (metrics_1['auc'] * 0.4) + (metrics_1['recall'] * 0.4) + (metrics_1['precision'] * 0.2)
score_2 = (metrics_2['auc'] * 0.4) + (metrics_2['recall'] * 0.4) + (metrics_2['precision'] * 0.2)

print(f"\nWeighted Score (AUC: 40%, Recall: 40%, Precision: 20%):")
print(f"  Model 1: {score_1:.4f}")
print(f"  Model 2: {score_2:.4f}")

if score_1 > score_2:
    best_model = model_1
    best_metrics = metrics_1
    best_name = "MODEL 1 (Conservative)"
    best_file = MODEL_1_FILE
else:
    best_model = model_2
    best_metrics = metrics_2
    best_name = "MODEL 2 (Balanced)"
    best_file = MODEL_2_FILE

print(f"\n🏆 SELECTED: {best_name}")
print(f"   Reason: Higher weighted score (AUC + Recall prioritized)")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: SAVE MODELS
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("STEP 6: SAVE MODELS")
print("─" * 90)

print("\nSaving Model 1 (Conservative)...")
joblib.dump(model_1, MODEL_1_FILE)
print(f"  ✓ {MODEL_1_FILE}")

print("Saving Model 2 (Balanced)...")
joblib.dump(model_2, MODEL_2_FILE)
print(f"  ✓ {MODEL_2_FILE}")

print(f"\nSaving BEST MODEL as {FINAL_MODEL_FILE}...")
joblib.dump(best_model, FINAL_MODEL_FILE)
print(f"  ✓ {FINAL_MODEL_FILE}")

print("\nSaving feature columns...")
joblib.dump(FEATURES, COLS_FILE)
print(f"  ✓ {COLS_FILE}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7: FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
print("─" * 90)

# Get feature importance from best model
importance = best_model.get_booster().get_score(importance_type='weight')
feature_importance = {f: importance.get(f, 0) for f in FEATURES}
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop Features ({best_name}):")
for i, (feat, imp) in enumerate(sorted_importance, 1):
    pct = (imp / sum(feature_importance.values())) * 100 if sum(feature_importance.values()) > 0 else 0
    bar = "█" * int(pct / 5)
    print(f"  {i}. {feat:30} {bar:20} {pct:5.1f}%")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8: SUMMARY REPORT
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 90)

summary = f"""
DUAL MODEL TRAINING RESULTS
═══════════════════════════════════════════════════════════════════════════

Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Training Set:  {len(X_train)} samples
Test Set:      {len(X_test)} samples
Features:      {len(FEATURES)}

─────────────────────────────────────────────────────────────────────────

MODEL 1 (CONSERVATIVE)
  Architecture: max_depth=3, n_estimators=250
  Regularization: Strong (alpha=0.5, lambda=2.0)
  Purpose: Reduce overfitting, prioritize stability
  
  Performance:
    AUC-ROC:  {metrics_1['auc']:.4f}
    Accuracy: {metrics_1['accuracy']:.4f}
    Recall:   {metrics_1['recall']:.4f}
    Precision:{metrics_1['precision']:.4f}
    F1 Score: {metrics_1['f1']:.4f}
  
  Saved: {MODEL_1_FILE}

─────────────────────────────────────────────────────────────────────────

MODEL 2 (BALANCED)
  Architecture: max_depth=5, n_estimators=300
  Regularization: Moderate (alpha=0.1, lambda=1.0)
  Purpose: Balance bias and variance
  
  Performance:
    AUC-ROC:  {metrics_2['auc']:.4f}
    Accuracy: {metrics_2['accuracy']:.4f}
    Recall:   {metrics_2['recall']:.4f}
    Precision:{metrics_2['precision']:.4f}
    F1 Score: {metrics_2['f1']:.4f}
  
  Saved: {MODEL_2_FILE}

─────────────────────────────────────────────────────────────────────────

BEST MODEL SELECTED: {best_name}
  Weighted Score: {max(score_1, score_2):.4f}
  AUC-ROC:        {best_metrics['auc']:.4f}
  Accuracy:       {best_metrics['accuracy']:.4f}
  Recall:         {best_metrics['recall']:.4f}
  Precision:      {best_metrics['precision']:.4f}
  F1 Score:       {best_metrics['f1']:.4f}
  
  Confusion Matrix:
    True Negatives:  {best_metrics['cm'][0,0]}
    False Positives: {best_metrics['cm'][0,1]}
    False Negatives: {best_metrics['cm'][1,0]}
    True Positives:  {best_metrics['cm'][1,1]}
  
  Saved (Primary): {FINAL_MODEL_FILE}
  Saved (Backup):  {best_file}

─────────────────────────────────────────────────────────────────────────

FILES GENERATED
  ✓ {FINAL_MODEL_FILE}          (Best model for production)
  ✓ {MODEL_1_FILE}              (Conservative model)
  ✓ {MODEL_2_FILE}              (Balanced model)
  ✓ {COLS_FILE}                 (Feature columns)
  ✓ training_results.txt         (This report)

═════════════════════════════════════════════════════════════════════════════

Next Step: Update ml_engine.py to load {FINAL_MODEL_FILE}

═════════════════════════════════════════════════════════════════════════════
"""

print(summary)

# Save report to file
with open(RESULTS_FILE, 'w') as f:
    f.write(summary)

print(f"\n✓ Report saved: {RESULTS_FILE}")
print("\n" + "=" * 90)
print("TRAINING PIPELINE COMPLETE")
print("=" * 90)
