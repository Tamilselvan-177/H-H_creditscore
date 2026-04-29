"""
SHAP Explainability
====================
Runs SHAP TreeExplainer on a single prediction.
Returns risk score, decision, base value, and per-feature SHAP values.
"""

import joblib
import shap
import numpy as np
import pandas as pd

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_FILE   = "credit_model.pkl"
COLS_FILE    = "feature_columns.pkl"
THRESHOLD    = 0.5   # risk_score > 0.5 → rejected

# ─── Load once at module level (cached) ───────────────────────────────────────

_model    = None
_features = None
_explainer = None


def _load():
    global _model, _features, _explainer
    if _model is None:
        _model    = joblib.load(MODEL_FILE)
        _features = joblib.load(COLS_FILE)
        _explainer = shap.TreeExplainer(_model)


def explain_prediction(feature_values: dict) -> dict:
    """
    Run SHAP on a single user's feature values.

    Parameters
    ----------
    feature_values : dict
        Keys must match the 8 feature column names.
        Example:
            {
                "income_regularity": 0.82,
                "spending_consistency": 0.74,
                "bill_payment_ratio": 0.60,
                "savings_rate": 0.12,
                "emi_burden_ratio": 0.65,
                "upi_activity_volume": 18.5,
                "rent_payment_regularity": 0.45,
                "cash_withdrawal_ratio": 0.22,
            }

    Returns
    -------
    dict with:
        risk_score   : float  (0–1, probability of default)
        decision     : str    ("approved" or "rejected")
        base_value   : float  (SHAP baseline)
        shap_values  : dict   {feature: shap_value}
        top_5_features : list of dicts sorted by abs(shap_value) desc
    """
    _load()

    # Build single-row DataFrame in correct column order
    row = pd.DataFrame([{f: feature_values.get(f, 0.0) for f in _features}])

    # Predict probability of default
    risk_score = float(_model.predict_proba(row)[0, 1])
    decision   = "rejected" if risk_score > THRESHOLD else "approved"

    # SHAP values
    shap_vals = _explainer.shap_values(row)

    # shap_vals shape depends on XGBoost version:
    # older → 2D array (n_samples, n_features)
    # newer → list of arrays or 3D
    if isinstance(shap_vals, list):
        # binary classification: index 1 = positive class
        sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    elif shap_vals.ndim == 3:
        sv = shap_vals[0, :, 1]
    else:
        sv = shap_vals[0]

    base_value = float(_explainer.expected_value)
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])

    shap_dict = {feat: round(float(sv[i]), 6) for i, feat in enumerate(_features)}

    # Sort by absolute SHAP value descending
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_5 = [
        {
            "feature":    feat,
            "shap_value": val,
            "impact":     "hurts" if val > 0 else "helps",
        }
        for feat, val in sorted_features[:5]
    ]

    return {
        "risk_score":     round(risk_score, 6),
        "decision":       decision,
        "threshold":      THRESHOLD,
        "base_value":     round(base_value, 6),
        "shap_values":    shap_dict,
        "top_5_features": top_5,
    }


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    sample = {
        "income_regularity":      0.82,
        "spending_consistency":   0.74,
        "bill_payment_ratio":     0.60,
        "savings_rate":           0.12,
        "emi_burden_ratio":       0.65,
        "upi_activity_volume":    18.5,
        "rent_payment_regularity": 0.45,
        "cash_withdrawal_ratio":  0.22,
    }

    result = explain_prediction(sample)
    print(json.dumps(result, indent=2))
    print(f"\nDecision: {result['decision'].upper()}  (risk={result['risk_score']:.4f})")
    print("\nTop 5 SHAP features:")
    for item in result["top_5_features"]:
        sign = "▲" if item["impact"] == "hurts" else "▼"
        print(f"  {sign} {item['feature']:<30} {item['shap_value']:+.4f}  ({item['impact']})")
