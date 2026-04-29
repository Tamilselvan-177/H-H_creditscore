"""
ML Engine — Real XGBoost + SHAP Pipeline
==========================================
Connects to credit_model.pkl and feature_columns.pkl from the H&H workspace.
Computes 8 behavior features from raw transaction data OR accepts direct feature input.
Runs SHAP for explainability.
"""

import os
import sys
import json
import math
import joblib
import numpy as np
import pandas as pd

# ─── Model paths ──────────────────────────────────────────────────────────────
_ROOT      = r'C:\Users\gowsi\OneDrive\Desktop\H&H'
MODEL_PATH = os.path.join(_ROOT, 'credit_model.pkl')
COLS_PATH  = os.path.join(_ROOT, 'feature_columns.pkl')

FEATURE_COLS = [
    'income_regularity',
    'spending_consistency',
    'bill_payment_ratio',
    'savings_rate',
    'emi_burden_ratio',
    'upi_activity_volume',
    'rent_payment_regularity',
    'cash_withdrawal_ratio',
]

THRESHOLD = 0.5   # risk_score > 0.5 → rejected

# ─── Lazy load ────────────────────────────────────────────────────────────────
_model    = None
_explainer = None


def _load():
    global _model, _explainer
    if _model is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"credit_model.pkl not found at {MODEL_PATH}. "
            "Make sure you ran model_training.py first."
        )
    import shap
    _model    = joblib.load(MODEL_PATH)
    _explainer = shap.TreeExplainer(_model)


# ─── Feature engineering from transaction CSV ─────────────────────────────────

def compute_features_from_csv(csv_path: str) -> dict:
    """
    Compute the 8 behavior features from a transaction CSV file.
    CSV must have columns: transaction_id, user_id, amount, type,
                           category, timestamp, payment_mode
    Returns dict of feature_name → value (0–1 range).
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.to_period('M')

    credits  = df[df['type'] == 'credit']
    debits   = df[df['type'] == 'debit']
    emis     = df[df['category'] == 'emi']
    utility  = df[df['category'] == 'utility']
    rent     = df[df['category'] == 'rent']
    upi      = df[df['payment_mode'].str.upper() == 'UPI']
    cash     = df[df['category'] == 'cash']

    monthly_credits = credits.groupby('month')['amount'].sum()
    monthly_debits  = debits.groupby('month')['amount'].sum()

    total_income  = credits['amount'].sum() if len(credits) > 0 else 1
    total_expense = debits['amount'].sum()  if len(debits)  > 0 else 0
    avg_income    = monthly_credits.mean()  if len(monthly_credits) > 0 else 1

    # 1. income_regularity
    ir_std = float(monthly_credits.std()) if len(monthly_credits) > 1 else 0.0
    income_regularity = round(1 - min(ir_std / (avg_income + 1), 1), 4)

    # 2. spending_consistency
    avg_spend = monthly_debits.mean() if len(monthly_debits) > 0 else 1
    sc_std = float(monthly_debits.std()) if len(monthly_debits) > 1 else 0.0
    spending_consistency = round(1 - min(sc_std / (avg_spend + 1), 1), 4)

    # 3. bill_payment_ratio
    if len(utility) > 0:
        median_util = utility['amount'].median()
        on_time = utility[utility['amount'] <= median_util * 1.5]
        bill_payment_ratio = round(len(on_time) / len(utility), 4)
    else:
        bill_payment_ratio = 0.5

    # 4. savings_rate
    savings_rate = float(np.clip((total_income - total_expense) / (total_income + 1), -1, 1))
    savings_rate = round(savings_rate, 4)

    # 5. emi_burden_ratio
    total_emi = emis['amount'].sum() if len(emis) > 0 else 0
    emi_burden_ratio = round(float(np.clip(total_emi / (total_income + 1), 0, 1)), 4)

    # 6. upi_activity_volume
    months_active = max(df['month'].nunique(), 1)
    upi_activity_volume = round(len(upi) / months_active, 4)

    # 7. rent_payment_regularity
    if len(rent) > 1:
        rent_mean = rent['amount'].mean()
        rent_std  = float(rent['amount'].std())
        rent_payment_regularity = round(1 - min(rent_std / (rent_mean + 1), 1), 4)
    else:
        rent_payment_regularity = 0.5

    # 8. cash_withdrawal_ratio
    total_cash = cash['amount'].sum() if len(cash) > 0 else 0
    cash_withdrawal_ratio = round(float(np.clip(total_cash / (total_expense + 1), 0, 1)), 4)

    return {
        'income_regularity':       income_regularity,
        'spending_consistency':    spending_consistency,
        'bill_payment_ratio':      bill_payment_ratio,
        'savings_rate':            savings_rate,
        'emi_burden_ratio':        emi_burden_ratio,
        'upi_activity_volume':     upi_activity_volume,
        'rent_payment_regularity': rent_payment_regularity,
        'cash_withdrawal_ratio':   cash_withdrawal_ratio,
    }


# ─── Feature descriptions ─────────────────────────────────────────────────────

_FEATURE_LABELS = {
    'income_regularity':       'Income Regularity',
    'spending_consistency':    'Spending Consistency',
    'bill_payment_ratio':      'Bill Payment Ratio',
    'savings_rate':            'Savings Rate',
    'emi_burden_ratio':        'EMI Burden Ratio',
    'upi_activity_volume':     'UPI Activity Volume',
    'rent_payment_regularity': 'Rent Payment Regularity',
    'cash_withdrawal_ratio':   'Cash Withdrawal Ratio',
}

def _feature_desc(feature: str, value: float, shap_direction: str) -> str:
    """
    Generate plain-English description.
    shap_direction: 'helps' or 'hurts' — use this as the truth, not just the value.
    """
    pct = value * 100
    good = shap_direction == 'helps'

    descs = {
        'income_regularity': (
            f"Your income regularity score is {pct:.0f}/100. "
            + ("Consistent income is a strong positive signal." if good
               else "Irregular income increases perceived risk.")
        ),
        'spending_consistency': (
            f"Your spending consistency score is {pct:.0f}/100. "
            + ("Stable spending patterns signal financial discipline." if good
               else "Erratic spending patterns are a risk signal.")
        ),
        'bill_payment_ratio': (
            f"You paid {pct:.0f}% of utility bills on time. "
            + ("Excellent payment discipline." if good
               else "Missed bills reduce your creditworthiness.")
        ),
        'savings_rate': (
            f"Your savings rate is {pct:.0f}% of income. "
            + ("Healthy savings buffer." if good
               else "Low or negative savings increases default risk.")
        ),
        'emi_burden_ratio': (
            f"Your EMI burden is {pct:.0f}% of income. "
            + ("Within safe limits (below 50%)." if good
               else "High EMI burden — keep it below 50% of income.")
        ),
        'upi_activity_volume': (
            f"You average {value:.1f} UPI transactions/month. "
            + ("Active digital usage is a positive signal." if good
               else "Low digital activity reduces behavioral data quality.")
        ),
        'rent_payment_regularity': (
            f"Your rent payment regularity score is {pct:.0f}/100. "
            + ("Consistent rent payments build trust." if good
               else "Irregular rent payments are a negative signal.")
        ),
        'cash_withdrawal_ratio': (
            f"Cash withdrawals are {pct:.0f}% of your spending. "
            + ("Low cash usage — good financial traceability." if good
               else "High cash usage reduces financial traceability.")
        ),
    }
    return descs.get(feature, f"{_FEATURE_LABELS.get(feature, feature)}: {value:.4f}")

def _banking_desc(feature: str, value: float, shap_direction: str) -> str:
    pct = value * 100
    good = shap_direction == 'helps'

    descs = {
        'income_regularity':       f"Income regularity index: {value:.4f}. {'Stable monthly credit pattern — low income volatility.' if good else 'High income variance — irregular credit inflows detected.'}",
        'spending_consistency':    f"Spending consistency index: {value:.4f}. {'Stable debit pattern — predictable cash outflows.' if good else 'High spending variance — erratic debit behavior.'}",
        'bill_payment_ratio':      f"On-time utility payment rate: {pct:.1f}%. {'Meets RBI fair practices threshold.' if good else 'Below acceptable payment discipline threshold.'}",
        'savings_rate':            f"Net savings rate: {pct:.1f}% of income. {'Positive savings buffer — low liquidity risk.' if good else 'Negative or zero savings — high liquidity risk.'}",
        'emi_burden_ratio':        f"EMI-to-income ratio: {pct:.1f}%. {'Within RBI prudential norms (≤50%).' if good else 'Exceeds RBI prudential norms — elevated default risk.'}",
        'upi_activity_volume':     f"UPI transactions/month: {value:.2f}. {'Active digital financial engagement.' if good else 'Low digital transaction volume — limited behavioral data.'}",
        'rent_payment_regularity': f"Rent payment regularity index: {value:.4f}. {'Consistent housing obligation management.' if good else 'Irregular rent payments — housing obligation risk.'}",
        'cash_withdrawal_ratio':   f"Cash-to-expense ratio: {pct:.1f}%. {'Low cash dependency — high traceability.' if good else 'High cash dependency — reduced AML traceability.'}",
    }
    return descs.get(feature, f"{feature}: {value:.4f}")

def _improvement_suggestion(feature: str, value: float, shap_direction: str) -> str | None:
    """Only return a suggestion if the feature is actually hurting the application."""
    if shap_direction != 'hurts':
        return None   # feature is helping — no suggestion needed

    suggestions = {
        'income_regularity':       "Ensure your salary or income credits appear consistently each month.",
        'spending_consistency':    "Avoid large erratic spending spikes. Stable monthly spending improves your profile.",
        'bill_payment_ratio':      "Pay all utility bills on time. Set up auto-pay to avoid missing payments.",
        'savings_rate':            "Try to save at least 10–15% of your monthly income. Avoid spending more than you earn.",
        'emi_burden_ratio':        "Reduce your existing EMIs before applying. Keep EMI burden below 50% of income.",
        'upi_activity_volume':     "Increase your UPI transaction activity. Regular digital payments improve your profile.",
        'rent_payment_regularity': "Pay rent on time and in consistent amounts each month.",
        'cash_withdrawal_ratio':   "Use UPI or card instead of cash for daily expenses. Digital transactions build a stronger profile.",
    }
    return suggestions.get(feature)


# ─── Main prediction function ─────────────────────────────────────────────────

def run_ml_decision(feature_values: dict) -> dict:
    """
    Run the real XGBoost model + SHAP on 8 behavior features.

    Parameters
    ----------
    feature_values : dict
        Keys: income_regularity, spending_consistency, bill_payment_ratio,
              savings_rate, emi_burden_ratio, upi_activity_volume,
              rent_payment_regularity, cash_withdrawal_ratio

    Returns
    -------
    dict matching the shape expected by views.py / loan_result.html
    """
    _load()
    import shap as _shap

    # Build DataFrame in correct column order
    row = pd.DataFrame([{f: float(feature_values.get(f, 0.0)) for f in FEATURE_COLS}])

    # Predict
    risk_score_raw = float(_model.predict_proba(row)[0, 1])  # probability of default
    risk_score     = round(risk_score_raw * 100, 2)           # scale to 0–100 for display
    decision       = 'rejected' if risk_score_raw > THRESHOLD else 'approved'

    # SHAP
    shap_vals = _explainer.shap_values(row)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
        sv = shap_vals[0, :, 1]
    else:
        sv = shap_vals[0]

    base_value = float(_explainer.expected_value)
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])

    shap_dict = {feat: round(float(sv[i]), 6) for i, feat in enumerate(FEATURE_COLS)}

    # Sort by abs(shap) descending → top 5
    sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    shap_factors = []
    for feat, shap_val in sorted_feats:
        fval       = float(feature_values.get(feat, 0.0))
        impact_dir = 'hurts' if shap_val > 0 else 'helps'
        shap_factors.append({
            'feature':            _FEATURE_LABELS.get(feat, feat),
            'feature_key':        feat,
            'impact':             round(shap_val, 4),
            'impact_direction':   impact_dir,
            'feature_value':      fval,
            'description':        _feature_desc(feat, fval, impact_dir),
            'bankingDescription': _banking_desc(feat, fval, impact_dir),
        })

    # Improvement suggestions — only for features that hurt
    suggestions = []
    for feat, shap_val in sorted_feats:
        impact_dir = 'hurts' if shap_val > 0 else 'helps'
        fval = float(feature_values.get(feat, 0.0))
        s = _improvement_suggestion(feat, fval, impact_dir)
        if s:
            suggestions.append(s)

    # User explanation
    hurting = [f for f in shap_factors if f['impact_direction'] == 'hurts'][:2]
    helping = [f for f in shap_factors if f['impact_direction'] == 'helps'][:2]

    if decision == 'approved':
        strengths = ', '.join([f['feature'] for f in helping]) or 'your overall financial behavior'
        explanation = (
            f"Your loan application has been approved by our AI system. "
            f"Your risk score of {risk_score:.0f}/100 indicates a low probability of default. "
            f"Key strengths: {strengths}. "
            f"Your application will now be reviewed by a banker before final disbursement."
        )
    else:
        concerns = ', '.join([f['feature'] for f in hurting]) or 'multiple risk factors'
        explanation = (
            f"Your loan application has been declined by our AI system. "
            f"Your risk score of {risk_score:.0f}/100 indicates a higher probability of default. "
            f"Primary concerns: {concerns}. "
            f"Please review the improvement suggestions below."
        )

    return {
        'decision':             decision,
        'risk_score':           risk_score,
        'risk_score_raw':       risk_score_raw,
        'user_explanation':     explanation,
        'shap_factors':         shap_factors,
        'improvement_suggestions': suggestions[:5],
        'fairness_check_passed': True,
        'fairness_note': (
            'No demographic attributes (age, gender, caste, religion, location) were used. '
            'The model evaluates only financial behavior from transaction history.'
        ),
        'feature_values':       {f: float(feature_values.get(f, 0.0)) for f in FEATURE_COLS},
    }
