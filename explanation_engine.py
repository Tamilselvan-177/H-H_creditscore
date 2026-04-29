"""
Explanation Engine
==================
Generates 3 views from the same prediction data:
  1. USER VIEW    — plain language, actionable
  2. AUDITOR VIEW — full technical detail
  3. REGULATOR VIEW — fairness and compliance notes
"""

from datetime import datetime


# ─── Plain English per feature ────────────────────────────────────────────────

_FEATURE_LABELS = {
    "income_regularity":       "Income Regularity",
    "spending_consistency":    "Spending Consistency",
    "bill_payment_ratio":      "Bill Payment Ratio",
    "savings_rate":            "Savings Rate",
    "emi_burden_ratio":        "EMI Burden",
    "upi_activity_volume":     "UPI Activity",
    "rent_payment_regularity": "Rent Payment Regularity",
    "cash_withdrawal_ratio":   "Cash Withdrawal Ratio",
}

_FEATURE_DESCRIPTIONS = {
    "emi_burden_ratio": (
        "Your EMI payments take up {pct:.0f}% of your income. "
        "Keep it below 50%."
    ),
    "savings_rate": (
        "Your savings rate is {pct:.0f}%. "
        "Try to save at least 10–15% of income monthly."
    ),
    "bill_payment_ratio": (
        "Only {pct:.0f}% of your utility bills were paid on time. "
        "Pay bills consistently."
    ),
    "cash_withdrawal_ratio": (
        "High cash usage ({pct:.0f}% of spending) reduces financial traceability."
    ),
    "income_regularity": (
        "Your income is irregular. Consistent income builds confidence."
    ),
    "spending_consistency": (
        "Your spending pattern is erratic. Stable spending is a positive signal."
    ),
    "upi_activity_volume": (
        "Low UPI activity ({val:.1f} transactions/month). "
        "More digital transactions improve your profile."
    ),
    "rent_payment_regularity": (
        "Irregular rent payments detected. Consistent rent improves trust."
    ),
}

_FEATURE_ADVICE = {
    "emi_burden_ratio": (
        "Reduce your existing EMIs before applying. "
        "Your current EMI burden is too high."
    ),
    "savings_rate": (
        "Try to save at least 10–15% of your monthly income. "
        "Avoid spending more than you earn."
    ),
    "bill_payment_ratio": (
        "Pay all utility bills on time every month. "
        "Set up auto-pay to avoid missing payments."
    ),
    "cash_withdrawal_ratio": (
        "Use UPI or card instead of cash for daily expenses. "
        "Digital transactions build a stronger financial profile."
    ),
    "income_regularity": (
        "Ensure your income credits appear consistently each month. "
        "Irregular income increases perceived risk."
    ),
    "spending_consistency": (
        "Avoid large erratic spending spikes. "
        "Stable monthly spending patterns signal financial discipline."
    ),
    "upi_activity_volume": (
        "Increase your UPI transaction activity. "
        "Regular digital payments demonstrate financial engagement."
    ),
    "rent_payment_regularity": (
        "Pay rent on time and in consistent amounts each month. "
        "Irregular rent payments are a negative signal."
    ),
}

_POLICY_ADVICE = {
    "min_monthly_income": (
        "Increase your monthly income or apply for a loan type "
        "with lower income requirements."
    ),
    "max_emi_nmi_ratio": (
        "Reduce your existing EMIs before applying. "
        "Your current EMI burden is too high."
    ),
    "min_cibil_score": (
        "Work on improving your CIBIL score to 750+ by paying bills "
        "on time and reducing outstanding debt."
    ),
    "cibil_required": (
        "This loan type requires a CIBIL score. "
        "Consider applying for xpress_credit or two_wheeler loan instead."
    ),
    "min_account_age_months": (
        "Your account needs more history. Apply again once your account "
        "has been active for the required period."
    ),
    "min_age": "You do not meet the minimum age requirement for this loan.",
    "max_age": "You exceed the maximum age limit for this loan.",
}


def _feature_desc(feature: str, value: float) -> str:
    """Format a plain-English description for a feature value."""
    template = _FEATURE_DESCRIPTIONS.get(feature)
    if not template:
        return f"{_FEATURE_LABELS.get(feature, feature)}: {value:.4f}"
    pct = value * 100
    return template.format(pct=pct, val=value)


def _format_txn(txn: dict) -> str:
    ts = txn.get("timestamp", "")[:10]
    return (
        f"  • {txn.get('category', '').capitalize()} on {ts} "
        f"— ₹{txn.get('amount', 0):,.0f} "
        f"({txn.get('payment_mode', '')})"
    )


# ─── Main function ────────────────────────────────────────────────────────────

def generate_explanation(
    user_info: dict,
    policy_result: dict,
    shap_result: dict | None,
    traced_transactions: list | None,
    feature_values: dict | None = None,
) -> dict:
    """
    Generate all 3 explanation views.

    Parameters
    ----------
    user_info : dict
        age, monthly_income, existing_emi, cibil_score, account_age_months,
        loan_type, user_id
    policy_result : dict
        Output of policy_engine.check_policy()
    shap_result : dict or None
        Output of shap_explainer.explain_prediction() — None if policy failed
    traced_transactions : list or None
        Output of transaction_backtracking.backtrack_transactions()

    Returns
    -------
    dict with keys: user, auditor, regulator
    """
    user_view      = _build_user_view(user_info, policy_result, shap_result, traced_transactions, feature_values)
    auditor_view   = _build_auditor_view(user_info, policy_result, shap_result, traced_transactions)
    regulator_view = _build_regulator_view(user_info, shap_result)

    return {
        "user":      user_view,
        "auditor":   auditor_view,
        "regulator": regulator_view,
    }


# ─── USER VIEW ────────────────────────────────────────────────────────────────

def _build_user_view(user_info, policy_result, shap_result, traced_transactions, feature_values=None) -> str:
    loan_type = user_info.get("loan_type", "loan")

    # ── Policy rejection ──────────────────────────────────────────────────────
    if not policy_result["passed"]:
        failed_rule = policy_result.get("failed_rule", "unknown")
        reason      = policy_result.get("reason", "A policy rule was not met.")
        advice      = _POLICY_ADVICE.get(failed_rule, "Please contact support for guidance.")

        return (
            f"Your loan application was not processed.\n\n"
            f"Reason: {reason}\n\n"
            f"What you can do:\n{advice}"
        )

    # ── ML decision ───────────────────────────────────────────────────────────
    if shap_result is None:
        return "Unable to generate explanation. Please contact support."

    decision   = shap_result["decision"]
    risk_score = shap_result["risk_score"]
    top_5      = shap_result.get("top_5_features", [])

    # Helper: get actual feature value (0–1 scale) for display
    fv = feature_values or {}

    def _get_fval(feat):
        """Return the real feature value (not SHAP value) for display."""
        return float(fv.get(feat, 0.0))

    # ── Approved ──────────────────────────────────────────────────────────────
    if decision == "approved":
        helpers = [f for f in top_5 if f["impact"] == "helps"][:3]
        strengths = "\n".join(
            f"  ✅ {_FEATURE_LABELS.get(f['feature'], f['feature'])}: "
            f"{_feature_desc(f['feature'], _get_fval(f['feature']))}"
            for f in helpers
        ) or "  ✅ Your overall financial behavior looks healthy."

        return (
            f"Your loan application is approved! 🎉\n\n"
            f"Your financial strengths:\n{strengths}\n\n"
            f"Risk score: {risk_score:.2f} (below threshold of 0.50)"
        )

    # ── Rejected by ML ────────────────────────────────────────────────────────
    hurters = [f for f in top_5 if f["impact"] == "hurts"][:3]

    reasons_lines = []
    for f in hurters:
        feat = f["feature"]
        val  = _get_fval(feat)   # use actual feature value, not SHAP value
        reasons_lines.append(f"  ❌ {_feature_desc(feat, val)}")

    reasons_text = "\n".join(reasons_lines) if reasons_lines else "  ❌ Multiple risk factors detected."

    # Transactions involved
    txn_lines = []
    if traced_transactions:
        for item in traced_transactions:
            if item["impact"] == "hurts" and item.get("top_transactions"):
                txn_lines.append(f"\n  {_FEATURE_LABELS.get(item['feature'], item['feature'])}:")
                for txn in item["top_transactions"][:2]:
                    txn_lines.append(_format_txn(txn))

    txn_text = "\n".join(txn_lines) if txn_lines else "  (Transaction details not available)"

    # Advice
    advice_lines = []
    for f in hurters:
        advice = _FEATURE_ADVICE.get(f["feature"])
        if advice:
            advice_lines.append(f"  • {advice}")

    advice_text = "\n".join(advice_lines) if advice_lines else "  • Improve your financial habits and reapply."

    return (
        f"Your loan application was declined.\n\n"
        f"Main reasons:\n{reasons_text}\n\n"
        f"Transactions involved:{txn_text}\n\n"
        f"How to improve:\n{advice_text}\n\n"
        f"Risk score: {risk_score:.2f} (threshold: 0.50)"
    )


# ─── AUDITOR VIEW ─────────────────────────────────────────────────────────────

def _build_auditor_view(user_info, policy_result, shap_result, traced_transactions) -> dict:
    view = {
        "loan_type":    user_info.get("loan_type"),
        "user_id":      user_info.get("user_id"),
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "policy_check": {
            "passed":        policy_result["passed"],
            "failed_rule":   policy_result.get("failed_rule"),
            "reason":        policy_result.get("reason"),
            "rules_checked": policy_result.get("rules_checked", []),
        },
    }

    if shap_result:
        view.update({
            "model_risk_score":    shap_result["risk_score"],
            "decision_threshold":  shap_result["threshold"],
            "final_decision":      shap_result["decision"],
            "shap_base_value":     shap_result["base_value"],
            "shap_values":         shap_result["shap_values"],
            "top_5_shap_features": shap_result["top_5_features"],
            "traced_transactions": traced_transactions or [],
        })
    else:
        view.update({
            "model_risk_score":   None,
            "final_decision":     "rejected",
            "rejection_stage":    "policy",
            "traced_transactions": [],
        })

    return view


# ─── REGULATOR VIEW ───────────────────────────────────────────────────────────

def _build_regulator_view(user_info, shap_result) -> dict:
    return {
        "note": (
            "Demographic data is not used as model input. "
            "Bias metrics require population-level analysis across all decisions."
        ),
        "model_used":    "XGBoost (behavior-based)",
        "features_used": [
            "income_regularity", "spending_consistency", "bill_payment_ratio",
            "savings_rate", "emi_burden_ratio", "upi_activity_volume",
            "rent_payment_regularity", "cash_withdrawal_ratio",
        ],
        "cibil_dependency": (
            "CIBIL score is used only when the specific loan policy requires it. "
            "Behavior features are the primary signal. "
            "Users without CIBIL are not automatically rejected."
        ),
        "fairness_approach": (
            "Behavior-based scoring reduces demographic proxy bias. "
            "Features measure financial discipline, not identity. "
            "No age, gender, state, or employment type is used in the ML model."
        ),
        "explanation_drift_note": (
            "Monitor SHAP top features monthly for drift detection. "
            "If feature importance ranking changes significantly, trigger model re-audit."
        ),
        "decision_threshold": shap_result["threshold"] if shap_result else None,
        "risk_score":         shap_result["risk_score"] if shap_result else None,
    }


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    user_info = {
        "user_id": "USR_0001",
        "age": 35,
        "monthly_income": 40000,
        "existing_emi": 8000,
        "cibil_score": None,
        "account_age_months": 36,
        "loan_type": "xpress_credit",
    }

    policy_result = {
        "passed": True,
        "failed_rule": None,
        "reason": None,
        "rules_checked": [
            {"rule": "min_age", "value": 35, "required": 18, "passed": True},
            {"rule": "max_age", "value": 35, "required": 76, "passed": True},
            {"rule": "min_monthly_income", "value": 40000, "required": 5000, "passed": True},
            {"rule": "max_emi_nmi_ratio", "value": 0.20, "required": 0.50, "passed": True},
            {"rule": "cibil_check", "value": None, "required": "not_required", "passed": True},
        ],
    }

    shap_result = {
        "risk_score": 0.73,
        "decision": "rejected",
        "threshold": 0.5,
        "base_value": 0.21,
        "shap_values": {
            "emi_burden_ratio":       0.34,
            "savings_rate":          -0.18,
            "bill_payment_ratio":     0.12,
            "income_regularity":     -0.09,
            "cash_withdrawal_ratio":  0.08,
            "spending_consistency":   0.03,
            "upi_activity_volume":   -0.02,
            "rent_payment_regularity": 0.01,
        },
        "top_5_features": [
            {"feature": "emi_burden_ratio",      "shap_value": 0.34,  "impact": "hurts"},
            {"feature": "savings_rate",           "shap_value": -0.18, "impact": "helps"},
            {"feature": "bill_payment_ratio",     "shap_value": 0.12,  "impact": "hurts"},
            {"feature": "income_regularity",      "shap_value": -0.09, "impact": "helps"},
            {"feature": "cash_withdrawal_ratio",  "shap_value": 0.08,  "impact": "hurts"},
        ],
    }

    traced = [
        {
            "feature": "emi_burden_ratio",
            "shap_value": 0.34,
            "impact": "hurts",
            "top_transactions": [
                {"transaction_id": "TXN_abc", "amount": 15000,
                 "category": "emi", "type": "debit",
                 "timestamp": "2024-01-05", "payment_mode": "NEFT"},
            ],
        }
    ]

    result = generate_explanation(user_info, policy_result, shap_result, traced)

    print("=" * 60)
    print("USER VIEW")
    print("=" * 60)
    print(result["user"])

    print("\n" + "=" * 60)
    print("AUDITOR VIEW")
    print("=" * 60)
    print(json.dumps(result["auditor"], indent=2))

    print("\n" + "=" * 60)
    print("REGULATOR VIEW")
    print("=" * 60)
    print(json.dumps(result["regulator"], indent=2))
