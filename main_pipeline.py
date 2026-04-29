"""
Main Pipeline — evaluate_application()
=======================================
Orchestrates all steps in order:
  1. Load user + transactions + features + lineage
  2. Policy eligibility check
  3. Feature extraction
  4. SHAP prediction
  5. Transaction backtracking
  6. Explanation generation (3 views)
  7. Audit logging

Single entry point: evaluate_application(user_id, loan_type, cibil_score=None)
"""

import pandas as pd

from policy_engine          import check_policy
from shap_explainer         import explain_prediction
from transaction_backtracking import backtrack_transactions
from explanation_engine     import generate_explanation
from audit_logger           import log_decision, verify_chain

# ─── Data files ───────────────────────────────────────────────────────────────

USERS_FILE       = "users.csv"
FEATURES_FILE    = "features.csv"
TRANSACTIONS_FILE = "transactions.csv"

FEATURE_COLS = [
    "income_regularity",
    "spending_consistency",
    "bill_payment_ratio",
    "savings_rate",
    "emi_burden_ratio",
    "upi_activity_volume",
    "rent_payment_regularity",
    "cash_withdrawal_ratio",
]

# ─── Lazy-loaded DataFrames ───────────────────────────────────────────────────

_users_df    = None
_features_df = None
_txn_df      = None


def _load_data():
    global _users_df, _features_df, _txn_df
    if _users_df is None:
        _users_df    = pd.read_csv(USERS_FILE)
        _features_df = pd.read_csv(FEATURES_FILE)
        _txn_df      = pd.read_csv(TRANSACTIONS_FILE)


# ─── Main function ────────────────────────────────────────────────────────────

def evaluate_application(
    user_id: str,
    loan_type: str,
    cibil_score: float | None = None,
) -> dict:
    """
    Full pipeline for one loan application.

    Parameters
    ----------
    user_id    : str   e.g. "USR_0001"
    loan_type  : str   one of: xpress_credit, xpress_power, two_wheeler
    cibil_score: float or None  (overrides stored value if provided)

    Returns
    -------
    dict with full decision, explanations, SHAP, traced transactions, audit info
    """
    _load_data()

    # ── Step 1: Load user ─────────────────────────────────────────────────────
    user_row = _users_df[_users_df["user_id"] == user_id]
    if user_row.empty:
        return {"error": f"User '{user_id}' not found in users.csv"}

    user = user_row.iloc[0]

    # Use provided cibil_score if given, else use stored value (may be NaN)
    stored_cibil = user.get("cibil_score", None)
    if cibil_score is None:
        # Convert NaN to None
        cibil_score = None if pd.isna(stored_cibil) else float(stored_cibil)

    user_info = {
        "user_id":             user_id,
        "age":                 int(user["age"]),
        "monthly_income":      float(user["monthly_income"]),
        "existing_emi":        float(user["existing_emi"]),
        "cibil_score":         cibil_score,
        "account_age_months":  int(user["account_age_months"]),
        "loan_type":           loan_type,
        "employment_type":     user.get("employment_type", "unknown"),
        "state":               user.get("state", "unknown"),
    }

    # ── Step 2: Policy check ──────────────────────────────────────────────────
    policy_result = check_policy(user_info, loan_type)

    if not policy_result["passed"]:
        # Policy rejection — skip ML entirely
        explanations = generate_explanation(
            user_info, policy_result, None, None, None
        )
        audit_id = log_decision(
            user_id, loan_type, policy_result, None, explanations
        )
        return _build_result(
            user_id, loan_type, "rejected", "policy",
            user_info, policy_result, None, None, None,
            explanations, audit_id,
        )

    # ── Step 3: Load features ─────────────────────────────────────────────────
    feat_row = _features_df[_features_df["user_id"] == user_id]
    if feat_row.empty:
        return {"error": f"Features not found for user '{user_id}'"}

    feature_values = {col: float(feat_row.iloc[0][col]) for col in FEATURE_COLS}

    # ── Step 4: SHAP prediction ───────────────────────────────────────────────
    shap_result = explain_prediction(feature_values)

    # ── Step 5: Transaction backtracking ──────────────────────────────────────
    user_txns = _txn_df[_txn_df["user_id"] == user_id]
    traced = backtrack_transactions(
        user_id, shap_result["top_5_features"], user_txns
    )

    # ── Step 6: Explanation generation ───────────────────────────────────────
    explanations = generate_explanation(user_info, policy_result, shap_result, traced, feature_values)

    # ── Step 7: Audit log ─────────────────────────────────────────────────────
    audit_id = log_decision(user_id, loan_type, policy_result, shap_result, explanations)

    return _build_result(
        user_id, loan_type,
        shap_result["decision"], "ml",
        user_info, policy_result,
        shap_result, feature_values, traced,
        explanations, audit_id,
    )


def _build_result(
    user_id, loan_type, decision, rejection_stage,
    user_info, policy_result, shap_result, feature_values,
    traced, explanations, audit_id,
) -> dict:
    return {
        "audit_id":        audit_id,
        "user_id":         user_id,
        "loan_type":       loan_type,
        "decision":        decision,
        "rejection_stage": rejection_stage if decision == "rejected" else None,
        "risk_score":      shap_result["risk_score"] if shap_result else None,
        "policy_result":   policy_result,
        "feature_values":  feature_values or {},
        "shap_values":     shap_result["shap_values"] if shap_result else {},
        "top_5_features":  shap_result["top_5_features"] if shap_result else [],
        "traced_transactions": traced or [],
        "explanations":    explanations,
        "audit_hash":      None,   # populated from DB if needed
    }


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Quick smoke test with first user
    result = evaluate_application("USR_0001", "xpress_credit")

    print("=" * 60)
    print(f"User:     {result['user_id']}")
    print(f"Loan:     {result['loan_type']}")
    print(f"Decision: {result['decision'].upper()}")
    if result["risk_score"] is not None:
        print(f"Risk:     {result['risk_score']:.4f}")
    print(f"Audit ID: {result['audit_id']}")
    print()
    print("USER EXPLANATION:")
    print(result["explanations"]["user"])

    chain = verify_chain()
    print(f"\nChain valid: {chain['valid']} | Records: {chain['total_records']}")
