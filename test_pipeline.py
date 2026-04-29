"""
Test Pipeline — 5 Cases
========================
Covers policy failures, ML decisions, and edge cases.
Prints full output for each case.
"""

import json
import pandas as pd
from main_pipeline import evaluate_application
from audit_logger  import verify_chain, get_all_records

# ─── Override user data for controlled test cases ─────────────────────────────
# We patch users.csv in-memory for cases that need specific conditions.

import main_pipeline as _mp

_original_load = _mp._load_data


def _patch_user(user_id: str, overrides: dict):
    """Temporarily override a user's data for testing."""
    _mp._load_data()
    idx = _mp._users_df[_mp._users_df["user_id"] == user_id].index
    if not idx.empty:
        for k, v in overrides.items():
            _mp._users_df.loc[idx[0], k] = v


def _print_result(case_num: int, description: str, result: dict):
    print(f"\n{'='*65}")
    print(f"  CASE {case_num}: {description}")
    print(f"{'='*65}")

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    print(f"  User:            {result['user_id']}")
    print(f"  Loan Type:       {result['loan_type']}")
    print(f"  Decision:        {result['decision'].upper()}")
    print(f"  Rejection Stage: {result['rejection_stage'] or 'N/A'}")
    print(f"  Risk Score:      {result['risk_score']:.4f}" if result["risk_score"] is not None else "  Risk Score:      N/A (policy rejection)")
    print(f"  Audit ID:        {result['audit_id']}")

    print(f"\n  ── Policy Result ──")
    pr = result["policy_result"]
    print(f"  Passed: {pr['passed']}")
    if not pr["passed"]:
        print(f"  Failed Rule: {pr['failed_rule']}")
        print(f"  Reason: {pr['reason']}")

    if result["top_5_features"]:
        print(f"\n  ── Top SHAP Features ──")
        for f in result["top_5_features"]:
            sign = "▲ hurts" if f["impact"] == "hurts" else "▼ helps"
            print(f"  {sign:<10} {f['feature']:<30} {f['shap_value']:+.4f}")

    print(f"\n  ── User Explanation ──")
    for line in result["explanations"]["user"].split("\n"):
        print(f"  {line}")


# ─── Test Cases ───────────────────────────────────────────────────────────────

def run_tests():
    print("\n" + "=" * 65)
    print("  CREDIT SCORING PIPELINE — TEST SUITE")
    print("=" * 65)

    # ── Case 1: xpress_power, no CIBIL → policy fail ──────────────────────────
    # Find a user with no CIBIL and patch loan type
    _mp._load_data()
    no_cibil_users = _mp._users_df[_mp._users_df["cibil_score"].isna()]
    uid1 = no_cibil_users.iloc[0]["user_id"]
    _patch_user(uid1, {
        "loan_type":       "xpress_power",
        "monthly_income":  60000,
        "existing_emi":    10000,
        "age":             30,
        "account_age_months": 24,
    })
    result1 = evaluate_application(uid1, "xpress_power", cibil_score=None)
    _print_result(1, "xpress_power — No CIBIL → should fail policy", result1)

    # ── Case 2: xpress_power, CIBIL=600 → policy fail ─────────────────────────
    uid2 = _mp._users_df.iloc[1]["user_id"]
    _patch_user(uid2, {
        "loan_type":       "xpress_power",
        "monthly_income":  60000,
        "existing_emi":    10000,
        "age":             30,
        "account_age_months": 24,
    })
    result2 = evaluate_application(uid2, "xpress_power", cibil_score=600)
    _print_result(2, "xpress_power — CIBIL=600 → should fail policy (below 750)", result2)

    # ── Case 3: two_wheeler, new account → policy fail ────────────────────────
    uid3 = _mp._users_df.iloc[2]["user_id"]
    _patch_user(uid3, {
        "loan_type":          "two_wheeler",
        "monthly_income":     20000,
        "existing_emi":       3000,
        "age":                25,
        "account_age_months": 6,   # below 12 months required
    })
    result3 = evaluate_application(uid3, "two_wheeler")
    _print_result(3, "two_wheeler — Account only 6 months old → should fail policy", result3)

    # ── Case 4: xpress_credit, normal salaried → ML decides ───────────────────
    # Find a salaried user with reasonable income
    salaried = _mp._users_df[
        (_mp._users_df["employment_type"] == "salaried") &
        (_mp._users_df["monthly_income"] > 30000) &
        (_mp._users_df["existing_emi"] / _mp._users_df["monthly_income"] < 0.4)
    ]
    uid4 = salaried.iloc[0]["user_id"] if not salaried.empty else _mp._users_df.iloc[3]["user_id"]
    _patch_user(uid4, {
        "loan_type":          "xpress_credit",
        "age":                35,
        "account_age_months": 36,
    })
    result4 = evaluate_application(uid4, "xpress_credit")
    _print_result(4, "xpress_credit — Normal salaried user → ML decides", result4)

    # ── Case 5: xpress_credit, high EMI burden → ML likely rejects ────────────
    # Find a user with high EMI ratio
    high_emi = _mp._users_df[
        _mp._users_df["existing_emi"] / _mp._users_df["monthly_income"] > 0.55
    ]
    uid5 = high_emi.iloc[0]["user_id"] if not high_emi.empty else _mp._users_df.iloc[4]["user_id"]
    _patch_user(uid5, {
        "loan_type":          "xpress_credit",
        "age":                40,
        "account_age_months": 48,
    })
    result5 = evaluate_application(uid5, "xpress_credit")
    _print_result(5, "xpress_credit — High EMI burden → ML likely rejects", result5)

    # ── Chain verification ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  AUDIT CHAIN VERIFICATION")
    print(f"{'='*65}")
    chain = verify_chain()
    print(f"  Valid:         {chain['valid']}")
    print(f"  Total Records: {chain['total_records']}")
    print(f"  Broken At:     {chain['broken_at'] or 'None — chain intact'}")

    print(f"\n{'='*65}")
    print("  AUDIT LOG SUMMARY")
    print(f"{'='*65}")
    records = get_all_records()
    for r in records[-5:]:   # show last 5
        print(f"  {r['audit_id'][:8]}...  {r['user_id']}  {r['loan_type']:<15}  {r['final_decision'].upper()}")

    print(f"\n✅ All test cases complete.\n")


if __name__ == "__main__":
    run_tests()
