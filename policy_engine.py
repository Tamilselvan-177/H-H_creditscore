"""
Policy Eligibility Engine
==========================
Hard-gate rules per loan type. Runs BEFORE any ML.
If any rule fails → reject immediately with plain English reason.
"""

# ─── Loan Configs ─────────────────────────────────────────────────────────────

LOAN_CONFIGS = {
    "xpress_credit": {
        "min_age": 18,
        "max_age": 76,
        "min_monthly_income": 5000,
        "max_emi_nmi_ratio": 0.50,
        "cibil_required": False,
        "min_cibil_score": None,
        "min_account_age_months": 0,
    },
    "xpress_power": {
        "min_age": 18,
        "max_age": 76,
        "min_monthly_income": 50000,
        "max_emi_nmi_ratio": 0.50,
        "cibil_required": True,
        "min_cibil_score": 750,
        "min_account_age_months": 0,
    },
    "two_wheeler": {
        "min_age": 18,
        "max_age": 65,
        "min_monthly_income": 10000,
        "max_emi_nmi_ratio": None,   # not enforced for two-wheeler
        "cibil_required": False,
        "min_cibil_score": None,
        "min_account_age_months": 12,
    },
}


def check_policy(user_info: dict, loan_type: str) -> dict:
    """
    Check all policy rules for a given loan type.

    Parameters
    ----------
    user_info : dict
        Keys: age, monthly_income, existing_emi, cibil_score (can be None),
              account_age_months
    loan_type : str
        One of: xpress_credit, xpress_power, two_wheeler

    Returns
    -------
    dict with keys:
        passed        : bool
        failed_rule   : str or None
        reason        : str or None  (plain English)
        rules_checked : list of dicts
    """
    if loan_type not in LOAN_CONFIGS:
        return {
            "passed": False,
            "failed_rule": "unknown_loan_type",
            "reason": f"Loan type '{loan_type}' is not supported.",
            "rules_checked": [],
        }

    cfg = LOAN_CONFIGS[loan_type]
    rules_checked = []

    age                 = user_info.get("age", 0)
    monthly_income      = user_info.get("monthly_income", 0)
    existing_emi        = user_info.get("existing_emi", 0)
    cibil_score         = user_info.get("cibil_score", None)
    account_age_months  = user_info.get("account_age_months", 0)

    emi_nmi_ratio = existing_emi / monthly_income if monthly_income > 0 else 1.0

    # ── Rule 1: min_age ───────────────────────────────────────────────────────
    passed = age >= cfg["min_age"]
    rules_checked.append({
        "rule": "min_age",
        "value": age,
        "required": cfg["min_age"],
        "passed": passed,
    })
    if not passed:
        return _fail(
            f"Your age {age} does not meet the minimum age requirement of "
            f"{cfg['min_age']} for this loan.",
            "min_age",
            rules_checked,
        )

    # ── Rule 2: max_age ───────────────────────────────────────────────────────
    passed = age <= cfg["max_age"]
    rules_checked.append({
        "rule": "max_age",
        "value": age,
        "required": cfg["max_age"],
        "passed": passed,
    })
    if not passed:
        return _fail(
            f"Your age {age} exceeds the maximum age limit of "
            f"{cfg['max_age']} for this loan.",
            "max_age",
            rules_checked,
        )

    # ── Rule 3: min_monthly_income ────────────────────────────────────────────
    passed = monthly_income >= cfg["min_monthly_income"]
    rules_checked.append({
        "rule": "min_monthly_income",
        "value": monthly_income,
        "required": cfg["min_monthly_income"],
        "passed": passed,
    })
    if not passed:
        return _fail(
            f"Your monthly income of ₹{monthly_income:,.0f} is below the minimum "
            f"required ₹{cfg['min_monthly_income']:,} for this loan.",
            "min_monthly_income",
            rules_checked,
        )

    # ── Rule 4: max_emi_nmi_ratio (if applicable) ─────────────────────────────
    if cfg["max_emi_nmi_ratio"] is not None:
        passed = emi_nmi_ratio <= cfg["max_emi_nmi_ratio"]
        rules_checked.append({
            "rule": "max_emi_nmi_ratio",
            "value": round(emi_nmi_ratio, 4),
            "required": cfg["max_emi_nmi_ratio"],
            "passed": passed,
        })
        if not passed:
            return _fail(
                f"Your EMI burden is {emi_nmi_ratio * 100:.1f}% of income. "
                f"Maximum allowed is {cfg['max_emi_nmi_ratio'] * 100:.0f}%.",
                "max_emi_nmi_ratio",
                rules_checked,
            )

    # ── Rule 5: CIBIL ─────────────────────────────────────────────────────────
    if cfg["cibil_required"]:
        if cibil_score is None:
            rules_checked.append({
                "rule": "cibil_required",
                "value": None,
                "required": cfg["min_cibil_score"],
                "passed": False,
            })
            return _fail(
                f"This loan requires a CIBIL score of {cfg['min_cibil_score']}+. "
                f"No CIBIL score found.",
                "cibil_required",
                rules_checked,
            )
        passed = cibil_score >= cfg["min_cibil_score"]
        rules_checked.append({
            "rule": "min_cibil_score",
            "value": cibil_score,
            "required": cfg["min_cibil_score"],
            "passed": passed,
        })
        if not passed:
            return _fail(
                f"Your CIBIL score of {cibil_score} is below the minimum required "
                f"{cfg['min_cibil_score']} for this loan.",
                "min_cibil_score",
                rules_checked,
            )
    else:
        # CIBIL not required — skip silently
        rules_checked.append({
            "rule": "cibil_check",
            "value": cibil_score,
            "required": "not_required",
            "passed": True,
        })

    # ── Rule 6: min_account_age_months ────────────────────────────────────────
    if cfg["min_account_age_months"] > 0:
        passed = account_age_months >= cfg["min_account_age_months"]
        rules_checked.append({
            "rule": "min_account_age_months",
            "value": account_age_months,
            "required": cfg["min_account_age_months"],
            "passed": passed,
        })
        if not passed:
            return _fail(
                f"Your account is {account_age_months} months old. "
                f"Minimum required is {cfg['min_account_age_months']} months.",
                "min_account_age_months",
                rules_checked,
            )

    # ── All rules passed ──────────────────────────────────────────────────────
    return {
        "passed": True,
        "failed_rule": None,
        "reason": None,
        "rules_checked": rules_checked,
    }


def _fail(reason: str, rule: str, rules_checked: list) -> dict:
    return {
        "passed": False,
        "failed_rule": rule,
        "reason": reason,
        "rules_checked": rules_checked,
    }


# ─── Quick standalone test ────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # (description, user_info, loan_type)
        ("xpress_power no CIBIL",
         {"age": 30, "monthly_income": 60000, "existing_emi": 10000,
          "cibil_score": None, "account_age_months": 24},
         "xpress_power"),

        ("xpress_power CIBIL=600",
         {"age": 30, "monthly_income": 60000, "existing_emi": 10000,
          "cibil_score": 600, "account_age_months": 24},
         "xpress_power"),

        ("two_wheeler new account",
         {"age": 25, "monthly_income": 20000, "existing_emi": 3000,
          "cibil_score": None, "account_age_months": 6},
         "two_wheeler"),

        ("xpress_credit normal salaried",
         {"age": 35, "monthly_income": 40000, "existing_emi": 8000,
          "cibil_score": None, "account_age_months": 36},
         "xpress_credit"),

        ("xpress_credit high EMI",
         {"age": 35, "monthly_income": 40000, "existing_emi": 25000,
          "cibil_score": None, "account_age_months": 36},
         "xpress_credit"),
    ]

    for desc, user_info, loan_type in tests:
        result = check_policy(user_info, loan_type)
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{status} | {desc}")
        if not result["passed"]:
            print(f"       Rule: {result['failed_rule']}")
            print(f"       Reason: {result['reason']}")
        print()
