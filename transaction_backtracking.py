"""
Transaction Backtracking
=========================
For each top SHAP feature, fetches the contributing transactions
from the lineage metadata and ranks them by amount contribution.

NOTE: This is APPROXIMATE attribution.
SHAP provides feature-level attribution only.
Transactions are ranked by their amount contribution to the feature value,
NOT by direct SHAP attribution. This is a proxy, not an exact mapping.
"""

import json
import pandas as pd

LINEAGE_FILE = "features_with_lineage.json"

# ─── Load lineage once ────────────────────────────────────────────────────────

_lineage_map = None


def _load_lineage():
    global _lineage_map
    if _lineage_map is None:
        with open(LINEAGE_FILE, "r") as f:
            records = json.load(f)
        _lineage_map = {r["user_id"]: r["lineage"] for r in records}


def backtrack_transactions(
    user_id: str,
    top_5_features: list,
    transactions_df: pd.DataFrame,
    top_n: int = 3,
) -> list:
    """
    For each top SHAP feature, find the transactions that contributed to it
    and rank them by amount.

    Parameters
    ----------
    user_id : str
    top_5_features : list of dicts from shap_explainer.explain_prediction()
        Each dict has: feature, shap_value, impact
    transactions_df : pd.DataFrame
        Full transactions table (all users OK — filtered by transaction_id)
    top_n : int
        How many transactions to return per feature (default 3)

    Returns
    -------
    list of dicts, one per feature:
        {
            feature, shap_value, impact,
            top_transactions: [
                { transaction_id, amount, category, type, timestamp, payment_mode }
            ]
        }
    """
    # NOTE: This is APPROXIMATE attribution.
    # SHAP gives feature-level attribution only.
    # Transaction ranking is proportional contribution to the feature value,
    # not direct SHAP attribution. Do not claim this is exact.

    _load_lineage()

    user_lineage = _lineage_map.get(user_id, {})
    results = []

    for feat_info in top_5_features:
        feature    = feat_info["feature"]
        shap_value = feat_info["shap_value"]
        impact     = feat_info["impact"]

        txn_ids = user_lineage.get(feature, [])

        if not txn_ids:
            results.append({
                "feature":         feature,
                "shap_value":      shap_value,
                "impact":          impact,
                "top_transactions": [],
                "note": "No lineage data available for this user/feature.",
            })
            continue

        # Filter transactions to only those in the lineage
        relevant = transactions_df[
            transactions_df["transaction_id"].isin(txn_ids)
        ].copy()

        if relevant.empty:
            results.append({
                "feature":         feature,
                "shap_value":      shap_value,
                "impact":          impact,
                "top_transactions": [],
                "note": "Lineage transaction IDs not found in transactions table.",
            })
            continue

        # Rank by amount descending (proxy for contribution magnitude)
        relevant = relevant.sort_values("amount", ascending=False).head(top_n)

        top_txns = []
        for _, row in relevant.iterrows():
            top_txns.append({
                "transaction_id": row["transaction_id"],
                "amount":         round(float(row["amount"]), 2),
                "category":       row.get("category", "unknown"),
                "type":           row.get("type", "unknown"),
                "timestamp":      str(row.get("timestamp", "")),
                "payment_mode":   row.get("payment_mode", "unknown"),
            })

        results.append({
            "feature":          feature,
            "shap_value":       shap_value,
            "impact":           impact,
            "top_transactions": top_txns,
        })

    return results


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json

    txn_df = pd.read_csv("transactions.csv")

    sample_top5 = [
        {"feature": "emi_burden_ratio",       "shap_value": 0.34,  "impact": "hurts"},
        {"feature": "savings_rate",            "shap_value": -0.18, "impact": "helps"},
        {"feature": "bill_payment_ratio",      "shap_value": 0.12,  "impact": "hurts"},
        {"feature": "income_regularity",       "shap_value": -0.09, "impact": "helps"},
        {"feature": "cash_withdrawal_ratio",   "shap_value": 0.08,  "impact": "hurts"},
    ]

    result = backtrack_transactions("USR_0001", sample_top5, txn_df)
    print(_json.dumps(result, indent=2))
