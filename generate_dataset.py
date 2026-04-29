"""
Credit Scoring Dataset Generator
=================================
Generates synthetic Indian banking transaction data
for training the XGBoost behavior-based credit scoring model.

OUTPUT FILES:
  - transactions.csv        → raw transaction records per user
  - users.csv               → user demographics + loan info
  - features.csv            → 8 engineered features per user (ready for ML)
  - features_with_lineage.json → features + source transaction IDs
  - train.csv               → 80% split (features + TARGET)
  - test.csv                → 20% split (features + TARGET)

Run: python generate_dataset.py
"""

import pandas as pd
import numpy as np
import json
import uuid
from datetime import datetime, timedelta
import random
import os

# ─── Config ───────────────────────────────────────────────────────────────────

RANDOM_SEED     = 42
NUM_USERS       = 2000          # total applicants
TXN_PER_USER    = (30, 120)     # min/max transactions per user
MONTHS_HISTORY  = 6             # transaction history window
OUTPUT_DIR      = "."           # save files here

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def random_date(months_back=6):
    end   = datetime.now()
    start = end - timedelta(days=30 * months_back)
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

def make_uuid():
    return str(uuid.uuid4())[:12]


# ─── Step 1: Generate User Profiles ───────────────────────────────────────────

print("Generating user profiles...")

employment_types = ["salaried", "self_employed", "gig_worker", "student", "retired", "farmer"]
loan_types       = ["xpress_credit", "xpress_power", "two_wheeler"]

users = []
for i in range(NUM_USERS):
    uid        = f"USR_{i+1:04d}"
    emp        = random.choices(employment_types, weights=[40, 20, 15, 10, 8, 7])[0]
    loan_type  = random.choice(loan_types)
    age        = np.random.randint(18, 72)

    # Income varies by employment type
    income_map = {
        "salaried":      np.random.randint(15000, 150000),
        "self_employed": np.random.randint(10000, 200000),
        "gig_worker":    np.random.randint(8000, 40000),
        "student":       np.random.randint(3000, 15000),
        "retired":       np.random.randint(10000, 60000),
        "farmer":        np.random.randint(5000, 30000),
    }
    monthly_income = income_map[emp]

    # CIBIL: many users won't have one (core use case)
    has_cibil   = random.random() < 0.55
    cibil_score = np.random.randint(550, 850) if has_cibil else None

    # Existing EMIs
    existing_emi = monthly_income * np.random.uniform(0.05, 0.70)

    users.append({
        "user_id":         uid,
        "age":             age,
        "employment_type": emp,
        "loan_type":       loan_type,
        "monthly_income":  round(monthly_income, 2),
        "cibil_score":     cibil_score,
        "existing_emi":    round(existing_emi, 2),
        "account_age_months": np.random.randint(1, 120),
        "state":           random.choice(["TN", "MH", "KA", "UP", "DL", "WB", "GJ", "RJ", "AP", "KL"]),
    })

users_df = pd.DataFrame(users)
print(f"  Created {len(users_df)} users")


# ─── Step 2: Generate Raw Transactions ────────────────────────────────────────

print("Generating transactions...")

categories    = ["salary", "emi", "utility", "rent", "upi", "cash", "other", "food", "shopping"]
payment_modes = ["UPI", "NEFT", "IMPS", "cash", "card", "cheque"]

all_transactions = []

for _, user in users_df.iterrows():
    uid     = user["user_id"]
    income  = user["monthly_income"]
    emi     = user["existing_emi"]
    n_txns  = np.random.randint(*TXN_PER_USER)

    for _ in range(n_txns):
        cat   = random.choices(
            categories,
            weights=[10, 15, 12, 8, 25, 8, 7, 8, 7]
        )[0]
        mode  = random.choices(
            payment_modes,
            weights=[35, 15, 15, 10, 20, 5]
        )[0]

        # Amount logic per category
        if cat == "salary":
            amount = income * np.random.uniform(0.9, 1.1)
            txn_type = "credit"
        elif cat == "emi":
            amount = emi / random.randint(1, 3)
            txn_type = "debit"
        elif cat == "rent":
            amount = income * np.random.uniform(0.15, 0.40)
            txn_type = "debit"
        elif cat == "utility":
            amount = np.random.uniform(200, 5000)
            txn_type = "debit"
        elif cat == "upi":
            amount = np.random.uniform(50, 10000)
            txn_type = random.choice(["credit", "debit"])
        elif cat == "cash":
            amount = np.random.uniform(500, 15000)
            txn_type = "debit"
        elif cat == "food":
            amount = np.random.uniform(100, 2000)
            txn_type = "debit"
        elif cat == "shopping":
            amount = np.random.uniform(200, 20000)
            txn_type = "debit"
        else:
            amount = np.random.uniform(100, 5000)
            txn_type = random.choice(["credit", "debit"])

        # Force payment_mode consistency
        if cat == "cash":
            mode = "cash"
        if cat == "upi":
            mode = "UPI"

        all_transactions.append({
            "transaction_id":  f"TXN_{make_uuid()}",
            "user_id":         uid,
            "amount":          round(amount, 2),
            "type":            txn_type,
            "category":        cat,
            "timestamp":       random_date(MONTHS_HISTORY).strftime("%Y-%m-%d %H:%M:%S"),
            "payment_mode":    mode,
            "merchant":        f"MERCH_{random.randint(100, 999)}",
        })

txn_df = pd.DataFrame(all_transactions)
print(f"  Created {len(txn_df)} transactions")


# ─── Step 3: Feature Engineering with Lineage ─────────────────────────────────

print("Engineering features...")

def compute_features(uid, txns):
    """Compute 8 behavior features + lineage for one user."""

    u_txns = txns[txns["user_id"] == uid].copy()
    u_txns["timestamp"] = pd.to_datetime(u_txns["timestamp"])
    u_txns["month"] = u_txns["timestamp"].dt.to_period("M")

    credits  = u_txns[u_txns["type"] == "credit"]
    debits   = u_txns[u_txns["type"] == "debit"]
    emis     = u_txns[u_txns["category"] == "emi"]
    utility  = u_txns[u_txns["category"] == "utility"]
    rent     = u_txns[u_txns["category"] == "rent"]
    upi      = u_txns[u_txns["payment_mode"] == "UPI"]
    cash     = u_txns[u_txns["category"] == "cash"]

    # Monthly aggregations
    monthly_credits = credits.groupby("month")["amount"].sum()
    monthly_debits  = debits.groupby("month")["amount"].sum()

    total_income  = credits["amount"].sum() if len(credits) > 0 else 1
    total_expense = debits["amount"].sum() if len(debits) > 0 else 0
    avg_income    = monthly_credits.mean() if len(monthly_credits) > 0 else 1

    # ── Feature 1: income_regularity ──────────────────────────────────────────
    income_regularity = float(monthly_credits.std()) if len(monthly_credits) > 1 else 0.0
    # Normalize: lower std relative to mean = more regular
    income_regularity_norm = 1 - min(income_regularity / (avg_income + 1), 1)

    # ── Feature 2: spending_consistency ───────────────────────────────────────
    spending_consistency = float(monthly_debits.std()) if len(monthly_debits) > 1 else 0.0
    avg_spend = monthly_debits.mean() if len(monthly_debits) > 0 else 1
    spending_consistency_norm = 1 - min(spending_consistency / (avg_spend + 1), 1)

    # ── Feature 3: bill_payment_ratio ─────────────────────────────────────────
    # Simulate: assume each utility payment is "on-time" if amount is within normal range
    if len(utility) > 0:
        median_util = utility["amount"].median()
        on_time = utility[utility["amount"] <= median_util * 1.5]
        bill_payment_ratio = len(on_time) / len(utility)
    else:
        bill_payment_ratio = 0.5  # neutral if no data

    # ── Feature 4: savings_rate ───────────────────────────────────────────────
    savings_rate = (total_income - total_expense) / (total_income + 1)
    savings_rate = float(np.clip(savings_rate, -1, 1))

    # ── Feature 5: emi_burden_ratio ───────────────────────────────────────────
    total_emi = emis["amount"].sum() if len(emis) > 0 else 0
    emi_burden_ratio = float(total_emi / (total_income + 1))
    emi_burden_ratio = float(np.clip(emi_burden_ratio, 0, 1))

    # ── Feature 6: upi_activity_volume ────────────────────────────────────────
    months_active = u_txns["month"].nunique()
    months_active = max(months_active, 1)
    upi_activity_volume = len(upi) / months_active  # avg UPI txns per month

    # ── Feature 7: rent_payment_regularity ────────────────────────────────────
    if len(rent) > 1:
        rent_std = float(rent["amount"].std())
        rent_mean = rent["amount"].mean()
        rent_payment_regularity = 1 - min(rent_std / (rent_mean + 1), 1)
    else:
        rent_payment_regularity = 0.5  # neutral

    # ── Feature 8: cash_withdrawal_ratio ──────────────────────────────────────
    total_cash = cash["amount"].sum() if len(cash) > 0 else 0
    cash_withdrawal_ratio = float(total_cash / (total_expense + 1))
    cash_withdrawal_ratio = float(np.clip(cash_withdrawal_ratio, 0, 1))

    # ── Lineage ───────────────────────────────────────────────────────────────
    lineage = {
        "income_regularity":        credits["transaction_id"].tolist()[:10],
        "spending_consistency":     debits["transaction_id"].tolist()[:10],
        "bill_payment_ratio":       utility["transaction_id"].tolist(),
        "savings_rate":             u_txns["transaction_id"].tolist()[:10],
        "emi_burden_ratio":         emis["transaction_id"].tolist(),
        "upi_activity_volume":      upi["transaction_id"].tolist()[:10],
        "rent_payment_regularity":  rent["transaction_id"].tolist(),
        "cash_withdrawal_ratio":    cash["transaction_id"].tolist()[:10],
    }

    return {
        "user_id":                   uid,
        "income_regularity":         round(income_regularity_norm, 4),
        "spending_consistency":      round(spending_consistency_norm, 4),
        "bill_payment_ratio":        round(bill_payment_ratio, 4),
        "savings_rate":              round(savings_rate, 4),
        "emi_burden_ratio":          round(emi_burden_ratio, 4),
        "upi_activity_volume":       round(upi_activity_volume, 4),
        "rent_payment_regularity":   round(rent_payment_regularity, 4),
        "cash_withdrawal_ratio":     round(cash_withdrawal_ratio, 4),
        "_lineage":                  lineage,
    }

feature_records     = []
lineage_records     = []

for uid in users_df["user_id"]:
    rec = compute_features(uid, txn_df)
    lineage_records.append({"user_id": uid, "lineage": rec.pop("_lineage")})
    feature_records.append(rec)

features_df = pd.DataFrame(feature_records)
print(f"  Engineered features for {len(features_df)} users")


# ─── Step 4: Generate TARGET Label (default = 1, no default = 0) ──────────────

print("Generating TARGET labels...")

def compute_target(row):
    """
    Simulate realistic default probability based on behavior features.
    High risk when: high EMI burden, low savings, low payment ratio, high cash usage.
    """
    risk = 0.0
    risk += (1 - row["savings_rate"])          * 0.25   # low savings → risky
    risk += row["emi_burden_ratio"]             * 0.30   # high EMI → risky
    risk += (1 - row["bill_payment_ratio"])     * 0.20   # missed bills → risky
    risk += row["cash_withdrawal_ratio"]        * 0.10   # high cash → risky
    risk += (1 - row["income_regularity"])      * 0.10   # irregular income → risky
    risk += (1 - row["spending_consistency"])   * 0.05   # erratic spending → risky

    # Clip and add noise
    risk = float(np.clip(risk, 0, 1))
    noise = np.random.normal(0, 0.08)
    prob_default = np.clip(risk + noise, 0, 1)
    return int(prob_default > 0.65)   # threshold → ~20% default rate

features_df["TARGET"] = features_df.apply(compute_target, axis=1)
default_rate = features_df["TARGET"].mean()
print(f"  Default rate: {default_rate:.1%}  (realistic: 15–30%)")


# ─── Step 5: Merge with User Info ─────────────────────────────────────────────

full_df = features_df.merge(
    users_df[["user_id", "age", "monthly_income", "cibil_score",
              "employment_type", "loan_type", "existing_emi",
              "account_age_months"]],
    on="user_id"
)


# ─── Step 6: Train / Test Split ───────────────────────────────────────────────

print("Splitting train/test...")

shuffled = full_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
split_at = int(len(shuffled) * 0.80)
train_df = shuffled.iloc[:split_at]
test_df  = shuffled.iloc[split_at:]

print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")


# ─── Step 7: Save Everything ──────────────────────────────────────────────────

print("Saving files...")

txn_df.to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False)
users_df.to_csv(os.path.join(OUTPUT_DIR, "users.csv"), index=False)
features_df.to_csv(os.path.join(OUTPUT_DIR, "features.csv"), index=False)
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "features_with_lineage.json"), "w") as f:
    json.dump(lineage_records[:50], f, indent=2)   # first 50 for review

print()
print("=" * 55)
print("  FILES GENERATED")
print("=" * 55)
print(f"  transactions.csv           → {len(txn_df):,} rows")
print(f"  users.csv                  → {len(users_df):,} rows")
print(f"  features.csv               → {len(features_df):,} rows")
print(f"  train.csv                  → {len(train_df):,} rows")
print(f"  test.csv                   → {len(test_df):,} rows")
print(f"  features_with_lineage.json → first 50 users")
print()
print("  FEATURE COLUMNS (ready for XGBoost):")
feat_cols = [
    "income_regularity", "spending_consistency", "bill_payment_ratio",
    "savings_rate", "emi_burden_ratio", "upi_activity_volume",
    "rent_payment_regularity", "cash_withdrawal_ratio"
]
for fc in feat_cols:
    print(f"    • {fc}")
print()
print(f"  TARGET column: TARGET  (1=default, 0=no default)")
print(f"  Default rate:  {default_rate:.1%}")
print("=" * 55)
print()
print("  NEXT STEP:")
print("  Give train.csv + test.csv to Cursor AI with the")
print("  model building prompt. Feature columns are ready.")
print("=" * 55)
