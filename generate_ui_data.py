"""Generate real pipeline data for the UI dashboard."""
import json
import pandas as pd
from main_pipeline import evaluate_application

users    = pd.read_csv('users.csv')
features = pd.read_csv('features.csv')

results = {}

# ── Approved ──────────────────────────────────────────────────────────────────
results['approved'] = evaluate_application('USR_0001', 'xpress_credit')

# ── ML Rejected ───────────────────────────────────────────────────────────────
high_risk = features[features['TARGET'] == 1]['user_id'].tolist()
for uid in high_risk:
    r = evaluate_application(uid, 'xpress_credit')
    if r.get('decision') == 'rejected' and r.get('rejection_stage') == 'ml':
        results['ml_rejected'] = r
        break

# ── Policy Rejected ───────────────────────────────────────────────────────────
no_cibil_uid = users[users['cibil_score'].isna()].iloc[0]['user_id']
results['policy_rejected'] = evaluate_application(no_cibil_uid, 'xpress_power')

with open('ui_data.json', 'w') as f:
    json.dump(results, f, indent=2)

for k, v in results.items():
    print(f"{k}: decision={v.get('decision')}, stage={v.get('rejection_stage')}, risk={v.get('risk_score')}")
