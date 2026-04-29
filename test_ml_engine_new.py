"""
TEST ML ENGINE WITH NEW TRAINED MODELS
======================================
Verify that ml_engine.py works correctly with the newly trained models.
"""

import sys
sys.path.insert(0, r'C:\Users\gowsi\OneDrive\Desktop\H&H\smart_lender\smart_lender')

from loans.ml_engine import run_ml_decision
import json

print("=" * 90)
print("ML ENGINE TEST - NEWLY TRAINED MODELS")
print("=" * 90)

# Test cases with different profiles
test_cases = {
    "Test 1: EXCELLENT Profile (Should APPROVE)": {
        'income_regularity': 0.95,
        'spending_consistency': 0.90,
        'bill_payment_ratio': 0.95,
        'savings_rate': 0.35,
        'emi_burden_ratio': 0.15,
        'upi_activity_volume': 8.0,
        'rent_payment_regularity': 0.92,
        'cash_withdrawal_ratio': 0.08,
    },
    
    "Test 2: GOOD Profile (Should APPROVE)": {
        'income_regularity': 0.85,
        'spending_consistency': 0.80,
        'bill_payment_ratio': 0.90,
        'savings_rate': 0.20,
        'emi_burden_ratio': 0.25,
        'upi_activity_volume': 6.0,
        'rent_payment_regularity': 0.85,
        'cash_withdrawal_ratio': 0.12,
    },
    
    "Test 3: AVERAGE Profile (Borderline)": {
        'income_regularity': 0.65,
        'spending_consistency': 0.60,
        'bill_payment_ratio': 0.70,
        'savings_rate': 0.05,
        'emi_burden_ratio': 0.40,
        'upi_activity_volume': 3.5,
        'rent_payment_regularity': 0.65,
        'cash_withdrawal_ratio': 0.35,
    },
    
    "Test 4: POOR Profile (Should REJECT)": {
        'income_regularity': 0.30,
        'spending_consistency': 0.25,
        'bill_payment_ratio': 0.35,
        'savings_rate': -0.15,
        'emi_burden_ratio': 0.70,
        'upi_activity_volume': 1.0,
        'rent_payment_regularity': 0.25,
        'cash_withdrawal_ratio': 0.85,
    },
    
    "Test 5: VERY BAD Profile (High Default Risk)": {
        'income_regularity': 0.10,
        'spending_consistency': 0.08,
        'bill_payment_ratio': 0.20,
        'savings_rate': -0.50,
        'emi_burden_ratio': 0.95,
        'upi_activity_volume': 0.3,
        'rent_payment_regularity': 0.10,
        'cash_withdrawal_ratio': 0.95,
    },
}

results = []

for test_name, features in test_cases.items():
    print(f"\n{'─' * 90}")
    print(test_name)
    print('─' * 90)
    
    try:
        # Run ML decision
        result = run_ml_decision(features)
        
        # Display results
        print(f"\nDecision:         {result['decision'].upper()}")
        print(f"Risk Score:       {result['risk_score']:.2f}/100")
        print(f"Risk Probability: {result.get('risk_score', 0) / 100:.2%}")
        
        print(f"\nTop 3 Influencing Factors:")
        shap_factors = result.get('shap_factors', [])
        for i, factor in enumerate(shap_factors[:3], 1):
            impact_dir = "↑ HURTS" if factor['impact_direction'] == 'hurts' else "↓ HELPS"
            print(f"  {i}. {factor['feature']:30} {impact_dir:10} ({factor['impact']:7.4f})")
        
        print(f"\nTop Improvement Suggestion:")
        suggestions = result.get('improvement_suggestions', [])
        if suggestions:
            print(f"  → {suggestions[0]}")
        else:
            print(f"  → No improvements needed!")
        
        print(f"\n✓ Test passed")
        results.append((test_name, result['decision'], result['risk_score'], True))
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append((test_name, 'ERROR', None, False))

# Summary
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(f"\nTest Results:")
print(f"{'Test Name':<50} {'Decision':<12} {'Risk Score':<12} {'Status':<10}")
print("─" * 90)

for test_name, decision, risk_score, passed in results:
    test_short = test_name[:48]
    status = "✓ PASS" if passed else "✗ FAIL"
    score_str = f"{risk_score:.2f}" if risk_score is not None else "N/A"
    print(f"{test_short:<50} {decision.upper():<12} {score_str:<12} {status:<10}")

passed_count = sum(1 for _, _, _, p in results if p)
total_count = len(results)

print(f"\n{'=' * 90}")
print(f"Tests Passed: {passed_count}/{total_count}")
print(f"{'=' * 90}\n")

if passed_count == total_count:
    print("✓ ALL TESTS PASSED - ML ENGINE IS WORKING CORRECTLY WITH NEW MODELS")
else:
    print("✗ SOME TESTS FAILED - CHECK ERRORS ABOVE")
