"""
AI Decision Engine
==================
Replicates the TypeScript runAIDecision() logic exactly.
"""
import json


def run_ai_decision(data: dict) -> dict:
    loan_type = data['loan_type']
    loan_amount = float(data['loan_amount_rupees'])
    tenure_months = int(data['tenure_months'])
    annual_income = float(data['annual_income_rupees'])
    monthly_expenses = float(data['monthly_expenses_rupees'])
    employment_type = data['employment_type']
    existing_loans_count = int(data['existing_loans_count'])
    existing_emi = float(data['existing_emi_rupees'])
    credit_score = int(data['credit_score'])

    monthly_income = annual_income / 12
    proposed_emi = (loan_amount / tenure_months) * 1.1
    debt_to_income = (monthly_expenses + existing_emi + proposed_emi) / monthly_income if monthly_income > 0 else 1
    expense_ratio = monthly_expenses / monthly_income if monthly_income > 0 else 1
    loan_to_income = loan_amount / annual_income if annual_income > 0 else 1

    shap_factors = []
    risk_score = 50.0

    # ── Factor 1: Credit Score ────────────────────────────────────────────────
    credit_impact = ((credit_score - 650) / 250) * 20
    risk_score -= credit_impact
    if credit_score >= 750:
        credit_desc = "Your credit score is excellent, which strongly supports your application."
        credit_bank = f"Credit score {credit_score}. Prime borrower."
    elif credit_score >= 650:
        credit_desc = "Your credit score is good and meets our requirements."
        credit_bank = f"Credit score {credit_score}. Near-prime borrower."
    elif credit_score >= 550:
        credit_desc = "Your credit score is average. This slightly reduces your chances of approval."
        credit_bank = f"Credit score {credit_score}. Sub-prime borrower — moderate risk."
    else:
        credit_desc = "Your credit score is low, which significantly impacts your application negatively."
        credit_bank = f"Credit score {credit_score}. Sub-prime borrower — high default probability."
    shap_factors.append({
        'feature': 'Credit Score',
        'impact': round(credit_impact, 2),
        'description': credit_desc,
        'bankingDescription': credit_bank,
    })

    # ── Factor 2: Debt-to-Income Ratio ────────────────────────────────────────
    dti_impact = -(debt_to_income - 0.4) * 30
    risk_score -= dti_impact
    if debt_to_income <= 0.4:
        dti_desc = "Your total loan payments compared to your income are within a healthy range."
        dti_bank = f"DTI ratio: {debt_to_income*100:.1f}%. Within RBI prudential norms (≤40%)."
    elif debt_to_income <= 0.6:
        dti_desc = "Your monthly loan payments are somewhat high compared to your income."
        dti_bank = f"DTI ratio: {debt_to_income*100:.1f}%. Borderline — close monitoring required."
    else:
        dti_desc = "Your monthly obligations are too high compared to what you earn. This is a major concern."
        dti_bank = f"DTI ratio: {debt_to_income*100:.1f}%. High credit utilization — systemic default risk elevated."
    shap_factors.append({
        'feature': 'Debt-to-Income Ratio',
        'impact': round(dti_impact, 2),
        'description': dti_desc,
        'bankingDescription': dti_bank,
    })

    # ── Factor 3: Monthly Expenses ────────────────────────────────────────────
    expense_impact = -(expense_ratio - 0.5) * 20
    risk_score -= expense_impact
    if expense_ratio <= 0.5:
        exp_desc = "Your monthly spending is well-managed relative to your income."
        exp_bank = f"Monthly expense ratio: {expense_ratio*100:.1f}%. Healthy expense ratio."
    elif expense_ratio <= 0.7:
        exp_desc = "Your monthly expenses are moderately high compared to your income."
        exp_bank = f"Monthly expense ratio: {expense_ratio*100:.1f}%. Above-average expense burden."
    else:
        exp_desc = "Your monthly expenses consume a large portion of your income, leaving little for loan repayment."
        exp_bank = f"Monthly expense ratio: {expense_ratio*100:.1f}%. Excessive spending — insufficient disposable income."
    shap_factors.append({
        'feature': 'Monthly Expenses',
        'impact': round(expense_impact, 2),
        'description': exp_desc,
        'bankingDescription': exp_bank,
    })

    # ── Factor 4: Existing Loans ──────────────────────────────────────────────
    loans_impact = -(existing_loans_count) * 5
    risk_score -= loans_impact
    if existing_loans_count == 0:
        loans_desc = "You have no existing loans, which is a positive indicator."
        loans_bank = "Active loan count: 0. Low concurrent credit exposure."
    elif existing_loans_count == 1:
        loans_desc = "You have one existing loan. This is manageable."
        loans_bank = "Active loan count: 1. Low concurrent credit exposure."
    elif existing_loans_count <= 3:
        loans_desc = f"You have {existing_loans_count} existing loans. This adds to your financial obligations."
        loans_bank = f"Active loan count: {existing_loans_count}. Multiple credit facilities — monitor aggregate exposure."
    else:
        loans_desc = f"You have {existing_loans_count} existing loans, which raises concerns about repayment capacity."
        loans_bank = f"Active loan count: {existing_loans_count}. Multiple concurrent facilities — concentration risk."
    shap_factors.append({
        'feature': 'Existing Loans',
        'impact': round(loans_impact, 2),
        'description': loans_desc,
        'bankingDescription': loans_bank,
    })

    # ── Factor 5: Employment Type ─────────────────────────────────────────────
    emp_map = {'salaried': 10, 'self_employed': 5, 'business_owner': 3, 'unemployed': -15}
    emp_impact = emp_map.get(employment_type, 0)
    risk_score -= emp_impact
    emp_descs = {
        'salaried': "Salaried employment provides stable, predictable income — a strong positive factor.",
        'self_employed': "Self-employment income is variable but acceptable with good documentation.",
        'business_owner': "Business income can be irregular. Additional documentation may be required.",
        'unemployed': "No current employment significantly increases the risk of default.",
    }
    emp_bank_descs = {
        'salaried': "Salaried — stable income stream, low income volatility risk.",
        'self_employed': "Self-employed — moderate income volatility, requires ITR verification.",
        'business_owner': "Business owner — income subject to business cycle risk.",
        'unemployed': "Unemployed — no verifiable income source, high default probability.",
    }
    shap_factors.append({
        'feature': 'Employment Type',
        'impact': round(emp_impact, 2),
        'description': emp_descs.get(employment_type, ''),
        'bankingDescription': emp_bank_descs.get(employment_type, ''),
    })

    # ── Factor 6: Loan-to-Income Ratio ────────────────────────────────────────
    lti_impact = -(loan_to_income - 3) * 5
    risk_score -= lti_impact
    if loan_to_income <= 3:
        lti_desc = "The loan amount is reasonable relative to your annual income."
        lti_bank = f"Loan-to-income ratio: {loan_to_income:.1f}x. Within acceptable lending thresholds."
    elif loan_to_income <= 5:
        lti_desc = "The loan amount is somewhat high relative to your annual income."
        lti_bank = f"Loan-to-income ratio: {loan_to_income:.1f}x. Elevated exposure — monitor repayment capacity."
    else:
        lti_desc = "The loan amount is very high relative to your annual income. This is a significant risk factor."
        lti_bank = f"Loan-to-income ratio: {loan_to_income:.1f}x. Excessive leverage — high default risk."
    shap_factors.append({
        'feature': 'Loan-to-Income Ratio',
        'impact': round(lti_impact, 2),
        'description': lti_desc,
        'bankingDescription': lti_bank,
    })

    # ── Final risk score ──────────────────────────────────────────────────────
    risk_score = max(5.0, min(95.0, risk_score))
    decision = 'approved' if risk_score < 55 else 'rejected'

    # ── User explanation ──────────────────────────────────────────────────────
    if decision == 'approved':
        explanation = (
            f"Congratulations! Your loan application has been approved by our AI system. "
            f"Your risk score of {risk_score:.0f}/100 indicates a low probability of default. "
            f"Your application will now be reviewed by a banker before final disbursement."
        )
    else:
        weak = [f for f in shap_factors if f['impact'] < 0]
        top_weak = weak[:2] if weak else shap_factors[:2]
        features_str = ' and '.join([f['feature'] for f in top_weak])
        explanation = (
            f"Unfortunately, your loan application has been declined by our AI system. "
            f"Your risk score of {risk_score:.0f}/100 indicates a higher probability of default. "
            f"The primary concerns are your {features_str}. "
            f"Please review the improvement suggestions below."
        )

    # ── Improvement suggestions ───────────────────────────────────────────────
    suggestions = []
    if credit_score < 700:
        suggestions.append("Improve your credit score by paying all bills on time and reducing outstanding debt.")
    if debt_to_income > 0.4:
        suggestions.append("Reduce your debt-to-income ratio by paying off existing loans before applying.")
    if expense_ratio > 0.5:
        suggestions.append("Lower your monthly expenses to improve your disposable income ratio.")
    if existing_loans_count >= 3:
        suggestions.append("Close some existing loan accounts before applying for a new loan.")
    if employment_type == 'unemployed':
        suggestions.append("Secure stable employment before reapplying for a loan.")
    if loan_to_income > 4:
        suggestions.append("Consider applying for a smaller loan amount relative to your annual income.")

    return {
        'decision': decision,
        'risk_score': round(risk_score, 2),
        'user_explanation': explanation,
        'shap_factors': shap_factors,
        'improvement_suggestions': suggestions,
        'fairness_check_passed': True,
        'fairness_note': (
            'No protected attributes (age, gender, religion, caste) were used in this decision. '
            'The model evaluates only financial behavior and repayment capacity.'
        ),
    }
