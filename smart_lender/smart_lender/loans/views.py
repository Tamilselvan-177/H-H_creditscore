import json
import math
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.db.models import Avg, Sum
from django.http import JsonResponse

from .models import LoanApplication, UserProfile, CsvUserFeature, CsvTransaction, DemoProfile
from .forms import RegisterForm, LoanApplicationForm, BankerReviewForm
from .ml_engine import run_ml_decision


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_role(user):
    try:
        return user.profile.role
    except Exception:
        return 'user'


def compute_monthly_expenses(csv_user_id):
    """
    Compute actual monthly living expenses from CsvTransaction data.
    
    Excludes EMI (already counted separately as existing_emi).
    Includes: rent, food, utilities, shopping, cash withdrawals, etc.
    
    Returns: Average monthly expenses (float)
    """
    if not csv_user_id:
        return 0
    
    try:
        from django.db.models import Sum, Count
        from django.db.models.functions import Extract
        from django.db.models import Q
        
        # Get all debit transactions EXCLUDING EMI (which is tracked separately)
        transactions = CsvTransaction.objects.filter(
            csv_user_id=csv_user_id,
            type='debit'
        ).exclude(
            category__in=['emi', 'loan_payment']  # These are tracked separately
        )
        
        if not transactions.exists():
            return 0
        
        # Group by year-month and sum
        monthly_data = transactions.values('timestamp__year', 'timestamp__month').annotate(
            total=Sum('amount')
        )
        
        if not monthly_data:
            return 0
        
        # Calculate average
        months = len(monthly_data)
        total_expenses = sum(m['total'] for m in monthly_data)
        avg_monthly = total_expenses / months if months > 0 else 0
        
        return float(avg_monthly)
        
    except Exception as e:
        print(f"Error computing monthly expenses: {e}")
        return 0


# ─── Auth ─────────────────────────────────────────────────────────────────────

def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            role       = form.cleaned_data.get('role', 'user')
            csv_uid    = form.cleaned_data.get('csv_user_id', '').strip()

            # Try to pull demographics from dataset if csv_user_id given
            profile_kwargs = dict(user=user, role=role, csv_user_id=csv_uid or None)
            if csv_uid:
                feat = CsvUserFeature.objects.filter(csv_user_id=csv_uid).first()
                # UserProfile demographics come from UserProfile.csv_user_id lookup
                # (populated by import_dataset command)
                existing = UserProfile.objects.filter(csv_user_id=csv_uid).first()
                if existing:
                    profile_kwargs.update(
                        age=existing.age,
                        employment_type=existing.employment_type,
                        monthly_income=existing.monthly_income,
                        cibil_score=existing.cibil_score,
                        existing_emi=existing.existing_emi,
                        account_age_months=existing.account_age_months,
                        state=existing.state,
                        loan_type_preference=existing.loan_type_preference,
                    )

            UserProfile.objects.create(**profile_kwargs)
            login(request, user)
            return redirect('home')
    else:
        form = RegisterForm()
    return render(request, 'loans/register.html', {'form': form})


def home_view(request):
    if not request.user.is_authenticated:
        return render(request, 'loans/landing.html')
    role = get_role(request.user)
    if role == 'banker':
        return redirect('banker_dashboard')
    if role == 'regulator':
        return redirect('regulator_dashboard')
    return redirect('user_dashboard')


# ─── User Views ───────────────────────────────────────────────────────────────

@login_required
def user_dashboard(request):
    if get_role(request.user) != 'user':
        return redirect('home')
    loans = LoanApplication.objects.filter(user=request.user)
    stats = {
        'total':    loans.count(),
        'approved': loans.filter(final_status='approved').count(),
        'rejected': loans.filter(final_status='rejected').count(),
        'pending':  loans.filter(final_status='pending').count(),
    }
    # Get user's profile info for display
    try:
        profile = request.user.profile
    except Exception:
        profile = None

    return render(request, 'loans/user_dashboard.html', {
        'loans':   loans[:20],
        'stats':   stats,
        'profile': profile,
    })


@login_required
def apply_loan(request):
    if get_role(request.user) != 'user':
        return redirect('home')

    # Check if user has pre-loaded features
    try:
        profile = request.user.profile
        csv_uid = profile.csv_user_id
    except Exception:
        profile = None
        csv_uid = None

    feature_record = CsvUserFeature.objects.filter(csv_user_id=csv_uid).first() if csv_uid else None
    demo_profile   = DemoProfile.objects.filter(csv_user_id=csv_uid).first() if csv_uid else None

    if request.method == 'POST':
        form = LoanApplicationForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data

            # ── Get features ──────────────────────────────────────────────────
            if feature_record:
                feature_values = feature_record.as_dict()
            else:
                # No pre-loaded data — use neutral defaults for demo
                feature_values = {
                    'income_regularity':       0.5,
                    'spending_consistency':    0.5,
                    'bill_payment_ratio':      0.7,
                    'savings_rate':            0.1,
                    'emi_burden_ratio':        0.3,
                    'upi_activity_volume':     5.0,
                    'rent_payment_regularity': 0.6,
                    'cash_withdrawal_ratio':   0.15,
                }

            # ── Run real ML model ─────────────────────────────────────────────
            try:
                ml_result = run_ml_decision(feature_values)
            except Exception as e:
                messages.error(request, f'AI model error: {e}')
                return render(request, 'loans/apply_loan.html', {
                    'form': form,
                    'profile': profile,
                    'has_features': bool(feature_record),
                    'demo': demo_profile,
                    'csv_uid': csv_uid,
                })

            # ── Save application — final_status stays PENDING ─────────────────
            # Compute actual monthly expenses from transactions (excluding EMI)
            computed_expenses = compute_monthly_expenses(csv_uid) if csv_uid else 0

            # ── Get previous hash for chain ───────────────────────────────────
            last_app = LoanApplication.objects.order_by('-created_at').first()
            prev_hash = last_app.current_hash if last_app and last_app.current_hash else '0' * 64

            app = LoanApplication.objects.create(
                user=request.user,
                loan_type=data['loan_type'],
                loan_amount_rupees=data['loan_amount_rupees'],
                tenure_months=data['tenure_months'],
                vehicle_purpose=data.get('vehicle_purpose') or None,
                # Demographics from profile
                annual_income_rupees=profile.monthly_income * 12 if profile else 0,
                monthly_expenses_rupees=computed_expenses,
                employment_type=profile.employment_type if profile else 'salaried',
                existing_loans_count=0,
                existing_emi_rupees=profile.existing_emi if profile else 0,
                credit_score=int(profile.cibil_score) if profile and profile.cibil_score else 0,
                # AI result
                ai_decision=ml_result['decision'],
                risk_score=ml_result['risk_score'],
                user_explanation=ml_result['user_explanation'],
                shap_factors_json=json.dumps(ml_result['shap_factors']),
                improvement_suggestions_json=json.dumps(ml_result['improvement_suggestions']),
                fairness_check_passed=ml_result['fairness_check_passed'],
                fairness_note=ml_result['fairness_note'],
                feature_values_json=json.dumps(ml_result['feature_values']),
                # Banker fields
                banker_decision='pending',
                final_status='pending',
                # Hash chain
                previous_hash=prev_hash,
            )
            # Compute and save current hash now that we have the ID
            app.current_hash = app.compute_hash(prev_hash)
            app.save(update_fields=['current_hash'])
            messages.success(request, 'Application submitted successfully! A banker will review it shortly.')
            return redirect('loan_result', pk=app.pk)

    else:
        form = LoanApplicationForm()

    return render(request, 'loans/apply_loan.html', {
        'form':         form,
        'profile':      profile,
        'has_features': bool(feature_record),
        'csv_uid':      csv_uid,
        'features':     feature_record,
        'demo':         demo_profile,   # ← all extra fields pre-filled
    })


@login_required
def loan_result(request, pk):
    if get_role(request.user) != 'user':
        return redirect('home')
    app = get_object_or_404(LoanApplication, pk=pk, user=request.user)

    # Only show full AI explanation AFTER banker has decided
    decided = app.final_status in ('approved', 'rejected')

    shap_data = json.dumps(app.shap_factors) if decided else '[]'

    return render(request, 'loans/loan_result.html', {
        'app':      app,
        'decided':  decided,
        'shap_data': shap_data,
    })


# ─── What-If Simulator ────────────────────────────────────────────────────────

# Grok AI integration disabled — using built-in fallback explanations
# To enable: GROK_API_KEY = os.environ.get('GROK_API_KEY')

FEATURE_META = {
    'income_regularity':       {'label': 'Income Regularity',       'min': 0.0,  'max': 1.0,  'step': 0.01, 'good': 'above', 'threshold': 0.5,  'unit': ''},
    'spending_consistency':    {'label': 'Spending Consistency',     'min': 0.0,  'max': 1.0,  'step': 0.01, 'good': 'above', 'threshold': 0.5,  'unit': ''},
    'bill_payment_ratio':      {'label': 'Bill Payment Ratio',       'min': 0.0,  'max': 1.0,  'step': 0.01, 'good': 'above', 'threshold': 0.8,  'unit': ''},
    'savings_rate':            {'label': 'Savings Rate',             'min': -1.0, 'max': 1.0,  'step': 0.01, 'good': 'above', 'threshold': 0.1,  'unit': ''},
    'emi_burden_ratio':        {'label': 'EMI Burden Ratio',         'min': 0.0,  'max': 1.0,  'step': 0.01, 'good': 'below', 'threshold': 0.5,  'unit': ''},
    'upi_activity_volume':     {'label': 'UPI Activity (txns/mo)',   'min': 0.0,  'max': 20.0, 'step': 0.1,  'good': 'above', 'threshold': 5.0,  'unit': ''},
    'rent_payment_regularity': {'label': 'Rent Payment Regularity',  'min': 0.0,  'max': 1.0,  'step': 0.01, 'good': 'above', 'threshold': 0.6,  'unit': ''},
    'cash_withdrawal_ratio':   {'label': 'Cash Withdrawal Ratio',    'min': 0.0,  'max': 1.0,  'step': 0.01, 'good': 'below', 'threshold': 0.2,  'unit': ''},
}


def recourse_api(request, pk):
    """
    Compute recourse validity on page load — no slider interaction needed.
    Returns the minimum improvements to flip the decision.
    """
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Login required'}, status=401)

    if get_role(request.user) != 'user':
        return JsonResponse({'error': 'Unauthorized'}, status=403)

    app = get_object_or_404(LoanApplication, pk=pk, user=request.user)

    if app.final_status not in ('approved', 'rejected'):
        return JsonResponse({'needed': False, 'message': 'Decision pending.'})

    from .ml_engine import run_ml_decision
    fv = app.feature_values
    try:
        result = run_ml_decision(fv)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    recourse = _compute_recourse(fv, fv, result)
    return JsonResponse(recourse)


@login_required
def whatif_simulator(request, pk):
    """What-If simulator page — user only, only after a decision."""
    if get_role(request.user) != 'user':
        return redirect('home')
    app = get_object_or_404(LoanApplication, pk=pk, user=request.user)

    if app.final_status not in ('approved', 'rejected'):
        messages.info(request, 'What-If simulator is available after the banker reviews your application.')
        return redirect('loan_result', pk=pk)

    fv = app.feature_values
    feature_list = [
        {
            'key':       k,
            'label':     FEATURE_META[k]['label'],
            'value':     round(fv.get(k, 0.0), 4),
            'min':       FEATURE_META[k]['min'],
            'max':       FEATURE_META[k]['max'],
            'step':      FEATURE_META[k]['step'],
            'threshold': FEATURE_META[k]['threshold'],
            'good':      FEATURE_META[k]['good'],
        }
        for k in FEATURE_META
    ]

    return render(request, 'loans/whatif_simulator.html', {
        'app':          app,
        'feature_list': feature_list,
        'feature_json': json.dumps({k: round(fv.get(k, 0.0), 4) for k in FEATURE_META}),
        'original_risk':     app.risk_score,
        'original_decision': app.ai_decision,
    })


def whatif_api(request):
    """
    AJAX endpoint — receives modified feature values,
    runs real XGBoost model, calls Grok for explanation,
    returns JSON with new score + decision + Grok reasoning.
    """
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Login required'}, status=401)

    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    if get_role(request.user) != 'user':
        return JsonResponse({'error': 'Unauthorized'}, status=403)

    try:
        body = json.loads(request.body)
        features = body.get('features', {})
        original = body.get('original', {})
        app_id   = body.get('app_id')
    except Exception:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    # ── Run real model ────────────────────────────────────────────────────────
    from .ml_engine import run_ml_decision
    try:
        result = run_ml_decision(features)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    new_score    = result['risk_score']
    new_decision = result['decision']
    orig_score   = float(original.get('risk_score', new_score))

    # ── Compute what changed ──────────────────────────────────────────────────
    changes = []
    for k, meta in FEATURE_META.items():
        orig_val = float(original.get(k, features.get(k, 0)))
        new_val  = float(features.get(k, 0))
        delta    = new_val - orig_val
        if abs(delta) > 0.001:
            direction = 'improved' if (
                (meta['good'] == 'above' and delta > 0) or
                (meta['good'] == 'below' and delta < 0)
            ) else 'worsened'
            changes.append({
                'feature':   meta['label'],
                'key':       k,
                'from':      round(orig_val, 3),
                'to':        round(new_val, 3),
                'delta':     round(delta, 3),
                'direction': direction,
            })

    # ── Grok AI reasoning ─────────────────────────────────────────────────────
    grok_explanation = _call_grok(
        features=features,
        original=original,
        changes=changes,
        orig_score=orig_score,
        new_score=new_score,
        orig_decision=original.get('decision', 'rejected'),
        new_decision=new_decision,
    )

    return JsonResponse({
        'new_score':       round(new_score, 1),
        'new_decision':    new_decision,
        'orig_score':      round(orig_score, 1),
        'score_delta':     round(orig_score - new_score, 1),
        'changes':         changes,
        'grok_explanation': grok_explanation,
        'shap_factors':    result['shap_factors'],
        'recourse':        _compute_recourse(features, original, result),
    })


def _compute_recourse(current_features: dict, original: dict, current_result: dict) -> dict:
    """
    Recourse Validity Score:
    1. Find the top hurting features from SHAP
    2. Set each to its optimal value
    3. Run the model to verify the decision flips
    4. Return: which improvements are needed, approval probability after, validity confirmed
    """
    from .ml_engine import run_ml_decision, FEATURE_COLS

    OPTIMAL_VALUES = {
        'income_regularity':       0.75,
        'spending_consistency':    0.70,
        'bill_payment_ratio':      0.90,
        'savings_rate':            0.15,
        'emi_burden_ratio':        0.30,
        'upi_activity_volume':     10.0,
        'rent_payment_regularity': 0.80,
        'cash_withdrawal_ratio':   0.10,
    }

    # Already approved — no recourse needed
    if current_result['decision'] == 'approved':
        return {
            'needed': False,
            'message': 'Already approved — no recourse needed.',
            'steps': [],
            'approval_prob_after': round((1 - current_result['risk_score_raw']) * 100, 1),
            'valid': True,
        }

    # Get top hurting features sorted by SHAP impact (highest positive = most hurting)
    hurters = sorted(
        [f for f in current_result['shap_factors'] if f.get('impact_direction') == 'hurts'],
        key=lambda x: abs(x['impact']),
        reverse=True,
    )

    if not hurters:
        return {'needed': False, 'message': 'No specific improvements identified.', 'steps': [], 'valid': False}

    # Build recourse steps — fix top 3 hurting features
    steps = []
    improved = dict(current_features)

    for factor in hurters[:3]:
        key = factor.get('feature_key', '')
        if not key or key not in OPTIMAL_VALUES:
            continue
        current_val = float(current_features.get(key, 0))
        optimal_val = OPTIMAL_VALUES[key]

        # Only suggest if there's meaningful room to improve
        if abs(optimal_val - current_val) < 0.05:
            continue

        improved[key] = optimal_val

        # Human-readable action
        action_map = {
            'income_regularity':       f'Ensure consistent monthly income (target: 75% regularity)',
            'spending_consistency':    f'Stabilize monthly spending patterns (target: 70% consistency)',
            'bill_payment_ratio':      f'Pay all utility bills on time (target: 90% on-time)',
            'savings_rate':            f'Save at least 15% of monthly income',
            'emi_burden_ratio':        f'Reduce EMI payments to below 30% of income',
            'upi_activity_volume':     f'Increase digital transactions to 10+ per month',
            'rent_payment_regularity': f'Pay rent consistently every month (target: 80% regularity)',
            'cash_withdrawal_ratio':   f'Reduce cash usage to below 10% of spending',
        }
        steps.append({
            'feature':     factor['feature'],
            'feature_key': key,
            'current':     round(current_val, 3),
            'target':      optimal_val,
            'action':      action_map.get(key, f'Improve {key}'),
            'shap_impact': round(abs(factor['impact']), 4),
        })

    if not steps:
        return {'needed': False, 'message': 'Your values are already near optimal.', 'steps': [], 'valid': False}

    # Validate: run model with improved values
    try:
        validated = run_ml_decision(improved)
        flips     = validated['decision'] == 'approved'
        prob_after = round((1 - validated['risk_score_raw']) * 100, 1)
        score_after = round(validated['risk_score'], 1)
    except Exception:
        flips      = False
        prob_after = 0
        score_after = 0

    return {
        'needed':            True,
        'steps':             steps,
        'flips_decision':    flips,
        'approval_prob_after': prob_after,
        'score_after':       score_after,
        'valid':             flips,
        'message': (
            f'If you make these {len(steps)} improvements, your approval probability becomes {prob_after}% '
            f'and the decision {"FLIPS TO APPROVED ✓" if flips else "remains rejected (more work needed)"}.'
        ),
    }


def _call_grok(features, original, changes, orig_score, new_score, orig_decision, new_decision):
    """Call Groq API for plain-language What-If reasoning. Falls back to built-in logic if API fails."""
    if not changes:
        return "No changes were made to the features. Adjust the sliders to see what-if scenarios."

    delta = orig_score - new_score
    changes_text = '\n'.join([f"- {c['feature']}: {c['from']} → {c['to']} ({c['direction']})" for c in changes])
    decision_change = (
        f"The decision CHANGED from {orig_decision.upper()} to {new_decision.upper()}."
        if orig_decision != new_decision else
        f"The decision remains {new_decision.upper()}."
    )
    prompt = (
        f"You are an AI credit advisor. Explain this What-If scenario in 2-3 simple sentences.\n\n"
        f"Original: {orig_score:.0f}/100 ({orig_decision.upper()}) → New: {new_score:.0f}/100 ({new_decision.upper()})\n"
        f"Changes: {changes_text}\n{decision_change}\n\n"
        f"Be encouraging, under 70 words, plain paragraph, no bullet points."
    )

    try:
        import requests as req_lib
        response = req_lib.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {GROK_API_KEY}', 'Content-Type': 'application/json'},
            json={'model': 'llama-3.3-70b-versatile', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 120, 'temperature': 0.7},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception:
        pass  # Fall through to built-in logic
    print(new_decision)
    # Built-in fallback
    if orig_decision != new_decision:
        return (f"Great news! By adjusting these factors, your risk score changed from {orig_score:.0f} to {new_score:.0f}/100 "
                f"and the decision flipped to {new_decision.upper()}. Maintain these improvements in your actual financial behavior.")
    if delta > 10:
        return (f"These changes reduced your risk score by {delta:.0f} points to {new_score:.0f}/100. "
                f"You're moving toward the approval threshold (below 50). Keep improving these factors.")
    if delta > 3:
        return (f"Good progress — score improved by {delta:.0f} points to {new_score:.0f}/100. "
                f"Focus on the highest-impact factors (Savings Rate, EMI Burden) to reach approval.")
    if delta > 0:
        return (f"Small improvement to {new_score:.0f}/100. Focus on Income Regularity, EMI Burden, and Savings Rate for bigger gains.")
    return (f"These changes worsened your score to {new_score:.0f}/100. Adjust in the direction marked 'good' for better results.")



# ─── Banker Views ─────────────────────────────────────────────────────────────

@login_required
def banker_dashboard(request):
    if get_role(request.user) != 'banker':
        return redirect('home')

    status_filter = request.GET.get('status', 'pending')
    all_apps = LoanApplication.objects.all()

    if status_filter == 'pending':
        apps = all_apps.filter(banker_decision='pending')
    elif status_filter == 'approved':
        apps = all_apps.filter(banker_decision='approved')
    elif status_filter == 'rejected':
        apps = all_apps.filter(banker_decision='rejected')
    else:
        apps = all_apps

    stats = {
        'pending_review':  all_apps.filter(banker_decision='pending').count(),
        'avg_risk':        round(all_apps.aggregate(avg=Avg('risk_score'))['avg'] or 0, 1),
        'total_approved':  all_apps.filter(banker_decision='approved').count(),
        'total_disbursed': all_apps.filter(banker_decision='approved').aggregate(
            total=Sum('loan_amount_rupees'))['total'] or 0,
    }

    return render(request, 'loans/banker_dashboard.html', {
        'apps':          apps,
        'stats':         stats,
        'status_filter': status_filter,
    })


@login_required
def banker_loan_detail(request, pk):
    if get_role(request.user) != 'banker':
        return redirect('home')

    app = get_object_or_404(LoanApplication, pk=pk)

    if request.method == 'POST':
        decision = request.POST.get('decision')
        note     = request.POST.get('note', '').strip()
        if decision in ('approved', 'rejected'):
            app.banker_decision    = decision
            app.banker_note        = note
            app.banker_reviewed_at = timezone.now()
            app.banker             = request.user
            app.final_status       = decision
            app.save()
            messages.success(request, f'Application #{app.pk} has been {decision}.')
            return redirect('banker_dashboard')

    shap_data = json.dumps(app.shap_factors)
    traced    = _trace_transactions(app)

    # ── Layer 1: Policy eligibility checks ───────────────────────────────────
    policy_checks = _compute_policy_checks(app)
    policy_passed = all(c['passed'] for c in policy_checks)

    # ── Layer 2: 4-point ML summary for banker ────────────────────────────────
    ml_summary = _compute_ml_summary(app)

    return render(request, 'loans/banker_loan_detail.html', {
        'app':           app,
        'shap_data':     shap_data,
        'traced':        traced,
        'policy_checks': policy_checks,
        'policy_passed': policy_passed,
        'ml_summary':    ml_summary,
    })


def _compute_policy_checks(app):
    """
    SBI loan policy eligibility — visual checklist for banker.
    Returns list of {label, value, required, passed, note}
    """
    profile = None
    try:
        profile = app.user.profile
    except Exception:
        pass

    monthly_income = profile.monthly_income if profile else 0
    annual_income  = monthly_income * 12
    existing_emi   = profile.existing_emi if profile else 0
    cibil          = profile.cibil_score if profile else None
    age            = profile.age if profile else 0
    acct_months    = profile.account_age_months if profile else 0
    emi_ratio      = (existing_emi / monthly_income) if monthly_income > 0 else 0

    # Loan-type specific rules
    if app.loan_type == 'vehicle':
        min_income  = 10000
        need_cibil  = False
        min_acct    = 12
        label_type  = 'SBI Two Wheeler Loan'
    else:
        # personal — use xpress_credit rules (most common)
        min_income  = 5000
        need_cibil  = False
        min_acct    = 0
        label_type  = 'SBI Personal Loan (Xpress Credit)'

    checks = [
        {
            'label':    'Minimum Age',
            'value':    f'{age} years',
            'required': '18+ years',
            'passed':   age >= 18,
            'note':     'Applicant must be at least 18 years old.',
        },
        {
            'label':    'Maximum Age',
            'value':    f'{age} years',
            'required': '≤ 76 years',
            'passed':   age <= 76,
            'note':     'Applicant must be below 76 years.',
        },
        {
            'label':    'Monthly Income',
            'value':    f'₹{monthly_income:,.0f}',
            'required': f'≥ ₹{min_income:,}',
            'passed':   monthly_income >= min_income,
            'note':     f'Minimum income for {label_type}.',
        },
        {
            'label':    'EMI Burden',
            'value':    f'{emi_ratio*100:.1f}% of income',
            'required': '≤ 50%',
            'passed':   emi_ratio <= 0.50,
            'note':     'Existing EMI should not exceed 50% of monthly income.',
        },
        {
            'label':    'Account Age',
            'value':    f'{acct_months} months',
            'required': f'≥ {min_acct} months' if min_acct > 0 else 'No minimum',
            'passed':   acct_months >= min_acct,
            'note':     'Account history requirement.',
        },
        {
            'label':    'CIBIL Score',
            'value':    f'{int(cibil)}' if cibil else 'Not available',
            'required': 'Not required for this loan' if not need_cibil else '≥ 750',
            'passed':   True if not need_cibil else (cibil is not None and cibil >= 750),
            'note':     'CIBIL not mandatory for this loan type — behavior-based scoring used.',
        },
    ]
    return checks


def _compute_ml_summary(app):
    """
    4 key decision points for the banker based on real ML analysis.
    Analyses the actual feature values + SHAP to give honest, nuanced points.
    """
    factors  = app.shap_factors   # sorted by abs(shap) desc
    fv       = app.feature_values  # raw 0-1 feature values
    risk_raw = app.risk_score      # already 0-100 scale

    # ── Point 1: Behavioral Score (computed from actual feature values) ───────
    # Weighted score based on the most important features
    savings       = fv.get('savings_rate', 0)
    emi_burden    = fv.get('emi_burden_ratio', 0)
    bill_ratio    = fv.get('bill_payment_ratio', 0)
    cash_ratio    = fv.get('cash_withdrawal_ratio', 0)
    income_reg    = fv.get('income_regularity', 0)
    spending_con  = fv.get('spending_consistency', 0)
    upi_vol       = fv.get('upi_activity_volume', 0)
    rent_reg      = fv.get('rent_payment_regularity', 0)

    # Count how many features are in "good" range
    good_count = sum([
        savings >= 0.10,
        emi_burden <= 0.50,
        bill_ratio >= 0.80,
        cash_ratio <= 0.20,
        income_reg >= 0.50,
        spending_con >= 0.50,
        upi_vol >= 5.0,
        rent_reg >= 0.60,
    ])
    bad_count = 8 - good_count

    # Behavioral health score out of 100
    behavior_score = round((good_count / 8) * 100)

    if behavior_score >= 70:
        beh_status = 'good'
        beh_label  = f'{behavior_score}% — Strong financial behavior'
    elif behavior_score >= 45:
        beh_status = 'warn'
        beh_label  = f'{behavior_score}% — Mixed financial behavior'
    else:
        beh_status = 'bad'
        beh_label  = f'{behavior_score}% — Weak financial behavior'

    weak_features = []
    if income_reg < 0.50:   weak_features.append('irregular income')
    if spending_con < 0.50: weak_features.append('erratic spending')
    if bill_ratio < 0.80:   weak_features.append('missed bills')
    if cash_ratio > 0.20:   weak_features.append('high cash usage')
    if emi_burden > 0.50:   weak_features.append('high EMI burden')
    if savings < 0.10:      weak_features.append('low savings')

    beh_detail = (
        f'{good_count}/8 behavior features in healthy range. '
        + (f'Concerns: {", ".join(weak_features)}.' if weak_features else 'No major concerns.')
    )

    # ── Point 2: Repayment Capacity ───────────────────────────────────────────
    # Based on savings + EMI burden (the two strongest predictors)
    if savings >= 0.15 and emi_burden <= 0.35:
        cap_status = 'good'
        cap_value  = 'Strong'
        cap_detail = (
            f'Savings rate {savings*100:.0f}% + EMI burden {emi_burden*100:.0f}% — '
            f'sufficient disposable income to service new loan.'
        )
    elif savings >= 0.05 and emi_burden <= 0.55:
        cap_status = 'warn'
        cap_value  = 'Moderate'
        cap_detail = (
            f'Savings rate {savings*100:.0f}% + EMI burden {emi_burden*100:.0f}% — '
            f'borderline repayment capacity. Monitor closely.'
        )
    else:
        cap_status = 'bad'
        cap_value  = 'Weak'
        cap_detail = (
            f'Savings rate {savings*100:.0f}% + EMI burden {emi_burden*100:.0f}% — '
            f'insufficient capacity. High default risk.'
        )

    # ── Point 3: Risk Flags ───────────────────────────────────────────────────
    flags = []
    if income_reg < 0.30:
        flags.append(f'Very irregular income (score: {income_reg:.2f})')
    if cash_ratio > 0.30:
        flags.append(f'Very high cash usage ({cash_ratio*100:.0f}% of spending)')
    if emi_burden > 0.60:
        flags.append(f'EMI burden critical ({emi_burden*100:.0f}% of income)')
    if savings < 0.0:
        flags.append(f'Negative savings rate ({savings*100:.0f}%)')
    if bill_ratio < 0.60:
        flags.append(f'Poor bill payment ({bill_ratio*100:.0f}% on time)')

    if not flags:
        flag_status = 'good'
        flag_value  = 'No Red Flags'
        flag_detail = 'No critical risk indicators detected in behavioral data.'
    elif len(flags) == 1:
        flag_status = 'warn'
        flag_value  = '1 Red Flag'
        flag_detail = flags[0]
    else:
        flag_status = 'bad'
        flag_value  = f'{len(flags)} Red Flags'
        flag_detail = ' | '.join(flags)

    # ── Point 4: Honest AI Recommendation ────────────────────────────────────
    # Use BOTH the model score AND the behavioral analysis
    model_says_approve = app.ai_decision == 'approved'
    behavior_ok        = behavior_score >= 50
    capacity_ok        = cap_status in ('good', 'warn')
    no_critical_flags  = flag_status != 'bad'

    # Consensus: model + behavior + capacity + no critical flags
    strong_approve = model_says_approve and behavior_ok and capacity_ok and no_critical_flags
    strong_reject  = (not model_says_approve) or (behavior_score < 40) or (cap_status == 'bad') or (flag_status == 'bad')

    if strong_approve:
        rec_status = 'good'
        rec_value  = 'APPROVE'
        rec_detail = (
            f'Model risk {risk_raw:.1f}/100 + {good_count}/8 behavior features healthy + '
            f'{cap_value.lower()} repayment capacity. Consistent approval signal.'
        )
    elif strong_reject:
        rec_status = 'bad'
        rec_value  = 'REJECT'
        # Explain why
        reasons = []
        if not model_says_approve:
            reasons.append(f'model risk {risk_raw:.1f}/100 above threshold')
        if behavior_score < 40:
            reasons.append(f'only {good_count}/8 behavior features healthy')
        if cap_status == 'bad':
            reasons.append('weak repayment capacity')
        if flag_status == 'bad':
            reasons.append(f'{len(flags)} critical risk flags')
        rec_detail = f'Rejection signals: {"; ".join(reasons)}.'
    else:
        rec_status = 'warn'
        rec_value  = 'REVIEW CAREFULLY'
        rec_detail = (
            f'Mixed signals — model says {"approve" if model_says_approve else "reject"} '
            f'(risk {risk_raw:.1f}/100) but {good_count}/8 behavior features healthy. '
            f'Banker judgment required.'
        )

    return [
        {
            'title':  'Behavioral Health Score',
            'value':  beh_label,
            'status': beh_status,
            'detail': beh_detail,
            'icon':   '📈',
        },
        {
            'title':  'Repayment Capacity',
            'value':  cap_value,
            'status': cap_status,
            'detail': cap_detail,
            'icon':   '💰',
        },
        {
            'title':  'Risk Flags',
            'value':  flag_value,
            'status': flag_status,
            'detail': flag_detail,
            'icon':   '🚩',
        },
        {
            'title':  'AI Recommendation',
            'value':  rec_value,
            'status': rec_status,
            'detail': rec_detail,
            'icon':   '🤖',
        },
    ]


def _trace_transactions(app):
    """Fetch top transactions for the top 3 SHAP features (banker view)."""
    CATEGORY_MAP = {
        'emi_burden_ratio':        ['emi'],
        'savings_rate':            ['salary', 'upi'],
        'bill_payment_ratio':      ['utility'],
        'cash_withdrawal_ratio':   ['cash'],
        'income_regularity':       ['salary'],
        'spending_consistency':    ['shopping', 'food', 'other'],
        'upi_activity_volume':     ['upi'],
        'rent_payment_regularity': ['rent'],
    }
    try:
        profile = app.user.profile
        csv_uid = profile.csv_user_id
    except Exception:
        return []

    if not csv_uid:
        return []

    factors = app.shap_factors[:3]
    result  = []
    for f in factors:
        feat_key = f.get('feature_key', '')
        cats     = CATEGORY_MAP.get(feat_key, [])
        txns     = CsvTransaction.objects.filter(
            csv_user_id=csv_uid,
            category__in=cats,
        ).order_by('-amount')[:3] if cats else []

        result.append({
            'feature':    f.get('feature', feat_key),
            'shap_value': f.get('impact', 0),
            'impact':     f.get('impact_direction', 'hurts'),
            'transactions': [
                {
                    'id':       t.transaction_id,
                    'amount':   t.amount,
                    'category': t.category,
                    'type':     t.txn_type,
                    'date':     t.timestamp[:10],
                    'mode':     t.payment_mode,
                }
                for t in txns
            ],
        })
    return result


# ─── Regulator Views ──────────────────────────────────────────────────────────

@login_required
def audit_chain(request):
    """Tamper-evident audit chain page — visible to regulator and banker."""
    role = get_role(request.user)
    if role not in ('regulator', 'banker'):
        return redirect('home')

    import hashlib

    apps = LoanApplication.objects.order_by('created_at')
    chain_records = []
    expected_prev = '0' * 64
    chain_valid   = True
    first_broken  = None

    for app in apps:
        # Recompute hash to verify
        recomputed = app.compute_hash(app.previous_hash)
        hash_ok    = (recomputed == app.current_hash) and (app.previous_hash == expected_prev)

        if not hash_ok and chain_valid:
            chain_valid  = False
            first_broken = app.id

        chain_records.append({
            'id':             app.id,
            'user':           app.user.username,
            'loan_type':      app.loan_type,
            'amount':         app.loan_amount_rupees,
            'ai_decision':    app.ai_decision,
            'risk_score':     app.risk_score,
            'final_status':   app.final_status,
            'created_at':     app.created_at,
            'previous_hash':  app.previous_hash[:16] + '...' if app.previous_hash else '—',
            'current_hash':   app.current_hash[:16] + '...' if app.current_hash else '—',
            'full_hash':      app.current_hash,
            'hash_ok':        hash_ok,
        })
        expected_prev = app.current_hash or expected_prev

    return render(request, 'loans/audit_chain.html', {
        'chain_records': chain_records,
        'chain_valid':   chain_valid,
        'first_broken':  first_broken,
        'total':         len(chain_records),
    })


@login_required
def regulator_dashboard(request):
    if get_role(request.user) != 'regulator':
        return redirect('home')

    # ── Real application data ─────────────────────────────────────────────────
    all_apps  = LoanApplication.objects.all()
    total     = all_apps.count()
    approved  = all_apps.filter(final_status='approved').count()
    rejected  = all_apps.filter(final_status='rejected').count()
    pending   = all_apps.filter(final_status='pending').count()
    ai_approved = all_apps.filter(ai_decision='approved').count()
    ai_rejected = all_apps.filter(ai_decision='rejected').count()
    avg_risk  = round(all_apps.aggregate(avg=Avg('risk_score'))['avg'] or 0, 1)
    approval_rate = round((approved / total * 100), 1) if total > 0 else 0
    ai_approval_rate = round((ai_approved / total * 100), 1) if total > 0 else 0

    # Banker override stats
    banker_overrode_ai = 0
    for app in all_apps:
        if app.banker_decision and app.banker_decision != 'pending':
            if app.banker_decision != app.ai_decision:
                banker_overrode_ai += 1

    # ── Feature health from all 2000 dataset users ────────────────────────────
    from loans.models import CsvUserFeature
    features = CsvUserFeature.objects.all()
    feat_count = features.count()

    def feat_stats(field):
        vals = list(features.values_list(field, flat=True))
        if not vals:
            return {'avg': 0, 'good': 0, 'bad': 0}
        avg = sum(vals) / len(vals)
        return {'avg': round(avg, 3), 'vals': vals}

    # Feature averages for radar/bar chart
    feature_avgs = {
        'Income Regularity':       round(sum(features.values_list('income_regularity', flat=True)) / feat_count, 3),
        'Spending Consistency':    round(sum(features.values_list('spending_consistency', flat=True)) / feat_count, 3),
        'Bill Payment Ratio':      round(sum(features.values_list('bill_payment_ratio', flat=True)) / feat_count, 3),
        'Savings Rate':            round(sum(features.values_list('savings_rate', flat=True)) / feat_count, 3),
        'EMI Burden Ratio':        round(sum(features.values_list('emi_burden_ratio', flat=True)) / feat_count, 3),
        'UPI Activity':            round(sum(features.values_list('upi_activity_volume', flat=True)) / feat_count, 3),
        'Rent Regularity':         round(sum(features.values_list('rent_payment_regularity', flat=True)) / feat_count, 3),
        'Cash Withdrawal':         round(sum(features.values_list('cash_withdrawal_ratio', flat=True)) / feat_count, 3),
    }

    # Benchmark thresholds (good = green, bad = red)
    feature_benchmarks = {
        'Income Regularity':    {'threshold': 0.5,  'direction': 'above'},
        'Spending Consistency': {'threshold': 0.5,  'direction': 'above'},
        'Bill Payment Ratio':   {'threshold': 0.8,  'direction': 'above'},
        'Savings Rate':         {'threshold': 0.1,  'direction': 'above'},
        'EMI Burden Ratio':     {'threshold': 0.5,  'direction': 'below'},
        'UPI Activity':         {'threshold': 5.0,  'direction': 'above'},
        'Rent Regularity':      {'threshold': 0.6,  'direction': 'above'},
        'Cash Withdrawal':      {'threshold': 0.2,  'direction': 'below'},
    }

    feature_health = []
    for name, avg in feature_avgs.items():
        bench = feature_benchmarks[name]
        if bench['direction'] == 'above':
            status = 'good' if avg >= bench['threshold'] else 'bad'
            pct = round(avg * 100, 1) if name not in ('UPI Activity',) else round(avg, 1)
        else:
            status = 'good' if avg <= bench['threshold'] else 'bad'
            pct = round(avg * 100, 1)
        feature_health.append({
            'name':      name,
            'avg':       avg,
            'threshold': bench['threshold'],
            'direction': bench['direction'],
            'status':    status,
            'display':   f"{avg:.1f}" if name == 'UPI Activity' else f"{avg*100:.0f}%",
        })

    # ── Risk distribution ─────────────────────────────────────────────────────
    # Use all 2000 feature records for population-level view
    low_risk = medium_risk = high_risk = 0
    default_count = features.filter(target=1).count()
    no_default_count = features.filter(target=0).count()

    # Approximate from feature data (savings_rate is strongest predictor)
    for f in features:
        sr = f.savings_rate
        emi = f.emi_burden_ratio
        # Simple heuristic matching model behavior
        if sr >= 0.1 and emi <= 0.4:
            low_risk += 1
        elif sr >= -0.3 and emi <= 0.6:
            medium_risk += 1
        else:
            high_risk += 1

    risk_dist = [
        {'label': 'Low Risk',    'count': low_risk,    'color': '#059669', 'pct': round(low_risk/feat_count*100, 1)},
        {'label': 'Medium Risk', 'count': medium_risk, 'color': '#f59e0b', 'pct': round(medium_risk/feat_count*100, 1)},
        {'label': 'High Risk',   'count': high_risk,   'color': '#e11d48', 'pct': round(high_risk/feat_count*100, 1)},
    ]

    # ── Employment breakdown ──────────────────────────────────────────────────
    from loans.models import UserProfile
    profiles = UserProfile.objects.filter(csv_user_id__isnull=False)
    emp_raw = {}
    for p in profiles:
        emp_raw[p.employment_type] = emp_raw.get(p.employment_type, 0) + 1
    emp_total = sum(emp_raw.values()) or 1
    employment_data = [
        {'label': k.replace('_', ' ').title(), 'count': v, 'pct': round(v/emp_total*100, 1)}
        for k, v in sorted(emp_raw.items(), key=lambda x: -x[1])
    ]

    # ── CIBIL availability ────────────────────────────────────────────────────
    has_cibil = profiles.filter(cibil_score__isnull=False).count()
    no_cibil  = profiles.filter(cibil_score__isnull=True).count()
    cibil_total = has_cibil + no_cibil or 1

    # ── Fairness / bias metrics ───────────────────────────────────────────────
    # Approval rates by employment type (from actual applications)
    emp_approval = []
    for emp_type in ['salaried', 'self_employed', 'gig_worker', 'student', 'retired', 'farmer']:
        emp_apps = all_apps.filter(employment_type=emp_type)
        emp_total_apps = emp_apps.count()
        emp_approved = emp_apps.filter(final_status='approved').count()
        if emp_total_apps > 0:
            emp_approval.append({
                'group': emp_type.replace('_', ' ').title(),
                'total': emp_total_apps,
                'approved': emp_approved,
                'rate': round(emp_approved / emp_total_apps * 100, 1),
            })

    # Bias score — max disparity in approval rates
    if len(emp_approval) >= 2:
        rates = [e['rate'] for e in emp_approval]
        disparity = max(rates) - min(rates)
        bias_score = round(min(1.0, disparity / 100 * 2), 3)
    else:
        bias_score = 0.0

    bias_flags = []
    if bias_score > 0.15:
        bias_flags.append('Approval rate disparity detected across employment groups')
    if no_cibil / cibil_total > 0.4:
        bias_flags.append(f'{round(no_cibil/cibil_total*100)}% of applicants have no CIBIL — behavior-based scoring active')

    # ── Model drift (simulated monthly trend) ────────────────────────────────
    drift_months = ['Nov 2024', 'Dec 2024', 'Jan 2025', 'Feb 2025', 'Mar 2025', 'Apr 2025']
    drift_points = [
        {'month': m, 'approval_rate': round(0.55 + math.sin(i * 0.8) * 0.12, 2),
         'avg_risk': round(48 + math.cos(i * 0.5) * 8, 1), 'shift': round(0.02 + i * 0.005, 3)}
        for i, m in enumerate(drift_months)
    ]
    latest_shift   = drift_points[-1]['shift']
    drift_detected = latest_shift > 0.03
    drift_severity = 'high' if latest_shift > 0.05 else ('medium' if latest_shift > 0.03 else 'low')

    return render(request, 'loans/regulator_dashboard.html', {
        # Application stats
        'total':            total,
        'approved':         approved,
        'rejected':         rejected,
        'pending':          pending,
        'ai_approved':      ai_approved,
        'ai_rejected':      ai_rejected,
        'avg_risk':         avg_risk,
        'approval_rate':    approval_rate,
        'ai_approval_rate': ai_approval_rate,
        'banker_overrode':  banker_overrode_ai,
        # Feature health
        'feature_health':   feature_health,
        'feature_avgs_json': json.dumps({f['name']: f['avg'] for f in feature_health}),
        'feat_count':       feat_count,
        # Risk distribution
        'risk_dist':        risk_dist,
        'risk_dist_json':   json.dumps(risk_dist),
        'default_count':    default_count,
        'no_default_count': no_default_count,
        # Employment
        'employment_data':  employment_data,
        'employment_json':  json.dumps(employment_data),
        # CIBIL
        'has_cibil':        has_cibil,
        'no_cibil':         no_cibil,
        'cibil_pct':        round(no_cibil / cibil_total * 100, 1),
        # Fairness
        'bias_score':       bias_score,
        'bias_flags':       bias_flags,
        'emp_approval':     emp_approval,
        'emp_approval_json': json.dumps(emp_approval),
        # Drift
        'drift_points':     json.dumps(drift_points),
        'drift_detected':   drift_detected,
        'drift_severity':   drift_severity,
        'last_retrained':   '15 Oct 2024',
        # Report
        'report_id':        f'SML-RPT-2026-{total:04d}',
        'standards': [
            'RBI Master Circular on Fair Practices Code',
            'GDPR Article 22 — Automated Decision-Making',
            'CFPB Equal Credit Opportunity Act',
        ],
    })
