from django.db import models
from django.contrib.auth.models import User
import json


# ─── User Profile ─────────────────────────────────────────────────────────────

class UserProfile(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('banker', 'Banker'),
        ('regulator', 'Regulator'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')

    # Link to dataset user (e.g. USR_0001)
    csv_user_id = models.CharField(max_length=20, blank=True, null=True, db_index=True)

    # Demographics from users.csv
    age = models.IntegerField(default=30)
    employment_type = models.CharField(max_length=30, default='salaried')
    monthly_income = models.FloatField(default=0)
    cibil_score = models.FloatField(null=True, blank=True)
    existing_emi = models.FloatField(default=0)
    account_age_months = models.IntegerField(default=0)
    state = models.CharField(max_length=10, blank=True)
    loan_type_preference = models.CharField(max_length=20, default='personal')

    def __str__(self):
        return f"{self.user.username} ({self.role}) [{self.csv_user_id or 'no-csv'}]"


# ─── Demo Profile (extra fields for hackathon demo) ──────────────────────────

class DemoProfile(models.Model):
    """
    Extended applicant details for demo purposes.
    One record per csv_user_id — pre-seeded with realistic fake data.
    Auto-fills the apply form during demonstration.
    """
    csv_user_id = models.CharField(max_length=20, unique=True, db_index=True)

    # Personal
    full_name       = models.CharField(max_length=100, default='')
    dob             = models.CharField(max_length=12, default='')   # YYYY-MM-DD string
    gender          = models.CharField(max_length=10, default='male')
    mobile          = models.CharField(max_length=15, default='')
    pan             = models.CharField(max_length=12, default='')
    aadhaar         = models.CharField(max_length=16, default='')
    marital_status  = models.CharField(max_length=15, default='single')

    # Address
    address         = models.CharField(max_length=200, default='')
    city            = models.CharField(max_length=50, default='')
    pincode         = models.CharField(max_length=8, default='')
    residence_type  = models.CharField(max_length=20, default='rented')
    years_at_address = models.IntegerField(default=2)

    # Employment
    company_name    = models.CharField(max_length=100, default='')
    job_role        = models.CharField(max_length=80, default='')
    work_experience = models.IntegerField(default=2)
    salary_credit   = models.CharField(max_length=20, default='bank_transfer')

    # Banking
    bank_name       = models.CharField(max_length=50, default='SBI')
    account_number  = models.CharField(max_length=20, default='')
    ifsc            = models.CharField(max_length=12, default='')
    account_type    = models.CharField(max_length=15, default='savings')

    def __str__(self):
        return f"DemoProfile({self.csv_user_id}) — {self.full_name}"


# ─── CSV Dataset Tables ───────────────────────────────────────────────────────

class CsvUserFeature(models.Model):
    """Pre-computed 8 behavior features from features.csv"""
    csv_user_id = models.CharField(max_length=20, unique=True, db_index=True)

    income_regularity       = models.FloatField(default=0)
    spending_consistency    = models.FloatField(default=0)
    bill_payment_ratio      = models.FloatField(default=0)
    savings_rate            = models.FloatField(default=0)
    emi_burden_ratio        = models.FloatField(default=0)
    upi_activity_volume     = models.FloatField(default=0)
    rent_payment_regularity = models.FloatField(default=0)
    cash_withdrawal_ratio   = models.FloatField(default=0)
    target                  = models.IntegerField(default=0)  # 0=no default, 1=default

    def as_dict(self):
        return {
            'income_regularity':       self.income_regularity,
            'spending_consistency':    self.spending_consistency,
            'bill_payment_ratio':      self.bill_payment_ratio,
            'savings_rate':            self.savings_rate,
            'emi_burden_ratio':        self.emi_burden_ratio,
            'upi_activity_volume':     self.upi_activity_volume,
            'rent_payment_regularity': self.rent_payment_regularity,
            'cash_withdrawal_ratio':   self.cash_withdrawal_ratio,
        }

    def __str__(self):
        return f"Features({self.csv_user_id})"


class CsvTransaction(models.Model):
    """Raw transactions from transactions.csv"""
    transaction_id = models.CharField(max_length=50, unique=True, db_index=True)
    csv_user_id    = models.CharField(max_length=20, db_index=True)
    amount         = models.FloatField()
    txn_type       = models.CharField(max_length=10)   # credit / debit
    category       = models.CharField(max_length=30)
    timestamp      = models.CharField(max_length=30)   # stored as string for simplicity
    payment_mode   = models.CharField(max_length=20)
    merchant       = models.CharField(max_length=30, blank=True)

    def __str__(self):
        return f"{self.transaction_id} ({self.csv_user_id})"


# ─── Loan Application ─────────────────────────────────────────────────────────

class LoanApplication(models.Model):
    LOAN_TYPE_CHOICES = [
        ('personal', 'Personal Loan'),
        ('vehicle',  'Vehicle Loan'),
    ]
    DECISION_CHOICES = [
        ('pending',  'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]

    # Applicant
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='applications')

    # Loan details (user fills these in)
    loan_type          = models.CharField(max_length=20, choices=LOAN_TYPE_CHOICES)
    loan_amount_rupees = models.FloatField()
    tenure_months      = models.IntegerField()
    vehicle_purpose    = models.CharField(max_length=255, blank=True, null=True)

    # Demographics (copied from UserProfile at time of application)
    annual_income_rupees    = models.FloatField(default=0)
    monthly_expenses_rupees = models.FloatField(default=0)
    employment_type         = models.CharField(max_length=30, default='salaried')
    existing_loans_count    = models.IntegerField(default=0)
    existing_emi_rupees     = models.FloatField(default=0)
    credit_score            = models.IntegerField(default=0)

    # AI decision (computed immediately on submit)
    ai_decision                  = models.CharField(max_length=20, choices=DECISION_CHOICES, default='pending')
    risk_score                   = models.FloatField(default=50)
    user_explanation             = models.TextField(default='')
    shap_factors_json            = models.TextField(default='[]')
    improvement_suggestions_json = models.TextField(default='[]')
    fairness_check_passed        = models.BooleanField(default=True)
    fairness_note                = models.TextField(blank=True, null=True)
    feature_values_json          = models.TextField(default='{}', blank=True)
    
    # Uploaded documents
    kyc_document   = models.FileField(upload_to='loan_docs/%Y/%m/%d', null=True, blank=True)
    address_proof  = models.FileField(upload_to='loan_docs/%Y/%m/%d', null=True, blank=True)
    income_proof   = models.FileField(upload_to='loan_docs/%Y/%m/%d', null=True, blank=True)

    # Banker review (happens after submission)
    banker_decision    = models.CharField(max_length=20, choices=DECISION_CHOICES, blank=True, null=True)
    banker_note        = models.TextField(blank=True, null=True)
    banker_reviewed_at = models.DateTimeField(blank=True, null=True)
    banker             = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='reviewed_applications'
    )

    # Final status — PENDING until banker acts
    final_status = models.CharField(max_length=20, choices=DECISION_CHOICES, default='pending')

    # SHA-256 hash chain fields
    previous_hash = models.CharField(max_length=64, default='0' * 64)
    current_hash  = models.CharField(max_length=64, default='')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"#{self.id} {self.user.username} {self.loan_type} {self.final_status}"

    @property
    def shap_factors(self):
        try:
            return json.loads(self.shap_factors_json)
        except Exception:
            return []

    @property
    def improvement_suggestions(self):
        try:
            return json.loads(self.improvement_suggestions_json)
        except Exception:
            return []

    @property
    def feature_values(self):
        try:
            return json.loads(self.feature_values_json)
        except Exception:
            return {}

    def compute_hash(self, previous_hash: str) -> str:
        """Compute SHA-256 hash for this record."""
        import hashlib
        payload = (
            f"{previous_hash}"
            f"{self.id}"
            f"{self.user_id}"
            f"{self.loan_type}"
            f"{self.loan_amount_rupees}"
            f"{self.ai_decision}"
            f"{self.risk_score}"
            f"{self.shap_factors_json[:100]}"   # first 100 chars of SHAP
        )
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    @property
    def risk_score_display(self):
        try:
            risk = float(self.risk_score)
        except Exception:
            return 50

        if risk <= 1:
            return risk * 100
        return risk
