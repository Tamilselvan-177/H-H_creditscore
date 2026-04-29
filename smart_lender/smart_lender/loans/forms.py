from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile


class RegisterForm(UserCreationForm):
    ROLE_CHOICES = [
        ('user',      'Applicant (User)'),
        ('banker',    'Banker'),
        ('regulator', 'Regulator'),
    ]
    email    = forms.EmailField(required=True)
    role     = forms.ChoiceField(choices=ROLE_CHOICES, initial='user')
    # For demo: user picks which dataset user they are
    csv_user_id = forms.CharField(
        required=False,
        label='Dataset User ID (optional)',
        widget=forms.TextInput(attrs={'placeholder': 'e.g. USR_0001'}),
        help_text='Link to a pre-loaded dataset user for demo purposes.',
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2', 'role', 'csv_user_id']


class LoanApplicationForm(forms.Form):
    """
    Simple form — user just picks loan type, amount, tenure.
    Features are fetched automatically from the pre-loaded dataset.
    """
    LOAN_TYPE_CHOICES = [
        ('personal', 'Personal Loan'),
        ('vehicle',  'Vehicle Loan'),
    ]

    loan_type = forms.ChoiceField(
        choices=LOAN_TYPE_CHOICES,
        label='Loan Type',
    )
    loan_amount_rupees = forms.FloatField(
        min_value=10000, max_value=10000000,
        label='Loan Amount (₹)',
        widget=forms.NumberInput(attrs={'placeholder': '100000'}),
    )
    tenure_months = forms.IntegerField(
        min_value=3, max_value=360,
        label='Tenure (Months)',
        widget=forms.NumberInput(attrs={'placeholder': '12'}),
    )
    vehicle_purpose = forms.CharField(
        required=False,
        label='Vehicle Purpose',
        widget=forms.TextInput(attrs={'placeholder': 'e.g. Daily commute, Business use'}),
    )

    def clean(self):
        cleaned = super().clean()
        if cleaned.get('loan_type') == 'vehicle' and not cleaned.get('vehicle_purpose'):
            self.add_error('vehicle_purpose', 'Vehicle purpose is required for vehicle loans.')
        return cleaned


class BankerReviewForm(forms.Form):
    DECISION_CHOICES = [
        ('approved', 'Approve'),
        ('rejected', 'Reject'),
    ]
    decision = forms.ChoiceField(choices=DECISION_CHOICES, widget=forms.RadioSelect)
    note = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'rows': 3,
            'placeholder': 'Add a note explaining your decision (optional)...',
        }),
        label='Banker Note',
    )
