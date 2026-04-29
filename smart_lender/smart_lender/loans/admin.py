from django.contrib import admin
from django.utils.html import format_html
from .models import LoanApplication, UserProfile, CsvUserFeature, CsvTransaction


# ─── User Profile ─────────────────────────────────────────────────────────────

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display  = ['user', 'role', 'csv_user_id', 'employment_type',
                     'monthly_income_display', 'cibil_score', 'state']
    list_filter   = ['role', 'employment_type', 'state']
    search_fields = ['user__username', 'user__email', 'csv_user_id']
    ordering      = ['user__username']

    def monthly_income_display(self, obj):
        return f'₹{obj.monthly_income:,.0f}' if obj.monthly_income else '—'
    monthly_income_display.short_description = 'Monthly Income'
# ─── Loan Application ─────────────────────────────────────────────────────────

@admin.register(LoanApplication)
class LoanApplicationAdmin(admin.ModelAdmin):
    list_display  = [
        'id', 'user_link', 'loan_type', 'amount_display',
        'risk_score_badge', 'ai_decision_badge',
        'banker_decision_badge', 'final_status_badge', 'created_at',
    ]
    list_filter   = ['loan_type', 'ai_decision', 'banker_decision',
                     'final_status', 'fairness_check_passed']
    search_fields = ['user__username', 'user__email']
    ordering      = ['-created_at']
    readonly_fields = [
        'ai_decision', 'risk_score', 'user_explanation',
        'shap_factors_json', 'improvement_suggestions_json',
        'fairness_check_passed', 'fairness_note', 'feature_values_json',
        'created_at', 'updated_at',
    ]
    fieldsets = (
        ('Applicant', {
            'fields': ('user', 'loan_type', 'loan_amount_rupees',
                       'tenure_months', 'vehicle_purpose')
        }),
        ('Demographics', {
            'fields': ('annual_income_rupees', 'monthly_expenses_rupees',
                       'employment_type', 'existing_loans_count',
                       'existing_emi_rupees', 'credit_score'),
            'classes': ('collapse',),
        }),
        ('AI Decision (read-only)', {
            'fields': ('ai_decision', 'risk_score', 'user_explanation',
                       'fairness_check_passed', 'fairness_note',
                       'feature_values_json'),
        }),
        ('Banker Review', {
            'fields': ('banker_decision', 'banker_note',
                       'banker_reviewed_at', 'banker', 'final_status'),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )

    def user_link(self, obj):
        return format_html(
            '<a href="/admin/auth/user/{}/change/">{}</a>',
            obj.user.pk, obj.user.username
        )
    user_link.short_description = 'Applicant'

    def amount_display(self, obj):
        return f'₹{obj.loan_amount_rupees:,.0f}'
    amount_display.short_description = 'Amount'

    def risk_score_badge(self, obj):
        score = obj.risk_score
        if score < 30:
            color = '#059669'; bg = '#f0fdf4'
        elif score < 60:
            color = '#d97706'; bg = '#fffbeb'
        else:
            color = '#e11d48'; bg = '#fff1f2'
        return format_html(
            '<span style="padding:2px 8px;border-radius:9999px;font-size:11px;'
            'font-weight:600;background:{};color:{};">{}</span>',
            bg, color, f'{score:.0f}'
        )
    risk_score_badge.short_description = 'Risk'

    def ai_decision_badge(self, obj):
        color = '#059669' if obj.ai_decision == 'approved' else '#e11d48'
        bg    = '#f0fdf4' if obj.ai_decision == 'approved' else '#fff1f2'
        return format_html(
            '<span style="padding:2px 8px;border-radius:9999px;font-size:11px;'
            'font-weight:600;background:{};color:{};">{}</span>',
            bg, color, obj.ai_decision.upper()
        )
    ai_decision_badge.short_description = 'AI'

    def banker_decision_badge(self, obj):
        if not obj.banker_decision or obj.banker_decision == 'pending':
            return format_html(
                '<span style="padding:2px 8px;border-radius:9999px;font-size:11px;'
                'font-weight:600;background:#fffbeb;color:#d97706;">PENDING</span>'
            )
        color = '#059669' if obj.banker_decision == 'approved' else '#e11d48'
        bg    = '#f0fdf4' if obj.banker_decision == 'approved' else '#fff1f2'
        return format_html(
            '<span style="padding:2px 8px;border-radius:9999px;font-size:11px;'
            'font-weight:600;background:{};color:{};">{}</span>',
            bg, color, obj.banker_decision.upper()
        )
    banker_decision_badge.short_description = 'Banker'

    def final_status_badge(self, obj):
        colors = {
            'approved': ('#059669', '#f0fdf4'),
            'rejected': ('#e11d48', '#fff1f2'),
            'pending':  ('#d97706', '#fffbeb'),
        }
        color, bg = colors.get(obj.final_status, ('#64748b', '#f8fafc'))
        return format_html(
            '<span style="padding:2px 8px;border-radius:9999px;font-size:11px;'
            'font-weight:600;background:{};color:{};">{}</span>',
            bg, color, obj.final_status.upper()
        )
    final_status_badge.short_description = 'Status'


# ─── CSV User Features ────────────────────────────────────────────────────────

@admin.register(CsvUserFeature)
class CsvUserFeatureAdmin(admin.ModelAdmin):
    list_display  = [
        'csv_user_id', 'income_regularity', 'spending_consistency',
        'bill_payment_ratio', 'savings_rate_display', 'emi_burden_ratio',
        'upi_activity_volume', 'cash_withdrawal_ratio', 'target_badge',
    ]
    list_filter   = ['target']
    search_fields = ['csv_user_id']
    ordering      = ['csv_user_id']

    def savings_rate_display(self, obj):
        color = '#059669' if obj.savings_rate >= 0.1 else '#e11d48'
        val = f'{obj.savings_rate:.3f}'
        return format_html(
            '<span style="color:{};">{}</span>', color, val
        )
    savings_rate_display.short_description = 'Savings Rate'

    def target_badge(self, obj):
        if obj.target == 1:
            return format_html(
                '<span style="padding:2px 8px;border-radius:9999px;font-size:11px;'
                'font-weight:600;background:#fff1f2;color:#e11d48;">DEFAULT</span>'
            )
        return format_html(
            '<span style="padding:2px 8px;border-radius:9999px;font-size:11px;'
            'font-weight:600;background:#f0fdf4;color:#059669;">NO DEFAULT</span>'
        )
    target_badge.short_description = 'Target'


# ─── CSV Transactions ─────────────────────────────────────────────────────────

@admin.register(CsvTransaction)
class CsvTransactionAdmin(admin.ModelAdmin):
    list_display  = [
        'transaction_id', 'csv_user_id', 'amount_display',
        'txn_type_badge', 'category', 'payment_mode', 'timestamp',
    ]
    list_filter   = ['txn_type', 'category', 'payment_mode']
    search_fields = ['transaction_id', 'csv_user_id', 'category']
    ordering      = ['-timestamp']

    def amount_display(self, obj):
        color = '#059669' if obj.txn_type == 'credit' else '#e11d48'
        amount_str = f'₹{obj.amount:,.0f}'
        return format_html(
            '<span style="color:{};font-weight:600;">{}</span>',
            color, amount_str
        )
    amount_display.short_description = 'Amount'

    def txn_type_badge(self, obj):
        if obj.txn_type == 'credit':
            return format_html(
                '<span style="padding:2px 8px;border-radius:9999px;font-size:11px;'
                'font-weight:600;background:#f0fdf4;color:#059669;">CREDIT</span>'
            )
        return format_html(
            '<span style="padding:2px 8px;border-radius:9999px;font-size:11px;'
            'font-weight:600;background:#fff1f2;color:#e11d48;">DEBIT</span>'
        )
    txn_type_badge.short_description = 'Type'
