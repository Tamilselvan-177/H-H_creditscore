from django.contrib import admin
from .models import LoanApplication, UserProfile

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'role']

@admin.register(LoanApplication)
class LoanApplicationAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'loan_type', 'loan_amount_rupees', 'ai_decision', 'final_status', 'created_at']
    list_filter = ['loan_type', 'ai_decision', 'final_status', 'employment_type']
    search_fields = ['user__username', 'user__email']
