from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('register/', views.register_view, name='register'),

    # User
    path('dashboard/', views.user_dashboard, name='user_dashboard'),
    path('apply/', views.apply_loan, name='apply_loan'),
    path('loans/<int:pk>/', views.loan_result, name='loan_result'),

    # Banker
    path('banker/', views.banker_dashboard, name='banker_dashboard'),
    path('banker/loans/<int:pk>/', views.banker_loan_detail, name='banker_loan_detail'),

    # Regulator
    path('regulator/', views.regulator_dashboard, name='regulator_dashboard'),
]
