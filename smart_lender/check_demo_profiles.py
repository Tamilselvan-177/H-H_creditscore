import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smart_lender.settings')
import django
django.setup()
from loans.models import DemoProfile
qs = DemoProfile.objects.filter(csv_user_id__regex=r'^USR_\d{4}$')
print('Total DemoProfiles (USR_####):', qs.count())
print('First 5 ids:', list(qs.values_list('csv_user_id', flat=True)[:5]))
