"""
Management command: seed_demo_profiles
========================================
Creates DemoProfile records for all imported dataset users (USR_0001–USR_0050+).
Generates realistic fake Indian personal/banking data for each user.
Used purely for hackathon demo — auto-fills the apply form.

Usage:
    python manage.py seed_demo_profiles
    python manage.py seed_demo_profiles --clear
"""

import random
from django.core.management.base import BaseCommand
from loans.models import DemoProfile, UserProfile

# ─── Fake data pools ──────────────────────────────────────────────────────────

FIRST_NAMES_M = ['Arjun', 'Rahul', 'Vikram', 'Suresh', 'Karthik', 'Arun', 'Deepak',
                 'Rajesh', 'Manoj', 'Sanjay', 'Pradeep', 'Anand', 'Ravi', 'Mohan',
                 'Ganesh', 'Venkat', 'Harish', 'Naveen', 'Dinesh', 'Ramesh']
FIRST_NAMES_F = ['Priya', 'Divya', 'Kavya', 'Ananya', 'Sneha', 'Pooja', 'Meena',
                 'Lakshmi', 'Sunita', 'Rekha', 'Nisha', 'Asha', 'Geeta', 'Sita',
                 'Radha', 'Usha', 'Mala', 'Saranya', 'Deepa', 'Revathi']
LAST_NAMES    = ['Kumar', 'Sharma', 'Patel', 'Singh', 'Reddy', 'Nair', 'Iyer',
                 'Gupta', 'Verma', 'Joshi', 'Pillai', 'Rao', 'Mishra', 'Tiwari',
                 'Pandey', 'Shah', 'Mehta', 'Desai', 'Bose', 'Das']

COMPANIES = {
    'salaried':      ['Infosys Ltd', 'TCS', 'Wipro Technologies', 'HCL Technologies',
                      'Tech Mahindra', 'Cognizant', 'Accenture India', 'IBM India',
                      'Reliance Industries', 'HDFC Bank', 'ICICI Bank', 'Axis Bank',
                      'Bajaj Auto', 'Maruti Suzuki', 'L&T Construction'],
    'self_employed': ['Self-Employed', 'Freelance Consultant', 'Independent Contractor'],
    'gig_worker':    ['Swiggy', 'Zomato', 'Ola', 'Uber', 'Urban Company', 'Dunzo'],
    'student':       ['N/A (Student)', 'Part-time work'],
    'retired':       ['Retired (Government)', 'Retired (Private Sector)'],
    'farmer':        ['Agriculture (Self)', 'Farming'],
}

JOB_ROLES = {
    'salaried':      ['Software Engineer', 'Senior Analyst', 'Project Manager',
                      'Business Analyst', 'HR Executive', 'Finance Manager',
                      'Operations Lead', 'Sales Executive', 'Team Lead'],
    'self_employed': ['Business Owner', 'Consultant', 'Freelancer'],
    'gig_worker':    ['Delivery Partner', 'Driver Partner', 'Service Professional'],
    'student':       ['Student'],
    'retired':       ['Retired'],
    'farmer':        ['Farmer'],
}

CITIES_BY_STATE = {
    'TN': ('Chennai', '600001'),
    'MH': ('Mumbai', '400001'),
    'KA': ('Bengaluru', '560001'),
    'UP': ('Lucknow', '226001'),
    'DL': ('New Delhi', '110001'),
    'WB': ('Kolkata', '700001'),
    'GJ': ('Ahmedabad', '380001'),
    'RJ': ('Jaipur', '302001'),
    'AP': ('Hyderabad', '500001'),
    'KL': ('Kochi', '682001'),
}

BANKS = ['State Bank of India', 'HDFC Bank', 'ICICI Bank', 'Axis Bank',
         'Punjab National Bank', 'Bank of Baroda', 'Canara Bank', 'Union Bank']

IFSC_PREFIXES = {
    'State Bank of India': 'SBIN',
    'HDFC Bank':           'HDFC',
    'ICICI Bank':          'ICIC',
    'Axis Bank':           'UTIB',
    'Punjab National Bank':'PUNB',
    'Bank of Baroda':      'BARB',
    'Canara Bank':         'CNRB',
    'Union Bank':          'UBIN',
}

STREETS = ['MG Road', 'Anna Salai', 'Brigade Road', 'Linking Road', 'Park Street',
           'Connaught Place', 'Banjara Hills', 'Koramangala', 'Andheri West',
           'Sector 18', 'Civil Lines', 'Jubilee Hills', 'Indiranagar', 'Salt Lake']


def fake_pan():
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return (random.choice(letters) + random.choice(letters) + random.choice(letters) +
            random.choice(letters) + random.choice(letters) +
            str(random.randint(1000, 9999)) + random.choice(letters))


def fake_aadhaar():
    return f"{random.randint(2000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"


def fake_account():
    return str(random.randint(10000000000, 99999999999999))


def fake_mobile():
    return f"9{random.randint(100000000, 999999999)}"


def fake_dob(age):
    year  = 2024 - age
    month = random.randint(1, 12)
    day   = random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


class Command(BaseCommand):
    help = 'Seed DemoProfile records for all imported dataset users'

    def add_arguments(self, parser):
        parser.add_argument('--clear', action='store_true',
                            help='Clear existing demo profiles before seeding')

    def handle(self, *args, **options):
        if options['clear']:
            DemoProfile.objects.all().delete()
            self.stdout.write('  Cleared existing demo profiles.')

        profiles = UserProfile.objects.filter(
            csv_user_id__isnull=False, role='user'
        ).select_related('user')

        created = skipped = 0

        for profile in profiles:
            uid = profile.csv_user_id
            if DemoProfile.objects.filter(csv_user_id=uid).exists():
                skipped += 1
                continue

            random.seed(hash(uid))   # deterministic per user

            # Gender
            gender = random.choice(['male', 'female'])
            first  = random.choice(FIRST_NAMES_M if gender == 'male' else FIRST_NAMES_F)
            last   = random.choice(LAST_NAMES)
            name   = f"{first} {last}"

            # Employment
            emp = profile.employment_type or 'salaried'
            company  = random.choice(COMPANIES.get(emp, COMPANIES['salaried']))
            job_role = random.choice(JOB_ROLES.get(emp, JOB_ROLES['salaried']))

            # Location
            state = profile.state or 'TN'
            city, pincode = CITIES_BY_STATE.get(state, ('Chennai', '600001'))
            street = random.choice(STREETS)
            house  = random.randint(1, 999)
            address = f"#{house}, {street}"

            # Banking
            bank = random.choice(BANKS)
            ifsc_prefix = IFSC_PREFIXES.get(bank, 'SBIN')
            ifsc = f"{ifsc_prefix}0{random.randint(100000, 999999)}"

            DemoProfile.objects.create(
                csv_user_id     = uid,
                full_name       = name,
                dob             = fake_dob(profile.age),
                gender          = gender,
                mobile          = fake_mobile(),
                pan             = fake_pan(),
                aadhaar         = fake_aadhaar(),
                marital_status  = random.choice(['single', 'married', 'married', 'single']),
                address         = address,
                city            = city,
                pincode         = pincode,
                residence_type  = random.choice(['rented', 'owned', 'family', 'rented']),
                years_at_address = random.randint(1, 8),
                company_name    = company,
                job_role        = job_role,
                work_experience = max(1, profile.age - 22 - random.randint(0, 3)),
                salary_credit   = 'bank_transfer',
                bank_name       = bank,
                account_number  = fake_account(),
                ifsc            = ifsc,
                account_type    = random.choice(['savings', 'salary', 'savings']),
            )
            created += 1

        self.stdout.write(self.style.SUCCESS(
            f'\n✅ Demo profiles seeded: {created} created, {skipped} already existed.'
        ))
