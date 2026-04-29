"""
Management command: seed_first50_demo_profiles
=============================================
Creates DemoProfile records for USR_0001..USR_0050 if they don't exist.
This helps auto-fill the apply form for the first 50 demo users.

Usage:
    python manage.py seed_first50_demo_profiles
"""

import random
from django.core.management.base import BaseCommand
from loans.models import DemoProfile

FIRST_NAMES_M = ['Arjun', 'Rahul', 'Vikram', 'Suresh', 'Karthik', 'Arun', 'Deepak',
                 'Rajesh', 'Manoj', 'Sanjay', 'Pradeep', 'Anand', 'Ravi', 'Mohan',
                 'Ganesh', 'Venkat', 'Harish', 'Naveen', 'Dinesh', 'Ramesh']
FIRST_NAMES_F = ['Priya', 'Divya', 'Kavya', 'Ananya', 'Sneha', 'Pooja', 'Meena',
                 'Lakshmi', 'Sunita', 'Rekha', 'Nisha', 'Asha', 'Geeta', 'Sita',
                 'Radha', 'Usha', 'Mala', 'Saranya', 'Deepa', 'Revathi']
LAST_NAMES    = ['Kumar', 'Sharma', 'Patel', 'Singh', 'Reddy', 'Nair', 'Iyer',
                 'Gupta', 'Verma', 'Joshi', 'Pillai', 'Rao', 'Mishra', 'Tiwari',
                 'Pandey', 'Shah', 'Mehta', 'Desai', 'Bose', 'Das']

STATES = ['TN','MH','KA','UP','DL','WB','GJ','RJ','AP','KL']
STREETS = ['MG Road', 'Anna Salai', 'Brigade Road', 'Linking Road', 'Park Street',
           'Connaught Place', 'Banjara Hills', 'Koramangala', 'Andheri West',
           'Sector 18', 'Civil Lines', 'Jubilee Hills', 'Indiranagar', 'Salt Lake']

BANKS = ['State Bank of India', 'HDFC Bank', 'ICICI Bank', 'Axis Bank',
         'Punjab National Bank', 'Bank of Baroda', 'Canara Bank', 'Union Bank']
IFSC_PREFIXES = {
    'State Bank of India': 'SBIN', 'HDFC Bank': 'HDFC', 'ICICI Bank': 'ICIC',
    'Axis Bank': 'UTIB', 'Punjab National Bank':'PUNB','Bank of Baroda':'BARB',
    'Canara Bank':'CNRB','Union Bank':'UBIN'
}


def fake_pan():
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return (random.choice(letters) + random.choice(letters) + random.choice(letters) +
            random.choice(letters) + random.choice(letters) + str(random.randint(1000, 9999)) + random.choice(letters))


def fake_aadhaar():
    return f"{random.randint(2000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"


def fake_account():
    return str(random.randint(10000000000, 99999999999999))


def fake_mobile():
    return f"9{random.randint(100000000, 999999999)}"


class Command(BaseCommand):
    help = 'Seed DemoProfile for USR_0001..USR_0050'

    def handle(self, *args, **options):
        created = 0
        for i in range(1, 51):
            uid = f"USR_{i:04d}"
            if DemoProfile.objects.filter(csv_user_id=uid).exists():
                continue

            random.seed(hash(uid))
            gender = random.choice(['male','female'])
            first = random.choice(FIRST_NAMES_M if gender == 'male' else FIRST_NAMES_F)
            last = random.choice(LAST_NAMES)
            name = f"{first} {last}"
            age = random.randint(22, 55)
            state = random.choice(STATES)
            street = random.choice(STREETS)
            house = random.randint(1, 999)
            address = f"#{house}, {street}"
            city = 'City'
            pincode = f"{random.randint(100000,999999)}"
            bank = random.choice(BANKS)
            ifsc = f"{IFSC_PREFIXES.get(bank,'SBIN')}0{random.randint(100000,999999)}"

            DemoProfile.objects.create(
                csv_user_id=uid,
                full_name=name,
                dob=f"{2024 - age}-01-01",
                gender=gender,
                mobile=fake_mobile(),
                pan=fake_pan(),
                aadhaar=fake_aadhaar(),
                marital_status=random.choice(['single','married']),
                address=address,
                city=city,
                pincode=pincode,
                residence_type=random.choice(['rented','owned','family']),
                years_at_address=random.randint(1,6),
                company_name=random.choice(['Infosys','TCS','Wipro','Self-Employed']),
                job_role=random.choice(['Software Engineer','Analyst','Manager','Consultant']),
                work_experience=max(1, age-22),
                salary_credit='bank_transfer',
                bank_name=bank,
                account_number=fake_account(),
                ifsc=ifsc,
                account_type=random.choice(['savings','salary']),
            )
            created += 1

        self.stdout.write(self.style.SUCCESS(f"Created {created} DemoProfiles for USR_0001..USR_0050"))
