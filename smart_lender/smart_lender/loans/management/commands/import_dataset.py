"""
Management command: import_dataset
===================================
Loads users.csv, features.csv, and transactions.csv from the H&H workspace
into the Django database.

Usage:
    python manage.py import_dataset

What it creates:
  - One Django User + UserProfile per row in users.csv
    (username = user_id, password = "demo1234", role = "user")
  - One CsvUserFeature record per row in features.csv
  - CsvTransaction records from transactions.csv (first 50k rows for speed)
"""

import os
import csv
import json
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from loans.models import UserProfile, CsvUserFeature, CsvTransaction

# Path to H&H workspace — hardcoded absolute path
_ROOT = r'C:\Users\gowsi\OneDrive\Desktop\H&H'

USERS_CSV    = os.path.join(_ROOT, 'users.csv')
FEATURES_CSV = os.path.join(_ROOT, 'features.csv')
TXN_CSV      = os.path.join(_ROOT, 'transactions.csv')

MAX_TXN_ROWS = 100000   # limit for speed
MAX_USERS    = 50       # only first 50 users needed for demo


class Command(BaseCommand):
    help = 'Import users.csv, features.csv, transactions.csv into Django DB'

    def add_arguments(self, parser):
        parser.add_argument('--clear', action='store_true',
                            help='Clear existing imported data before importing')

    def handle(self, *args, **options):
        if options['clear']:
            self.stdout.write('Clearing existing data...')
            CsvTransaction.objects.all().delete()
            CsvUserFeature.objects.all().delete()
            User.objects.filter(username__startswith='USR_').delete()
            self.stdout.write('  Cleared.')

        self._import_users()
        self._import_features()
        self._import_transactions()
        self.stdout.write(self.style.SUCCESS('\n✅ Dataset import complete.'))

    # ── Users ─────────────────────────────────────────────────────────────────

    def _import_users(self):
        self.stdout.write('Importing users.csv (first 50)...')
        if not os.path.exists(USERS_CSV):
            self.stdout.write(self.style.WARNING(f'  Not found: {USERS_CSV}'))
            return

        created = skipped = 0
        with open(USERS_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= MAX_USERS:
                    break
                uid = row['user_id'].strip()
                if User.objects.filter(username=uid).exists():
                    skipped += 1
                    continue

                user = User.objects.create_user(
                    username=uid,
                    email=f"{uid.lower()}@demo.sml",
                    password='demo1234',
                    first_name=uid,
                )
                UserProfile.objects.create(
                    user=user,
                    role='user',
                    csv_user_id=uid,
                    age=int(row.get('age', 30)),
                    employment_type=row.get('employment_type', 'salaried'),
                    monthly_income=float(row.get('monthly_income', 0)),
                    cibil_score=float(row['cibil_score']) if row.get('cibil_score') else None,
                    existing_emi=float(row.get('existing_emi', 0)),
                    account_age_months=int(row.get('account_age_months', 0)),
                    state=row.get('state', ''),
                    loan_type_preference=row.get('loan_type', 'personal'),
                )
                created += 1

        self.stdout.write(f'  Created: {created}  Skipped (already exist): {skipped}')

    # ── Features ──────────────────────────────────────────────────────────────

    def _import_features(self):
        self.stdout.write('Importing features.csv (first 50)...')
        if not os.path.exists(FEATURES_CSV):
            self.stdout.write(self.style.WARNING(f'  Not found: {FEATURES_CSV}'))
            return

        created = skipped = 0
        with open(FEATURES_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= MAX_USERS:
                    break
                uid = row['user_id'].strip()
                if CsvUserFeature.objects.filter(csv_user_id=uid).exists():
                    skipped += 1
                    continue
                CsvUserFeature.objects.create(
                    csv_user_id=uid,
                    income_regularity=float(row.get('income_regularity', 0)),
                    spending_consistency=float(row.get('spending_consistency', 0)),
                    bill_payment_ratio=float(row.get('bill_payment_ratio', 0)),
                    savings_rate=float(row.get('savings_rate', 0)),
                    emi_burden_ratio=float(row.get('emi_burden_ratio', 0)),
                    upi_activity_volume=float(row.get('upi_activity_volume', 0)),
                    rent_payment_regularity=float(row.get('rent_payment_regularity', 0)),
                    cash_withdrawal_ratio=float(row.get('cash_withdrawal_ratio', 0)),
                    target=int(row.get('TARGET', 0)),
                )
                created += 1

        self.stdout.write(f'  Created: {created}  Skipped (already exist): {skipped}')

    # ── Transactions ──────────────────────────────────────────────────────────

    def _import_transactions(self):
        self.stdout.write('Importing transactions for first 50 users...')
        if not os.path.exists(TXN_CSV):
            self.stdout.write(self.style.WARNING(f'  Not found: {TXN_CSV}'))
            return

        if CsvTransaction.objects.exists():
            self.stdout.write('  Already imported — skipping.')
            return

        # Only import transactions for the first 50 user IDs
        allowed_uids = set(f'USR_{i:04d}' for i in range(1, MAX_USERS + 1))

        batch = []
        count = 0
        with open(TXN_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get('user_id', '')
                if uid not in allowed_uids:
                    continue
                batch.append(CsvTransaction(
                    transaction_id=row.get('transaction_id', ''),
                    csv_user_id=uid,
                    amount=float(row.get('amount', 0)),
                    txn_type=row.get('type', 'debit'),
                    category=row.get('category', 'other'),
                    timestamp=row.get('timestamp', ''),
                    payment_mode=row.get('payment_mode', ''),
                    merchant=row.get('merchant', ''),
                ))
                count += 1
                if len(batch) >= 500:
                    CsvTransaction.objects.bulk_create(batch, ignore_conflicts=True)
                    batch = []

        if batch:
            CsvTransaction.objects.bulk_create(batch, ignore_conflicts=True)

        self.stdout.write(f'  Imported: {count} transactions for {MAX_USERS} users')
