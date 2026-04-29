"""
Tamper-Proof Audit Logger
==========================
Append-only SQLite log with SHA-256 hash chaining.
Every decision is logged. No record can be modified after creation.

Hash chain: H_n = SHA256(H_(n-1) || audit_id || timestamp || decision || shap_values_str)
"""

import sqlite3
import hashlib
import json
import uuid
from datetime import datetime

DB_FILE = "audit_log.db"


# ─── Init ─────────────────────────────────────────────────────────────────────

def init_db(db_path: str = DB_FILE):
    """Create the audit_log table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            rowid              INTEGER PRIMARY KEY AUTOINCREMENT,
            audit_id           TEXT    NOT NULL UNIQUE,
            user_id            TEXT    NOT NULL,
            loan_type          TEXT    NOT NULL,
            timestamp          TEXT    NOT NULL,
            policy_passed      INTEGER NOT NULL,
            failed_rule        TEXT,
            risk_score         REAL,
            final_decision     TEXT    NOT NULL,
            shap_values        TEXT,
            explanation_user   TEXT,
            explanation_auditor TEXT,
            previous_hash      TEXT    NOT NULL,
            current_hash       TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ─── Hash ─────────────────────────────────────────────────────────────────────

def compute_hash(
    previous_hash: str,
    audit_id: str,
    timestamp: str,
    final_decision: str,
    shap_values_str: str,
) -> str:
    """Compute SHA-256 hash for a log entry."""
    payload = f"{previous_hash}{audit_id}{timestamp}{final_decision}{shap_values_str}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─── Log Decision ─────────────────────────────────────────────────────────────

def log_decision(
    user_id: str,
    loan_type: str,
    policy_result: dict,
    shap_result: dict | None,
    explanations: dict,
    db_path: str = DB_FILE,
) -> str:
    """
    Append a decision to the audit log.

    Returns
    -------
    str : audit_id of the new record
    """
    init_db(db_path)

    audit_id       = str(uuid.uuid4())
    timestamp      = datetime.utcnow().isoformat() + "Z"
    policy_passed  = int(policy_result.get("passed", False))
    failed_rule    = policy_result.get("failed_rule")
    risk_score     = shap_result["risk_score"] if shap_result else None
    final_decision = (
        shap_result["decision"] if shap_result
        else "rejected"
    )
    shap_values_str = json.dumps(shap_result["shap_values"] if shap_result else {})

    explanation_user    = explanations.get("user", "")
    explanation_auditor = json.dumps(explanations.get("auditor", {}))

    conn = sqlite3.connect(db_path)

    # Get last hash
    row = conn.execute(
        "SELECT current_hash FROM audit_log ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    previous_hash = row[0] if row else "0" * 64

    current_hash = compute_hash(
        previous_hash, audit_id, timestamp, final_decision, shap_values_str
    )

    conn.execute(
        """
        INSERT INTO audit_log (
            audit_id, user_id, loan_type, timestamp,
            policy_passed, failed_rule, risk_score, final_decision,
            shap_values, explanation_user, explanation_auditor,
            previous_hash, current_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            audit_id, user_id, loan_type, timestamp,
            policy_passed, failed_rule, risk_score, final_decision,
            shap_values_str, explanation_user, explanation_auditor,
            previous_hash, current_hash,
        ),
    )
    conn.commit()
    conn.close()

    return audit_id


# ─── Verify Chain ─────────────────────────────────────────────────────────────

def verify_chain(db_path: str = DB_FILE) -> dict:
    """
    Recompute every hash in the chain and verify integrity.

    Returns
    -------
    dict:
        valid         : bool
        total_records : int
        broken_at     : audit_id or None
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT audit_id, timestamp, final_decision, shap_values,
               previous_hash, current_hash
        FROM audit_log ORDER BY rowid ASC
        """
    ).fetchall()
    conn.close()

    if not rows:
        return {"valid": True, "total_records": 0, "broken_at": None}

    expected_prev = "0" * 64

    for audit_id, timestamp, final_decision, shap_values_str, prev_hash, stored_hash in rows:
        # Verify previous_hash matches what we expect
        if prev_hash != expected_prev:
            return {"valid": False, "total_records": len(rows), "broken_at": audit_id}

        # Recompute current hash
        recomputed = compute_hash(prev_hash, audit_id, timestamp, final_decision, shap_values_str)
        if recomputed != stored_hash:
            return {"valid": False, "total_records": len(rows), "broken_at": audit_id}

        expected_prev = stored_hash

    return {"valid": True, "total_records": len(rows), "broken_at": None}


# ─── Get Record ───────────────────────────────────────────────────────────────

def get_record(audit_id: str, db_path: str = DB_FILE) -> dict | None:
    """Retrieve a full audit record by audit_id."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT * FROM audit_log WHERE audit_id = ?", (audit_id,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    cols = [
        "rowid", "audit_id", "user_id", "loan_type", "timestamp",
        "policy_passed", "failed_rule", "risk_score", "final_decision",
        "shap_values", "explanation_user", "explanation_auditor",
        "previous_hash", "current_hash",
    ]
    record = dict(zip(cols, row))

    # Parse JSON fields
    for field in ("shap_values", "explanation_auditor"):
        try:
            record[field] = json.loads(record[field]) if record[field] else {}
        except (json.JSONDecodeError, TypeError):
            pass

    return record


def get_all_records(db_path: str = DB_FILE) -> list:
    """Return all audit records ordered by rowid."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT audit_id, user_id, loan_type, timestamp, final_decision, risk_score "
        "FROM audit_log ORDER BY rowid ASC"
    ).fetchall()
    conn.close()

    return [
        {
            "audit_id":       r[0],
            "user_id":        r[1],
            "loan_type":      r[2],
            "timestamp":      r[3],
            "final_decision": r[4],
            "risk_score":     r[5],
        }
        for r in rows
    ]


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    TEST_DB = "test_audit.db"
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    init_db(TEST_DB)
    print("DB initialized.")

    # Log 3 fake decisions
    for i in range(3):
        aid = log_decision(
            user_id=f"USR_{i:04d}",
            loan_type="xpress_credit",
            policy_result={"passed": True, "failed_rule": None},
            shap_result={
                "risk_score": 0.3 + i * 0.2,
                "decision": "approved" if i < 2 else "rejected",
                "shap_values": {"emi_burden_ratio": 0.1 * i},
            },
            explanations={"user": f"Test explanation {i}", "auditor": {}},
            db_path=TEST_DB,
        )
        print(f"  Logged audit_id: {aid}")

    # Verify chain
    result = verify_chain(TEST_DB)
    print(f"\nChain verification: {result}")

    # Fetch one record
    all_recs = get_all_records(TEST_DB)
    rec = get_record(all_recs[0]["audit_id"], TEST_DB)
    print(f"\nFirst record: {rec['audit_id']} | {rec['final_decision']} | {rec['risk_score']}")

    os.remove(TEST_DB)
    print("\n✅ audit_logger.py test complete.")
