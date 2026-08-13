import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "memory.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS call_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT NOT NULL UNIQUE,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_seconds INTEGER DEFAULT 0,
    channel TEXT DEFAULT 'unknown',
    outcome TEXT DEFAULT 'in_progress' CHECK (outcome IN ('in_progress','success','failed')),
    failure_reason TEXT,
    created_at TEXT NOT NULL
)
"""


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_connection()
    try:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def _normalize_outcome(outcome: Optional[str]) -> str:
    value = (outcome or "failed").strip().lower()
    if value not in {"in_progress", "success", "failed"}:
        return "failed"
    return value


def _safe_iso_utc(ts: Optional[str]) -> Optional[str]:
    if not ts:
        return None
    value = ts.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        datetime.fromisoformat(value)
        return ts
    except ValueError:
        return None


def start_call_record(channel: str = "unknown") -> str:
    call_id = f"CALL-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8].upper()}"
    started_at = datetime.now(timezone.utc).isoformat()
    created_at = started_at
    normalized_channel = (channel or "unknown").strip().lower()
    if normalized_channel not in {"browser", "sip", "unknown"}:
        normalized_channel = "unknown"

    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO call_analytics (call_id, started_at, channel, outcome, created_at)
            VALUES (?, ?, ?, 'in_progress', ?)
            """,
            (call_id, started_at, normalized_channel, created_at),
        )
        conn.commit()
        return call_id
    finally:
        conn.close()


def close_call_record(
    call_id: str,
    outcome: Optional[str] = None,
    failure_reason: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    ended_at: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not call_id:
        return None

    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT call_id, started_at, ended_at, duration_seconds, outcome FROM call_analytics WHERE call_id = ?",
            (call_id,),
        ).fetchone()
        if row is None:
            return None

        final_outcome = _normalize_outcome(outcome)
        end_iso = ended_at or datetime.now(timezone.utc).isoformat()
        start_iso = row["started_at"]
        calculated_seconds = 0
        if duration_seconds is None:
            try:
                start_dt = datetime.fromisoformat(_safe_iso_utc(start_iso) or start_iso)
                end_dt = datetime.fromisoformat(_safe_iso_utc(end_iso) or end_iso)
                calculated_seconds = max(0, int((end_dt - start_dt).total_seconds()))
            except ValueError:
                calculated_seconds = int(row["duration_seconds"] or 0)
        else:
            calculated_seconds = max(0, int(duration_seconds))

        if final_outcome == "success":
            final_failure_reason = None
        else:
            final_failure_reason = failure_reason or "session_ended_without_success"

        conn.execute(
            """
            UPDATE call_analytics
            SET ended_at = ?,
                duration_seconds = ?,
                outcome = ?,
                failure_reason = ?
            WHERE call_id = ?
            """,
            (end_iso, calculated_seconds, final_outcome, final_failure_reason, call_id),
        )
        conn.commit()

        updated = conn.execute(
            "SELECT call_id, started_at, ended_at, duration_seconds, channel, outcome, failure_reason FROM call_analytics WHERE call_id = ?",
            (call_id,),
        ).fetchone()
        return dict(updated) if updated else None
    finally:
        conn.close()


def get_analytics_summary() -> Dict[str, Any]:
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_calls,
                SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) AS successful_calls,
                SUM(CASE WHEN outcome = 'failed' THEN 1 ELSE 0 END) AS failed_calls
            FROM call_analytics
            """
        ).fetchone()

        total_calls = int(row["total_calls"] or 0)
        successful_calls = int(row["successful_calls"] or 0)
        failed_calls = int(row["failed_calls"] or 0)

        completed_calls = successful_calls + failed_calls
        success_rate = 0.0
        if completed_calls:
            success_rate = round((successful_calls / completed_calls) * 100, 2)

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": success_rate,
        }
    finally:
        conn.close()


def get_recent_calls(limit: int = 20) -> List[Dict[str, Any]]:
    safe_limit = max(1, int(limit))
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT call_id, started_at, ended_at, duration_seconds, channel, outcome, failure_reason
            FROM call_analytics
            WHERE outcome IN ('success', 'failed')
            ORDER BY ended_at DESC, started_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
