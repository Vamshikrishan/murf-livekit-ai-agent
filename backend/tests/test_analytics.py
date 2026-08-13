import sqlite3
from pathlib import Path

import pytest

from analytics import (
    close_call_record,
    get_analytics_summary,
    get_recent_calls,
    init_db,
    start_call_record,
)


@pytest.fixture
def analytics_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "memory.db"
    monkeypatch.setattr("analytics.DB_PATH", db_path)
    init_db()
    return db_path


def test_analytics_table_creation(analytics_db: Path) -> None:
    with sqlite3.connect(analytics_db) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='call_analytics'"
        ).fetchall()
    assert tables


def test_successful_call_record(analytics_db: Path) -> None:
    call_id = start_call_record(channel="browser")
    close_call_record(call_id, outcome="success", failure_reason=None)
    summary = get_analytics_summary()
    assert summary["total_calls"] == 1
    assert summary["successful_calls"] == 1
    assert summary["failed_calls"] == 0
    assert summary["success_rate"] == 100.0


def test_failed_call_record(analytics_db: Path) -> None:
    call_id = start_call_record(channel="sip")
    close_call_record(call_id, outcome="failed", failure_reason="caller_disconnected")
    summary = get_analytics_summary()
    assert summary["total_calls"] == 1
    assert summary["successful_calls"] == 0
    assert summary["failed_calls"] == 1
    assert summary["success_rate"] == 0.0


def test_total_call_count_and_aggregates(analytics_db: Path) -> None:
    start_call_record(channel="browser")
    close_call_record(start_call_record(channel="sip"), outcome="success")
    close_call_record(start_call_record(channel="unknown"), outcome="failed")
    summary = get_analytics_summary()
    assert summary["total_calls"] == 3
    assert summary["successful_calls"] == 1
    assert summary["failed_calls"] == 1
    assert summary["success_rate"] == 50.0


def test_empty_database_behavior(analytics_db: Path) -> None:
    summary = get_analytics_summary()
    assert summary == {
        "total_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "success_rate": 0,
    }


def test_call_duration_calculation(analytics_db: Path) -> None:
    call_id = start_call_record(channel="browser")
    close_call_record(call_id, outcome="success", duration_seconds=42)
    recent = get_recent_calls(limit=1)
    assert recent[0]["duration_seconds"] == 42


def test_unexpected_failed_call_termination(analytics_db: Path) -> None:
    call_id = start_call_record(channel="browser")
    close_call_record(call_id, outcome="failed", failure_reason="unexpected_disconnect")
    recent = get_recent_calls(limit=1)
    assert recent[0]["outcome"] == "failed"
    assert recent[0]["failure_reason"] == "unexpected_disconnect"
