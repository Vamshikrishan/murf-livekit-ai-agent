import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "memory.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_memory (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    language_preference TEXT,
    facts TEXT,
    last_interaction TEXT
)
"""


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _serialize_facts(facts: Any) -> str:
    if facts is None:
        return json.dumps({})
    if isinstance(facts, str):
        try:
            json.loads(facts)
            return facts
        except json.JSONDecodeError:
            return json.dumps({"note": facts}, ensure_ascii=False)
    return json.dumps(facts, ensure_ascii=False)


def _deserialize_facts(facts_text: Optional[str]) -> Dict[str, Any]:
    if not facts_text:
        return {}
    try:
        return json.loads(facts_text)
    except json.JSONDecodeError:
        return {"raw": facts_text}


def init_db() -> None:
    conn = _get_connection()
    try:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def lookup_user(user_id: str) -> Optional[Dict[str, Any]]:
    if not user_id:
        return None
    try:
        conn = _get_connection()
        row = conn.execute(
            "SELECT user_id, name, language_preference, facts, last_interaction FROM user_memory WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None

        return {
            "user_id": row["user_id"],
            "name": row["name"],
            "language_preference": row["language_preference"],
            "facts": _deserialize_facts(row["facts"]),
            "last_interaction": row["last_interaction"],
        }
    except sqlite3.Error:
        return None
    finally:
        if "conn" in locals():
            conn.close()


def save_user_memory(
    user_id: str,
    name: Optional[str] = None,
    language_preference: Optional[str] = None,
    facts: Any = None,
    last_interaction: Optional[str] = None,
) -> Dict[str, Any]:
    if not user_id:
        raise ValueError("user_id is required to save memory")

    current = lookup_user(user_id)
    existing = current or {
        "name": None,
        "language_preference": None,
        "facts": {},
    }

    merged_name = name if name is not None else existing.get("name")
    merged_language = (
        language_preference
        if language_preference is not None
        else existing.get("language_preference")
    )
    merged_facts = existing.get("facts", {}) if existing else {}
    if facts is not None:
        if isinstance(facts, dict):
            merged_facts = {**merged_facts, **facts}
        else:
            merged_facts = {**merged_facts, "note": facts}

    facts_text = _serialize_facts(merged_facts)
    last_interaction = last_interaction or datetime.utcnow().isoformat() + "Z"

    try:
        conn = _get_connection()
        conn.execute(
            "INSERT INTO user_memory (user_id, name, language_preference, facts, last_interaction) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "name = excluded.name, "
            "language_preference = excluded.language_preference, "
            "facts = excluded.facts, "
            "last_interaction = excluded.last_interaction",
            (user_id, merged_name, merged_language, facts_text, last_interaction),
        )
        conn.commit()

        return {
            "user_id": user_id,
            "name": merged_name,
            "language_preference": merged_language,
            "facts": merged_facts,
            "last_interaction": last_interaction,
        }
    finally:
        conn.close()
