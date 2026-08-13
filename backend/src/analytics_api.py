import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from analytics import get_analytics_summary, get_recent_calls, init_db


def main() -> int:
    init_db()
    mode = (sys.argv[1] if len(sys.argv) > 1 else "summary").strip().lower()

    if mode == "summary":
        payload = get_analytics_summary()
    elif mode == "calls":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        payload = get_recent_calls(limit=limit)
    else:
        payload = {"error": "unsupported_mode"}

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
