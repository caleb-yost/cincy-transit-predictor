"""Print the current model's stored metrics + metadata.

Run: ``python ml/evaluate.py``. Reads ml/artifacts/metrics.json written by train.py.
"""

from __future__ import annotations

import json
from pathlib import Path

METRICS_PATH = Path(__file__).resolve().parent / "artifacts" / "metrics.json"


def main() -> None:
    if not METRICS_PATH.exists():
        print("No metrics found — run `python ml/train.py` first.")
        return
    metrics = json.loads(METRICS_PATH.read_text())
    print(json.dumps(metrics, indent=2))
    if metrics.get("is_smoke_model"):
        print("\nNOTE: smoke model — accrue more data (a week of ingestion) for meaningful numbers.")


if __name__ == "__main__":
    main()
