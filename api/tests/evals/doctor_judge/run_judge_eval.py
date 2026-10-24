"""Judge eval for Doctor verification grading (plan R7/D24).

Measures how often the Claude rubric judge agrees with human labels on real
workflow 7 turns. Gate: agreement >= 90% and at most 1 false pass on items
marked critical. Only human-labelled items (``label`` set) are scored.

Run (needs ANTHROPIC_API_KEY; spends real tokens):
    cd ~/dograh && set -a && source api/.env && set +a && \
    ./venv/Scripts/python.exe -m api.tests.evals.doctor_judge.run_judge_eval
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

LABELS = Path(__file__).with_name("labels.json")
#: Per-item agent prompts from each run's pinned definition (snapshot_knowledge.py).
KNOWLEDGE = Path(__file__).with_name("knowledge.json")
MIN_AGREEMENT = 0.90
MAX_CRITICAL_FALSE_PASSES = 1


def load_labelled() -> list[dict]:
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    return [item for item in data["items"] if item.get("label") in ("pass", "fail")]


def score(items: list[dict], predictions: dict[str, bool]) -> dict:
    agree = 0
    critical_false_passes = 0
    disagreements = []
    for item in items:
        predicted_pass = predictions[item["id"]]
        human_pass = item["label"] == "pass"
        if predicted_pass == human_pass:
            agree += 1
            continue
        disagreements.append(item["id"])
        if item.get("critical") and predicted_pass and not human_pass:
            critical_false_passes += 1
    total = len(items)
    agreement = agree / total if total else 0.0
    return {
        "total": total,
        "agreement": agreement,
        "critical_false_passes": critical_false_passes,
        "disagreements": disagreements,
        "passed": total > 0
        and agreement >= MIN_AGREEMENT
        and critical_false_passes <= MAX_CRITICAL_FALSE_PASSES,
    }


async def _predict(items: list[dict]) -> dict[str, bool]:
    from api.services.workflow.doctor.contracts import SampleResult
    from api.services.workflow.doctor.verify import grade

    knowledge = (
        json.loads(KNOWLEDGE.read_text(encoding="utf-8")) if KNOWLEDGE.exists() else {}
    )
    predictions: dict[str, bool] = {}
    for item in items:
        samples = [SampleResult(text=item["bot"], passed=False)]
        graded = await grade(
            samples,
            item["expectation"],
            context=f"Caller said: {item['user']}",
            agent_knowledge=knowledge.get(item["id"]),
        )
        predictions[item["id"]] = graded[0].passed
    return predictions


def main() -> int:
    items = load_labelled()
    if not items:
        print("No human-labelled items yet: set 'label' in labels.json first.")
        return 2
    predictions = asyncio.run(_predict(items))
    result = score(items, predictions)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
