"""Snapshot the agent knowledge for each labelled judge-eval item.

The judge needs the prompts the agent actually had on that call, so facts it
was given are not graded as inventions. Prompts changed between iterations,
so each item uses its own run's pinned definition. The result is written to
knowledge.json so the eval runs without a database.

Run once against the dev DB (re-run when labels.json gains new runs):

    set -a && source api/.env && set +a
    python -m api.tests.evals.doctor_judge.snapshot_knowledge
"""

import asyncio
import json

from api.db import db_client
from api.services.workflow.doctor.verify import agent_knowledge_for
from api.tests.evals.doctor_judge.run_judge_eval import KNOWLEDGE, LABELS


async def _snapshot() -> dict[str, str]:
    items = json.loads(LABELS.read_text(encoding="utf-8"))["items"]
    definitions: dict[int, dict] = {}
    knowledge: dict[str, str] = {}
    for item in items:
        run_id = item["run_id"]
        if run_id not in definitions:
            run = await db_client.get_workflow_run(run_id)
            if run is None or run.definition is None:
                raise SystemExit(f"Run {run_id} or its definition is missing")
            definitions[run_id] = run.definition.workflow_json
        knowledge[item["id"]] = agent_knowledge_for(definitions[run_id], item["node"])
    return knowledge


def main() -> None:
    knowledge = asyncio.run(_snapshot())
    KNOWLEDGE.write_text(
        json.dumps(knowledge, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote knowledge for {len(knowledge)} items to {KNOWLEDGE}")


if __name__ == "__main__":
    main()
