"""Offline checks for the Doctor judge eval harness (no network)."""

import json

from api.tests.evals.doctor_judge.run_judge_eval import LABELS, score


def test_label_set_is_well_formed():
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    items = data["items"]
    assert len(items) == 40
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids))
    for item in items:
        assert item["expectation"]
        assert item["suggested_label"] in ("pass", "fail")
        assert item["label"] in (None, "pass", "fail")
    assert any(item["critical"] for item in items)


def test_score_gate():
    items = [
        {"id": "a", "label": "fail", "critical": True},
        {"id": "b", "label": "pass", "critical": False},
    ]
    clean = score(items, {"a": False, "b": True})
    assert clean["passed"] and clean["agreement"] == 1.0

    false_pass = score(items, {"a": True, "b": True})
    assert false_pass["critical_false_passes"] == 1
    assert not false_pass["passed"]  # 50% agreement
