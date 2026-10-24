"""Dry-run tool execution for Agent Doctor replays (O1).

A ``doctor_replay`` run must never let a request leave Dograh. External tools
(HTTP API, MCP, transfer call) therefore return a recorded or synthetic result
instead of executing. Built-ins (calculator, timezone, knowledge-base
retrieval, end call, node transitions) still run for real.

The replay configuration is stored on the replay run's
``annotations["doctor_replay"]`` by ``doctor/verify.py``::

    {
        "source_run_id": 123,
        "turn": 6,
        "tool_results": {"lookup_order": [<result>, ...]},
        "failure_mocks": {"lookup_order": {"status": "error", "error": "..."}},
    }

``tool_results`` are the results the source call actually got, per function
name, in call order; ``failure_mocks`` force an error result for a function.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

DOCTOR_REPLAY_ANNOTATION_KEY = "doctor_replay"

#: Synthetic results used when the source call recorded nothing for a tool.
SYNTHETIC_HTTP_RESULT: dict[str, Any] = {
    "status": "success",
    "status_code": 200,
    "data": {},
}
SYNTHETIC_MCP_RESULT: dict[str, Any] = {"status": "success", "content": []}
SYNTHETIC_TRANSFER_RESULT: dict[str, Any] = {
    "status": "failed",
    "message": "I'm sorry, but call transfers are not available right now.",
    "action": "transfer_failed",
    "reason": "doctor_replay_dry_run",
}


def _parse_recorded_result(value: Any) -> Any:
    """Recorded results are persisted as JSON strings; restore dicts/lists."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


@dataclass
class ToolDryRun:
    """Per-run dry-run state. One instance per replay engine."""

    tool_results: dict[str, list[Any]] = field(default_factory=dict)
    failure_mocks: dict[str, Any] = field(default_factory=dict)
    #: (function_name, arguments) of every intercepted call, for tests/audit.
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    _cursor: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_annotations(cls, annotations: dict[str, Any] | None) -> ToolDryRun:
        config = (annotations or {}).get(DOCTOR_REPLAY_ANNOTATION_KEY) or {}
        raw_results = config.get("tool_results") or {}
        return cls(
            tool_results={
                str(name): [_parse_recorded_result(v) for v in (values or [])]
                for name, values in raw_results.items()
            },
            failure_mocks=dict(config.get("failure_mocks") or {}),
        )

    def result_for(
        self,
        function_name: str,
        arguments: dict[str, Any] | None,
        *,
        synthetic: dict[str, Any],
    ) -> Any:
        """Return the failure mock, else the next recorded result, else synthetic.

        Recorded results are replayed in call order; once exhausted the last
        one repeats (a re-rolled turn may call a tool more often than the
        source call did).
        """
        self.calls.append((function_name, dict(arguments or {})))
        if function_name in self.failure_mocks:
            mock = self.failure_mocks[function_name]
            return (
                copy.deepcopy(mock)
                if mock
                else {"status": "error", "error": "Simulated tool failure"}
            )
        recorded = self.tool_results.get(function_name) or []
        if recorded:
            index = self._cursor.get(function_name, 0)
            self._cursor[function_name] = index + 1
            return copy.deepcopy(recorded[min(index, len(recorded) - 1)])
        return copy.deepcopy(synthetic)


def get_tool_dry_run(engine: Any) -> ToolDryRun | None:
    """The engine's dry-run state, or None for every normal run.

    Only a real ``ToolDryRun`` counts, so a stand-in engine (e.g. a test
    ``MagicMock``, whose every attribute is truthy) never flips a normal run
    into dry-run.
    """
    dry_run = getattr(engine, "tool_dry_run", None)
    return dry_run if isinstance(dry_run, ToolDryRun) else None
