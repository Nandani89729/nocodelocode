"""Shared contracts for Self-improve (milestone 2b).

Self-improve is NOT a new system: every N real conversations it runs the
existing Agent Doctor on exactly those calls, then one copilot turn in
``mode="self_improve"`` that may propose fixes (a normal copilot Proposal).
The user reviews it in the copilot panel's Self-improve tab.

Plan + review ledger: ~/.claude/plans/i-want-this-claude-luminous-ember.md
("Milestone 2b", decisions D1-D6, RI-1..6, O1-O8, design DS-1..DS-10).

    completion job ──(real conversation? AI on? not paused)──► counted run
          │ window full (N x multiplier) AND previous round reviewed (D4)
          ▼
     round queued ─► diagnosing ─► proposing ─► ready ─(applied/discarded/stale)─► reviewed
                         │fail          │             └► no_changes (multiplier x2, max 4)
                         ▼              ▼
                       failed (2 in a row ─► paused, Resume)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Allowed values for "every N calls".
ALLOWED_EVERY_N_CALLS: tuple[int, ...] = (5, 10, 20)
DEFAULT_EVERY_N_CALLS = 10

#: Backoff after a no_changes round: N, 2N, 4N (D6).
MAX_WINDOW_MULTIPLIER = 4

#: Consecutive failed rounds that pause Self-improve (D6).
FAILURES_BEFORE_PAUSE = 2

#: A run counts as a real conversation when the caller spoke at least this
#: many times (D3).
MIN_USER_TURNS = 1

#: Next-calls check: fewer eligible calls than this = insufficient evidence (O3).
MIN_CALLS_FOR_EVIDENCE = 3

#: Rounds stuck in queued/diagnosing/proposing longer than this are re-enqueued
#: by the sweep (O5). Matches the Doctor job timeout.
STUCK_ROUND_SECONDS = 1800

#: A round still unfinished this long after it was created is failed by the
#: sweep instead of re-enqueued again (no endless retries).
MAX_ROUND_AGE_SECONDS = 4 * STUCK_ROUND_SECONDS

#: arq job name (registered in api/tasks/arq.py).
SELF_IMPROVE_JOB = "self_improve_iteration"


# ---------------------------------------------------------------------------
# Settings + progress
# ---------------------------------------------------------------------------


class SelfImproveState(str, Enum):
    """What the status line shows (DS-3)."""

    off = "off"
    ai_disabled = "ai_disabled"  # org not opted in
    no_key = "no_key"
    collecting = "collecting"
    waiting = "waiting"  # a round is waiting for the user's review (D4)
    running = "running"  # a round is queued / diagnosing / proposing
    paused = "paused"  # 2 failed rounds in a row (D6)


class SelfImproveSettings(BaseModel):
    enabled: bool = False
    every_n_calls: int = DEFAULT_EVERY_N_CALLS


class SelfImproveStatus(BaseModel):
    """GET /workflow/{id}/self-improve."""

    workflow_id: int
    settings: SelfImproveSettings
    state: SelfImproveState
    #: calls counted in the current window, and the window size (N x multiplier)
    collected: int = 0
    window_size: int = DEFAULT_EVERY_N_CALLS
    window_multiplier: int = 1
    paused_reason: Optional[str] = None
    #: number of rounds waiting for review (0 or 1 under D4), for DS-2 counts
    to_review: int = 0
    #: the round to show first (DS-1): waiting/running round, else latest
    current_iteration: Optional["IterationResponse"] = None
    #: True when the workflow has no brief yet (DS-6 prompt on turn-on)
    has_brief: bool = False


class UpdateSelfImproveRequest(BaseModel):
    enabled: bool
    every_n_calls: int = DEFAULT_EVERY_N_CALLS


# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------


class IterationStatus(str, Enum):
    queued = "queued"
    diagnosing = "diagnosing"
    proposing = "proposing"
    ready = "ready"  # proposal waiting for review
    no_changes = "no_changes"
    failed = "failed"


#: Round statuses during which the job is still running.
ACTIVE_ITERATION_STATUSES = (
    IterationStatus.queued,
    IterationStatus.diagnosing,
    IterationStatus.proposing,
)


class FixOutcome(str, Enum):
    """Next-calls check result for one earlier accepted change (DS-4, O1/O3)."""

    fixed = "fixed"  # ✓ 0 of N calls since the fix
    still_happening = "still_happening"  # ✗ still in X of N calls
    insufficient = "insufficient"  # – not enough calls yet
    waiting_for_publish = "waiting_for_publish"  # – draft not published


class FixCarryover(BaseModel):
    """One line under a round: how an earlier accepted change is doing."""

    proposal_uuid: str
    change_id: str
    title: str  # the hypothesis title, e.g. "Welcome step answers pricing"
    outcome: FixOutcome
    #: calls since publish that reached the target node, and how many still
    #: showed the problem
    eligible_calls: int = 0
    affected_calls: int = 0
    published_at: Optional[str] = None


class IterationResponse(BaseModel):
    iteration_id: int
    number: int
    status: IterationStatus
    run_ids: list[int] = Field(default_factory=list)
    report_uuid: Optional[str] = None
    proposal_uuid: Optional[str] = None
    #: reviewed = proposal applied/discarded/stale, or no_changes/failed
    reviewed: bool = False
    carryover: list[FixCarryover] = Field(default_factory=list)
    #: actionable clusters in the round's report (DS-3 "Writing fixes for N problems")
    problem_count: Optional[int] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None


class IterationListResponse(BaseModel):
    iterations: list[IterationResponse]


class RunNowResponse(BaseModel):
    started: bool
    iteration: Optional[IterationResponse] = None
    #: why the round did not start
    reason: Optional[
        Literal[
            "waiting_for_review",
            "running",
            "no_calls",
            "ai_disabled",
            "no_key",
            "off",
            "paused",
        ]
    ] = None


SelfImproveStatus.model_rebuild()
