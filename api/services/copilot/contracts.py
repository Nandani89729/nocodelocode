"""Shared contracts for the Agent Copilot (milestone 2).

The copilot is a chat panel that plans, builds and edits workflows. Every
copilot module, the DB client, the routes and the UI code against this file;
it is the seam between parallel work streams. Change it deliberately.

Flow (plan: ~/.claude/plans/i-want-this-claude-luminous-ember.md, M2):

    POST /copilot/sessions                      -> CopilotSession (per workflow, or none = build new)
    POST /copilot/sessions/{uuid}/messages      -> text/event-stream of CopilotEvent
        agent loop (llm.py) <-> tools (tools.py): read tools reuse the MCP tools
        in-process; `present_plan` emits a plan; `propose_changes` parses SDK
        code, diffs it against the working copy and stores a Proposal
    POST /copilot/proposals/{uuid}/apply        -> saves ONLY accepted changes as a
        draft (optimistic lock) or creates the workflow; publishing stays manual

Guardrails (design doc): deployment-level key + per-org AI opt-in (same as the
Doctor), drafts only, transcripts/tool output are data not instructions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Claude model for the copilot (OpenAI fallback via DOCTOR_LLM_PROVIDER=openai,
#: same switch as the Doctor; see api/services/workflow/doctor/llm.py).
COPILOT_MODEL = "claude-opus-5-5"

#: Max model<->tool round trips for one user message before the loop stops
#: and tells the user it ran out of steps.
MAX_TOOL_ITERATIONS_PER_TURN = 25

#: propose_changes may fail validation; the model may retry this many times
#: per user message (self-fix rounds) before it must report the errors.
MAX_PROPOSAL_FIX_ROUNDS = 2

#: Messages kept in a session transcript sent to the model (oldest dropped,
#: the system prompt and the first user message are always kept).
MAX_TRANSCRIPT_MESSAGES = 80

#: Tool results longer than this are truncated before going to the model.
MAX_TOOL_RESULT_CHARS = 40_000

#: usage_context tag for metering copilot LLM usage.
USAGE_CONTEXT_COPILOT = "copilot"


# ---------------------------------------------------------------------------
# Transcript (provider-neutral; llm.py maps it to Anthropic / OpenAI shapes)
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class TranscriptMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    #: user/assistant text; for role=tool the tool result (stringified JSON)
    content: str = ""
    #: assistant only: tool calls made in this message
    tool_calls: list[ToolCall] = Field(default_factory=list)
    #: tool only: which call this result answers, and the tool name
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    #: tool only: True when the tool failed (result content is the error)
    is_error: bool = False
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Plans and proposals
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    title: str  # e.g. "Add a 'Qualify lead' agent node after the greeting"
    detail: Optional[str] = None


class Plan(BaseModel):
    """Emitted by the `present_plan` tool; the UI renders it with an
    'Approve plan' button (which sends a normal user message)."""

    summary: str
    steps: list[PlanStep]
    #: open questions for the user before building, if any
    questions: list[str] = Field(default_factory=list)


class ChangeTarget(str, Enum):
    node = "node"
    edge = "edge"
    workflow = "workflow"  # workflow-level settings (name, configurations)
    #: a new reusable tool created on apply (M2c); ref_id "t:<name>"
    tool = "tool"


class ChangeOp(str, Enum):
    add = "add"
    update = "update"
    remove = "remove"


class FieldDiff(BaseModel):
    field: str  # dotted path inside node/edge data, e.g. "prompt", "tool_uuids"
    before: Any = None
    after: Any = None


class ChangeItem(BaseModel):
    """One reviewable change. Node changes replace the whole node (its data
    and type); edge changes the whole edge. Positions never count."""

    change_id: str  # stable within a proposal, e.g. "n:3", "e:2-4:end_call"
    target: ChangeTarget
    op: ChangeOp
    #: node id / edge id / "workflow"
    ref_id: str
    #: human label, e.g. 'Update node "Welcome and Source" (prompt)'
    label: str
    before: Optional[dict[str, Any]] = None
    after: Optional[dict[str, Any]] = None
    field_diffs: list[FieldDiff] = Field(default_factory=list)
    #: change_ids that must also be accepted (an added edge needs its added
    #: endpoint nodes; removing a node requires removing its edges)
    depends_on: list[str] = Field(default_factory=list)


class ProposalKind(str, Enum):
    edit = "edit"  # changes to an existing workflow's working copy
    create = "create"  # a brand-new workflow (accepted as a whole)


class ProposalStatus(str, Enum):
    pending = "pending"
    #: an apply is in progress (atomic claim); NOT "reviewed" (M2b O2)
    applying = "applying"
    applied = "applied"
    discarded = "discarded"
    #: the working copy changed since the proposal was made (hash mismatch)
    stale = "stale"


class Proposal(BaseModel):
    proposal_uuid: str
    session_uuid: str
    kind: ProposalKind
    workflow_id: Optional[int] = None  # None for kind=create until applied
    summary: str
    #: compute_definition_hash of the working copy the diff was made against
    base_hash: Optional[str] = None
    #: M2c: hash of the working copy's settings (workflow_configurations +
    #: template_context_variables) when the proposal changes settings; None
    #: for graph-only proposals (their staleness check is unchanged)
    settings_base_hash: Optional[str] = None
    changes: list[ChangeItem] = Field(default_factory=list)
    status: ProposalStatus = ProposalStatus.pending
    #: for create: the workflow name from `new Workflow({ name })`
    workflow_name: Optional[str] = None
    created_at: Optional[str] = None
    #: set when the proposal came from a Self-improve round (M2b)
    iteration_id: Optional[int] = None
    #: after apply: which change_ids were applied and the resulting definition
    accepted_change_ids: list[str] = Field(default_factory=list)
    applied_definition_id: Optional[int] = None
    #: per change_id: the Doctor cluster it targets (next-calls check, M2b O3)
    hypotheses: dict[str, "Hypothesis"] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    """What an accepted change is expected to fix (M2b next-calls check)."""

    cluster_id: Optional[str] = None
    title: str
    category: Optional[str] = None
    #: node name (or "edge:A->B", "kb", ...) the problem was attributed to
    fix_target: Optional[str] = None


# ---------------------------------------------------------------------------
# Workflow brief / North Star (M2b R15/D2; design doc change C)
# ---------------------------------------------------------------------------


class WorkflowBrief(BaseModel):
    goal: str = ""
    must_do: list[str] = Field(default_factory=list)
    must_never: list[str] = Field(default_factory=list)
    accepted_tradeoffs: list[str] = Field(default_factory=list)
    notes: str = ""


class BriefResponse(BaseModel):
    workflow_id: int
    brief: WorkflowBrief
    #: increments on every confirmed change; used as the optimistic lock
    version: int = 0
    updated_at: Optional[str] = None
    updated_by: Optional[Literal["user", "copilot"]] = None
    #: a copilot-proposed change waiting for the user, if any
    pending_change: Optional["BriefChange"] = None


class BriefChangeStatus(str, Enum):
    pending = "pending"
    confirmed = "confirmed"
    dismissed = "dismissed"
    #: the brief changed since this was proposed (version mismatch)
    stale = "stale"


class BriefChange(BaseModel):
    """A copilot-proposed brief update; never applied without the user (O8)."""

    change_uuid: str
    workflow_id: int
    base_version: int
    proposed: WorkflowBrief
    reason: str = ""
    status: BriefChangeStatus = BriefChangeStatus.pending
    created_at: Optional[str] = None


class UpdateBriefRequest(BaseModel):
    brief: WorkflowBrief
    expected_version: int


# ---------------------------------------------------------------------------
# Streaming events (SSE: `event: <type>` + `data: <json>`)
# ---------------------------------------------------------------------------


class CopilotEventType(str, Enum):
    text_delta = "text_delta"  # {text}
    tool_call = "tool_call"  # {id, name, summary}
    tool_result = "tool_result"  # {id, name, ok, summary}
    plan = "plan"  # {plan: Plan}
    proposal = "proposal"  # {proposal: Proposal}
    brief_change = "brief_change"  # {brief_change: BriefChange}
    message_done = "message_done"  # {}  end of the assistant turn
    error = "error"  # {message}


class CopilotEvent(BaseModel):
    type: CopilotEventType
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API shapes
# ---------------------------------------------------------------------------


#: Session kinds: chat sessions are what the panel resumes; self_improve
#: sessions are internal to Self-improve rounds and never listed in chat (RI-1).
SessionKind = Literal["chat", "self_improve"]


class CreateSessionRequest(BaseModel):
    #: None = "Build with Copilot" from /workflow/create (proposals are kind=create)
    workflow_id: Optional[int] = None


class SendMessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class DisplayMessage(BaseModel):
    """What the UI renders for history (tool messages collapsed to chips)."""

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_name: Optional[str] = None
    is_error: bool = False
    created_at: Optional[str] = None


class SessionResponse(BaseModel):
    session_uuid: str
    workflow_id: Optional[int] = None
    title: Optional[str] = None
    messages: list[DisplayMessage] = Field(default_factory=list)
    proposals: list[Proposal] = Field(default_factory=list)
    #: last plan presented in this session, if any
    plan: Optional[Plan] = None
    created_at: str
    updated_at: str


class ApplyProposalRequest(BaseModel):
    #: change_ids to apply (edit). Ignored for create (all-or-nothing).
    accepted_change_ids: list[str] = Field(default_factory=list)
    #: optimistic lock: the working copy's version_number (int) or hash (str)
    expected_draft_version: Optional[int | str] = None


class ApplyProposalResponse(BaseModel):
    saved: bool
    workflow_id: Optional[int] = None
    #: hash of the new working copy after the save
    draft_hash: Optional[str] = None
    #: error_code values: draft_conflict, stale, missing_dependency,
    #: validation_error (incl. an empty or unknown selection), graph_validation,
    #: not_found, already_applied (also: another apply is in progress),
    #: discarded, plus save-path codes passed through: schema_validation,
    #: node_instance_validation, transition_name_validation,
    #: trigger_path_conflict, bridge_error.
    #: (Tool-level errors the model sees from propose_changes: no_changes,
    #: missing_name, not_found, invalid_input, and the save-path codes.)
    error_code: Optional[str] = None
    error: Optional[str] = None
    proposal: Optional[Proposal] = None


# ---------------------------------------------------------------------------
# Tool interface (tools.py implements; llm.py / the agent loop calls)
# ---------------------------------------------------------------------------


class ToolSpec(BaseModel):
    name: str
    description: str
    #: JSON Schema for the tool input (object)
    input_schema: dict[str, Any]
    #: True for tools that only read (safe to run without user review)
    read_only: bool = True


class ToolContext(BaseModel):
    """Who is acting and where. Tools must scope every lookup to
    organization_id (tenant isolation)."""

    model_config = {"arbitrary_types_allowed": True}

    session_uuid: str
    organization_id: int
    user_id: int
    workflow_id: Optional[int] = None
    #: proposal fix rounds used in the current user turn
    fix_rounds_used: int = 0
    #: "self_improve": no present_plan, read-only brief, report bound to the
    #: round, at most one proposal (RI-4, O4, O5, O8)
    mode: Literal["chat", "self_improve"] = "chat"
    #: self_improve only: the round and its Doctor report
    iteration_id: Optional[int] = None
    report_uuid: Optional[str] = None


class ToolOutcome(BaseModel):
    ok: bool = True
    #: what the model sees (JSON-serialisable; truncated by the loop)
    content: Any = None
    #: short human line for the UI chip, e.g. "Read 5 node types"
    summary: str = ""
    #: set by present_plan / propose_changes so the loop can emit events
    plan: Optional[Plan] = None
    proposal: Optional[Proposal] = None
    brief_change: Optional[BriefChange] = None


Proposal.model_rebuild()
BriefResponse.model_rebuild()


# ---------------------------------------------------------------------------
# Settings (M2c): everything the Settings tab can change, as reviewable changes
# ---------------------------------------------------------------------------

#: ChangeItem.ref_id prefixes: settings "s:<path>", new tools "t:<name>".
SETTINGS_REF_PREFIX = "s:"
TOOL_REF_PREFIX = "t:"

#: Placeholder a node's tool_uuids may use for a tool created by the same
#: proposal: "tool:<name>" (replaced by the real uuid on apply).
NEW_TOOL_PLACEHOLDER_PREFIX = "tool:"


class ServiceModelChoice(BaseModel):
    """One service's model choice for this agent (no secrets, ever)."""

    provider: str
    model: Optional[str] = None
    #: TTS only
    voice: Optional[str] = None
    #: provider-specific non-secret extras (e.g. language, speed)
    options: dict[str, Any] = Field(default_factory=dict)


class SettingsPatch(BaseModel):
    """What propose_changes(settings=...) carries. Keys are dotted paths into
    the settings view (see settings.get_settings_view), e.g.
    "max_call_duration", "turn_stop_strategy", "voicemail_detection.enabled",
    "template_variables.company", "models.tts". Value null removes a
    template variable / clears a model override for that service."""

    values: dict[str, Any] = Field(default_factory=dict)
