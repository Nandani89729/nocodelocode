"""Settings engine for the Agent Copilot (milestone 2c).

Lets the copilot read and change everything in a workflow's Settings tab as
reviewable changes, exactly like graph changes:

- `get_settings_view`: current values (masked; never secrets), template
  variables, the effective models (agent override vs organization default)
  and a per-path schema with allowed values/ranges.
- `list_model_options` / `list_voices`: what this organization can pick.
- `plan_settings_changes`: a `SettingsPatch` (dotted paths -> values) becomes
  one `ChangeItem` per changed path plus candidate configurations/variables.
- `merge_accepted_settings`: applies ONLY the accepted paths onto the CURRENT
  working values (unrelated edits made meanwhile survive) and re-validates.
- `settings_hash`: the staleness token for proposals that change settings.

Everything a Settings card saves lives in the draft definition:
`workflow_configurations` (schema `WorkflowConfigurationDefaults`, plus the
`voicemail_detection` / `transcript_configuration` extras and the per-agent
model override under `model_configuration_v2_override`) and
`template_context_variables`.

Model changes are THIS AGENT ONLY: they edit the workflow's v2 model override,
never organization defaults, and never carry API keys from the copilot; keys
are copied server-side from the organization configuration (or the existing
override) for providers that already have one.

Security: `SettingsPlan.candidate_configurations` holds real secrets (the
compiled model override) and a private bookkeeping key
(`CANDIDATE_META_KEY`); it is server-side state for `merge_accepted_settings`
only. Never send it to the model or the UI, and never save it directly.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Annotated, Any, Iterable

from pydantic import Field, TypeAdapter, ValidationError

from api.db import db_client
from api.schemas.ai_model_configuration import (
    DOGRAH_DEFAULT_LANGUAGE,
    DOGRAH_DEFAULT_VOICE,
    DOGRAH_SPEED_MAX,
    DOGRAH_SPEED_MIN,
    EffectiveAIModelConfiguration,
    OrganizationAIModelConfigurationV2,
    compile_ai_model_configuration_v2,
)
from api.schemas.answer_supervisor import AnswerSupervisorConfig
from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.configuration.ai_model_configuration import (
    WORKFLOW_MODEL_CONFIGURATION_V2_OVERRIDE_KEY,
    check_for_masked_keys_in_ai_model_configuration_v2,
    convert_legacy_ai_model_configuration_to_v2,
    get_resolved_ai_model_configuration,
)
from api.services.configuration.masking import SERVICE_SECRET_FIELDS
from api.services.configuration.model_catalog import (
    get_model_configuration_v2_defaults,
)
from api.services.configuration.registry import (
    DOGRAH_STT_LANGUAGES,
    REGISTRY,
    ServiceProviders,
    ServiceType,
)
from api.services.configuration.resolve import (
    enrich_overrides_with_api_keys,
    resolve_effective_config,
)
from api.services.copilot.contracts import (
    SETTINGS_REF_PREFIX,
    ChangeItem,
    ChangeOp,
    ChangeTarget,
    FieldDiff,
    ServiceModelChoice,
    SettingsPatch,
)
from api.services.mps_service_key_client import mps_service_key_client
from api.services.organization_preferences import external_pbx_integrations_enabled
from api.services.workflow.sdk_authoring import compute_definition_hash

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERRIDE_KEY = WORKFLOW_MODEL_CONFIGURATION_V2_OVERRIDE_KEY
LEGACY_OVERRIDE_KEY = "model_overrides"

#: Private key inside `SettingsPlan.candidate_configurations` holding the
#: model-override base (with secrets) so a partial accept can merge per
#: service. Stripped by `merge_accepted_settings`; never persisted.
CANDIDATE_META_KEY = "__copilot_settings_meta__"

TEMPLATE_VARIABLES_PREFIX = "template_variables."
MODELS_PATH = "models"
MODELS_PREFIX = "models."

#: Services the copilot can change per agent. "realtime" only exists when the
#: model configuration is BYOK realtime (speech-to-speech).
MODEL_SERVICES = ("llm", "tts", "stt", "realtime")
_SERVICE_TYPES = {
    "llm": ServiceType.LLM,
    "tts": ServiceType.TTS,
    "stt": ServiceType.STT,
    "realtime": ServiceType.REALTIME,
}
_SERVICE_LABELS = {
    "llm": "Model (LLM)",
    "tts": "Voice (TTS)",
    "stt": "Transcription (STT)",
    "realtime": "Realtime model",
}
_PIPELINE_SERVICES = ("llm", "tts", "stt")
_REALTIME_SERVICES = ("realtime", "llm")

#: Non-secret fields shown for a service (everything else stays server-side).
_PUBLIC_SERVICE_FIELDS = ("provider", "model", "voice", "language", "speed")

#: Option keys the copilot may never set: secrets and anything that decides
#: WHERE a stored key is sent (a changed base_url would leak the key).
_FORBIDDEN_OPTION_FRAGMENTS = ("url", "endpoint", "host", "proxy")
_RESERVED_OPTION_KEYS = {"provider", "model", "voice"}

#: TTS providers with a voice catalog through MPS; mirrors
#: `api.routes.user.TTSProvider` (a test pins the two together).
VOICE_CATALOG_PROVIDERS = (
    "elevenlabs",
    "deepgram",
    "sarvam",
    "cartesia",
    "dograh",
    "rime",
)
VOICE_CACHE_TTL_SECONDS = 600.0
_VOICE_FIELDS = (
    "voice_id",
    "name",
    "description",
    "accent",
    "gender",
    "language",
    "preview_url",
)

_TEMPLATE_VARIABLE_NAME = TypeAdapter(
    Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,99}$")]
)
_MAX_TEMPLATE_VARIABLE_VALUE_LENGTH = 10_000
_DOGRAH_SPEED = TypeAdapter(
    Annotated[float, Field(ge=DOGRAH_SPEED_MIN, le=DOGRAH_SPEED_MAX)]
)

_MISSING = object()


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class SettingsValidationError(Exception):
    """One or more settings paths are invalid. `errors` are plain messages,
    each naming the path, for the model to fix and re-propose."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors) or "invalid settings")


@dataclass
class SettingsPlan:
    changes: list[ChangeItem]
    #: server-side only: real secrets + CANDIDATE_META_KEY (see module doc)
    candidate_configurations: dict
    candidate_template_variables: dict


# ---------------------------------------------------------------------------
# Settings path catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SettingSpec:
    path: str
    label: str
    adapter: TypeAdapter
    description: str = ""
    default: Any = _MISSING
    #: static read-only reason (dynamic gating, e.g. external PBX, is separate)
    read_only: str | None = None
    #: True for top-level WorkflowConfigurationDefaults fields: the view uses
    #: the model-validated (normalized) value
    top_level: bool = False


def _field_adapter(name: str, *extra: Any) -> TypeAdapter:
    info = WorkflowConfigurationDefaults.model_fields[name]
    metadata = [*info.metadata, *extra]
    annotation = (
        Annotated[(info.annotation, *metadata)] if metadata else info.annotation
    )
    return TypeAdapter(annotation)


def _field_default(name: str) -> Any:
    info = WorkflowConfigurationDefaults.model_fields[name]
    value = info.get_default(call_default_factory=True)
    return TypeAdapter(info.annotation).dump_python(value, mode="json")


def _top(
    name: str,
    label: str,
    *extra: Any,
    description: str = "",
    read_only: str | None = None,
) -> _SettingSpec:
    info = WorkflowConfigurationDefaults.model_fields[name]
    return _SettingSpec(
        path=name,
        label=label,
        adapter=_field_adapter(name, *extra),
        description=description or info.description or "",
        default=_field_default(name),
        read_only=read_only,
        top_level=True,
    )


def _answer_field(name: str, label: str, *extra: Any, read_only: str | None = None):
    info = AnswerSupervisorConfig.model_fields[name]
    metadata = [*info.metadata, *extra]
    annotation = (
        Annotated[(info.annotation, *metadata)] if metadata else info.annotation
    )
    default = info.get_default(call_default_factory=True)
    return _SettingSpec(
        path=f"voicemail_detection.{name}",
        label=label,
        adapter=TypeAdapter(annotation),
        description=info.description or "",
        default=TypeAdapter(info.annotation).dump_python(default, mode="json"),
        read_only=read_only,
    )


_RO_UPLOAD = "Set by uploading an audio file in the Settings tab."
_RO_RECORDING = "Pick a recording in the Settings tab (Voicemail & Screening)."
_RO_NOT_IN_UI = "Not editable in the Settings tab."
_RO_LISTENING = "Controlled by the Start node's Delayed Start setting."
_RO_VM_PROVIDER = (
    "The classifier provider goes with its API key, which can only be entered "
    "in the Settings tab."
)
_RO_EXTERNAL_PBX = (
    "External PBX integrations are disabled for this organization (enable them "
    "in Platform Settings first)."
)
_EXTERNAL_PBX_PATHS = ("external_pbx_field_mappings", "external_pbx_lead_headers")


def _build_specs() -> dict[str, _SettingSpec]:
    specs = [
        # General: call management
        _top(
            "max_call_duration",
            "Max call duration",
            description="Hard limit for one call, in seconds.",
        ),
        _top(
            "max_user_idle_timeout",
            "Max user idle timeout",
            Field(ge=1),
            description="Seconds of caller silence before the agent ends the call.",
        ),
        # General: turn detection / interruption
        _top(
            "turn_stop_strategy",
            "Turn detection strategy",
            description=(
                "How the agent decides the caller finished speaking: "
                "'transcription' (best for short answers) or 'turn_analyzer' "
                "(ML model, best for longer answers with pauses)."
            ),
        ),
        _top(
            "smart_turn_stop_secs",
            "Incomplete turn timeout",
            Field(ge=0.5, le=10),
            description=(
                "Max silence (seconds) before ending an incomplete turn. Used "
                "with turn_stop_strategy 'turn_analyzer'."
            ),
        ),
        _top(
            "turn_start_strategy",
            "Interruption strategy",
            description=(
                "When caller speech interrupts the agent: 'default' (platform "
                "default) or 'min_words' (wait for turn_start_min_words words)."
            ),
        ),
        _top(
            "turn_start_min_words",
            "Minimum words before interruption",
            Field(ge=1, le=10),
            description="Used with turn_start_strategy 'min_words'.",
        ),
        _top(
            "context_compaction_enabled",
            "Context compaction",
            description=(
                "Summarize context on node transitions (ignored in realtime mode)."
            ),
        ),
        _top("tts_cache_enabled", "Speech caching"),
        _top("call_dispositions", "Call dispositions"),
        _top(
            "dictionary",
            "Dictionary",
            description=(
                "Comma-separated words the speech recognizer should listen for "
                "(jargon, names). May add provider cost."
            ),
        ),
        _top(
            "text_chat_inactivity_timeout_seconds",
            "Text chat inactivity timeout",
            description="Website chat widget: end inactive text chats after this many seconds.",
        ),
        _top("external_pbx_field_mappings", "External PBX field mappings"),
        _top("external_pbx_lead_headers", "External PBX lead fields"),
        # General: ambient noise
        _SettingSpec(
            path="ambient_noise_configuration.enabled",
            label="Ambient noise",
            adapter=TypeAdapter(bool),
            description="Play background ambient noise during calls.",
            default=False,
        ),
        _SettingSpec(
            path="ambient_noise_configuration.volume",
            label="Ambient noise volume",
            adapter=TypeAdapter(Annotated[float, Field(ge=0, le=1)]),
            description="Ambient noise volume, 0 to 1.",
            default=0.3,
        ),
        _SettingSpec(
            path="ambient_noise_configuration.original_filename",
            label="Ambient noise audio file",
            adapter=TypeAdapter(str | None),
            description="Custom ambient audio file name.",
            default=None,
            read_only=_RO_UPLOAD,
        ),
        _SettingSpec(
            path="ambient_noise_configuration.storage_key",
            label="Ambient noise audio storage key",
            adapter=TypeAdapter(str | None),
            default=None,
            read_only=_RO_UPLOAD,
        ),
        # General: transcript
        _SettingSpec(
            path="transcript_configuration.include_end_timestamps",
            label="Enhanced timestamped transcript",
            adapter=TypeAdapter(bool),
            description="Include start and stop timestamps per speaker in the transcript.",
            default=False,
        ),
        # Voicemail & screening
        _SettingSpec(
            path="voicemail_detection.enabled",
            label="Voicemail and screening handling",
            adapter=TypeAdapter(bool),
            description=(
                "Detect voicemail / call screening on outbound calls (cascade "
                "pipelines only)."
            ),
            default=False,
        ),
        _answer_field("voicemail_action", "When voicemail is detected"),
        _SettingSpec(
            path="voicemail_detection.voicemail_message.text",
            label="Voicemail message",
            adapter=TypeAdapter(Annotated[str, Field(max_length=5000)]),
            description="Message left when voicemail_action is 'leave_message'.",
            default="",
        ),
        _SettingSpec(
            path="voicemail_detection.voicemail_message.recording_id",
            label="Voicemail message recording",
            adapter=TypeAdapter(str | None),
            default=None,
            read_only=_RO_RECORDING,
        ),
        _SettingSpec(
            path="voicemail_detection.screening_message.text",
            label="Screening message",
            adapter=TypeAdapter(Annotated[str, Field(max_length=5000)]),
            description=(
                "Caller introduction for call screeners (name and reason for "
                "calling); empty ends screened calls."
            ),
            default="",
        ),
        _SettingSpec(
            path="voicemail_detection.screening_message.recording_id",
            label="Screening message recording",
            adapter=TypeAdapter(str | None),
            default=None,
            read_only=_RO_RECORDING,
        ),
        _answer_field(
            "screening_wait_ms",
            "Screening wait (ms)",
            Field(ge=1000, le=60000),
        ),
        _SettingSpec(
            path="voicemail_detection.use_workflow_llm",
            label="Voicemail classifier uses workflow LLM",
            adapter=TypeAdapter(bool),
            description=(
                "Classify answers with the agent's own LLM. False needs a "
                "separate classifier API key entered in the Settings tab."
            ),
            default=True,
        ),
        _SettingSpec(
            path="voicemail_detection.provider",
            label="Voicemail classifier provider",
            adapter=TypeAdapter(str | None),
            default="openai",
            read_only=_RO_VM_PROVIDER,
        ),
        _SettingSpec(
            path="voicemail_detection.model",
            label="Voicemail classifier model",
            adapter=TypeAdapter(Annotated[str, Field(min_length=1, max_length=200)]),
            description="Classifier model when use_workflow_llm is false.",
            default="gpt-4.1",
        ),
        _SettingSpec(
            path="voicemail_detection.api_key_configured",
            label="Voicemail classifier API key configured",
            adapter=TypeAdapter(bool),
            description="Whether a separate classifier key is stored (value never shown).",
            default=False,
            read_only="API keys are entered only in the Settings tab.",
        ),
        _SettingSpec(
            path="voicemail_detection.system_prompt",
            label="Voicemail classifier instructions",
            adapter=TypeAdapter(Annotated[str, Field(min_length=1, max_length=20000)]),
            description=(
                "Overrides the built-in classifier instructions; null restores "
                "the built-in text."
            ),
            default=None,
        ),
        _answer_field(
            "listening_window_ms", "Listening window (ms)", read_only=_RO_LISTENING
        ),
        _answer_field(
            "human_utterance_max_ms",
            "Human utterance max (ms)",
            read_only=_RO_NOT_IN_UI,
        ),
        _answer_field(
            "machine_utterance_cap_ms",
            "Machine utterance cap (ms)",
            read_only=_RO_NOT_IN_UI,
        ),
        _answer_field(
            "classify_budget_ms", "Classify budget (ms)", read_only=_RO_NOT_IN_UI
        ),
        _answer_field(
            "max_screening_rearms", "Max screening attempts", read_only=_RO_NOT_IN_UI
        ),
    ]
    return {spec.path: spec for spec in specs}


SETTINGS_SPECS: dict[str, _SettingSpec] = _build_specs()

#: view-only paths computed from stored values (never written)
_DERIVED_PATHS = {"voicemail_detection.api_key_configured"}

_SCHEMA_KEYS = (
    "type",
    "enum",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "items",
    "properties",
    "required",
    "anyOf",
)


def _inline_refs(node: Any, defs: dict) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return _inline_refs(defs.get(name, {}), defs)
        return {
            key: _inline_refs(value, defs)
            for key, value in node.items()
            if key not in ("$defs", "title")
        }
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    return node


def _spec_schema(spec: _SettingSpec, *, read_only: str | None) -> dict:
    raw = spec.adapter.json_schema()
    raw = _inline_refs(raw, raw.get("$defs", {}))
    entry: dict[str, Any] = {k: raw[k] for k in _SCHEMA_KEYS if k in raw}
    # Integers: express "> 0" as "minimum: 1" (what the model should use).
    if entry.get("type") == "integer" and "exclusiveMinimum" in entry:
        floor = int(entry.pop("exclusiveMinimum")) + 1
        entry["minimum"] = max(entry.get("minimum", floor), floor)
    entry["description"] = spec.description or spec.label
    entry["label"] = spec.label
    if spec.default is not _MISSING:
        entry["default"] = spec.default
    if read_only:
        entry["read_only"] = True
        entry["read_only_reason"] = read_only
    return entry


def _extra_schema_entries() -> dict[str, dict]:
    services = ", ".join(f"models.{s}" for s in MODEL_SERVICES)
    return {
        "template_variables.<name>": {
            "type": "string",
            "description": (
                "Template variable available in prompts as {{name}}. New names: "
                "letters, digits and underscores, not starting with a digit. "
                "null removes the variable."
            ),
        },
        "models.<service>": {
            "type": "object",
            "description": (
                f"This agent's model for one service ({services}); value "
                "{provider, model?, voice? (tts/realtime), options?}. Only "
                "providers listed by list_model_options are accepted (API keys "
                "are reused server-side, never passed). null resets that service "
                "to the organization default."
            ),
        },
        MODELS_PATH: {
            "type": "null",
            "description": (
                "null removes this agent's whole model override (all services "
                "use the organization defaults)."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Small path helpers
# ---------------------------------------------------------------------------


def _get_path(data: dict | None, path: str, default: Any = _MISSING) -> Any:
    node: Any = data or {}
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _set_path(data: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _del_path(data: dict, path: str) -> None:
    parts = path.split(".")
    node: Any = data
    for part in parts[:-1]:
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def _format_errors(exc: ValidationError, prefix: str) -> list[str]:
    """Plain messages naming the path; never echoes input values (secrets)."""
    messages = []
    for error in exc.errors(include_input=False, include_url=False):
        loc = ".".join(str(part) for part in error.get("loc", ()) if part != "")
        path = ".".join(part for part in (prefix, loc) if part)
        msg = error.get("msg", "invalid value")
        messages.append(f"{path}: {msg}" if path else msg)
    return messages or [f"{prefix}: invalid value"]


def _short(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return f"{len(value)} item" + ("" if len(value) == 1 else "s")
    if isinstance(value, dict):
        return "{…}"
    text = str(value).replace("\n", " ")
    return text if len(text) <= 40 else text[:39] + "…"


def _label_for_path(path: str) -> str:
    spec = SETTINGS_SPECS.get(path)
    if spec is not None:
        return spec.label
    return path.replace("_", " ").replace(".", " / ").capitalize()


# ---------------------------------------------------------------------------
# Working copy
# ---------------------------------------------------------------------------


async def load_working_settings(workflow: Any) -> tuple[dict, dict]:
    """(workflow_configurations, template_context_variables) of the working
    copy, chosen like `sdk_authoring.select_working_copy` (draft > published
    > legacy columns). Raw values (with secrets): server-side only."""
    draft = await db_client.get_draft_version(workflow.id)
    if draft is not None and draft.workflow_json:
        source = draft
    elif (
        getattr(workflow, "released_definition", None) is not None
        and workflow.released_definition.workflow_json
    ):
        source = workflow.released_definition
    else:
        source = workflow
    return (
        copy.deepcopy(getattr(source, "workflow_configurations", None) or {}),
        copy.deepcopy(getattr(source, "template_context_variables", None) or {}),
    )


def settings_hash(configurations: dict | None, template_variables: dict | None) -> str:
    """Canonical sha256 of a working copy's settings (staleness token)."""
    return compute_definition_hash(
        {
            "workflow_configurations": configurations or {},
            "template_context_variables": template_variables or {},
        }
    )


# ---------------------------------------------------------------------------
# Effective values (view and diff baseline)
# ---------------------------------------------------------------------------


def _validated_top_level(configurations: dict) -> dict:
    try:
        return WorkflowConfigurationDefaults.model_validate(configurations).model_dump(
            mode="json"
        )
    except ValidationError:
        return {}


def _effective_value(
    spec: _SettingSpec, configurations: dict, validated: dict | None = None
) -> Any:
    if spec.path == "voicemail_detection.api_key_configured":
        return bool(_get_path(configurations, "voicemail_detection.api_key", None))
    if spec.top_level and validated and spec.path in validated:
        return validated[spec.path]
    value = _get_path(configurations, spec.path, None)
    if value is None:
        return None if spec.default is _MISSING else copy.deepcopy(spec.default)
    return value


def _read_only_reason(spec: _SettingSpec, *, external_pbx_enabled: bool) -> str | None:
    if spec.read_only:
        return spec.read_only
    if spec.path in _EXTERNAL_PBX_PATHS and not external_pbx_enabled:
        return _RO_EXTERNAL_PBX
    return None


# ---------------------------------------------------------------------------
# Model configuration helpers (dict level; secrets stay inside)
# ---------------------------------------------------------------------------


def _provider_value(provider: Any) -> str:
    return str(getattr(provider, "value", provider))


def _registry(service: str) -> dict[str, type]:
    return {
        _provider_value(provider): cls
        for provider, cls in REGISTRY[_SERVICE_TYPES[service]].items()
    }


def _provider_title(service: str, provider: str) -> str:
    cls = _registry(service).get(provider)
    title = cls.model_config.get("title") if cls is not None else None
    return title or provider


def _config_services(config: dict) -> tuple[str, ...]:
    if config.get("mode") == "dograh":
        return _PIPELINE_SERVICES
    byok = config.get("byok") or {}
    return _REALTIME_SERVICES if byok.get("mode") == "realtime" else _PIPELINE_SERVICES


def _config_shape(config: dict | None) -> tuple[str | None, str | None]:
    if not config:
        return None, None
    if config.get("mode") == "dograh":
        return "dograh", None
    return "byok", (config.get("byok") or {}).get("mode")


def _get_section(config: dict, service: str) -> dict | None:
    """One service's provider section (dict, may include secrets)."""
    if config.get("mode") == "dograh":
        managed = config.get("dograh") or {}
        dograh = ServiceProviders.DOGRAH.value
        if service == "llm":
            return {"provider": dograh, "model": "default"}
        if service == "tts":
            return {
                "provider": dograh,
                "model": "default",
                "voice": managed.get("voice", DOGRAH_DEFAULT_VOICE),
                "speed": managed.get("speed", 1.0),
            }
        if service == "stt":
            return {
                "provider": dograh,
                "model": "default",
                "language": managed.get("language", DOGRAH_DEFAULT_LANGUAGE),
            }
        return None
    byok = config.get("byok") or {}
    mode = byok.get("mode")
    if service not in _config_services(config):
        return None
    container = byok.get(mode) or {}
    section = container.get(service)
    return section if isinstance(section, dict) else None


def _set_section(config: dict, service: str, section: dict) -> None:
    if config.get("mode") == "dograh":
        managed = config.setdefault("dograh", {})
        if service == "tts":
            managed["voice"] = section.get("voice", DOGRAH_DEFAULT_VOICE)
            managed["speed"] = section.get("speed", 1.0)
        elif service == "stt":
            managed["language"] = section.get("language", DOGRAH_DEFAULT_LANGUAGE)
        return
    byok = config.setdefault("byok", {})
    container = byok.setdefault(byok.get("mode") or "pipeline", {})
    container[service] = copy.deepcopy(section)


def _public(section: dict | None) -> dict | None:
    if not section:
        return None
    return {
        key: _provider_value(section[key]) if key == "provider" else section[key]
        for key in _PUBLIC_SERVICE_FIELDS
        if section.get(key) is not None
    }


def _public_from_service(service_cfg: Any) -> dict | None:
    if service_cfg is None:
        return None
    return _public(service_cfg.model_dump(mode="json"))


def _iter_sections(config: dict | None) -> Iterable[tuple[str, dict]]:
    if not config:
        return
    for service in ("llm", "tts", "stt", "realtime", "embeddings"):
        if config.get("mode") == "dograh":
            continue
        byok = config.get("byok") or {}
        container = byok.get(byok.get("mode")) or {}
        section = container.get(service)
        if isinstance(section, dict) and section.get("provider"):
            yield service, section


def _required_secret_fields(cls: type) -> list[str]:
    return [
        name
        for name in SERVICE_SECRET_FIELDS
        if name in cls.model_fields and cls.model_fields[name].is_required()
    ]


def _has_secret(section: dict) -> bool:
    return any(section.get(name) for name in SERVICE_SECRET_FIELDS)


def _find_key_source(
    provider: str, service: str, key_sources: list[dict]
) -> tuple[dict, bool] | None:
    """A section of `provider` holding its key: (section, same_service)."""
    cls = _registry(service).get(provider)
    if cls is None:
        return None
    fallback: dict | None = None
    for config in key_sources:
        for section_service, section in _iter_sections(config):
            if _provider_value(section.get("provider")) != provider:
                continue
            if not _has_secret(section):
                continue
            if section_service == service:
                return section, True
            required = _required_secret_fields(cls)
            if fallback is None and all(section.get(name) for name in required):
                fallback = section
    return (fallback, False) if fallback is not None else None


def _byok_available_providers(service: str, key_sources: list[dict]) -> list[str]:
    available = []
    for provider in _registry(service):
        if provider == ServiceProviders.DOGRAH.value:
            continue
        if _find_key_source(provider, service, key_sources) is not None:
            available.append(provider)
    return available


def _field_choices(cls: type, field_name: str) -> dict:
    schema = cls.model_json_schema().get("properties", {}).get(field_name)
    if not schema:
        return {}
    values = schema.get("enum") or schema.get("examples") or []
    return {
        "values": list(values),
        "default": schema.get("default"),
        "allow_custom": bool(schema.get("allow_custom_input")) or not values,
    }


def _is_forbidden_option(key: str) -> bool:
    lowered = key.lower()
    return key in SERVICE_SECRET_FIELDS or any(
        fragment in lowered for fragment in _FORBIDDEN_OPTION_FRAGMENTS
    )


def _check_choice_value(
    service: str, provider: str, field_name: str, value: str, errors: list[str]
) -> None:
    cls = _registry(service)[provider]
    choices = _field_choices(cls, field_name)
    if choices and not choices["allow_custom"] and value not in choices["values"]:
        errors.append(
            f"models.{service}.{field_name}: '{value}' is not offered for "
            f"{_provider_title(service, provider)}; choose one of {choices['values']}"
        )


def _build_byok_section(
    *,
    service: str,
    choice: ServiceModelChoice,
    current: dict | None,
    key_sources: list[dict],
    errors: list[str],
) -> dict | None:
    path = f"models.{service}"
    provider = choice.provider
    if provider == ServiceProviders.DOGRAH.value:
        errors.append(
            f"{path}: this agent uses your own provider keys (BYOK); switching to "
            "Dograh-managed models is done in the Settings tab."
        )
        return None
    registry = _registry(service)
    cls = registry.get(provider)
    if cls is None:
        errors.append(
            f"{path}: unknown provider '{provider}'. Available: "
            f"{_byok_available_providers(service, key_sources)}"
        )
        return None

    if current and _provider_value(current.get("provider")) == provider:
        section = copy.deepcopy(current)
    else:
        found = _find_key_source(provider, service, key_sources)
        if found is None:
            errors.append(
                f"{path}: provider '{provider}' has no API key configured for "
                "this organization, so it cannot be selected here (keys are "
                "added in Model Configurations). Available: "
                f"{_byok_available_providers(service, key_sources)}"
            )
            return None
        source, same_service = found
        if same_service:
            section = copy.deepcopy(source)
        else:
            # Same provider used by another service: reuse its key and the
            # connection settings that go with it (endpoint, region, ...).
            section = {"provider": provider}
            for name, value in source.items():
                if name in ("provider", "model", "voice", "language", "speed"):
                    continue
                if name in cls.model_fields:
                    section[name] = copy.deepcopy(value)
        section["provider"] = provider

    if choice.model is not None:
        if "model" not in cls.model_fields:
            errors.append(f"{path}.model: {provider} has no model setting")
        else:
            _check_choice_value(service, provider, "model", choice.model, errors)
            section["model"] = choice.model
    if choice.voice is not None:
        if "voice" not in cls.model_fields:
            errors.append(f"{path}.voice: voice applies to TTS/realtime providers only")
        else:
            _check_choice_value(service, provider, "voice", choice.voice, errors)
            section["voice"] = choice.voice
    for key, value in choice.options.items():
        if key in _RESERVED_OPTION_KEYS:
            errors.append(f"{path}.options.{key}: set '{key}' directly, not in options")
        elif _is_forbidden_option(key):
            errors.append(
                f"{path}.options.{key}: secrets and endpoints can only be changed "
                "in the Settings tab"
            )
        elif key not in cls.model_fields:
            errors.append(f"{path}.options.{key}: unknown option for {provider}")
        else:
            section[key] = value

    try:
        validated = cls.model_validate(section)
    except ValidationError as exc:
        errors.extend(_format_errors(exc, path))
        return None
    return validated.model_dump(mode="json", exclude_none=True)


def _build_dograh_section(
    *, service: str, choice: ServiceModelChoice, current: dict, errors: list[str]
) -> dict | None:
    path = f"models.{service}"
    if choice.provider != ServiceProviders.DOGRAH.value:
        errors.append(
            f"{path}: this agent uses Dograh-managed models; other providers need "
            "your own keys (BYOK), set up in Model Configurations / the Settings tab."
        )
        return None
    if choice.model not in (None, "default"):
        errors.append(f"{path}.model: Dograh-managed models have one tier ('default')")
        return None
    section = copy.deepcopy(current)
    allowed_options = {"tts": {"speed"}, "stt": {"language"}}.get(service, set())
    if choice.voice is not None:
        if service != "tts":
            errors.append(f"{path}.voice: voice applies to TTS only")
            return None
        section["voice"] = choice.voice
    for key, value in choice.options.items():
        if key not in allowed_options:
            errors.append(
                f"{path}.options.{key}: not adjustable for Dograh-managed {service}"
                + (f" (allowed: {sorted(allowed_options)})" if allowed_options else "")
            )
            return None
        section[key] = value
    if service == "tts":
        try:
            section["speed"] = _DOGRAH_SPEED.validate_python(section.get("speed", 1.0))
        except ValidationError as exc:
            errors.extend(_format_errors(exc, f"{path}.options.speed"))
            return None
    if service == "stt" and section.get("language") not in DOGRAH_STT_LANGUAGES:
        errors.append(
            f"{path}.options.language: unsupported language "
            f"'{section.get('language')}' (use 'multi' for auto-detect)"
        )
        return None
    return section


def _legacy_override_to_v2(
    model_overrides: dict, org_effective: EffectiveAIModelConfiguration
) -> dict:
    """Same conversion `PUT /workflow/{id}` applies to a legacy v1 override."""
    enriched = enrich_overrides_with_api_keys(model_overrides, org_effective)
    effective = resolve_effective_config(org_effective, enriched)
    return convert_legacy_ai_model_configuration_to_v2(effective).model_dump(
        mode="json", exclude_none=True
    )


def _validate_override(override: dict, errors: list[str]) -> None:
    try:
        parsed = OrganizationAIModelConfigurationV2.model_validate(override)
        check_for_masked_keys_in_ai_model_configuration_v2(parsed)
    except ValidationError as exc:
        errors.extend(_format_errors(exc, MODELS_PATH))
    except ValueError as exc:
        errors.append(f"{MODELS_PATH}: {exc}")


# ---------------------------------------------------------------------------
# Read: settings view, model options, voices
# ---------------------------------------------------------------------------


def _models_view(
    configurations: dict, resolved: Any
) -> tuple[dict[str, dict], dict[str, Any]]:
    org_effective: EffectiveAIModelConfiguration = resolved.effective
    override = configurations.get(OVERRIDE_KEY)
    legacy = configurations.get(LEGACY_OVERRIDE_KEY)
    effective = org_effective
    has_override = False
    note = None
    if override:
        try:
            effective = compile_ai_model_configuration_v2(
                OrganizationAIModelConfigurationV2.model_validate(override)
            )
            has_override = True
        except (ValidationError, ValueError):
            note = "The saved agent model override is invalid; re-save it in Settings."
    elif isinstance(legacy, dict) and legacy:
        try:
            effective = resolve_effective_config(org_effective, legacy)
            has_override = True
            note = "This agent uses a legacy model override."
        except (ValidationError, ValueError, TypeError):
            note = "The saved legacy model override is invalid."

    services = ("realtime", "llm") if effective.is_realtime else _PIPELINE_SERVICES
    models: dict[str, dict] = {}
    for service in services:
        public = _public_from_service(getattr(effective, service, None))
        if public is None:
            continue
        org_public = _public_from_service(getattr(org_effective, service, None))
        public["source"] = (
            "agent_override"
            if has_override and public != org_public
            else "organization_default"
        )
        models[service] = public

    org_config = resolved.organization_configuration
    mode, byok_mode = (
        _config_shape(org_config.model_dump(mode="json"))
        if org_config
        else (None, None)
    )
    if override and has_override:
        mode, byok_mode = _config_shape(override)
    info: dict[str, Any] = {
        "mode": mode,
        "byok_mode": byok_mode,
        "has_agent_override": has_override,
        "organization_configured": org_config is not None,
    }
    if note:
        info["note"] = note
    return models, info


async def get_settings_view(*, workflow: Any, organization_id: int) -> dict:
    """What the copilot sees about the Settings tab (never secrets)."""
    configurations, template_variables = await load_working_settings(workflow)
    external_pbx_enabled = await external_pbx_integrations_enabled(organization_id)
    resolved = await get_resolved_ai_model_configuration(
        organization_id=organization_id
    )

    validated = _validated_top_level(configurations)
    settings: dict[str, Any] = {}
    schema: dict[str, dict] = {}
    for spec in SETTINGS_SPECS.values():
        read_only = _read_only_reason(spec, external_pbx_enabled=external_pbx_enabled)
        value = _effective_value(spec, configurations, validated)
        if read_only and value is None:
            # Read-only and unset (e.g. no custom audio): nothing to show.
            schema[spec.path] = _spec_schema(spec, read_only=read_only)
            continue
        _set_path(settings, spec.path, value)
        schema[spec.path] = _spec_schema(spec, read_only=read_only)
    schema.update(_extra_schema_entries())

    models, model_mode = _models_view(configurations, resolved)
    return {
        "settings": settings,
        "template_variables": copy.deepcopy(template_variables),
        "models": models,
        "model_mode": model_mode,
        "schema": schema,
    }


def _provider_option(service: str, provider: str, *, current: bool) -> dict:
    cls = _registry(service)[provider]
    option: dict[str, Any] = {
        "provider": provider,
        "title": _provider_title(service, provider),
        "current_org_default": current,
    }
    model = _field_choices(cls, "model")
    if model:
        option["models"] = model["values"]
        option["default_model"] = model["default"]
        option["allow_custom_model"] = model["allow_custom"]
    if "voice" in cls.model_fields:
        voice = _field_choices(cls, "voice")
        option["voices"] = voice.get("values", [])
        option["default_voice"] = voice.get("default")
        option["allow_custom_voice"] = voice.get("allow_custom", True)
        option["voice_catalog"] = (
            service == "tts" and provider in VOICE_CATALOG_PROVIDERS
        )
    return option


async def list_model_options(*, organization_id: int) -> dict:
    """Per service: providers this org can use (dograh-managed vs BYOK with a
    key configured) and their models/voices. No secrets."""
    resolved = await get_resolved_ai_model_configuration(
        organization_id=organization_id
    )
    org_config = resolved.organization_configuration
    if org_config is None:
        return {
            "mode": None,
            "byok_mode": None,
            "services": {},
            "note": (
                "This organization has no model configuration yet; set one up in "
                "Model Configurations first."
            ),
        }
    config = org_config.model_dump(mode="json", exclude_none=True)
    mode, byok_mode = _config_shape(config)
    services: dict[str, list[dict]] = {}
    if mode == "dograh":
        managed = get_model_configuration_v2_defaults()["dograh"]
        current_tts = _get_section(config, "tts") or {}
        current_stt = _get_section(config, "stt") or {}
        dograh = ServiceProviders.DOGRAH.value
        services = {
            "llm": [
                {
                    "provider": dograh,
                    "title": "Dograh",
                    "models": ["default"],
                    "current_org_default": True,
                }
            ],
            "tts": [
                {
                    "provider": dograh,
                    "title": "Dograh",
                    "models": ["default"],
                    "voices": managed["voices"],
                    "allow_custom_voice": managed["allow_custom_input"],
                    "voice_catalog": True,
                    "current_voice": current_tts.get("voice"),
                    "options": {
                        "speed": {
                            **managed["speed_range"],
                            "suggested": managed["speeds"],
                            "current": current_tts.get("speed"),
                        }
                    },
                    "current_org_default": True,
                }
            ],
            "stt": [
                {
                    "provider": dograh,
                    "title": "Dograh",
                    "models": ["default"],
                    "options": {
                        "language": {
                            "values": list(managed["languages"]),
                            "multilingual_autodetect": list(
                                managed["multilingual_languages"]
                            ),
                            "current": current_stt.get("language"),
                        }
                    },
                    "current_org_default": True,
                }
            ],
        }
        note = (
            "Dograh-managed: one model tier; the voice, speech speed and "
            "transcription language can be changed per agent."
        )
    else:
        for service in _config_services(config):
            current_provider = _provider_value(
                (_get_section(config, service) or {}).get("provider")
            )
            services[service] = [
                _provider_option(
                    service, provider, current=provider == current_provider
                )
                for provider in _byok_available_providers(service, [config])
            ]
        note = (
            "BYOK: only providers with an API key already configured for this "
            "organization are listed; keys are reused server-side."
        )
    return {"mode": mode, "byok_mode": byok_mode, "services": services, "note": note}


_voice_cache: dict[tuple[int, str], tuple[float, list[dict]]] = {}


def clear_voice_cache() -> None:
    _voice_cache.clear()


async def list_voices(*, organization_id: int, provider: str, user: Any) -> list[dict]:
    """Voices for a TTS provider via MPS (same call as the voices route),
    cached per (organization, provider) for 10 minutes."""
    if provider not in VOICE_CATALOG_PROVIDERS:
        raise SettingsValidationError(
            [
                f"provider: no voice catalog for '{provider}'. Supported: "
                f"{list(VOICE_CATALOG_PROVIDERS)}"
            ]
        )
    key = (organization_id, provider)
    now = time.monotonic()
    cached = _voice_cache.get(key)
    if cached is not None and now - cached[0] < VOICE_CACHE_TTL_SECONDS:
        return copy.deepcopy(cached[1])
    result = await mps_service_key_client.get_voices(
        provider=provider,
        model=None,
        language=None,
        q=None,
        gender=None,
        accent=None,
        organization_id=organization_id,
        created_by=getattr(user, "provider_id", None),
    )
    voices = [
        {name: voice.get(name) for name in _VOICE_FIELDS if voice.get(name) is not None}
        for voice in (result or {}).get("voices", [])
        if isinstance(voice, dict) and voice.get("voice_id")
    ]
    _voice_cache[key] = (now, voices)
    return copy.deepcopy(voices)


# ---------------------------------------------------------------------------
# Write: plan, merge
# ---------------------------------------------------------------------------


def _change(
    path: str,
    op: ChangeOp,
    label: str,
    before: Any,
    after: Any,
    field_diffs: list[FieldDiff] | None = None,
) -> ChangeItem:
    ref = f"{SETTINGS_REF_PREFIX}{path}"
    return ChangeItem(
        change_id=ref,
        target=ChangeTarget.workflow,
        op=op,
        ref_id=ref,
        label=label,
        before=None if op == ChangeOp.add else {"path": path, "value": before},
        after=None if op == ChangeOp.remove else {"path": path, "value": after},
        field_diffs=field_diffs or [FieldDiff(field=path, before=before, after=after)],
    )


def _validate_voicemail(configurations: dict, errors: list[str]) -> None:
    voicemail = configurations.get("voicemail_detection")
    if voicemail is None:
        return
    if not isinstance(voicemail, dict):
        errors.append("voicemail_detection: must be an object")
        return
    try:
        AnswerSupervisorConfig.model_validate(voicemail)
    except ValidationError as exc:
        errors.extend(_format_errors(exc, "voicemail_detection"))
        return
    if not voicemail.get("enabled"):
        return
    if voicemail.get("use_workflow_llm", True) is False and not voicemail.get(
        "api_key"
    ):
        errors.append(
            "voicemail_detection.use_workflow_llm: a separate classifier model "
            "needs its own API key, which can only be entered in the Settings "
            "tab; keep use_workflow_llm true or add the key there first."
        )
    if voicemail.get("voicemail_action") == "leave_message":
        message = voicemail.get("voicemail_message") or {}
        if not (
            str(message.get("text") or "").strip()
            or message.get("recording_id")
            or message.get("recording_pk")
        ):
            errors.append(
                "voicemail_detection.voicemail_message.text: enter a voicemail "
                "message (or pick a recording in Settings) when voicemail_action "
                "is 'leave_message'."
            )


def _validate_configurations(configurations: dict, errors: list[str]) -> None:
    try:
        WorkflowConfigurationDefaults.model_validate(configurations)
    except ValidationError as exc:
        errors.extend(_format_errors(exc, ""))
    _validate_voicemail(configurations, errors)
    transcript = configurations.get("transcript_configuration")
    if transcript is not None and not isinstance(transcript, dict):
        errors.append("transcript_configuration: must be an object")
    override = configurations.get(OVERRIDE_KEY)
    if override:
        _validate_override(override, errors)


def _normalize_top_level(candidate: dict, paths: list[str]) -> None:
    """Store the schema-normalized value (e.g. stripped codes) for top-level
    fields touched by the patch."""
    try:
        validated = WorkflowConfigurationDefaults.model_validate(candidate)
    except ValidationError:
        return
    dumped = validated.model_dump(mode="json")
    for path in paths:
        if path in candidate and path in dumped:
            candidate[path] = dumped[path]


async def plan_settings_changes(
    *,
    workflow: Any,
    organization_id: int,
    patch: SettingsPatch,
    working_configurations: dict | None,
    working_template_variables: dict | None,
) -> SettingsPlan:
    """Apply `patch` onto copies of the working values, validate, and return
    one ChangeItem per changed path (unchanged paths are skipped)."""
    if working_configurations is None or working_template_variables is None:
        loaded_configurations, loaded_variables = await load_working_settings(workflow)
        if working_configurations is None:
            working_configurations = loaded_configurations
        if working_template_variables is None:
            working_template_variables = loaded_variables
    base_configurations = copy.deepcopy(working_configurations or {})
    base_variables = copy.deepcopy(working_template_variables or {})
    candidate = copy.deepcopy(base_configurations)
    candidate_variables = copy.deepcopy(base_variables)

    errors: list[str] = []
    changes: list[ChangeItem] = []
    model_paths: list[tuple[str, Any]] = []
    touched_top_level: list[str] = []
    external_pbx_enabled: bool | None = None
    validated_before = _validated_top_level(base_configurations)

    for path, value in patch.values.items():
        if not isinstance(path, str) or not path:
            errors.append("settings: every key must be a dotted path string")
            continue
        if path == MODELS_PATH or path.startswith(MODELS_PREFIX):
            model_paths.append((path, value))
            continue
        if path.startswith(TEMPLATE_VARIABLES_PREFIX):
            change = _plan_template_variable(
                path, value, candidate_variables, base_variables, errors
            )
            if change is not None:
                changes.append(change)
            continue
        spec = SETTINGS_SPECS.get(path)
        if spec is None:
            errors.append(
                f"{path}: unknown setting. Use get_workflow_settings for the "
                "editable paths (template variables: template_variables.<name>; "
                "models: models.<service>)."
            )
            continue
        if path in _EXTERNAL_PBX_PATHS and external_pbx_enabled is None:
            external_pbx_enabled = await external_pbx_integrations_enabled(
                organization_id
            )
        read_only = _read_only_reason(
            spec, external_pbx_enabled=bool(external_pbx_enabled)
        )
        if read_only or path in _DERIVED_PATHS:
            errors.append(f"{path}: read-only. {read_only or ''}".strip())
            continue
        before = _effective_value(spec, base_configurations, validated_before)
        if value is None:
            after = None if spec.default is _MISSING else copy.deepcopy(spec.default)
            if _get_path(base_configurations, path, None) is None:
                continue
            _del_path(candidate, path)
        else:
            try:
                coerced = spec.adapter.validate_python(value)
            except ValidationError as exc:
                errors.extend(_format_errors(exc, path))
                continue
            after = spec.adapter.dump_python(coerced, mode="json")
            _set_path(candidate, path, after)
            if spec.top_level:
                touched_top_level.append(path)
        if before == after:
            # A no-op (also when explicitly resetting to an equal default):
            # keep the stored value exactly as it was.
            stored = _get_path(base_configurations, path, _MISSING)
            if stored is _MISSING:
                _del_path(candidate, path)
            else:
                _set_path(candidate, path, copy.deepcopy(stored))
            continue
        label = f"Setting: {spec.label} {_short(before)} → {_short(after)}"
        changes.append(_change(path, ChangeOp.update, label, before, after))

    if model_paths:
        model_changes = await _plan_model_changes(
            organization_id=organization_id,
            model_paths=model_paths,
            candidate=candidate,
            errors=errors,
        )
        changes.extend(model_changes)

    if errors:
        raise SettingsValidationError(errors)

    _normalize_top_level(candidate, touched_top_level)
    _validate_configurations(_strip_meta(candidate), errors)
    if errors:
        raise SettingsValidationError(errors)

    if not changes:
        return SettingsPlan(
            changes=[],
            candidate_configurations=base_configurations,
            candidate_template_variables=base_variables,
        )
    return SettingsPlan(
        changes=changes,
        candidate_configurations=candidate,
        candidate_template_variables=candidate_variables,
    )


def _plan_template_variable(
    path: str,
    value: Any,
    candidate_variables: dict,
    base_variables: dict,
    errors: list[str],
) -> ChangeItem | None:
    name = path[len(TEMPLATE_VARIABLES_PREFIX) :]
    exists = name in base_variables
    if not name:
        errors.append(f"{path}: missing variable name")
        return None
    if value is None:
        if not exists:
            return None
        before = base_variables[name]
        candidate_variables.pop(name, None)
        return _change(
            path,
            ChangeOp.remove,
            f"Template variable: remove {name} ({_short(before)})",
            before,
            None,
        )
    if not exists:
        try:
            _TEMPLATE_VARIABLE_NAME.validate_python(name)
        except ValidationError:
            errors.append(
                f"{path}: invalid variable name; use letters, digits and "
                "underscores, not starting with a digit"
            )
            return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        errors.append(f"{path}: value must be text (or a number)")
        return None
    if isinstance(value, str) and len(value) > _MAX_TEMPLATE_VARIABLE_VALUE_LENGTH:
        errors.append(
            f"{path}: value is longer than {_MAX_TEMPLATE_VARIABLE_VALUE_LENGTH} characters"
        )
        return None
    if exists and base_variables[name] == value:
        return None
    candidate_variables[name] = value
    if not exists:
        return _change(
            path,
            ChangeOp.add,
            f"Template variable: add {name} = {_short(value)}",
            None,
            value,
        )
    before = base_variables[name]
    return _change(
        path,
        ChangeOp.update,
        f"Template variable: {name} {_short(before)} → {_short(value)}",
        before,
        value,
    )


def _strip_meta(configurations: dict) -> dict:
    if CANDIDATE_META_KEY not in configurations:
        return configurations
    return {k: v for k, v in configurations.items() if k != CANDIDATE_META_KEY}


def _describe_service(service: str, public: dict | None) -> str:
    if not public:
        return "none"
    provider = public.get("provider", "")
    parts = [_provider_title(service, provider)]
    if service in ("tts", "realtime") and public.get("voice"):
        if public.get("model") and service == "realtime":
            parts.append(str(public["model"]))
        parts.append(str(public["voice"]))
    elif public.get("model"):
        parts.append(str(public["model"]))
    if service == "stt" and provider == ServiceProviders.DOGRAH.value:
        parts.append(f"({public.get('language')})")
    return " ".join(parts)


async def _plan_model_changes(
    *,
    organization_id: int,
    model_paths: list[tuple[str, Any]],
    candidate: dict,
    errors: list[str],
) -> list[ChangeItem]:
    resolved = await get_resolved_ai_model_configuration(
        organization_id=organization_id
    )
    org_config = resolved.organization_configuration
    if org_config is None:
        errors.append(
            "models: this organization has no model configuration yet; set one "
            "up in Model Configurations first."
        )
        return []
    org_dict = org_config.model_dump(mode="json", exclude_none=True)

    current_override = candidate.get(OVERRIDE_KEY)
    legacy = candidate.get(LEGACY_OVERRIDE_KEY)
    try:
        if current_override:
            base = OrganizationAIModelConfigurationV2.model_validate(
                current_override
            ).model_dump(mode="json", exclude_none=True)
        elif isinstance(legacy, dict) and legacy:
            base = _legacy_override_to_v2(legacy, resolved.effective)
        else:
            base = copy.deepcopy(org_dict)
    except (ValidationError, ValueError) as exc:
        errors.append(
            "models: the agent's saved model override is invalid; fix it in the "
            f"Settings tab first ({type(exc).__name__})."
        )
        return []
    has_override = bool(current_override) or (isinstance(legacy, dict) and bool(legacy))

    # Whole-override removal ("models": null) wins over per-service edits.
    if any(path == MODELS_PATH for path, _ in model_paths):
        for path, value in model_paths:
            if path == MODELS_PATH and value is not None:
                errors.append(
                    "models: only null is accepted here (removes the agent "
                    "override); change one service with models.<service>"
                )
                return []
        if len(model_paths) > 1:
            errors.append(
                "models: do not combine removing the whole override with "
                "models.<service> changes"
            )
            return []
        if not has_override:
            return []
        candidate.pop(OVERRIDE_KEY, None)
        candidate.pop(LEGACY_OVERRIDE_KEY, None)
        candidate[CANDIDATE_META_KEY] = {"base": base, "org": org_dict}
        diffs = []
        for service in _config_services(base):
            before = _public(_get_section(base, service))
            after = _public(_get_section(org_dict, service))
            if before != after:
                diffs.append(
                    FieldDiff(field=f"models.{service}", before=before, after=after)
                )
        return [
            _change(
                MODELS_PATH,
                ChangeOp.update,
                "Models: remove this agent's model override (use organization defaults)",
                {"agent_override": True},
                {"agent_override": False},
                field_diffs=diffs
                or [FieldDiff(field=MODELS_PATH, before="agent override", after=None)],
            )
        ]

    new = copy.deepcopy(base)
    key_sources = [base, org_dict]
    services = _config_services(base)
    changes: list[ChangeItem] = []
    for path, value in model_paths:
        service = path[len(MODELS_PREFIX) :]
        if service not in MODEL_SERVICES:
            errors.append(
                f"{path}: unknown service; use one of "
                f"{[f'models.{s}' for s in services]}"
            )
            continue
        if service not in services:
            errors.append(
                f"{path}: not part of this agent's model setup "
                f"(available: {[f'models.{s}' for s in services]})"
            )
            continue
        current = _get_section(new, service)
        if value is None:
            if _config_shape(base) != _config_shape(org_dict):
                errors.append(
                    f"{path}: the agent override uses a different mode than the "
                    "organization, so one service cannot be reset alone; use "
                    "models: null to remove the whole override"
                )
                continue
            section = copy.deepcopy(_get_section(org_dict, service))
        else:
            try:
                choice = ServiceModelChoice.model_validate(value)
            except ValidationError as exc:
                errors.extend(_format_errors(exc, path))
                continue
            if new.get("mode") == "dograh":
                section = _build_dograh_section(
                    service=service, choice=choice, current=current or {}, errors=errors
                )
            else:
                section = _build_byok_section(
                    service=service,
                    choice=choice,
                    current=current,
                    key_sources=key_sources,
                    errors=errors,
                )
        if section is None:
            continue
        before_public = _public(current)
        _set_section(new, service, section)
        after_public = _public(_get_section(new, service))
        if before_public == after_public:
            continue
        diffs = [
            FieldDiff(
                field=f"{path}.{key}",
                before=(before_public or {}).get(key),
                after=(after_public or {}).get(key),
            )
            for key in _PUBLIC_SERVICE_FIELDS
            if (before_public or {}).get(key) != (after_public or {}).get(key)
        ]
        target = "organization default" if value is None else ""
        after_text = _describe_service(service, after_public)
        if target:
            after_text = f"{after_text} ({target})"
        label = (
            f"{_SERVICE_LABELS[service]}: "
            f"{_describe_service(service, before_public)} → {after_text}"
        )
        changes.append(
            _change(path, ChangeOp.update, label, before_public, after_public, diffs)
        )

    if errors or not changes:
        return changes

    _validate_override(new, errors)
    if errors:
        return changes
    if new == org_dict:
        candidate.pop(OVERRIDE_KEY, None)
    else:
        candidate[OVERRIDE_KEY] = new
    candidate.pop(LEGACY_OVERRIDE_KEY, None)
    candidate[CANDIDATE_META_KEY] = {"base": base, "org": org_dict}
    return changes


def merge_accepted_settings(
    *,
    current_configurations: dict | None,
    current_template_variables: dict | None,
    plan_changes: list[ChangeItem],
    candidate_configurations: dict,
    candidate_template_variables: dict,
    accepted_change_ids: list[str],
) -> tuple[dict, dict]:
    """Apply ONLY the accepted settings paths (from the candidate) onto the
    CURRENT working values, then re-validate. Non-settings change ids are
    ignored. Raises SettingsValidationError."""
    configurations = copy.deepcopy(current_configurations or {})
    variables = copy.deepcopy(current_template_variables or {})
    candidate = candidate_configurations or {}
    candidate_variables = candidate_template_variables or {}
    meta = candidate.get(CANDIDATE_META_KEY) or {}

    plan_ids = [
        change.change_id
        for change in plan_changes
        if change.change_id.startswith(SETTINGS_REF_PREFIX)
    ]
    accepted = {
        cid for cid in accepted_change_ids if cid.startswith(SETTINGS_REF_PREFIX)
    }
    errors = [
        f"{cid}: not part of this proposal" for cid in accepted if cid not in plan_ids
    ]
    if errors:
        raise SettingsValidationError(errors)

    for change_id in plan_ids:
        if change_id not in accepted:
            continue
        path = change_id[len(SETTINGS_REF_PREFIX) :]
        if path.startswith(TEMPLATE_VARIABLES_PREFIX):
            name = path[len(TEMPLATE_VARIABLES_PREFIX) :]
            if name in candidate_variables:
                variables[name] = copy.deepcopy(candidate_variables[name])
            else:
                variables.pop(name, None)
        elif path == MODELS_PATH:
            configurations.pop(OVERRIDE_KEY, None)
            configurations.pop(LEGACY_OVERRIDE_KEY, None)
        elif path.startswith(MODELS_PREFIX):
            _merge_model_service(
                configurations, candidate, meta, path[len(MODELS_PREFIX) :], errors
            )
        else:
            value = _get_path(candidate, path, _MISSING)
            if value is _MISSING:
                _del_path(configurations, path)
            else:
                _set_path(configurations, path, copy.deepcopy(value))

    configurations.pop(CANDIDATE_META_KEY, None)
    if not errors:
        _validate_configurations(configurations, errors)
    if errors:
        raise SettingsValidationError(errors)
    return configurations, variables


def _merge_model_service(
    configurations: dict,
    candidate: dict,
    meta: dict,
    service: str,
    errors: list[str],
) -> None:
    path = f"models.{service}"
    org = meta.get("org")
    candidate_override = candidate.get(OVERRIDE_KEY) or copy.deepcopy(org)
    if not candidate_override:
        errors.append(f"{path}: proposal data is incomplete; propose the change again")
        return
    current_override = configurations.get(OVERRIDE_KEY)
    target = copy.deepcopy(current_override or meta.get("base") or org)
    if _config_shape(target) != _config_shape(candidate_override):
        errors.append(
            f"{path}: the agent's model setup changed since this was proposed; "
            "propose the change again"
        )
        return
    section = _get_section(candidate_override, service)
    if section is None:
        errors.append(f"{path}: not part of this agent's model setup")
        return
    _set_section(target, service, section)
    if org is not None and target == org:
        configurations.pop(OVERRIDE_KEY, None)
    else:
        configurations[OVERRIDE_KEY] = target
    configurations.pop(LEGACY_OVERRIDE_KEY, None)


__all__ = [
    "CANDIDATE_META_KEY",
    "SETTINGS_SPECS",
    "SettingsPlan",
    "SettingsValidationError",
    "VOICE_CATALOG_PROVIDERS",
    "clear_voice_cache",
    "get_settings_view",
    "list_model_options",
    "list_voices",
    "load_working_settings",
    "merge_accepted_settings",
    "plan_settings_changes",
    "settings_hash",
]
