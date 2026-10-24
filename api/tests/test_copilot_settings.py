"""`api.services.copilot.settings`: the copilot settings engine (M2c).

DB, org-preference, org-model-configuration and MPS access are mocked at the
settings module boundary; schemas and validation run for real.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock

import pytest

from api.schemas.ai_model_configuration import (
    OrganizationAIModelConfigurationV2,
    compile_ai_model_configuration_v2,
)
from api.services.configuration.ai_model_configuration import (
    ResolvedAIModelConfiguration,
)
from api.services.copilot import settings
from api.services.copilot.contracts import ChangeOp, ChangeTarget, SettingsPatch
from api.services.copilot.settings import (
    CANDIDATE_META_KEY,
    OVERRIDE_KEY,
    SettingsValidationError,
    merge_accepted_settings,
    plan_settings_changes,
    settings_hash,
)

ORG = 11

LLM_KEY = "sk-org-llm-SECRET-0000000000001111"
TTS_KEY = "el-org-tts-SECRET-0000000000002222"
STT_KEY = "dg-org-stt-SECRET-0000000000003333"
OVERRIDE_TTS_KEY = "el-agent-tts-SECRET-000000004444"
VOICEMAIL_KEY = "vm-classifier-SECRET-000000005555"
DOGRAH_KEY = "dgr-managed-SECRET-000000006666"
SECRETS = (LLM_KEY, TTS_KEY, STT_KEY, OVERRIDE_TTS_KEY, VOICEMAIL_KEY, DOGRAH_KEY)


def _byok_org() -> dict:
    return {
        "version": 2,
        "mode": "byok",
        "byok": {
            "mode": "pipeline",
            "pipeline": {
                "llm": {"provider": "openai", "api_key": LLM_KEY, "model": "gpt-4.1"},
                "tts": {
                    "provider": "elevenlabs",
                    "api_key": TTS_KEY,
                    "voice": "21m00Tcm4TlvDq8ikWAM",
                    "model": "eleven_flash_v2_5",
                },
                "stt": {
                    "provider": "deepgram",
                    "api_key": STT_KEY,
                    "model": "nova-3-general",
                },
            },
        },
    }


def _normalized_byok_org() -> dict:
    """The org config as stored (defaults filled, like the upsert writes it)."""
    return OrganizationAIModelConfigurationV2.model_validate(_byok_org()).model_dump(
        mode="json", exclude_none=True
    )


def _dograh_org() -> dict:
    return {
        "version": 2,
        "mode": "dograh",
        "dograh": {
            "api_key": DOGRAH_KEY,
            "voice": "default",
            "speed": 1.0,
            "language": "multi",
        },
    }


def _resolved(org: dict | None) -> ResolvedAIModelConfiguration:
    if org is None:
        from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration

        return ResolvedAIModelConfiguration(
            effective=EffectiveAIModelConfiguration(), source="empty"
        )
    parsed = OrganizationAIModelConfigurationV2.model_validate(org)
    return ResolvedAIModelConfiguration(
        effective=compile_ai_model_configuration_v2(parsed),
        source="organization_v2",
        organization_configuration=parsed,
    )


def _workflow(configurations: dict | None = None, variables: dict | None = None):
    released = SimpleNamespace(
        workflow_json={"nodes": [], "edges": []},
        workflow_configurations=configurations or {},
        template_context_variables=variables or {},
    )
    return SimpleNamespace(
        id=7,
        released_definition=released,
        workflow_definition=None,
        workflow_configurations={},
        template_context_variables={},
    )


@pytest.fixture
def env(monkeypatch):
    """Mocks the settings module's I/O; `env.org` picks the org config."""
    state = SimpleNamespace(org=_byok_org(), pbx=False, draft=None)

    async def resolved(*, organization_id):
        assert organization_id == ORG
        return _resolved(state.org)

    async def pbx(organization_id):
        assert organization_id == ORG
        return state.pbx

    async def draft(workflow_id):
        return state.draft

    monkeypatch.setattr(settings, "get_resolved_ai_model_configuration", resolved)
    monkeypatch.setattr(settings, "external_pbx_integrations_enabled", pbx)
    monkeypatch.setattr(settings.db_client, "get_draft_version", draft)
    settings.clear_voice_cache()
    yield state
    settings.clear_voice_cache()


async def _plan(values: dict, configurations=None, variables=None):
    return await plan_settings_changes(
        workflow=_workflow(),
        organization_id=ORG,
        patch=SettingsPatch(values=values),
        working_configurations=configurations if configurations is not None else {},
        working_template_variables=variables if variables is not None else {},
    )


def _assert_no_secrets(payload) -> None:
    dumped = json.dumps(payload, default=str)
    for secret in SECRETS:
        assert secret not in dumped
        assert secret[-8:] not in dumped  # not even a masked tail
    assert '"api_key"' not in dumped
    assert '"credentials"' not in dumped


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------


async def test_view_masks_secrets_and_reports_override_vs_org_default(env):
    override = copy.deepcopy(_byok_org())
    override["byok"]["pipeline"]["tts"] = {
        "provider": "elevenlabs",
        "api_key": OVERRIDE_TTS_KEY,
        "voice": "Aria",
        "model": "eleven_flash_v2_5",
    }
    configurations = {
        "max_call_duration": 600,
        "voicemail_detection": {
            "enabled": True,
            "use_workflow_llm": False,
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "api_key": VOICEMAIL_KEY,
        },
        OVERRIDE_KEY: override,
        "some_unknown_extra": {"api_key": "should-never-show"},
    }
    env.draft = SimpleNamespace(
        workflow_json={"nodes": []},
        workflow_configurations=configurations,
        template_context_variables={"company": "Dograh"},
    )

    view = await settings.get_settings_view(workflow=_workflow(), organization_id=ORG)

    _assert_no_secrets(view)
    assert "should-never-show" not in json.dumps(view)
    assert view["settings"]["max_call_duration"] == 600
    assert view["settings"]["turn_stop_strategy"] == "transcription"  # default
    assert view["settings"]["voicemail_detection"]["api_key_configured"] is True
    assert view["settings"]["voicemail_detection"]["model"] == "gpt-4.1-mini"
    assert view["template_variables"] == {"company": "Dograh"}
    assert view["models"]["tts"] == {
        "provider": "elevenlabs",
        "model": "eleven_flash_v2_5",
        "voice": "Aria",
        "speed": 1.0,
        "source": "agent_override",
    }
    assert view["models"]["llm"]["source"] == "organization_default"
    assert view["models"]["stt"]["source"] == "organization_default"
    assert view["model_mode"]["has_agent_override"] is True
    assert view["model_mode"]["mode"] == "byok"


async def test_view_without_override_is_org_default(env):
    view = await settings.get_settings_view(workflow=_workflow(), organization_id=ORG)
    assert {m["source"] for m in view["models"].values()} == {"organization_default"}
    assert view["models"]["llm"]["provider"] == "openai"
    assert view["model_mode"]["has_agent_override"] is False
    _assert_no_secrets(view)


async def test_schema_enums_ranges_and_read_only(env):
    view = await settings.get_settings_view(workflow=_workflow(), organization_id=ORG)
    schema = view["schema"]
    assert schema["turn_stop_strategy"]["enum"] == ["transcription", "turn_analyzer"]
    assert schema["turn_start_strategy"]["enum"] == ["default", "min_words"]
    assert schema["max_call_duration"]["minimum"] == 1
    assert schema["max_call_duration"]["maximum"] == 1200
    assert schema["smart_turn_stop_secs"]["minimum"] == 0.5
    assert schema["voicemail_detection.screening_wait_ms"]["minimum"] == 1000
    assert schema["call_dispositions"]["type"] == "array"
    # Not editable in the Settings tab / gated / secret-bearing
    assert schema["external_pbx_lead_headers"]["read_only"] is True
    assert schema["voicemail_detection.provider"]["read_only"] is True
    assert schema["voicemail_detection.classify_budget_ms"]["read_only"] is True
    assert schema["ambient_noise_configuration.storage_key"]["read_only"] is True
    assert "read_only" not in schema["max_call_duration"]
    assert "models.<service>" in schema and "template_variables.<name>" in schema
    assert "voicemail_detection.api_key" not in schema


async def test_external_pbx_editable_when_org_enabled(env):
    env.pbx = True
    view = await settings.get_settings_view(workflow=_workflow(), organization_id=ORG)
    assert "read_only" not in view["schema"]["external_pbx_lead_headers"]
    plan = await _plan({"external_pbx_lead_headers": ["first_name"]})
    assert plan.candidate_configurations["external_pbx_lead_headers"] == ["first_name"]


# ---------------------------------------------------------------------------
# Plan: plain settings and template variables
# ---------------------------------------------------------------------------


async def test_plan_simple_field(env):
    plan = await _plan({"max_call_duration": 600}, {"max_call_duration": 300})
    (change,) = plan.changes
    assert change.change_id == change.ref_id == "s:max_call_duration"
    assert change.target == ChangeTarget.workflow
    assert change.op == ChangeOp.update
    assert change.label == "Setting: Max call duration 300 → 600"
    assert change.field_diffs[0].before == 300
    assert change.field_diffs[0].after == 600
    assert plan.candidate_configurations["max_call_duration"] == 600


async def test_plan_default_baseline_and_strategy(env):
    plan = await _plan({"turn_stop_strategy": "turn_analyzer"})
    (change,) = plan.changes
    assert change.label == (
        "Setting: Turn detection strategy transcription → turn_analyzer"
    )
    assert plan.candidate_configurations == {"turn_stop_strategy": "turn_analyzer"}


async def test_plan_nested_fields(env):
    working = {"voicemail_detection": {"enabled": False, "use_workflow_llm": True}}
    plan = await _plan(
        {
            "voicemail_detection.enabled": True,
            "ambient_noise_configuration.volume": 0.5,
        },
        working,
    )
    assert [c.change_id for c in plan.changes] == [
        "s:voicemail_detection.enabled",
        "s:ambient_noise_configuration.volume",
    ]
    assert plan.candidate_configurations["voicemail_detection"] == {
        "enabled": True,
        "use_workflow_llm": True,
    }
    assert plan.candidate_configurations["ambient_noise_configuration"] == {
        "volume": 0.5
    }
    assert working["voicemail_detection"]["enabled"] is False  # not mutated


async def test_plan_template_variables_add_update_remove(env):
    plan = await _plan(
        {
            "template_variables.company": "Dograh",
            "template_variables.city": "Paris",
            "template_variables.old": None,
            "template_variables.missing": None,
        },
        variables={"city": "London", "old": "x", "keep": "y"},
    )
    ops = {c.change_id: c.op for c in plan.changes}
    assert ops == {
        "s:template_variables.company": ChangeOp.add,
        "s:template_variables.city": ChangeOp.update,
        "s:template_variables.old": ChangeOp.remove,
    }
    assert plan.candidate_template_variables == {
        "company": "Dograh",
        "city": "Paris",
        "keep": "y",
    }


async def test_plan_template_variable_invalid_name(env):
    with pytest.raises(SettingsValidationError) as exc:
        await _plan({"template_variables.1bad name": "x"})
    assert "template_variables.1bad name" in exc.value.errors[0]


async def test_plan_call_dispositions_list(env):
    plan = await _plan(
        {
            "call_dispositions": [
                {"code": "booked", "description": "  Demo booked.  "},
                {"code": "no_answer", "description": "Nobody answered."},
            ]
        }
    )
    (change,) = plan.changes
    assert change.label == "Setting: Call dispositions 0 items → 2 items"
    assert plan.candidate_configurations["call_dispositions"][0] == {
        "code": "booked",
        "description": "Demo booked.",
    }


async def test_plan_call_dispositions_duplicate_codes_rejected(env):
    with pytest.raises(SettingsValidationError) as exc:
        await _plan(
            {
                "call_dispositions": [
                    {"code": "a", "description": "x"},
                    {"code": "A", "description": "y"},
                ]
            }
        )
    assert any("call_dispositions" in e and "unique" in e for e in exc.value.errors)


async def test_plan_invalid_value_names_path(env):
    with pytest.raises(SettingsValidationError) as exc:
        await _plan({"max_call_duration": 5000})
    assert exc.value.errors[0].startswith("max_call_duration:")
    assert "1200" in exc.value.errors[0]


async def test_plan_ui_bounds_and_enum(env):
    with pytest.raises(SettingsValidationError) as exc:
        await _plan({"smart_turn_stop_secs": 30, "turn_stop_strategy": "vad"})
    assert len(exc.value.errors) == 2
    assert exc.value.errors[0].startswith("smart_turn_stop_secs:")
    assert exc.value.errors[1].startswith("turn_stop_strategy:")


async def test_plan_unknown_and_read_only_paths(env):
    with pytest.raises(SettingsValidationError) as exc:
        await _plan(
            {
                "max_call_length": 10,
                "voicemail_detection.api_key": "x",
                "voicemail_detection.provider": "groq",
                "external_pbx_lead_headers": ["a"],
            }
        )
    errors = exc.value.errors
    assert errors[0].startswith("max_call_length: unknown setting")
    assert errors[1].startswith("voicemail_detection.api_key: unknown setting")
    assert errors[2].startswith("voicemail_detection.provider: read-only")
    assert errors[3].startswith("external_pbx_lead_headers: read-only")


async def test_plan_no_op_returns_empty(env):
    working = {"max_call_duration": 300, "dictionary": "foo"}
    plan = await _plan(
        {
            "max_call_duration": 300,
            "turn_start_strategy": "default",  # equals the default
            "dictionary": "foo",
            "template_variables.a": "1",
        },
        working,
        {"a": "1"},
    )
    assert plan.changes == []
    assert plan.candidate_configurations == working
    assert plan.candidate_template_variables == {"a": "1"}


async def test_plan_null_resets_to_default(env):
    plan = await _plan({"max_call_duration": None}, {"max_call_duration": 900})
    (change,) = plan.changes
    assert change.label == "Setting: Max call duration 900 → 300"
    assert "max_call_duration" not in plan.candidate_configurations


async def test_plan_voicemail_rules(env):
    with pytest.raises(SettingsValidationError) as exc:
        await _plan(
            {
                "voicemail_detection.enabled": True,
                "voicemail_detection.voicemail_action": "leave_message",
                "voicemail_detection.use_workflow_llm": False,
            }
        )
    joined = " | ".join(exc.value.errors)
    assert "voicemail_detection.use_workflow_llm" in joined
    assert "voicemail_detection.voicemail_message.text" in joined

    plan = await _plan(
        {
            "voicemail_detection.enabled": True,
            "voicemail_detection.voicemail_action": "leave_message",
            "voicemail_detection.voicemail_message.text": "Hi, call us back.",
        }
    )
    assert plan.candidate_configurations["voicemail_detection"] == {
        "enabled": True,
        "voicemail_action": "leave_message",
        "voicemail_message": {"text": "Hi, call us back."},
    }


async def test_plan_voicemail_field_diffs_masked(env):
    working = {
        "voicemail_detection": {
            "enabled": True,
            "use_workflow_llm": False,
            "api_key": VOICEMAIL_KEY,
            "model": "gpt-4.1",
        }
    }
    plan = await _plan({"voicemail_detection.model": "gpt-4.1-mini"}, working)
    _assert_no_secrets([c.model_dump() for c in plan.changes])
    # The stored key survives in the (server-side) candidate.
    assert plan.candidate_configurations["voicemail_detection"]["api_key"] == (
        VOICEMAIL_KEY
    )


# ---------------------------------------------------------------------------
# Plan: models
# ---------------------------------------------------------------------------


async def test_model_change_keeps_other_services_and_secrets(env):
    plan = await _plan({"models.tts": {"provider": "elevenlabs", "voice": "Aria"}})
    (change,) = plan.changes
    assert change.change_id == "s:models.tts"
    assert (
        change.label == "Voice (TTS): ElevenLabs 21m00Tcm4TlvDq8ikWAM → ElevenLabs Aria"
    )
    assert [d.field for d in change.field_diffs] == ["models.tts.voice"]
    _assert_no_secrets(change.model_dump())

    override = plan.candidate_configurations[OVERRIDE_KEY]
    pipeline = override["byok"]["pipeline"]
    org_pipeline = _normalized_byok_org()["byok"]["pipeline"]
    assert pipeline["llm"] == org_pipeline["llm"]
    assert pipeline["stt"] == org_pipeline["stt"]
    assert pipeline["tts"]["voice"] == "Aria"
    assert pipeline["tts"]["api_key"] == TTS_KEY
    OrganizationAIModelConfigurationV2.model_validate(override)


async def test_model_change_starts_from_existing_override(env):
    override = copy.deepcopy(_byok_org())
    override["byok"]["pipeline"]["llm"]["model"] = "gpt-4.1-mini"
    override["byok"]["pipeline"]["tts"]["api_key"] = OVERRIDE_TTS_KEY
    plan = await _plan(
        {"models.tts": {"provider": "elevenlabs", "voice": "Aria"}},
        {OVERRIDE_KEY: override},
    )
    pipeline = plan.candidate_configurations[OVERRIDE_KEY]["byok"]["pipeline"]
    assert pipeline["llm"]["model"] == "gpt-4.1-mini"
    assert pipeline["tts"]["api_key"] == OVERRIDE_TTS_KEY
    assert pipeline["tts"]["voice"] == "Aria"


async def test_model_provider_switch_reuses_key_of_same_provider(env):
    plan = await _plan({"models.tts": {"provider": "openai", "voice": "nova"}})
    tts = plan.candidate_configurations[OVERRIDE_KEY]["byok"]["pipeline"]["tts"]
    assert tts["provider"] == "openai"
    assert tts["api_key"] == LLM_KEY
    assert tts["voice"] == "nova"
    assert plan.changes[0].label.startswith("Voice (TTS): ElevenLabs")


async def test_model_rejects_unavailable_provider(env):
    with pytest.raises(SettingsValidationError) as exc:
        await _plan({"models.tts": {"provider": "cartesia"}})
    assert exc.value.errors[0].startswith(
        "models.tts: provider 'cartesia' has no API key"
    )
    assert "elevenlabs" in exc.value.errors[0]


async def test_model_rejects_secret_and_endpoint_options(env):
    with pytest.raises(SettingsValidationError) as exc:
        await _plan(
            {
                "models.llm": {
                    "provider": "openai",
                    "options": {"base_url": "https://evil.example", "api_key": "x"},
                }
            }
        )
    assert all("models.llm.options." in e for e in exc.value.errors)
    assert len(exc.value.errors) == 2


async def test_model_rejects_unoffered_model_and_wrong_service(env):
    with pytest.raises(SettingsValidationError) as exc:
        await _plan(
            {
                "models.tts": {"provider": "openai", "model": "tts-9000"},
                "models.realtime": {"provider": "openai_realtime"},
            }
        )
    assert exc.value.errors[0].startswith("models.tts.model:")
    assert exc.value.errors[1].startswith("models.realtime: not part of")


async def test_model_null_clears_service_back_to_org_default(env):
    override = copy.deepcopy(_byok_org())
    override["byok"]["pipeline"]["tts"]["voice"] = "Aria"
    plan = await _plan(
        {"models.tts": None}, {OVERRIDE_KEY: override, "dictionary": "a"}
    )
    (change,) = plan.changes
    assert change.label == (
        "Voice (TTS): ElevenLabs Aria → ElevenLabs 21m00Tcm4TlvDq8ikWAM "
        "(organization default)"
    )
    # Every service now equals the org config -> the override is dropped.
    assert OVERRIDE_KEY not in plan.candidate_configurations
    assert plan.candidate_configurations["dictionary"] == "a"


async def test_model_null_keeps_override_for_other_changed_services(env):
    override = copy.deepcopy(_byok_org())
    override["byok"]["pipeline"]["tts"]["voice"] = "Aria"
    override["byok"]["pipeline"]["llm"]["model"] = "gpt-4.1-mini"
    plan = await _plan({"models.tts": None}, {OVERRIDE_KEY: override})
    pipeline = plan.candidate_configurations[OVERRIDE_KEY]["byok"]["pipeline"]
    assert pipeline["tts"]["voice"] == "21m00Tcm4TlvDq8ikWAM"
    assert pipeline["llm"]["model"] == "gpt-4.1-mini"


async def test_models_null_removes_whole_override(env):
    override = copy.deepcopy(_byok_org())
    override["byok"]["pipeline"]["tts"]["voice"] = "Aria"
    plan = await _plan({"models": None}, {OVERRIDE_KEY: override})
    (change,) = plan.changes
    assert change.change_id == "s:models"
    assert OVERRIDE_KEY not in plan.candidate_configurations
    assert (await _plan({"models": None})).changes == []


async def test_model_same_choice_is_no_op(env):
    plan = await _plan({"models.llm": {"provider": "openai", "model": "gpt-4.1"}})
    assert plan.changes == []
    assert OVERRIDE_KEY not in plan.candidate_configurations


async def test_dograh_managed_models(env):
    env.org = _dograh_org()
    plan = await _plan(
        {
            "models.tts": {
                "provider": "dograh",
                "voice": "calm-female",
                "options": {"speed": 1.2},
            },
            "models.stt": {"provider": "dograh", "options": {"language": "en"}},
        }
    )
    assert [c.change_id for c in plan.changes] == ["s:models.tts", "s:models.stt"]
    assert plan.changes[0].label == "Voice (TTS): Dograh default → Dograh calm-female"
    managed = plan.candidate_configurations[OVERRIDE_KEY]["dograh"]
    assert managed == {
        "api_key": DOGRAH_KEY,
        "voice": "calm-female",
        "speed": 1.2,
        "language": "en",
    }
    _assert_no_secrets([c.model_dump() for c in plan.changes])

    with pytest.raises(SettingsValidationError) as exc:
        await _plan({"models.llm": {"provider": "openai", "model": "gpt-4.1"}})
    assert "Dograh-managed" in exc.value.errors[0]

    with pytest.raises(SettingsValidationError) as exc:
        await _plan({"models.tts": {"provider": "dograh", "options": {"speed": 5}}})
    assert exc.value.errors[0].startswith("models.tts.options.speed:")


async def test_models_without_org_configuration(env):
    env.org = None
    with pytest.raises(SettingsValidationError) as exc:
        await _plan({"models.llm": {"provider": "openai"}})
    assert "no model configuration" in exc.value.errors[0]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


async def test_merge_applies_only_accepted_over_concurrent_edits(env):
    working = {"max_call_duration": 300, "dictionary": "a"}
    plan = await _plan(
        {
            "max_call_duration": 600,
            "dictionary": "b",
            "template_variables.company": "Dograh",
            "template_variables.drop": None,
        },
        working,
        {"drop": "x"},
    )
    current = {**working, "turn_stop_strategy": "turn_analyzer"}  # edited meanwhile
    configurations, variables = merge_accepted_settings(
        current_configurations=current,
        current_template_variables={"drop": "x", "added_meanwhile": "1"},
        plan_changes=plan.changes,
        candidate_configurations=plan.candidate_configurations,
        candidate_template_variables=plan.candidate_template_variables,
        accepted_change_ids=[
            "s:max_call_duration",
            "s:template_variables.drop",
            "n:3",  # graph changes are ignored here
        ],
    )
    assert configurations == {
        "max_call_duration": 600,
        "dictionary": "a",
        "turn_stop_strategy": "turn_analyzer",
    }
    assert variables == {"added_meanwhile": "1"}
    assert current["max_call_duration"] == 300  # inputs untouched


async def test_merge_null_reset_and_nested(env):
    working = {"max_call_duration": 900, "voicemail_detection": {"enabled": False}}
    plan = await _plan(
        {"max_call_duration": None, "voicemail_detection.enabled": True}, working
    )
    configurations, _ = merge_accepted_settings(
        current_configurations={**working, "dictionary": "z"},
        current_template_variables=None,
        plan_changes=plan.changes,
        candidate_configurations=plan.candidate_configurations,
        candidate_template_variables=plan.candidate_template_variables,
        accepted_change_ids=[c.change_id for c in plan.changes],
    )
    assert configurations == {
        "voicemail_detection": {"enabled": True},
        "dictionary": "z",
    }


async def test_merge_models_per_service(env):
    plan = await _plan(
        {
            "models.tts": {"provider": "elevenlabs", "voice": "Aria"},
            "models.llm": {"provider": "openai", "model": "gpt-4.1-mini"},
        }
    )
    assert CANDIDATE_META_KEY in plan.candidate_configurations
    configurations, _ = merge_accepted_settings(
        current_configurations={"dictionary": "x"},
        current_template_variables={},
        plan_changes=plan.changes,
        candidate_configurations=plan.candidate_configurations,
        candidate_template_variables=plan.candidate_template_variables,
        accepted_change_ids=["s:models.tts"],
    )
    assert CANDIDATE_META_KEY not in configurations
    pipeline = configurations[OVERRIDE_KEY]["byok"]["pipeline"]
    assert pipeline["tts"]["voice"] == "Aria"
    assert pipeline["tts"]["api_key"] == TTS_KEY
    assert pipeline["llm"] == _normalized_byok_org()["byok"]["pipeline"]["llm"]
    assert configurations["dictionary"] == "x"

    # Declining every model change leaves no override behind.
    configurations, _ = merge_accepted_settings(
        current_configurations={},
        current_template_variables={},
        plan_changes=plan.changes,
        candidate_configurations=plan.candidate_configurations,
        candidate_template_variables=plan.candidate_template_variables,
        accepted_change_ids=[],
    )
    assert configurations == {}


async def test_merge_model_onto_concurrently_changed_override(env):
    plan = await _plan({"models.tts": {"provider": "elevenlabs", "voice": "Aria"}})
    current_override = copy.deepcopy(_byok_org())
    current_override["byok"]["pipeline"]["llm"]["model"] = "gpt-5"  # edited meanwhile
    configurations, _ = merge_accepted_settings(
        current_configurations={OVERRIDE_KEY: current_override},
        current_template_variables={},
        plan_changes=plan.changes,
        candidate_configurations=plan.candidate_configurations,
        candidate_template_variables=plan.candidate_template_variables,
        accepted_change_ids=["s:models.tts"],
    )
    pipeline = configurations[OVERRIDE_KEY]["byok"]["pipeline"]
    assert pipeline["llm"]["model"] == "gpt-5"
    assert pipeline["tts"]["voice"] == "Aria"


async def test_merge_rejects_unknown_ids_and_revalidates(env):
    plan = await _plan({"max_call_duration": 600})
    with pytest.raises(SettingsValidationError) as exc:
        merge_accepted_settings(
            current_configurations={},
            current_template_variables={},
            plan_changes=plan.changes,
            candidate_configurations=plan.candidate_configurations,
            candidate_template_variables=plan.candidate_template_variables,
            accepted_change_ids=["s:dictionary"],
        )
    assert exc.value.errors == ["s:dictionary: not part of this proposal"]

    with pytest.raises(SettingsValidationError) as exc:
        merge_accepted_settings(
            current_configurations={"turn_stop_strategy": "bogus"},
            current_template_variables={},
            plan_changes=plan.changes,
            candidate_configurations=plan.candidate_configurations,
            candidate_template_variables=plan.candidate_template_variables,
            accepted_change_ids=["s:max_call_duration"],
        )
    assert exc.value.errors[0].startswith("turn_stop_strategy:")


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------


def test_settings_hash_stable():
    a = settings_hash({"b": 1, "a": {"y": 2, "x": 1}}, {"k": "v"})
    b = settings_hash({"a": {"x": 1, "y": 2}, "b": 1}, {"k": "v"})
    assert a == b and len(a) == 64
    assert settings_hash(None, None) == settings_hash({}, {})
    assert settings_hash({"b": 2}, {"k": "v"}) != b
    assert settings_hash({"b": 1, "a": {"y": 2, "x": 1}}, {"k": "w"}) != a


# ---------------------------------------------------------------------------
# Catalog: model options and voices
# ---------------------------------------------------------------------------


async def test_list_model_options_byok_only_providers_with_keys(env):
    options = await settings.list_model_options(organization_id=ORG)
    assert options["mode"] == "byok" and options["byok_mode"] == "pipeline"
    llm = {o["provider"] for o in options["services"]["llm"]}
    tts = {o["provider"] for o in options["services"]["tts"]}
    stt = {o["provider"] for o in options["services"]["stt"]}
    assert llm == {"openai"}
    # elevenlabs (its own key), openai + deepgram (keys from other services)
    assert tts == {"elevenlabs", "openai", "deepgram"}
    assert "cartesia" not in tts and "dograh" not in tts
    assert stt == {"deepgram", "openai", "elevenlabs"}
    openai_llm = options["services"]["llm"][0]
    assert openai_llm["current_org_default"] is True
    assert "gpt-4.1" in openai_llm["models"]
    eleven = next(
        o for o in options["services"]["tts"] if o["provider"] == "elevenlabs"
    )
    assert eleven["voice_catalog"] is True
    _assert_no_secrets(options)


async def test_list_model_options_managed(env):
    env.org = _dograh_org()
    options = await settings.list_model_options(organization_id=ORG)
    assert options["mode"] == "dograh"
    assert [o["provider"] for o in options["services"]["tts"]] == ["dograh"]
    tts = options["services"]["tts"][0]
    assert tts["options"]["speed"]["min"] == 0.5
    assert tts["voice_catalog"] is True
    assert "multi" in options["services"]["stt"][0]["options"]["language"]["values"]
    _assert_no_secrets(options)


async def test_list_model_options_without_org_configuration(env):
    env.org = None
    options = await settings.list_model_options(organization_id=ORG)
    assert options["services"] == {} and options["mode"] is None


async def test_list_voices_cached_per_org_and_provider(env, monkeypatch):
    get_voices = AsyncMock(
        return_value={
            "provider": "elevenlabs",
            "voices": [
                {"voice_id": "v1", "name": "Rachel", "gender": "female", "extra": 1},
                {"name": "no id"},
            ],
        }
    )
    monkeypatch.setattr(settings.mps_service_key_client, "get_voices", get_voices)
    clock = [1000.0]
    monkeypatch.setattr(settings.time, "monotonic", lambda: clock[0])
    user = SimpleNamespace(provider_id="user-abc")

    first = await settings.list_voices(
        organization_id=ORG, provider="elevenlabs", user=user
    )
    assert first == [{"voice_id": "v1", "name": "Rachel", "gender": "female"}]
    first.append({"mutated": True})
    second = await settings.list_voices(
        organization_id=ORG, provider="elevenlabs", user=user
    )
    assert second == [{"voice_id": "v1", "name": "Rachel", "gender": "female"}]
    assert get_voices.await_count == 1
    get_voices.assert_awaited_with(
        provider="elevenlabs",
        model=None,
        language=None,
        q=None,
        gender=None,
        accent=None,
        organization_id=ORG,
        created_by="user-abc",
    )

    await settings.list_voices(organization_id=ORG, provider="cartesia", user=user)
    await settings.list_voices(
        organization_id=ORG + 1, provider="elevenlabs", user=user
    )
    assert get_voices.await_count == 3

    clock[0] += settings.VOICE_CACHE_TTL_SECONDS + 1
    await settings.list_voices(organization_id=ORG, provider="elevenlabs", user=user)
    assert get_voices.await_count == 4


async def test_list_voices_rejects_unsupported_provider(env):
    with pytest.raises(SettingsValidationError):
        await settings.list_voices(
            organization_id=ORG,
            provider="openai",
            user=SimpleNamespace(provider_id="u"),
        )


def test_voice_catalog_providers_match_voices_route():
    from api.routes.user import TTSProvider

    assert set(settings.VOICE_CATALOG_PROVIDERS) == set(get_args(TTSProvider))


def test_defaults_route_uses_lifted_catalog():
    from api.services.configuration.model_catalog import (
        get_model_configuration_v2_defaults,
    )

    catalog = get_model_configuration_v2_defaults()
    assert set(catalog) == {"dograh", "byok"}
    assert "dograh" not in catalog["byok"]["pipeline"]["llm"]
    assert catalog["byok"]["pipeline"]["default_providers"]["llm"] == "openai"


async def test_defaults_route_returns_the_catalog_unchanged():
    from api.routes import organization as organization_routes
    from api.services.configuration.model_catalog import (
        get_model_configuration_v2_defaults,
    )

    response = await organization_routes.get_model_configuration_v2_defaults(
        user=SimpleNamespace(selected_organization_id=ORG)
    )
    assert response == get_model_configuration_v2_defaults()
