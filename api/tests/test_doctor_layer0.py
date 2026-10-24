"""Doctor Layer 0 per-run platform checks."""

from api.services.workflow.doctor.layer0 import (
    check_greeting,
    check_pipeline_errors,
    check_prompt_load,
    run_layer0,
)

BIG_PROMPT = "You are Sarah from Acme Plumbing. " * 200  # ~6.6k chars ≈ 1.6k tokens

WORKFLOW = {
    "nodes": [
        {
            "id": "1",
            "type": "startCall",
            "data": {
                "name": "Welcome",
                "prompt": BIG_PROMPT,
                "greeting_type": "text",
                "greeting": "Hi {{first_name}}, this is Sarah from Acme Plumbing.",
            },
        },
        {"id": "2", "type": "endCall", "data": {"name": "End", "prompt": "Bye."}},
    ],
    "edges": [],
}


def run(run_id=1, first_bot="Hi Ann, this is Sarah from Acme Plumbing.", **extra):
    timeline = []
    if first_bot is not None:
        timeline.append(
            {"kind": "bot", "turn": 0, "node": "Welcome", "text": first_bot}
        )
    view = {
        "run_id": run_id,
        "definition_id": 10,
        "has_events": True,
        "initial_context": {"first_name": "Ann"},
        "gathered_context": {},
        "usage_info": {},
        "timeline": timeline,
    }
    view.update(extra)
    return view


def test_greeting_matches_rendered_template():
    flag, insufficient = check_greeting(run(), WORKFLOW)
    assert flag is None and not insufficient


def test_generic_first_line_is_flagged():
    flag, _ = check_greeting(
        run(first_bot="Hello! How can I help you today?"), WORKFLOW
    )
    assert flag is not None and flag.check == "greeting_mismatch"


def test_greeting_followed_by_more_text_still_matches():
    flag, _ = check_greeting(
        run(
            first_bot="Hi Ann, this is Sarah from Acme Plumbing. Is now a good time "
            "to talk about your boiler service?"
        ),
        WORKFLOW,
    )
    assert flag is None


def test_greeting_skipped_without_greeting():
    no_greeting = {
        "nodes": [
            {"id": "1", "type": "startCall", "data": {"name": "S", "prompt": "x"}}
        ],
        "edges": [],
    }
    flag, insufficient = check_greeting(run(first_bot="Anything"), no_greeting)
    assert flag is None and not insufficient


def test_prompt_tokens_far_below_estimate_flags_system_prompt_missing():
    r = run(usage_info={"llm_turns": [{"prompt_tokens": 15}]})
    flag, insufficient = check_prompt_load(r, WORKFLOW)
    assert flag is not None and flag.check == "system_prompt_missing"
    assert not insufficient

    ok = run(usage_info={"llm_turns": [{"prompt_tokens": 2400}]})
    assert check_prompt_load(ok, WORKFLOW) == (None, False)


def test_aggregate_only_usage_is_insufficient_data():
    r = run(usage_info={"llm": {"OpenAILLMService#0|||gpt-4.1": {"prompt_tokens": 9}}})
    assert check_prompt_load(r, WORKFLOW) == (None, True)


def test_pipeline_errors_fatal_or_repeated_processor():
    def err(proc, fatal=False):
        return {"kind": "error", "processor": proc, "fatal": fatal, "error": "x"}

    single = run(timeline=[err("TTS")])
    assert check_pipeline_errors(single) == (None, False)
    repeated = run(timeline=[err("TTS"), err("TTS")])
    assert check_pipeline_errors(repeated)[0].check == "pipeline_errors"
    fatal = run(timeline=[err("LLM", fatal=True)])
    assert "fatal" in check_pipeline_errors(fatal)[0].detail


def test_legacy_run_counts_as_insufficient_never_flagged():
    legacy = run(run_id=2, first_bot=None, has_events=False, timeline=[])
    no_bot = run(run_id=3, first_bot=None)
    generic = run(run_id=4, first_bot="Hello! How can I help you today?")
    summary = run_layer0([legacy, no_bot, generic], {10: WORKFLOW})
    assert summary.runs_checked == 3
    assert summary.runs_flagged == 1
    assert {f.run_id for f in summary.flags} == {4}
    assert summary.insufficient_data["greeting_mismatch"] == 2
    assert summary.insufficient_data["system_prompt_missing"] == 3
    assert summary.insufficient_data["pipeline_errors"] == 1
