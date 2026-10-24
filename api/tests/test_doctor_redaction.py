"""Doctor per-run redaction: consistent placeholder tokens."""

from api.services.workflow.doctor.redaction import RunRedactor, redact_run


def test_same_value_same_token_within_a_run():
    r = RunRedactor()
    a = r.redact_text("Call me on +1 (555) 123-4567 or mail ann@example.com")
    b = r.redact_text("So that's 555 123 4567 and ANN@example.com, right?")
    assert a == "Call me on <PHONE_1> or mail <EMAIL_1>"
    assert b == "So that's <PHONE_1> and <EMAIL_1>, right?"
    c = r.redact_text("Other number 555 987 6543")
    assert c == "Other number <PHONE_2>"


def test_card_numbers_luhn_and_dates_untouched():
    r = RunRedactor()
    out = r.redact_text("Card 4111 1111 1111 1111 expires on 2026-10-01")
    assert out == "Card <CARD_1> expires on 2026-10-01"
    assert r.redact_text("Price is 1234.50") == "Price is 1234.50"


def test_new_run_restarts_numbering_and_timeline_keys_preserved():
    view = {
        "run_id": 1,
        "initial_context": {"phone_number": "+15551234567", "first_name": "Ann"},
        "gathered_context": {"extracted_variables": {}},
        "extracted_variables": {"callback": "5551234567"},
        "timeline": [
            {
                "turn": 1,
                "node": "Start",
                "t": "2026-10-01T10:00:00+00:00",
                "kind": "user",
                "text": "my number is 555-123-4567",
            },
            {
                "turn": 1,
                "node": "Start",
                "t": None,
                "kind": "tool_result",
                "fn": "lookup",
                "result": {"phone": "+1 555 123 4567", "email": "a@b.co"},
            },
        ],
        "usage_info": {"llm": {}},
    }
    out = redact_run(view)
    assert out["initial_context"]["phone_number"] == "<PHONE_1>"
    assert out["initial_context"]["first_name"] == "Ann"  # residual PII, disclosed
    assert out["extracted_variables"]["callback"] == "<PHONE_1>"
    assert out["timeline"][0]["text"] == "my number is <PHONE_1>"
    assert out["timeline"][0]["t"] == "2026-10-01T10:00:00+00:00"
    assert out["timeline"][1]["result"] == {"phone": "<PHONE_1>", "email": "<EMAIL_1>"}
    assert "usage_info" not in out and "gathered_context" not in out
    # Original untouched.
    assert view["initial_context"]["phone_number"] == "+15551234567"

    other = redact_run({"run_id": 2, "timeline": [{"kind": "user", "text": "a@b.co"}]})
    assert other["timeline"][0]["text"] == "<EMAIL_1>"


def test_registered_pii_variable_value_masked_in_free_text():
    r = RunRedactor()
    r.register_call_variables({"account_number": "AC-99812"})
    assert r.redact_text("Your account AC-99812 is active") == (
        "Your account <PII_1> is active"
    )
