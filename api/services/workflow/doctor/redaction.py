"""Per-run PII redaction before anything is sent to Claude.

Scope (design doc "Safety, privacy, limits"): transcripts, tool results and
call variables. Phone numbers, emails, card-like numbers and the values of
call variables that look like PII are replaced with **consistent placeholder
tokens per run** (``<PHONE_1>``, ``<EMAIL_1>``, ``<CARD_1>``), so a correct
read-back still matches its source and the Doctor can tell
"repeated correctly" from "hallucinated".

Not masked in v1 (accepted, disclosed in the opt-in text): names, addresses
and account identifiers spoken in free text, and numbers spoken as words.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# 13-19 digits, optionally grouped by single spaces or dashes.
_CARD_RE = re.compile(r"(?<![\w+])(?:\d[ \-]?){12,18}\d(?!\w)")
# 7-15 digits with an optional leading +, separators and one bracketed group.
_PHONE_RE = re.compile(r"(?<![\w])\+?(?:\(?\d[\d\s().\-]{5,18}\d)(?!\w)")
_DECIMAL_OR_DATE_RE = re.compile(
    r"\d+\.\d+|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
)
_TOKEN_RE = re.compile(r"<(?:PHONE|EMAIL|CARD|PII)_\d+>")

#: Call-variable keys whose values are masked whole (beyond regex matches).
PII_KEY_MARKERS = (
    "phone",
    "mobile",
    "email",
    "card",
    "ssn",
    "dob",
    "date_of_birth",
    "account_number",
    "iban",
    "caller_number",
    "called_number",
    "from_number",
    "to_number",
)


#: Timeline keys that never carry PII (timestamps would look like numbers).
_SKIP_KEYS = frozenset({"t", "turn", "kind", "secs", "total", "user_turn"})


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


class RunRedactor:
    """Stateful redactor for one run: same value → same token."""

    def __init__(self, pii_values: Optional[Iterable[str]] = None) -> None:
        self._tokens: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}
        self._literals: list[tuple[str, str]] = []
        for value in pii_values or ():
            self.add_pii_value(value)

    # ------------------------------------------------------------------ tokens
    def _token(self, kind: str, key: str) -> str:
        ident = (kind, key)
        if ident not in self._tokens:
            self._counters[kind] = self._counters.get(kind, 0) + 1
            self._tokens[ident] = f"<{kind}_{self._counters[kind]}>"
        return self._tokens[ident]

    def _phone_key(self, raw: str) -> str:
        digits = _digits(raw)
        # "+1 555 123 4567" and "5551234567" are the same number.
        return digits[-10:] if len(digits) > 10 else digits

    def add_pii_value(self, value: Any) -> None:
        """Mask an exact call-variable value wherever it appears."""
        if value is None or isinstance(value, bool):
            return
        text = str(value).strip()
        if len(text) < 3 or _TOKEN_RE.fullmatch(text):
            return
        if _EMAIL_RE.fullmatch(text):
            token = self._token("EMAIL", text.lower())
        elif 7 <= len(_digits(text)) <= 15 and not re.search(r"[A-Za-z]", text):
            token = self._token("PHONE", self._phone_key(text))
        else:
            token = self._token("PII", text.lower())
        if all(lit != text for lit, _ in self._literals):
            self._literals.append((text, token))
            # Longest first so a value containing another is replaced whole.
            self._literals.sort(key=lambda item: len(item[0]), reverse=True)

    # --------------------------------------------------------------- redaction
    def redact_text(self, text: Optional[str]) -> Optional[str]:
        if not text or not isinstance(text, str):
            return text
        for literal, token in self._literals:
            if literal in text:
                text = text.replace(literal, token)

        text = _EMAIL_RE.sub(lambda m: self._token("EMAIL", m.group().lower()), text)

        def _card(m: re.Match) -> str:
            digits = _digits(m.group())
            if 13 <= len(digits) <= 19 and _luhn_ok(digits):
                return self._token("CARD", digits)
            return m.group()

        text = _CARD_RE.sub(_card, text)

        def _phone(m: re.Match) -> str:
            raw = m.group()
            digits = _digits(raw)
            if not 7 <= len(digits) <= 15:
                return raw
            # Leave decimals and dates alone.
            if _DECIMAL_OR_DATE_RE.fullmatch(raw.strip()):
                return raw
            return self._token("PHONE", self._phone_key(raw))

        return _PHONE_RE.sub(_phone, text)

    def redact(self, obj: Any) -> Any:
        """Recursively redact strings inside dicts/lists (keys are kept)."""
        if isinstance(obj, str):
            return self.redact_text(obj)
        if isinstance(obj, dict):
            return {k: v if k in _SKIP_KEYS else self.redact(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.redact(v) for v in obj]
        if isinstance(obj, int) and not isinstance(obj, bool):
            redacted = self.redact_text(str(obj))
            return obj if redacted == str(obj) else redacted
        return obj

    def redact_call_variables(self, variables: Any) -> Any:
        """Register PII-looking keys' values, then redact the whole mapping."""
        self.register_call_variables(variables)
        return self.redact(variables)

    def register_call_variables(self, variables: Any) -> None:
        if not isinstance(variables, dict):
            return
        for key, value in variables.items():
            if isinstance(value, dict):
                self.register_call_variables(value)
            elif any(marker in str(key).lower() for marker in PII_KEY_MARKERS):
                self.add_pii_value(value)

    @property
    def mapping(self) -> dict[str, str]:
        """token → kind (never the raw value; safe to log)."""
        return {token: kind for (kind, _), token in self._tokens.items()}


def redact_run(run_view: dict) -> dict:
    """Redact a ``timeline.build_run_timeline`` dict for sending to Claude.

    Call variables are registered first so their exact values are masked in
    transcripts and tool results with the same token.
    """
    redactor = RunRedactor()
    redactor.register_call_variables(run_view.get("initial_context"))
    redactor.register_call_variables(run_view.get("gathered_context"))
    out = dict(run_view)
    out["initial_context"] = redactor.redact(run_view.get("initial_context") or {})
    out["extracted_variables"] = redactor.redact(run_view.get("extracted_variables"))
    out["timeline"] = redactor.redact(run_view.get("timeline") or [])
    out.pop("gathered_context", None)
    out.pop("usage_info", None)
    return out
