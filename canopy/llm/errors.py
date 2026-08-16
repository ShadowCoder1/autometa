"""Errors raised by the LLM layer (kept separate so providers and client can share them)."""
from __future__ import annotations


class LLMError(RuntimeError):
    """Base class for every LLM-layer failure."""


class MissingFixture(LLMError):
    """A replayed request has no fixture and live calls are not enabled."""


class LiveCallsDisabled(LLMError):
    """A network call was needed but `CANOPY_LIVE=1` / `allow_live=True` was not set."""


class RefusalError(LLMError):
    """The model refused (`stop_reason == "refusal"`)."""


class TruncatedOutput(LLMError):
    """The model hit `max_tokens` (even after one retry with twice the budget)."""


class BudgetExceeded(LLMError):
    """The configured USD budget would be exceeded by this call."""


class ParseError(LLMError):
    """A structured response was not valid JSON."""
