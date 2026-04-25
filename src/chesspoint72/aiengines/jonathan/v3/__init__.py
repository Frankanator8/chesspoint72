"""Calix v3 — Module Selector Agent driven by Claude Sonnet 4.6.

The package layout mirrors v1; the only behavioural difference is in
``agent.py``, which calls the Anthropic API to produce the EngineConfig
instead of running a deterministic rule cascade. If the API is
unavailable (no key, no network, malformed response) the agent silently
falls back to v1's rule-based defaults so the engine still plays.
"""
