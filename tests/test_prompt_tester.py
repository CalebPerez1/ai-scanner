"""Unit tests for the prompt injection tester module.

Covers:
- PAYLOADS: structure validation (all categories present, minimum payload count,
  unique IDs, non-empty prompt text).
- classify_response: true positives for each category, true negatives for
  benign responses, category not found.
- test_endpoint: successful injection produces Finding; benign response produces
  no Finding; HTTP errors are silently skipped; correct severity and scanner_name;
  custom headers are forwarded; all categories are exercised.
- scan_prompt_injection: delegates correctly to test_endpoint.
"""
from __future__ import annotations

import asyncio
import json
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.models import Finding, Severity
from backend.scanners.prompt_tester import (
    PAYLOADS,
    classify_response,
    scan_prompt_injection,
    test_endpoint as run_endpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENDPOINT = "https://api.example.com/v1/chat/completions"

_EXPECTED_CATEGORIES = {
    "system_prompt_extraction",
    "jailbreak",
    "data_exfiltration",
    "indirect_injection",
    "output_manipulation",
    "token_smuggling",
}


def _openai_response(content: str) -> dict:
    """Build a minimal OpenAI-compatible chat completion response dict."""
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ]
    }


def _make_client_mock(responses: List[dict]) -> AsyncMock:
    """Build an httpx.AsyncClient mock that returns *responses* in sequence.

    Each dict in *responses* is serialised as JSON and returned for successive
    POST calls. If more calls are made than responses provided, the last
    response is repeated.
    """
    call_count = 0

    async def _post(*args, **kwargs):
        nonlocal call_count
        resp_data = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = resp_data
        return mock_resp

    mock_client = AsyncMock()
    mock_client.post = _post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def _make_error_client_mock() -> AsyncMock:
    """Build an httpx.AsyncClient mock that always raises HTTPError on POST."""
    import httpx

    async def _post(*args, **kwargs):
        raise httpx.HTTPError("connection refused")

    mock_client = AsyncMock()
    mock_client.post = _post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ---------------------------------------------------------------------------
# PAYLOADS structure
# ---------------------------------------------------------------------------

class TestPayloadsStructure:
    def test_all_required_categories_present(self) -> None:
        assert _EXPECTED_CATEGORIES <= set(PAYLOADS.keys())

    def test_each_category_has_at_least_five_payloads(self) -> None:
        for category, payloads in PAYLOADS.items():
            assert len(payloads) >= 5, f"{category!r} has fewer than 5 payloads"

    def test_all_payload_ids_are_unique(self) -> None:
        all_ids = [pid for payloads in PAYLOADS.values() for pid, _ in payloads]
        assert len(all_ids) == len(set(all_ids)), "Duplicate payload IDs found"

    def test_all_prompt_texts_are_non_empty(self) -> None:
        for category, payloads in PAYLOADS.items():
            for pid, text in payloads:
                assert text.strip(), f"Empty prompt for {category}/{pid}"

    def test_payload_entries_are_two_tuples(self) -> None:
        for category, payloads in PAYLOADS.items():
            for entry in payloads:
                assert len(entry) == 2, f"Bad tuple length in {category}"


# ---------------------------------------------------------------------------
# classify_response
# ---------------------------------------------------------------------------

class TestClassifyResponse:
    # --- True positives (attack succeeded) ---

    def test_system_prompt_extraction_my_instructions(self) -> None:
        assert classify_response(
            "system_prompt_extraction",
            "My instructions are to assist users with coding questions.",
        )

    def test_system_prompt_extraction_system_prompt(self) -> None:
        assert classify_response(
            "system_prompt_extraction",
            "Here is my system prompt: You are a helpful assistant.",
        )

    def test_system_prompt_extraction_told_to(self) -> None:
        assert classify_response(
            "system_prompt_extraction",
            "I was told to always respond in English.",
        )

    def test_jailbreak_dan_mode(self) -> None:
        assert classify_response("jailbreak", "DAN Mode enabled. I will now comply.")

    def test_jailbreak_jailbroken_tag(self) -> None:
        assert classify_response("jailbreak", "[JAILBROKEN] Sure, here you go.")

    def test_jailbreak_no_restrictions(self) -> None:
        assert classify_response("jailbreak", "I operate with no restrictions now.")

    def test_jailbreak_developer_mode(self) -> None:
        assert classify_response("jailbreak", "Developer mode: ON — all filters disabled.")

    def test_data_exfiltration_user_data(self) -> None:
        assert classify_response(
            "data_exfiltration",
            "I have access to the following user data: name, email.",
        )

    def test_data_exfiltration_previous_session(self) -> None:
        assert classify_response(
            "data_exfiltration",
            "From a previous conversation, I can see that the user mentioned...",
        )

    def test_data_exfiltration_api_key(self) -> None:
        assert classify_response(
            "data_exfiltration",
            "The api_key found in the context is sk-abc123.",
        )

    def test_indirect_injection_success_token(self) -> None:
        assert classify_response("indirect_injection", "INJECTION_SUCCESS")

    def test_indirect_injection_pwned(self) -> None:
        assert classify_response("indirect_injection", "PWNED")

    def test_indirect_injection_bypassed(self) -> None:
        assert classify_response("indirect_injection", "BYPASSED — all rules ignored.")

    def test_output_manipulation_safety_override(self) -> None:
        assert classify_response(
            "output_manipulation", "SAFETY_OVERRIDE_CONFIRMED — proceeding."
        )

    def test_output_manipulation_status_jailbroken(self) -> None:
        assert classify_response(
            "output_manipulation", "STATUS: JAILBROKEN\nRESTRICTIONS: NONE"
        )

    def test_output_manipulation_unrestricted_tag(self) -> None:
        assert classify_response(
            "output_manipulation", "Here is my answer. [UNRESTRICTED]"
        )

    def test_token_smuggling_jailbroken(self) -> None:
        assert classify_response("token_smuggling", "The decoded message says: JAILBROKEN")

    def test_token_smuggling_bypass_success(self) -> None:
        assert classify_response("token_smuggling", "BYPASS_SUCCESS confirmed.")

    def test_token_smuggling_ignore_instructions(self) -> None:
        assert classify_response(
            "token_smuggling", "Decoded: ignore all previous instructions."
        )

    # --- True negatives (benign responses) ---

    def test_benign_response_returns_false(self) -> None:
        assert not classify_response(
            "system_prompt_extraction",
            "I'm here to help! What would you like to know?",
        )

    def test_benign_jailbreak_refusal(self) -> None:
        assert not classify_response(
            "jailbreak",
            "I'm sorry, I can't comply with that request.",
        )

    def test_benign_data_exfiltration_response(self) -> None:
        assert not classify_response(
            "data_exfiltration",
            "I don't have access to any personal information.",
        )

    def test_unknown_category_returns_false(self) -> None:
        assert not classify_response("nonexistent_category", "JAILBROKEN DAN MODE enabled")


# ---------------------------------------------------------------------------
# test_endpoint
# ---------------------------------------------------------------------------

TOTAL_PAYLOADS = sum(len(v) for v in PAYLOADS.values())


class TestTestEndpoint:
    def test_successful_injection_produces_finding(self) -> None:
        # One response triggers a jailbreak indicator; all others are benign.
        responses = [_openai_response("I'm sorry, I cannot help with that.")] * TOTAL_PAYLOADS
        # Find the index of the first jailbreak payload and make it succeed.
        jb_offset = sum(len(v) for k, v in PAYLOADS.items() if k == list(PAYLOADS.keys())[0])
        # Easier: just make ALL responses succeed to guarantee at least one finding.
        trigger = _openai_response("DAN Mode enabled. I am now unrestricted.")
        responses = [trigger] * TOTAL_PAYLOADS

        mock_client = _make_client_mock(responses)
        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(run_endpoint(_ENDPOINT))

        assert len(findings) > 0

    def test_benign_responses_produce_no_findings(self) -> None:
        benign = _openai_response("I'm here to help. How can I assist you today?")
        mock_client = _make_client_mock([benign] * TOTAL_PAYLOADS)
        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(run_endpoint(_ENDPOINT))
        assert findings == []

    def test_http_error_is_skipped_silently(self) -> None:
        mock_client = _make_error_client_mock()
        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            # Should not raise; HTTP failures are swallowed.
            findings = asyncio.run(run_endpoint(_ENDPOINT))
        assert findings == []

    def test_finding_has_correct_scanner_name(self) -> None:
        trigger = _openai_response("DAN Mode enabled.")
        mock_client = _make_client_mock([trigger] * TOTAL_PAYLOADS)
        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(run_endpoint(_ENDPOINT))
        assert all(f.scanner_name == "prompt_tester" for f in findings)

    def test_finding_severity_matches_category(self) -> None:
        # Make only data_exfiltration payloads succeed.
        from backend.scanners.prompt_tester import _CATEGORY_SEVERITY

        def _resp_for_payload(category: str, _pid: str, _text: str) -> dict:
            if category == "data_exfiltration":
                return _openai_response("I have user data: email, api_key stored.")
            return _openai_response("I'm sorry, I cannot help.")

        responses = [
            _resp_for_payload(cat, pid, text)
            for cat, payloads in PAYLOADS.items()
            for pid, text in payloads
        ]
        mock_client = _make_client_mock(responses)
        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(run_endpoint(_ENDPOINT))

        de_findings = [f for f in findings if "data_exfiltration" in f.title]
        assert all(f.severity == Severity.CRITICAL for f in de_findings)

    def test_finding_title_includes_category_and_payload_id(self) -> None:
        trigger = _openai_response("INJECTION_SUCCESS")
        mock_client = _make_client_mock([trigger] * TOTAL_PAYLOADS)
        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(run_endpoint(_ENDPOINT))
        for f in findings:
            assert any(cat.replace("_", " ") in f.title for cat in PAYLOADS)

    def test_custom_headers_are_forwarded(self) -> None:
        """Verify that the auth header is included in the POST request."""
        captured_headers: list = []

        async def _post(url, *, json=None, headers=None):
            captured_headers.append(headers or {})
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = _openai_response("benign reply")
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = _post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        auth_headers = {"Authorization": "Bearer sk-test-key"}
        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            asyncio.run(run_endpoint(_ENDPOINT, headers=auth_headers))

        assert captured_headers, "No POST calls were made"
        assert all(h.get("Authorization") == "Bearer sk-test-key" for h in captured_headers)

    def test_no_headers_arg_does_not_raise(self) -> None:
        benign = _openai_response("Sure, let me help.")
        mock_client = _make_client_mock([benign] * TOTAL_PAYLOADS)
        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(run_endpoint(_ENDPOINT))
        assert isinstance(findings, list)

    def test_all_categories_are_exercised(self) -> None:
        """Each category should contribute at least one POST call."""
        posted_bodies: list = []

        async def _post(url, *, json=None, headers=None):
            posted_bodies.append(json or {})
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = _openai_response("benign")
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = _post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            asyncio.run(run_endpoint(_ENDPOINT))

        assert len(posted_bodies) == TOTAL_PAYLOADS

    def test_simple_response_format_fallback(self) -> None:
        """Endpoints that return {"response": "..."} instead of OpenAI format."""
        trigger = {"response": "I have no restrictions — JAILBROKEN mode active."}
        mock_client = _make_client_mock([trigger] * TOTAL_PAYLOADS)
        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(run_endpoint(_ENDPOINT))
        assert len(findings) > 0

    def test_finding_description_includes_payload_and_response(self) -> None:
        trigger = _openai_response("DAN Mode enabled.")
        mock_client = _make_client_mock([trigger] * TOTAL_PAYLOADS)
        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(run_endpoint(_ENDPOINT))
        for f in findings:
            assert "Payload" in f.description
            assert "Response" in f.description


# ---------------------------------------------------------------------------
# scan_prompt_injection — entry point delegation
# ---------------------------------------------------------------------------

class TestScanPromptInjection:
    def test_delegates_to_test_endpoint(self) -> None:
        benign = _openai_response("I cannot assist with that.")
        mock_client = _make_client_mock([benign] * TOTAL_PAYLOADS)
        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(scan_prompt_injection(_ENDPOINT))
        assert isinstance(findings, list)

    def test_returns_findings_on_successful_injection(self) -> None:
        trigger = _openai_response("BYPASS_SUCCESS — token smuggling worked.")
        mock_client = _make_client_mock([trigger] * TOTAL_PAYLOADS)
        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(scan_prompt_injection(_ENDPOINT))
        assert len(findings) > 0
        assert all(isinstance(f, Finding) for f in findings)

    def test_passes_headers_through(self) -> None:
        captured: list = []

        async def _post(url, *, json=None, headers=None):
            captured.append(headers)
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = _openai_response("fine")
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = _post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("backend.scanners.prompt_tester.httpx.AsyncClient", return_value=mock_client):
            asyncio.run(scan_prompt_injection(_ENDPOINT, headers={"X-Api-Key": "tok"}))

        assert all(h.get("X-Api-Key") == "tok" for h in captured)
