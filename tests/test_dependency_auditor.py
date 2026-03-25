"""Unit tests for the dependency auditor scanner module.

Covers:
- parse_requirements: normal pins, extras, missing version, comments,
  blank lines, pip options, VCS URLs with and without #egg=, env markers,
  FileNotFoundError, string path input.
- detect_typosquats: clean package, dist-1 (HIGH), dist-2 (MEDIUM),
  exact popular match (no finding), unrelated package.
- query_osv_for_package: OSV response mapped to Finding objects,
  severity thresholds (CRITICAL/HIGH/MEDIUM/LOW), missing details
  fallback, no-version query, HTTP error propagation.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.core.models import Finding, Severity
from backend.scanners.dependency_auditor import (
    detect_typosquats,
    parse_requirements,
    query_osv_for_package,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_osv_client_mock(response_data: dict) -> AsyncMock:
    """Build a mock httpx.AsyncClient that returns *response_data* from POST.

    The returned object satisfies the ``async with httpx.AsyncClient() as c``
    protocol used inside query_osv_for_package.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = response_data

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def _osv_vuln(
    vuln_id: str = "GHSA-0000-0000-0000",
    summary: str = "A vulnerability",
    details: str | None = None,
    cvss_score: float | None = None,
    ref_url: str | None = None,
) -> dict:
    """Construct a minimal OSV vulnerability dict."""
    v: dict = {"id": vuln_id, "summary": summary}
    if details:
        v["details"] = details
    if cvss_score is not None:
        v["severity"] = [{"type": "CVSS_V3", "score": str(cvss_score)}]
    if ref_url:
        v["references"] = [{"url": ref_url}]
    return v


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def requirements_file(tmp_path: Path) -> Path:
    """Write a realistic requirements.txt covering many edge cases."""
    content = (
        "# --- production deps ---\n"
        "requests==2.28.0\n"
        "flask==2.3.1\n"
        "\n"                                              # blank line
        "numpy\n"                                         # no version pin
        "pydantic[email]==1.10.0\n"                       # extras
        "scipy >= 1.9.0\n"                                # inequality (no exact pin)
        'requests==2.28.0; python_version >= "3.6"\n'    # env marker — same pkg, ensures version still parsed
        "\n"
        "# --- pip options (should be skipped) ---\n"
        "-r other-requirements.txt\n"
        "--index-url https://pypi.org/simple/\n"
        "\n"
        "# --- VCS (egg present → extracted) ---\n"
        "git+https://github.com/psf/requests.git#egg=requests-dev\n"
        "\n"
        "# --- VCS (no egg → skipped) ---\n"
        "git+https://github.com/foo/bar.git\n"
        "\n"
        "# --- plain HTTPS URL (skipped) ---\n"
        "https://example.com/mypackage-1.0.tar.gz\n"
    )
    req_file = tmp_path / "requirements.txt"
    req_file.write_text(content, encoding="utf-8")
    return req_file


# ---------------------------------------------------------------------------
# parse_requirements
# ---------------------------------------------------------------------------

class TestParseRequirements:
    def test_pinned_packages_captured(self, requirements_file: Path) -> None:
        results = parse_requirements(requirements_file)
        assert ("requests", "2.28.0") in results
        assert ("flask", "2.3.1") in results

    def test_unpinned_package_has_none_version(self, requirements_file: Path) -> None:
        results = parse_requirements(requirements_file)
        numpy_entries = [(n, v) for n, v in results if n == "numpy"]
        assert len(numpy_entries) >= 1
        assert numpy_entries[0][1] is None

    def test_extras_stripped_from_name(self, requirements_file: Path) -> None:
        results = parse_requirements(requirements_file)
        names = [n for n, _ in results]
        # The name should be "pydantic", not "pydantic[email]"
        assert "pydantic" in names
        assert not any("[" in n for n in names)

    def test_extras_version_preserved(self, requirements_file: Path) -> None:
        results = parse_requirements(requirements_file)
        entry = next(((n, v) for n, v in results if n == "pydantic"), None)
        assert entry is not None
        assert entry[1] == "1.10.0"

    def test_environment_marker_does_not_affect_version(self, requirements_file: Path) -> None:
        # "requests==2.28.0; python_version >= '3.6'" must yield version "2.28.0"
        results = parse_requirements(requirements_file)
        versioned = [(n, v) for n, v in results if n == "requests" and v is not None]
        assert all(v == "2.28.0" for _, v in versioned)

    def test_comments_skipped(self, requirements_file: Path) -> None:
        results = parse_requirements(requirements_file)
        names = [n for n, _ in results]
        # Comment text must never appear as a package name
        assert "---" not in names
        assert "production" not in names

    def test_blank_lines_do_not_produce_entries(self, tmp_path: Path) -> None:
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("\n\n\nrequests==2.28.0\n\n\n", encoding="utf-8")
        results = parse_requirements(req_file)
        assert results == [("requests", "2.28.0")]

    def test_pip_options_skipped(self, requirements_file: Path) -> None:
        results = parse_requirements(requirements_file)
        names = [n for n, _ in results]
        assert "other-requirements.txt" not in names
        assert "https" not in names

    def test_vcs_url_with_egg_extracted(self, requirements_file: Path) -> None:
        results = parse_requirements(requirements_file)
        names = [n for n, _ in results]
        assert "requests-dev" in names

    def test_vcs_url_without_egg_skipped(self, requirements_file: Path) -> None:
        results = parse_requirements(requirements_file)
        names = [n for n, _ in results]
        assert "bar" not in names

    def test_plain_https_url_skipped(self, requirements_file: Path) -> None:
        results = parse_requirements(requirements_file)
        names = [n for n, _ in results]
        assert "mypackage-1.0.tar.gz" not in names

    def test_accepts_string_path(self, requirements_file: Path) -> None:
        # parse_requirements must accept both Path and str
        results = parse_requirements(str(requirements_file))
        assert ("flask", "2.3.1") in results

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_requirements(tmp_path / "does_not_exist.txt")

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("", encoding="utf-8")
        assert parse_requirements(req_file) == []

    def test_only_comments_returns_empty_list(self, tmp_path: Path) -> None:
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("# comment only\n# another\n", encoding="utf-8")
        assert parse_requirements(req_file) == []


# ---------------------------------------------------------------------------
# detect_typosquats
# ---------------------------------------------------------------------------

class TestDetectTyposquats:
    def test_exact_popular_name_returns_no_findings(self) -> None:
        # "requests" is in POPULAR_PYPI_PACKAGES — exact match is not a typosquat
        findings = detect_typosquats("requests")
        assert findings == []

    def test_unrelated_package_returns_no_findings(self) -> None:
        findings = detect_typosquats("completelyrandomxyzpackage99")
        assert findings == []

    def test_distance_one_typosquat_is_high_severity(self) -> None:
        # "reqests" is edit-distance 1 from "requests" (missing 'u')
        findings = detect_typosquats("reqests")
        assert len(findings) >= 1
        hit = next((f for f in findings if "requests" in f.title), None)
        assert hit is not None
        assert hit.severity == Severity.HIGH

    def test_distance_two_typosquat_is_medium_severity(self) -> None:
        # "reqeusts" is edit-distance 2 from "requests" (eu ↔ ue transposition)
        findings = detect_typosquats("reqeusts")
        assert len(findings) >= 1
        hit = next((f for f in findings if "requests" in f.title), None)
        assert hit is not None
        assert hit.severity == Severity.MEDIUM

    def test_finding_fields_are_populated(self) -> None:
        findings = detect_typosquats("reqests")
        assert len(findings) >= 1
        f = findings[0]
        assert isinstance(f, Finding)
        assert f.scanner_name == "dependency_auditor"
        assert f.recommendation is not None
        assert "reqests" in f.description

    def test_case_insensitive_comparison(self) -> None:
        # "Requests" (capitalised) is an exact match after lowercasing → no finding
        findings = detect_typosquats("Requests")
        assert findings == []

    def test_typosquat_near_flask(self) -> None:
        # "flassk" is edit-distance 1 from "flask"
        findings = detect_typosquats("flassk")
        hit = next((f for f in findings if "flask" in f.title), None)
        assert hit is not None
        assert hit.severity == Severity.HIGH


# ---------------------------------------------------------------------------
# query_osv_for_package
# ---------------------------------------------------------------------------

class TestQueryOsvForPackage:
    """All HTTP calls are mocked via unittest.mock.patch."""

    def test_no_vulns_returns_empty_list(self) -> None:
        mock_client = _make_osv_client_mock({"vulns": []})
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("requests", "2.28.0"))
        assert result == []

    def test_missing_vulns_key_returns_empty_list(self) -> None:
        # OSV may return {} when there are no results
        mock_client = _make_osv_client_mock({})
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("flask", "1.0.0"))
        assert result == []

    def test_single_vuln_produces_one_finding(self) -> None:
        mock_client = _make_osv_client_mock(
            {"vulns": [_osv_vuln("GHSA-aaaa-bbbb-cccc", cvss_score=9.8)]}
        )
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("requests", "2.28.0"))
        assert len(result) == 1

    def test_finding_type_is_finding(self) -> None:
        mock_client = _make_osv_client_mock(
            {"vulns": [_osv_vuln("CVE-2023-0001")]}
        )
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("django", "3.2.0"))
        assert all(isinstance(f, Finding) for f in result)

    def test_scanner_name_is_dependency_auditor(self) -> None:
        mock_client = _make_osv_client_mock({"vulns": [_osv_vuln()]})
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("django", "3.2.0"))
        assert result[0].scanner_name == "dependency_auditor"

    def test_vuln_id_in_title(self) -> None:
        mock_client = _make_osv_client_mock(
            {"vulns": [_osv_vuln("GHSA-1234-5678-9abc")]}
        )
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("requests", "2.28.0"))
        assert "GHSA-1234-5678-9abc" in result[0].title

    def test_reference_url_becomes_recommendation(self) -> None:
        ref = "https://github.com/advisories/GHSA-1234"
        mock_client = _make_osv_client_mock(
            {"vulns": [_osv_vuln(ref_url=ref)]}
        )
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("requests", "2.28.0"))
        assert result[0].recommendation == ref

    def test_no_references_leaves_recommendation_none(self) -> None:
        mock_client = _make_osv_client_mock({"vulns": [_osv_vuln()]})
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("flask", "2.0.0"))
        assert result[0].recommendation is None

    # -- Severity mapping --

    def test_severity_critical_at_9_or_above(self) -> None:
        for score in (9.0, 9.8, 10.0):
            mock_client = _make_osv_client_mock({"vulns": [_osv_vuln(cvss_score=score)]})
            with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
                result = asyncio.run(query_osv_for_package("pkg", "1.0"))
            assert result[0].severity == Severity.CRITICAL, f"score={score} should be CRITICAL"

    def test_severity_high_between_7_and_9(self) -> None:
        for score in (7.0, 7.5, 8.9):
            mock_client = _make_osv_client_mock({"vulns": [_osv_vuln(cvss_score=score)]})
            with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
                result = asyncio.run(query_osv_for_package("pkg", "1.0"))
            assert result[0].severity == Severity.HIGH, f"score={score} should be HIGH"

    def test_severity_medium_between_4_and_7(self) -> None:
        for score in (4.0, 5.5, 6.9):
            mock_client = _make_osv_client_mock({"vulns": [_osv_vuln(cvss_score=score)]})
            with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
                result = asyncio.run(query_osv_for_package("pkg", "1.0"))
            assert result[0].severity == Severity.MEDIUM, f"score={score} should be MEDIUM"

    def test_severity_low_below_4(self) -> None:
        for score in (0.0, 2.5, 3.9):
            mock_client = _make_osv_client_mock({"vulns": [_osv_vuln(cvss_score=score)]})
            with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
                result = asyncio.run(query_osv_for_package("pkg", "1.0"))
            assert result[0].severity == Severity.LOW, f"score={score} should be LOW"

    def test_severity_defaults_to_medium_when_absent(self) -> None:
        # No "severity" key in OSV response → should fall back to MEDIUM
        mock_client = _make_osv_client_mock({"vulns": [_osv_vuln()]})
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("flask", "2.0.0"))
        assert result[0].severity == Severity.MEDIUM

    def test_non_numeric_cvss_score_falls_back_to_medium(self) -> None:
        vuln = {
            "id": "CVE-2023-9999",
            "summary": "test",
            "severity": [{"type": "CVSS_V3", "score": "NONE"}],
        }
        mock_client = _make_osv_client_mock({"vulns": [vuln]})
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("pkg", "1.0"))
        assert result[0].severity == Severity.MEDIUM

    # -- details / summary fallback --

    def test_details_field_used_when_present(self) -> None:
        mock_client = _make_osv_client_mock(
            {"vulns": [_osv_vuln(summary="short", details="Long detailed description")]}
        )
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("pkg", "1.0"))
        assert result[0].description == "Long detailed description"

    def test_summary_used_as_fallback_when_details_absent(self) -> None:
        mock_client = _make_osv_client_mock(
            {"vulns": [_osv_vuln(summary="Only a summary", details=None)]}
        )
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("pkg", "1.0"))
        assert result[0].description == "Only a summary"

    def test_multiple_vulns_produce_multiple_findings(self) -> None:
        vulns = [_osv_vuln(f"CVE-2023-000{i}") for i in range(3)]
        mock_client = _make_osv_client_mock({"vulns": vulns})
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("requests", "2.28.0"))
        assert len(result) == 3

    # -- version handling --

    def test_query_without_version_succeeds(self) -> None:
        mock_client = _make_osv_client_mock({"vulns": []})
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(query_osv_for_package("requests", None))
        assert result == []

    def test_query_without_version_omits_version_from_payload(self) -> None:
        mock_client = _make_osv_client_mock({"vulns": []})
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            asyncio.run(query_osv_for_package("requests", None))
        posted_payload = mock_client.post.call_args.kwargs["json"]
        assert "version" not in posted_payload

    def test_query_with_version_includes_version_in_payload(self) -> None:
        mock_client = _make_osv_client_mock({"vulns": []})
        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            asyncio.run(query_osv_for_package("requests", "2.28.0"))
        posted_payload = mock_client.post.call_args.kwargs["json"]
        assert posted_payload["version"] == "2.28.0"

    # -- error handling --

    def test_http_status_error_propagates(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("backend.scanners.dependency_auditor.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(query_osv_for_package("requests", "2.28.0"))
