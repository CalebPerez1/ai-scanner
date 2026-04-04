"""Unit tests for backend.core.risk_engine.

Tests cover:
- run_all_scanners: orchestrates three file-system scanners concurrently;
  optionally adds the prompt injection scanner; collects all findings;
  survives individual scanner exceptions without aborting.
- generate_scan_result: creates a ScanResult with correct summary fields.
- scan_project: resolves local paths and GitHub URLs; derives project names;
  cleans up temporary clone directories after scanning.
- _is_git_url / _derive_project_name: URL detection and name derivation helpers.
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from backend.core.models import Finding, ScanResult, Severity
from backend.core.risk_engine import (
    _derive_project_name,
    _is_git_url,
    generate_scan_result,
    run_all_scanners,
    scan_project,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _finding(title: str, severity: Severity = Severity.LOW, scanner: str = "test") -> Finding:
    """Build a minimal Finding for use in tests."""
    return Finding(
        title=title,
        description=f"Description for {title}",
        severity=severity,
        scanner_name=scanner,
    )


@pytest.fixture()
def dep_findings() -> List[Finding]:
    return [_finding("CVE in numpy", Severity.HIGH, "dependency_auditor")]


@pytest.fixture()
def model_findings() -> List[Finding]:
    return [_finding("Unsafe torch.load", Severity.HIGH, "model_scanner")]


@pytest.fixture()
def config_findings() -> List[Finding]:
    return [_finding("DEBUG=True", Severity.MEDIUM, "config_analyzer")]


@pytest.fixture()
def prompt_findings() -> List[Finding]:
    return [_finding("Jailbreak succeeded", Severity.HIGH, "prompt_tester")]


# ---------------------------------------------------------------------------
# _is_git_url
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    # GitHub
    ("https://github.com/user/repo", True),
    ("http://github.com/user/repo", True),
    ("git@github.com:user/repo.git", True),
    ("git+https://github.com/user/repo.git", True),
    # GitLab
    ("https://gitlab.com/user/repo", True),
    ("http://gitlab.com/user/repo", True),
    ("git@gitlab.com:user/repo.git", True),
    # Local paths and other hosts
    ("/home/user/project", False),
    ("./relative/path", False),
    ("https://bitbucket.org/user/repo", False),
    ("", False),
])
def test_is_git_url(url: str, expected: bool) -> None:
    assert _is_git_url(url) is expected


# ---------------------------------------------------------------------------
# _derive_project_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("input_str,expected", [
    ("https://github.com/user/my-repo.git", "my-repo"),
    ("https://github.com/user/my-repo", "my-repo"),
    ("/home/user/projects/aiscan", "aiscan"),
    ("/home/user/projects/aiscan/", "aiscan"),
    ("simple", "simple"),
])
def test_derive_project_name(input_str: str, expected: str) -> None:
    assert _derive_project_name(input_str) == expected


# ---------------------------------------------------------------------------
# generate_scan_result
# ---------------------------------------------------------------------------

def test_generate_scan_result_empty() -> None:
    result = generate_scan_result("myproject", [])
    assert isinstance(result, ScanResult)
    assert result.project_name == "myproject"
    assert result.total_findings == 0
    assert result.findings == []


def test_generate_scan_result_counts_by_severity(
    dep_findings: List[Finding],
    model_findings: List[Finding],
    config_findings: List[Finding],
) -> None:
    all_findings = dep_findings + model_findings + config_findings
    result = generate_scan_result("proj", all_findings)

    assert result.total_findings == 3
    assert result.by_severity[Severity.HIGH] == 2
    assert result.by_severity[Severity.MEDIUM] == 1
    assert result.by_severity[Severity.LOW] == 0
    assert result.by_severity[Severity.CRITICAL] == 0


def test_generate_scan_result_preserves_project_name() -> None:
    result = generate_scan_result("special-name-123", [])
    assert result.project_name == "special-name-123"


# ---------------------------------------------------------------------------
# run_all_scanners — without LLM endpoint
# ---------------------------------------------------------------------------

def test_run_all_scanners_no_llm(
    tmp_path: Path,
    dep_findings: List[Finding],
    model_findings: List[Finding],
    config_findings: List[Finding],
) -> None:
    """Three file-system scanners are called; prompt tester is NOT called."""
    with (
        patch(
            "backend.core.risk_engine.scan_dependencies",
            new_callable=AsyncMock,
            return_value=dep_findings,
        ) as mock_dep,
        patch(
            "backend.core.risk_engine.scan_models",
            new_callable=AsyncMock,
            return_value=model_findings,
        ) as mock_model,
        patch(
            "backend.core.risk_engine.scan_configs",
            return_value=config_findings,
        ) as mock_cfg,
        patch(
            "backend.core.risk_engine.scan_prompt_injection",
            new_callable=AsyncMock,
        ) as mock_prompt,
    ):
        findings = asyncio.run(run_all_scanners(tmp_path))

    mock_dep.assert_awaited_once_with(tmp_path)
    mock_model.assert_awaited_once_with(tmp_path)
    mock_cfg.assert_called_once_with(tmp_path)
    mock_prompt.assert_not_awaited()

    assert len(findings) == 3
    titles = {f.title for f in findings}
    assert "CVE in numpy" in titles
    assert "Unsafe torch.load" in titles
    assert "DEBUG=True" in titles


def test_run_all_scanners_with_llm(
    tmp_path: Path,
    dep_findings: List[Finding],
    model_findings: List[Finding],
    config_findings: List[Finding],
    prompt_findings: List[Finding],
) -> None:
    """All four scanners are called when an LLM endpoint URL is provided."""
    llm_url = "http://localhost:8080/v1/chat/completions"

    with (
        patch(
            "backend.core.risk_engine.scan_dependencies",
            new_callable=AsyncMock,
            return_value=dep_findings,
        ),
        patch(
            "backend.core.risk_engine.scan_models",
            new_callable=AsyncMock,
            return_value=model_findings,
        ),
        patch(
            "backend.core.risk_engine.scan_configs",
            return_value=config_findings,
        ),
        patch(
            "backend.core.risk_engine.scan_prompt_injection",
            new_callable=AsyncMock,
            return_value=prompt_findings,
        ) as mock_prompt,
    ):
        findings = asyncio.run(run_all_scanners(tmp_path, llm_endpoint_url=llm_url))

    mock_prompt.assert_awaited_once_with(llm_url)
    assert len(findings) == 4


def test_run_all_scanners_combines_findings(
    tmp_path: Path,
    dep_findings: List[Finding],
    model_findings: List[Finding],
    config_findings: List[Finding],
) -> None:
    """Findings from all scanners are merged into a single flat list."""
    with (
        patch(
            "backend.core.risk_engine.scan_dependencies",
            new_callable=AsyncMock,
            return_value=dep_findings,
        ),
        patch(
            "backend.core.risk_engine.scan_models",
            new_callable=AsyncMock,
            return_value=model_findings,
        ),
        patch(
            "backend.core.risk_engine.scan_configs",
            return_value=config_findings,
        ),
    ):
        findings = asyncio.run(run_all_scanners(tmp_path))

    scanner_names = {f.scanner_name for f in findings}
    assert scanner_names == {"dependency_auditor", "model_scanner", "config_analyzer"}


def test_run_all_scanners_tolerates_scanner_exception(tmp_path: Path) -> None:
    """A scanner that raises is skipped; other scanners still contribute."""
    good_finding = _finding("Good finding", Severity.LOW, "model_scanner")

    with (
        patch(
            "backend.core.risk_engine.scan_dependencies",
            new_callable=AsyncMock,
            side_effect=RuntimeError("OSV API down"),
        ),
        patch(
            "backend.core.risk_engine.scan_models",
            new_callable=AsyncMock,
            return_value=[good_finding],
        ),
        patch(
            "backend.core.risk_engine.scan_configs",
            return_value=[],
        ),
    ):
        findings = asyncio.run(run_all_scanners(tmp_path))

    # The failed dependency_auditor is skipped; model_scanner result survives.
    assert len(findings) == 1
    assert findings[0].title == "Good finding"


def test_run_all_scanners_all_fail_returns_empty(tmp_path: Path) -> None:
    """If all scanners raise, an empty list is returned rather than an exception."""
    with (
        patch(
            "backend.core.risk_engine.scan_dependencies",
            new_callable=AsyncMock,
            side_effect=Exception("fail"),
        ),
        patch(
            "backend.core.risk_engine.scan_models",
            new_callable=AsyncMock,
            side_effect=Exception("fail"),
        ),
        patch(
            "backend.core.risk_engine.scan_configs",
            side_effect=Exception("fail"),
        ),
    ):
        findings = asyncio.run(run_all_scanners(tmp_path))

    assert findings == []


# ---------------------------------------------------------------------------
# scan_project — local path
# ---------------------------------------------------------------------------

def test_scan_project_local_path(
    tmp_path: Path,
    dep_findings: List[Finding],
    model_findings: List[Finding],
) -> None:
    """scan_project resolves a local path and returns a populated ScanResult."""
    all_findings = dep_findings + model_findings

    with (
        patch(
            "backend.core.risk_engine.scan_dependencies",
            new_callable=AsyncMock,
            return_value=dep_findings,
        ),
        patch(
            "backend.core.risk_engine.scan_models",
            new_callable=AsyncMock,
            return_value=model_findings,
        ),
        patch(
            "backend.core.risk_engine.scan_configs",
            return_value=[],
        ),
    ):
        result = asyncio.run(scan_project(tmp_path, project_name="my-project"))

    assert isinstance(result, ScanResult)
    assert result.project_name == "my-project"
    assert result.total_findings == len(all_findings)


def test_scan_project_derives_name_from_path(tmp_path: Path) -> None:
    """When project_name is omitted, it is derived from the directory basename."""
    named_dir = tmp_path / "cool-project"
    named_dir.mkdir()

    with (
        patch(
            "backend.core.risk_engine.scan_dependencies",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.scan_models",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.scan_configs",
            return_value=[],
        ),
    ):
        result = asyncio.run(scan_project(named_dir))

    assert result.project_name == "cool-project"


def test_scan_project_passes_llm_url_to_prompt_tester(
    tmp_path: Path,
    prompt_findings: List[Finding],
) -> None:
    """The llm_endpoint_url is forwarded to scan_prompt_injection."""
    llm_url = "http://llm.example.com/chat"

    with (
        patch(
            "backend.core.risk_engine.scan_dependencies",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.scan_models",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.scan_configs",
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.scan_prompt_injection",
            new_callable=AsyncMock,
            return_value=prompt_findings,
        ) as mock_prompt,
    ):
        result = asyncio.run(scan_project(tmp_path, llm_endpoint_url=llm_url))

    mock_prompt.assert_awaited_once_with(llm_url)
    assert result.total_findings == len(prompt_findings)


# ---------------------------------------------------------------------------
# scan_project — GitHub URL (clone + cleanup)
# ---------------------------------------------------------------------------

def test_scan_project_clones_github_url(tmp_path: Path) -> None:
    """When given a GitHub URL, _clone_repo is called and the clone is cleaned up."""
    fake_clone_dir = tmp_path / "cloned"
    fake_clone_dir.mkdir()

    with (
        patch(
            "backend.core.risk_engine._clone_repo",
            return_value=fake_clone_dir,
        ) as mock_clone,
        patch(
            "backend.core.risk_engine.scan_dependencies",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.scan_models",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.scan_configs",
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.shutil.rmtree",
        ) as mock_rmtree,
    ):
        result = asyncio.run(scan_project("https://github.com/user/my-repo.git"))

    mock_clone.assert_called_once_with("https://github.com/user/my-repo.git")
    mock_rmtree.assert_called_once_with(fake_clone_dir, ignore_errors=True)
    assert result.project_name == "my-repo"


def test_scan_project_cleans_up_even_on_scanner_error(tmp_path: Path) -> None:
    """Temporary clone directory is removed even if a scanner raises."""
    fake_clone_dir = tmp_path / "cloned"
    fake_clone_dir.mkdir()

    with (
        patch(
            "backend.core.risk_engine._clone_repo",
            return_value=fake_clone_dir,
        ),
        patch(
            "backend.core.risk_engine.scan_dependencies",
            new_callable=AsyncMock,
            side_effect=RuntimeError("unexpected"),
        ),
        patch(
            "backend.core.risk_engine.scan_models",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.scan_configs",
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.shutil.rmtree",
        ) as mock_rmtree,
    ):
        # Scanner exception is caught internally; finally block must still fire.
        asyncio.run(scan_project("https://github.com/user/repo"))

    mock_rmtree.assert_called_once()


def test_scan_project_derives_name_from_github_url(tmp_path: Path) -> None:
    """Project name is derived from the repo slug in the GitHub URL."""
    fake_clone_dir = tmp_path / "cloned"
    fake_clone_dir.mkdir()

    with (
        patch("backend.core.risk_engine._clone_repo", return_value=fake_clone_dir),
        patch(
            "backend.core.risk_engine.scan_dependencies",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.scan_models",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.scan_configs",
            return_value=[],
        ),
        patch("backend.core.risk_engine.shutil.rmtree"),
    ):
        result = asyncio.run(scan_project("https://github.com/org/awesome-scanner.git"))

    assert result.project_name == "awesome-scanner"


def test_scan_project_explicit_name_overrides_url_derivation(tmp_path: Path) -> None:
    """Explicit project_name takes precedence over URL-derived name."""
    fake_clone_dir = tmp_path / "cloned"
    fake_clone_dir.mkdir()

    with (
        patch("backend.core.risk_engine._clone_repo", return_value=fake_clone_dir),
        patch(
            "backend.core.risk_engine.scan_dependencies",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.scan_models",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "backend.core.risk_engine.scan_configs",
            return_value=[],
        ),
        patch("backend.core.risk_engine.shutil.rmtree"),
    ):
        result = asyncio.run(
            scan_project(
                "https://github.com/org/repo.git",
                project_name="Override Name",
            )
        )

    assert result.project_name == "Override Name"
