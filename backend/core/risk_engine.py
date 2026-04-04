"""Risk engine for AIScan — orchestrates all scanner modules.

This module is the main entry point for scanning a project. It runs all
four scanner modules concurrently, collects findings, computes a risk
summary, and returns a ScanResult.

Supported input formats:
- Local directory path (str or Path)
- GitHub repository URL (https://github.com/... or git@github.com:...)
- GitLab repository URL (https://gitlab.com/... or git@gitlab.com:...)
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from backend.core.models import Finding, ScanResult
from backend.scanners.config_analyzer import scan_configs
from backend.scanners.dependency_auditor import scan_dependencies
from backend.scanners.model_scanner import scan_models
from backend.scanners.prompt_tester import scan_prompt_injection


def _is_git_url(path_or_url: str) -> bool:
    """Return True if the input looks like a GitHub or GitLab remote URL.

    Args:
        path_or_url: A string that may be a local path or a remote URL.

    Returns:
        True if the string starts with a recognized git hosting prefix.
    """
    s = path_or_url.strip()
    return s.startswith((
        "https://github.com/",
        "http://github.com/",
        "git@github.com:",
        "https://gitlab.com/",
        "http://gitlab.com/",
        "git@gitlab.com:",
        "git+https://",
    ))


def _clone_repo(url: str) -> Path:
    """Clone a GitHub or GitLab repository to a fresh temporary directory.

    Uses ``git clone --depth=1`` so only the latest commit is fetched,
    keeping the clone fast regardless of repository history.

    Args:
        url: The repository URL to clone (https or ssh form).

    Returns:
        Path to the cloned repository root inside a temporary directory.

    Raises:
        subprocess.CalledProcessError: If git clone fails.
    """
    tmp_dir = tempfile.mkdtemp(prefix="aiscan_")
    # Strip git+ prefix used in some pip-style VCS URLs.
    clone_url = url.removeprefix("git+")
    subprocess.run(
        ["git", "clone", "--depth=1", clone_url, tmp_dir],
        check=True,
        capture_output=True,
    )
    return Path(tmp_dir)


def _derive_project_name(path_or_url: str) -> str:
    """Derive a human-friendly project name from a local path or repository URL.

    For URLs the last path segment is used (minus any ``.git`` suffix).
    For local paths the directory's basename is used.

    Args:
        path_or_url: Local path or repository URL string.

    Returns:
        A short, non-empty project name string.
    """
    stripped = path_or_url.rstrip("/")
    # Take the rightmost path/URL segment.
    name = stripped.rsplit("/", maxsplit=1)[-1]
    name = name.removesuffix(".git")
    # Fall back to the full string if the segment ended up empty (edge case).
    return name or stripped


async def run_all_scanners(
    project_path: Union[str, Path],
    llm_endpoint_url: Optional[str] = None,
) -> List[Finding]:
    """Run all scanner modules against a local project directory.

    The dependency auditor, model scanner, and config analyzer are launched
    concurrently. If ``llm_endpoint_url`` is provided the prompt injection
    tester is added to the concurrent batch as well.

    Individual scanner failures are caught and skipped so that a single
    broken scanner cannot abort the entire scan.

    Args:
        project_path: Local path to the project directory to scan.
        llm_endpoint_url: Optional URL of an LLM chat endpoint to probe for
            prompt injection vulnerabilities.

    Returns:
        A combined list of Finding objects from every scanner that succeeded.
    """
    path = Path(project_path)

    # scan_configs is synchronous — run it in a thread to avoid blocking the loop.
    async def _run_configs() -> List[Finding]:
        """Wrap the synchronous config scanner for use with asyncio.gather."""
        return await asyncio.to_thread(scan_configs, path)

    tasks: List[asyncio.Task] = [
        asyncio.create_task(scan_dependencies(path)),
        asyncio.create_task(scan_models(path)),
        asyncio.create_task(_run_configs()),
    ]

    if llm_endpoint_url:
        tasks.append(asyncio.create_task(scan_prompt_injection(llm_endpoint_url)))

    # return_exceptions=True prevents one scanner failure from cancelling the rest.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    findings: List[Finding] = []
    for result in results:
        if isinstance(result, Exception):
            # TODO (Phase 3): surface per-scanner errors in the ScanResult.
            continue
        findings.extend(result)

    return findings


def generate_scan_result(project_name: str, findings: List[Finding]) -> ScanResult:
    """Build a ScanResult from a project name and a flat list of findings.

    The ScanResult model's root validator automatically computes
    ``total_findings`` and ``by_severity`` from the findings list.

    Args:
        project_name: Human-readable name for the scanned project.
        findings: All Finding objects collected from scanner modules.

    Returns:
        A fully populated ScanResult instance.
    """
    return ScanResult(project_name=project_name, findings=findings)


async def scan_project(
    project_path: Union[str, Path],
    project_name: Optional[str] = None,
    llm_endpoint_url: Optional[str] = None,
) -> ScanResult:
    """Scan a project and return a structured security report.

    Accepts either a local directory path or a GitHub repository URL. When a
    URL is detected the repository is cloned to a temporary directory, scanned,
    and the temporary directory is always cleaned up — even if an exception is
    raised during scanning.

    Args:
        project_path: Local directory path or GitHub repository URL.
        project_name: Optional display name for the project. Derived from the
            path or URL basename when omitted.
        llm_endpoint_url: Optional LLM endpoint URL for prompt injection testing.

    Returns:
        A ScanResult containing all findings and a severity summary.
    """
    path_str = str(project_path)
    tmp_dir: Optional[Path] = None

    if _is_git_url(path_str):
        tmp_dir = _clone_repo(path_str)
        resolved_path: Path = tmp_dir
    else:
        resolved_path = Path(project_path)

    derived_name = project_name or _derive_project_name(path_str)

    try:
        findings = await run_all_scanners(resolved_path, llm_endpoint_url=llm_endpoint_url)
        return generate_scan_result(derived_name, findings)
    finally:
        # Always remove the temporary clone directory, if one was created.
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


__all__ = [
    "_is_git_url",
    "run_all_scanners",
    "generate_scan_result",
    "scan_project",
]
