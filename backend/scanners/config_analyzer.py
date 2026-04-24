"""Config analyzer scanner for AIScan.

Detects hardcoded secrets in project files, dangerous runtime misconfigurations
in Python code, and leaked credentials in Jupyter notebook cell outputs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional

from backend.core.models import Finding, Severity


SCANNER_NAME = "config_analyzer"

# ---------------------------------------------------------------------------
# Secret patterns
# ---------------------------------------------------------------------------

class _SecretPattern(NamedTuple):
    name: str
    regex: re.Pattern
    severity: Severity


# Ordered from most to least specific so the first match wins in descriptions.
SECRET_PATTERNS: List[_SecretPattern] = [
    _SecretPattern(
        "OpenAI API key",
        re.compile(r"\bsk-[a-zA-Z0-9]{48}\b"),
        Severity.CRITICAL,
    ),
    _SecretPattern(
        "HuggingFace token",
        re.compile(r"\bhf_[a-zA-Z0-9]{34}\b"),
        Severity.CRITICAL,
    ),
    _SecretPattern(
        "AWS access key ID",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        Severity.CRITICAL,
    ),
    _SecretPattern(
        "GitHub personal access token",
        re.compile(r"\bghp_[a-zA-Z0-9]{36}\b"),
        Severity.CRITICAL,
    ),
    _SecretPattern(
        "Slack webhook URL",
        re.compile(r"https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[a-zA-Z0-9]+"),
        Severity.HIGH,
    ),
    _SecretPattern(
        "PostgreSQL connection string with credentials",
        re.compile(r"postgres(?:ql)?://[^:]+:[^@]+@[^\s\"']+", re.IGNORECASE),
        Severity.HIGH,
    ),
    _SecretPattern(
        "MySQL connection string with credentials",
        re.compile(r"mysql(?:\+\w+)?://[^:]+:[^@]+@[^\s\"']+", re.IGNORECASE),
        Severity.HIGH,
    ),
]

# ---------------------------------------------------------------------------
# Misconfiguration patterns
# ---------------------------------------------------------------------------

class _MisconfigPattern(NamedTuple):
    name: str
    regex: re.Pattern
    severity: Severity
    recommendation: str


MISCONFIG_PATTERNS: List[_MisconfigPattern] = [
    _MisconfigPattern(
        "DEBUG mode enabled",
        re.compile(r"\bDEBUG\s*=\s*True\b"),
        Severity.HIGH,
        "Set DEBUG=False in production. Use environment variables to toggle debug mode.",
    ),
    _MisconfigPattern(
        "Flask debug mode enabled",
        re.compile(r"\bapp\.debug\s*=\s*True\b"),
        Severity.HIGH,
        "Never enable app.debug=True in production; it exposes an interactive debugger.",
    ),
    _MisconfigPattern(
        "Permissive CORS wildcard origin",
        re.compile(r'allow_origins\s*=\s*\[\s*["\']?\*["\']?\s*\]'),
        Severity.MEDIUM,
        "Restrict CORS origins to known domains instead of using a wildcard.",
    ),
    _MisconfigPattern(
        "Unauthenticated model inference endpoint",
        # Flags FastAPI/Flask routes whose path contains 'predict', 'infer', or
        # 'model' but have no apparent dependency injection for auth.
        re.compile(
            r'@(?:app|router)\.\w+\s*\(\s*["\'][^"\']*(?:predict|infer|model)[^"\']*["\']'
            r'(?:(?!Depends|login_required|require_auth).)*\n\s*(?:async\s+)?def\s+\w+\s*\([^)]*\)\s*:',
            re.MULTILINE,
        ),
        Severity.MEDIUM,
        (
            "Add authentication to model endpoints. Use FastAPI Depends() with an "
            "auth scheme, or Flask's login_required decorator."
        ),
    ),
]

# ---------------------------------------------------------------------------
# File walking helpers
# ---------------------------------------------------------------------------

# Directory names to skip when walking the project tree.
DEFAULT_EXCLUDE_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "venv", ".venv", "env", ".env",
    "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "site-packages",
    "tests",
})

# Extensions treated as binary; reading them as text is not useful.
_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".pt", ".pth", ".pkl", ".safetensors", ".onnx", ".h5",
    ".pyc", ".pyd",
    ".mp3", ".mp4", ".wav", ".ogg",
    ".ttf", ".woff", ".woff2",
})


def _iter_text_files(project_path: Path) -> Iterable[Path]:
    """Yield readable text files under *project_path*, skipping noisy dirs and binaries.

    Args:
        project_path: Root directory to walk.

    Yields:
        Path objects for files that are likely plain text.
    """
    for entry in project_path.rglob("*"):
        if not entry.is_file():
            continue
        if any(part in DEFAULT_EXCLUDE_DIRS for part in entry.parts):
            continue
        if entry.name.startswith("test_"):
            continue
        if entry.suffix.lower() in _BINARY_EXTENSIONS:
            continue
        yield entry


def _read_lines(path: Path) -> Optional[List[str]]:
    """Read *path* as UTF-8 text, returning lines or None on failure.

    Args:
        path: File to read.

    Returns:
        List of lines, or None if the file cannot be read as text.
    """
    try:
        return path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# Secret scanning helpers
# ---------------------------------------------------------------------------

def _secret_finding(
    pattern: _SecretPattern,
    rel_path: str,
    line_number: int,
    context: str,
) -> Finding:
    """Build a Finding for a detected secret.

    Args:
        pattern: The matched _SecretPattern.
        rel_path: Relative file path for the finding.
        line_number: Line number where the match was found.
        context: A short excerpt (redacted) for the description.

    Returns:
        A Finding object.
    """
    return Finding(
        title=f"Hardcoded {pattern.name}",
        description=(
            f"Potential {pattern.name} detected at {rel_path}:{line_number}. "
            f"Excerpt: {context!r}. Hardcoded credentials are a critical security risk."
        ),
        severity=pattern.severity,
        scanner_name=SCANNER_NAME,
        file_path=rel_path,
        line_number=line_number,
        recommendation=(
            "Remove the credential from source code. Store secrets in environment "
            "variables and load them with python-dotenv or a secrets manager."
        ),
    )


def _scan_lines_for_secrets(
    lines: List[str],
    rel_path: str,
    start_line: int = 1,
) -> List[Finding]:
    """Apply all secret regex patterns to a list of text lines.

    Args:
        lines: Text lines to scan.
        rel_path: Relative path used in Finding metadata.
        start_line: Line number offset (1-based) for the first line in *lines*.

    Returns:
        List of Finding objects for each secret match found.
    """
    findings: List[Finding] = []
    for offset, line in enumerate(lines):
        lineno = start_line + offset
        for pattern in SECRET_PATTERNS:
            if pattern.regex.search(line):
                # Truncate the line to avoid dumping the full secret in the report.
                excerpt = line.strip()[:80]
                findings.append(_secret_finding(pattern, rel_path, lineno, excerpt))
                # Only report the first matching pattern per line to avoid duplication.
                break
    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_for_secrets(project_path: Path | str) -> List[Finding]:
    """Walk all text files in a project and detect hardcoded secrets.

    Scans every non-binary file outside venv/node_modules directories for
    known secret patterns: OpenAI keys, HuggingFace tokens, AWS access key IDs,
    GitHub PATs, Slack webhook URLs, and database connection strings with
    embedded credentials.

    Args:
        project_path: Root directory of the project to scan.

    Returns:
        A list of CRITICAL or HIGH severity Finding objects.
    """
    root = Path(project_path)
    findings: List[Finding] = []
    for file_path in _iter_text_files(root):
        # Skip .ipynb files — notebook outputs are handled by scan_jupyter_notebooks.
        if file_path.suffix == ".ipynb":
            continue
        lines = _read_lines(file_path)
        if lines is None:
            continue
        rel = str(file_path.relative_to(root))
        findings.extend(_scan_lines_for_secrets(lines, rel))
    return findings


def scan_for_misconfigs(project_path: Path | str) -> List[Finding]:
    """Scan Python files for dangerous runtime misconfigurations.

    Checks for DEBUG=True, Flask app.debug=True, permissive CORS wildcard
    origins, and FastAPI/Flask model endpoints that appear to lack authentication.

    Args:
        project_path: Root directory of the project to scan.

    Returns:
        A list of MEDIUM or HIGH severity Finding objects.
    """
    root = Path(project_path)
    findings: List[Finding] = []

    for file_path in root.rglob("*.py"):
        if any(part in DEFAULT_EXCLUDE_DIRS for part in file_path.parts):
            continue
        if file_path.name.startswith("test_"):
            continue
        lines = _read_lines(file_path)
        if lines is None:
            continue
        rel = str(file_path.relative_to(root))
        full_text = "\n".join(lines)

        # Line-level checks (DEBUG, debug, CORS).
        for lineno, line in enumerate(lines, start=1):
            for mp in MISCONFIG_PATTERNS[:-1]:  # last pattern is multiline
                if mp.regex.search(line):
                    findings.append(Finding(
                        title=mp.name,
                        description=(
                            f"Found `{line.strip()[:80]}` at {rel}:{lineno}."
                        ),
                        severity=mp.severity,
                        scanner_name=SCANNER_NAME,
                        file_path=rel,
                        line_number=lineno,
                        recommendation=mp.recommendation,
                    ))

        # Multiline check for unauthenticated model endpoints.
        auth_pattern = MISCONFIG_PATTERNS[-1]
        for m in auth_pattern.regex.finditer(full_text):
            # Compute approximate line number from character offset.
            lineno = full_text[: m.start()].count("\n") + 1
            findings.append(Finding(
                title=auth_pattern.name,
                description=(
                    f"Route at {rel}:{lineno} appears to serve model inference "
                    "without authentication middleware."
                ),
                severity=auth_pattern.severity,
                scanner_name=SCANNER_NAME,
                file_path=rel,
                line_number=lineno,
                recommendation=auth_pattern.recommendation,
            ))

    return findings


def scan_jupyter_notebooks(project_path: Path | str) -> List[Finding]:
    """Scan Jupyter notebook cell outputs for leaked secrets.

    Reads every .ipynb file in the project, extracts all cell output text
    (stdout, stderr, plain-text display data, and error tracebacks), and
    applies the same secret patterns used by scan_for_secrets.

    Args:
        project_path: Root directory of the project to scan.

    Returns:
        A list of Finding objects for secrets found in notebook outputs.
    """
    root = Path(project_path)
    findings: List[Finding] = []

    for nb_path in root.rglob("*.ipynb"):
        if any(part in DEFAULT_EXCLUDE_DIRS for part in nb_path.parts):
            continue
        if nb_path.name.startswith("test_"):
            continue
        rel = str(nb_path.relative_to(root))
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue

        for cell_idx, cell in enumerate(nb.get("cells") or [], start=1):
            output_lines = _extract_notebook_output_lines(cell)
            if output_lines:
                # Use cell index as a human-readable location since notebooks
                # don't have traditional line numbers across cells.
                findings.extend(
                    _scan_lines_for_secrets(output_lines, rel, start_line=cell_idx)
                )

    return findings


def _extract_notebook_output_lines(cell: dict) -> List[str]:
    """Extract all text lines from a single Jupyter notebook cell's outputs.

    Handles code cell output types: stream (stdout/stderr), display_data,
    execute_result, and error (tracebacks).

    Args:
        cell: A parsed notebook cell dict.

    Returns:
        A flat list of text lines from all outputs in this cell.
    """
    lines: List[str] = []
    for output in cell.get("outputs") or []:
        output_type = output.get("output_type", "")
        if output_type in ("stream", ):
            # 'text' may be a list of strings or a single string.
            raw = output.get("text") or []
            lines.extend(raw if isinstance(raw, list) else raw.splitlines())
        elif output_type in ("display_data", "execute_result"):
            data = output.get("data") or {}
            raw = data.get("text/plain") or []
            lines.extend(raw if isinstance(raw, list) else raw.splitlines())
        elif output_type == "error":
            # Tracebacks can contain echoed values — check them too.
            for tb_line in output.get("traceback") or []:
                lines.extend(str(tb_line).splitlines())
    return lines


def scan_configs(project_path: Path | str) -> List[Finding]:
    """Main entry point for the config analyzer scanner.

    Runs all three checks — secret detection, misconfiguration analysis, and
    Jupyter notebook output scanning — and returns the combined findings.

    Args:
        project_path: Root directory of the project to scan.

    Returns:
        A combined list of Finding objects from all config-analysis checks.
    """
    return (
        scan_for_secrets(project_path)
        + scan_for_misconfigs(project_path)
        + scan_jupyter_notebooks(project_path)
    )


__all__ = [
    "scan_for_secrets",
    "scan_for_misconfigs",
    "scan_jupyter_notebooks",
    "scan_configs",
]
