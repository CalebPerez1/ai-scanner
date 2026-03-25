"""Dependency auditor scanner utilities.

This module provides helpers to parse a `requirements.txt` file and to
query the OSV.dev API for known vulnerabilities for a given PyPI
package and version.

Notes
-----
- Uses `httpx` for async HTTP requests per project guidance.
- Returns `Finding` objects (from `backend.core.models`) for vulnerabilities.
- Parsing is intentionally conservative; it handles common `==` pins and
  simple package lines. VCS/URL-based requirements are best-effort.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import List, Optional, Tuple

import httpx

from backend.core.models import Finding, Severity


REQ_LINE_RE = re.compile(r"^\s*(?P<name>[^=<>!~\s\[]+)(?:\[(?P<extras>[^\]]+)\])?\s*(?:==\s*(?P<version>[^\s;]+))?")


def parse_requirements(requirements_path: Path | str) -> List[Tuple[str, Optional[str]]]:
    """Read a requirements.txt and extract (package, version) pairs.

    This function is conservative: it skips comments, editable installs,
    and include directives. For VCS/URL installs that include `#egg=` it
    will try to extract the package name.

    Args:
        requirements_path: Path to the requirements file.

    Returns:
        A list of tuples `(package_name, version_or_None)` in the same
        order as the file.
    """

    p = Path(requirements_path)
    if not p.exists():
        raise FileNotFoundError(f"requirements file not found: {p}")

    results: List[Tuple[str, Optional[str]]] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # Skip pip options and include directives
        if line.startswith(('-', '--')) or line.startswith(('git+', 'hg+', 'svn+', 'http:', 'https:')):
            # Try to extract name from URL fragments like '#egg=packagename'
            egg_idx = line.find('#egg=')
            if egg_idx != -1:
                name = line[egg_idx + 5 :].split('&')[0]
                results.append((name, None))
            else:
                # Not a simple package line we can parse
                continue

        else:
            m = REQ_LINE_RE.match(line)
            if not m:
                continue
            name = m.group('name')
            version = m.group('version')
            results.append((name, version))

    return results


async def query_osv_for_package(package_name: str, version: Optional[str]) -> List[Finding]:
    """Query the OSV.dev API for vulnerabilities affecting a package/version.

    This function calls the OSV `v1/query` endpoint with the PyPI ecosystem
    and constructs `Finding` objects for each returned vulnerability.

    Args:
        package_name: The PyPI package name to check (e.g., "django").
        version: The installed version to check, or `None` to query by package only.

    Returns:
        A list of `Finding` objects representing discovered vulnerabilities.

    Raises:
        httpx.HTTPError: If the OSV request fails.
    """

    url = "https://api.osv.dev/v1/query"
    payload = {"package": {"name": package_name, "ecosystem": "PyPI"}}
    if version:
        payload["version"] = version

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    vulns = data.get("vulns") or []
    findings: List[Finding] = []

    for v in vulns:
        # Build a short title and description from OSV fields
        vid = v.get("id")
        summary = v.get("summary") or ""
        details = v.get("details") or summary

        # Map OSV severity (if present) to our Severity enum.
        # OSV may include a `severity` list with score info; attempt a best-effort mapping.
        sev = Severity.MEDIUM
        for s in v.get("severity", []) or []:
            # `s` may contain keys like 'type' and 'score'
            score = s.get("score")
            if score is None:
                continue
            try:
                numeric = float(score)
            except (TypeError, ValueError):
                # Score may be a non-numeric string (skip best-effort)
                numeric = None

            if numeric is not None:
                if numeric >= 9:
                    sev = Severity.CRITICAL
                elif numeric >= 7:
                    sev = Severity.HIGH
                elif numeric >= 4:
                    sev = Severity.MEDIUM
                else:
                    sev = Severity.LOW

        # Fallback: if the OSV entry references a CVE or has an explicit severity
        # not covered above, leave `sev` as MEDIUM. TODO: improve mapping.

        # Compose a remediation recommendation from references if available
        refs = v.get("references") or []
        recommendation = None
        if refs:
            first = refs[0]
            recommendation = first.get("url") if isinstance(first, dict) else None

        finding = Finding(
            title=f"{vid} - {package_name}",
            description=details,
            severity=sev,
            scanner_name="dependency_auditor",
            file_path=None,
            line_number=None,
            recommendation=recommendation,
        )
        findings.append(finding)

    return findings


__all__ = ["parse_requirements", "query_osv_for_package"]


def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    A straightforward dynamic programming implementation sufficient for
    short package names.
    """

    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            add = prev[j] + 1
            delete = cur[j - 1] + 1
            change = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = min(add, delete, change)
        prev = cur
    return prev[lb]


# A hard-coded list of popular PyPI packages (approx. top 50). This list
# is intentionally conservative and can be updated later from an external
# source if desired.
POPULAR_PYPI_PACKAGES = [
    "pip",
    "setuptools",
    "wheel",
    "numpy",
    "requests",
    "pandas",
    "flask",
    "django",
    "scipy",
    "matplotlib",
    "pytest",
    "pillow",
    "sqlalchemy",
    "urllib3",
    "six",
    "certifi",
    "idna",
    "chardet",
    "click",
    "pyyaml",
    "tqdm",
    "lxml",
    "jinja2",
    "protobuf",
    "scikit-learn",
    "torch",
    "tensorflow",
    "beautifulsoup4",
    "cryptography",
    "opencv-python",
    "sentry-sdk",
    "psycopg2",
    "paramiko",
    "kombu",
    "celery",
    "nose",
    "pycryptodome",
    "markupsafe",
    "uvicorn",
    "fastapi",
    "pydantic",
    "werkzeug",
    "bcrypt",
    "sqlparse",
    "alembic",
    "gunicorn",
    "docutils",
    "sphinx",
]


def detect_typosquats(package_name: str) -> List[Finding]:
    """Detect potential typosquatting by comparing to popular packages.

    This function computes the Levenshtein distance between `package_name`
    and each name in the `POPULAR_PYPI_PACKAGES` list. If the distance is
    1 or 2 and the names are not exact matches, a `Finding` is returned
    indicating a potential typosquat.

    Args:
        package_name: The package name to analyze.

    Returns:
        A list of `Finding` objects (empty if no potential typosquats found).
    """

    results: List[Finding] = []
    lname = package_name.lower()
    for popular in POPULAR_PYPI_PACKAGES:
        popular_l = popular.lower()
        if lname == popular_l:
            continue
        dist = _levenshtein(lname, popular_l)
        if dist in (1, 2):
            severity = Severity.HIGH if dist == 1 else Severity.MEDIUM
            finding = Finding(
                title=f"Potential typosquat: {package_name} ~ {popular}",
                description=(
                    f"Package name '{package_name}' is within edit distance {dist} of "
                    f"popular package '{popular}'. This may be a typosquatting attempt."
                ),
                severity=severity,
                scanner_name="dependency_auditor",
                file_path=None,
                line_number=None,
                recommendation=(
                    "Verify the intended package name and repository. If you meant '"
                    f"{popular}', replace '{package_name}' with the correct name."
                ),
            )
            results.append(finding)
    return results


import asyncio


async def scan_dependencies(project_path: Path | str) -> List[Finding]:
    """Scan a project's dependencies for OSV vulnerabilities and typosquats.

    This function ties together `parse_requirements`, `query_osv_for_package`,
    and `detect_typosquats`. It expects either a path to a directory that
    contains `requirements.txt` or a direct path to a requirements file.

    Args:
        project_path: Directory or file path to scan.

    Returns:
        A list of `Finding` objects representing issues found.
    """

    p = Path(project_path)
    req_file = p
    if p.is_dir():
        req_file = p / "requirements.txt"

    packages = parse_requirements(req_file)
    findings: List[Finding] = []

    # Run OSV checks concurrently
    tasks = [asyncio.create_task(query_osv_for_package(name, ver)) for name, ver in packages]
    if tasks:
        vuln_lists = await asyncio.gather(*tasks)
        for vlist in vuln_lists:
            findings.extend(vlist)

    # Check for typosquatting
    for name, _ in packages:
        findings.extend(detect_typosquats(name))

    return findings


__all__ = [
    "parse_requirements",
    "query_osv_for_package",
    "detect_typosquats",
    "scan_dependencies",
]
