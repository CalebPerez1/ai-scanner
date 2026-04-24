"""Model integrity scanner for AIScan.

Detects unsafe model-loading patterns in Python source files and flags
HuggingFace models that may pose supply-chain risks (pickle/bin format,
low community adoption, or unverified authors).
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import httpx

from backend.core.models import Finding, Severity


SCANNER_NAME = "model_scanner"

DEFAULT_EXCLUDE_DIRS = frozenset({
    "venv", ".venv", "node_modules", ".git", "__pycache__", "tests",
})

# ---------------------------------------------------------------------------
# Unsafe deserialization patterns
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, severity, short title, recommendation).
_UNSAFE_PATTERNS: List[Tuple[re.Pattern, Severity, str, str]] = [
    (
        re.compile(r"\bpickle\.loads?\s*\("),
        Severity.CRITICAL,
        "Unsafe pickle deserialization",
        "Avoid pickle for model loading; use safetensors or ONNX instead.",
    ),
    (
        re.compile(r"\bdill\.loads?\s*\("),
        Severity.HIGH,
        "Unsafe dill deserialization",
        "dill can execute arbitrary code on load; use a safe serialization format.",
    ),
    (
        re.compile(r"\bcloudpickle\.loads?\s*\("),
        Severity.HIGH,
        "Unsafe cloudpickle deserialization",
        "cloudpickle can execute arbitrary code on load; use a safe serialization format.",
    ),
    (
        re.compile(r"\btorch\.load\s*\("),
        Severity.HIGH,
        "Unsafe torch.load call",
        "Pass weights_only=True to torch.load to prevent arbitrary code execution.",
    ),
    (
        re.compile(r"\bjoblib\.load\s*\("),
        Severity.MEDIUM,
        "Potentially unsafe joblib.load",
        "joblib uses pickle internally; verify the file source is trusted.",
    ),
    (
        re.compile(r"\bnp\.load\s*\(.*allow_pickle\s*=\s*True"),
        Severity.MEDIUM,
        "numpy.load with allow_pickle=True",
        "Remove allow_pickle=True or ensure the .npy file comes from a trusted source.",
    ),
]

# Extracts model IDs from from_pretrained("org/model") or from_pretrained("model") calls.
_FROM_PRETRAINED_RE = re.compile(
    r'from_pretrained\s*\(\s*["\']([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)?)["\']'
)

# Organizations considered well-known for the unverified-author check.
_TRUSTED_HF_ORGS = frozenset({
    "google", "facebook", "meta-llama", "openai", "microsoft", "huggingface",
    "mistralai", "stabilityai", "eleutherai", "bigscience", "sentence-transformers",
    "bert-base-uncased", "roberta-base",
})

# Models with fewer downloads than this threshold are flagged as low-adoption.
_LOW_DOWNLOAD_THRESHOLD = 100


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _iter_python_files(project_path: Path) -> Iterable[Path]:
    """Yield .py files under *project_path*, skipping excluded dirs and test files."""
    for py_file in project_path.rglob("*.py"):
        if any(part in DEFAULT_EXCLUDE_DIRS for part in py_file.parts):
            continue
        if py_file.name.startswith("test_"):
            continue
        yield py_file


def _check_file_for_unsafe_loads(file_path: Path, project_root: Path) -> List[Finding]:
    """Scan one Python file for unsafe model-loading calls.

    Args:
        file_path: Absolute path to the .py file.
        project_root: Project root used to compute relative paths in findings.

    Returns:
        A list of Finding objects, one per unsafe pattern occurrence.
    """
    findings: List[Finding] = []
    rel = str(file_path.relative_to(project_root))
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return findings

    for lineno, line in enumerate(lines, start=1):
        for pattern, severity, title, recommendation in _UNSAFE_PATTERNS:
            if pattern.search(line):
                findings.append(Finding(
                    title=title,
                    description=(
                        f"Found `{pattern.pattern}` at {rel}:{lineno}. "
                        "Deserializing untrusted model files can execute arbitrary code."
                    ),
                    severity=severity,
                    scanner_name=SCANNER_NAME,
                    file_path=rel,
                    line_number=lineno,
                    recommendation=recommendation,
                ))
    return findings


def _extract_hf_model_ids(project_path: Path) -> Dict[str, Tuple[str, int]]:
    """Find HuggingFace model IDs referenced via from_pretrained in .py files.

    Args:
        project_path: Root of the project to scan.

    Returns:
        Dict mapping model_id -> (relative_file_path, line_number) for the
        first occurrence of each model ID found.
    """
    seen: Dict[str, Tuple[str, int]] = {}
    for py_file in _iter_python_files(project_path):
        rel = str(py_file.relative_to(project_path))
        try:
            lines = py_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            for m in _FROM_PRETRAINED_RE.finditer(line):
                model_id = m.group(1)
                if model_id not in seen:
                    seen[model_id] = (rel, lineno)
    return seen


async def _fetch_hf_metadata(model_id: str) -> Optional[dict]:
    """Query the HuggingFace Hub API for model metadata.

    Args:
        model_id: HuggingFace model identifier (e.g. "org/model-name").

    Returns:
        Parsed JSON dict on success, or None if the model is not found or
        the request fails.
    """
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError:
        return None


def _finding_pickle_format(
    model_id: str, file_path: str, line_number: int, file_names: set
) -> Optional[Finding]:
    """Return a Finding if the model uses pickle weights but no safetensors.

    Args:
        model_id: HuggingFace model identifier.
        file_path: Relative path where the model reference was found.
        line_number: Line number of the from_pretrained call.
        file_names: Set of filenames from the model repository siblings list.

    Returns:
        A Finding if pickle-only weights are detected, otherwise None.
    """
    has_safetensors = any(f.endswith(".safetensors") for f in file_names)
    has_pickle = any(f.endswith((".bin", ".pt", ".pth", ".pkl")) for f in file_names)
    if not (has_pickle and not has_safetensors):
        return None
    return Finding(
        title=f"Model uses pickle format: {model_id}",
        description=(
            f"'{model_id}' (referenced at {file_path}:{line_number}) stores weights "
            "in pickle/bin format with no safetensors alternative. "
            "Pickle files can execute arbitrary code when loaded."
        ),
        severity=Severity.HIGH,
        scanner_name=SCANNER_NAME,
        file_path=file_path,
        line_number=line_number,
        recommendation=(
            "Prefer models that publish safetensors weights, or convert the model "
            "yourself using `safetensors.torch.save_file`."
        ),
    )


def _finding_low_downloads(
    model_id: str, file_path: str, line_number: int, downloads: int
) -> Optional[Finding]:
    """Return a Finding if the model has very few downloads.

    Args:
        model_id: HuggingFace model identifier.
        file_path: Relative path where the model reference was found.
        line_number: Line number of the from_pretrained call.
        downloads: Total download count from the HF Hub API.

    Returns:
        A Finding if the download count is below the low-adoption threshold,
        otherwise None.
    """
    if downloads >= _LOW_DOWNLOAD_THRESHOLD:
        return None
    return Finding(
        title=f"Low-adoption HuggingFace model: {model_id}",
        description=(
            f"Model '{model_id}' has only {downloads} downloads. "
            "Low-adoption models carry higher risk of being malicious or unmaintained."
        ),
        severity=Severity.MEDIUM,
        scanner_name=SCANNER_NAME,
        file_path=file_path,
        line_number=line_number,
        recommendation=(
            "Audit the model card and repository history before using this model "
            "in a production pipeline."
        ),
    )


def _finding_unverified_author(
    model_id: str, file_path: str, line_number: int, author: str
) -> Optional[Finding]:
    """Return a Finding if the model's author is not a well-known organization.

    Args:
        model_id: HuggingFace model identifier.
        file_path: Relative path where the model reference was found.
        line_number: Line number of the from_pretrained call.
        author: Author or organization name from the HF Hub API.

    Returns:
        A Finding if the author is not in the trusted organizations list,
        otherwise None.
    """
    if author.lower() in _TRUSTED_HF_ORGS:
        return None
    return Finding(
        title=f"Unverified model author: {author}",
        description=(
            f"Model '{model_id}' is published by '{author}', which is not in the "
            "list of well-known, trusted HuggingFace organizations."
        ),
        severity=Severity.LOW,
        scanner_name=SCANNER_NAME,
        file_path=file_path,
        line_number=line_number,
        recommendation=(
            "Review the author's profile and model card. Prefer models from "
            "established organizations or with high community trust."
        ),
    )


def _hf_metadata_to_findings(
    model_id: str,
    metadata: dict,
    file_path: str,
    line_number: int,
) -> List[Finding]:
    """Convert HuggingFace model metadata into security Findings.

    Runs three checks: pickle-only format, low download count, and
    unverified author. Each check is delegated to a focused helper.

    Args:
        model_id: The HuggingFace model identifier.
        metadata: Parsed JSON response from the HF Hub API.
        file_path: Relative path where the model reference was found.
        line_number: Line number of the from_pretrained call.

    Returns:
        A list of Finding objects for any issues detected.
    """
    siblings = metadata.get("siblings") or []
    file_names = {s.get("rfilename", "") for s in siblings}
    downloads = metadata.get("downloads", 0) or 0
    # Prefer the explicit `author` field; fall back to the org prefix in the ID.
    author = metadata.get("author") or model_id.split("/")[0]

    checks = [
        _finding_pickle_format(model_id, file_path, line_number, file_names),
        _finding_low_downloads(model_id, file_path, line_number, downloads),
        _finding_unverified_author(model_id, file_path, line_number, author),
    ]
    return [f for f in checks if f is not None]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_python_files(project_path: Path | str) -> List[Finding]:
    """Walk a project directory and detect unsafe model-loading calls in .py files.

    Checks for dangerous deserialization functions — torch.load(), pickle.load(),
    joblib.load(), dill.load(), cloudpickle.load(), and numpy.load with
    allow_pickle=True — which can execute arbitrary code when loading untrusted
    model artifacts.

    Args:
        project_path: Root directory of the project to scan.

    Returns:
        A list of Finding objects, one per unsafe pattern occurrence detected.
    """
    root = Path(project_path)
    findings: List[Finding] = []
    for py_file in _iter_python_files(root):
        findings.extend(_check_file_for_unsafe_loads(py_file, root))
    return findings


async def check_huggingface_models(project_path: Path | str) -> List[Finding]:
    """Extract HuggingFace model references and audit them via the Hub API.

    Finds all from_pretrained("model-id") calls in .py files, queries the
    HuggingFace Hub API for each model concurrently, and flags models that:
    - store weights in pickle/bin format with no safetensors alternative,
    - have a low download count (under the community-trust threshold), or
    - are published by an unrecognized author.

    Args:
        project_path: Root directory of the project to scan.

    Returns:
        A list of Finding objects for any flagged model references.
    """
    root = Path(project_path)
    model_refs = _extract_hf_model_ids(root)
    if not model_refs:
        return []

    model_ids = list(model_refs.keys())
    tasks = [asyncio.create_task(_fetch_hf_metadata(mid)) for mid in model_ids]
    results = await asyncio.gather(*tasks)

    findings: List[Finding] = []
    for model_id, metadata in zip(model_ids, results):
        file_path, line_number = model_refs[model_id]
        if metadata is None:
            # A model ID that resolves to nothing on HF Hub is itself suspicious.
            findings.append(Finding(
                title=f"HuggingFace model not found: {model_id}",
                description=(
                    f"The model '{model_id}' referenced at {file_path}:{line_number} "
                    "could not be found on the HuggingFace Hub."
                ),
                severity=Severity.MEDIUM,
                scanner_name=SCANNER_NAME,
                file_path=file_path,
                line_number=line_number,
                recommendation="Verify the model ID is correct and the repository is public.",
            ))
            continue
        findings.extend(_hf_metadata_to_findings(model_id, metadata, file_path, line_number))

    return findings


async def scan_models(project_path: Path | str) -> List[Finding]:
    """Main entry point for the model integrity scanner.

    Combines static analysis for unsafe deserialization calls with the
    HuggingFace Hub model audit, returning all findings together.

    Args:
        project_path: Root directory of the project to scan.

    Returns:
        A combined list of Finding objects from all model-integrity checks.
    """
    static_findings = scan_python_files(project_path)
    hf_findings = await check_huggingface_models(project_path)
    return static_findings + hf_findings


__all__ = ["scan_python_files", "check_huggingface_models", "scan_models"]
