"""Unit tests for the model integrity scanner module.

Covers:
- scan_python_files: detects pickle, dill, cloudpickle, torch.load, joblib.load,
  numpy allow_pickle; handles unreadable files; returns no findings for clean code.
- check_huggingface_models: extracts model IDs, flags pickle-only format, low
  downloads, unverified author, and missing models; no-op on projects with no
  from_pretrained calls.
- scan_models: combines both scanners; returns all findings together.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.models import Finding, Severity
from backend.scanners.model_scanner import (
    check_huggingface_models,
    scan_models,
    scan_python_files,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_py(tmp_path: Path, name: str, content: str) -> Path:
    """Write *content* to *tmp_path/name* and return the file path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _make_hf_client_mock(response_data: dict, status_code: int = 200) -> AsyncMock:
    """Build a mock httpx.AsyncClient that returns *response_data* from GET.

    Satisfies the ``async with httpx.AsyncClient() as c`` protocol used
    inside _fetch_hf_metadata.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = response_data

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def _hf_model_metadata(
    model_id: str = "org/model",
    downloads: int = 5000,
    author: str = "google",
    file_names: list[str] | None = None,
) -> dict:
    """Construct a minimal HuggingFace Hub API response dict."""
    if file_names is None:
        file_names = ["model.safetensors"]
    return {
        "id": model_id,
        "author": author,
        "downloads": downloads,
        "siblings": [{"rfilename": f} for f in file_names],
    }


# ---------------------------------------------------------------------------
# scan_python_files — unsafe deserialization detection
# ---------------------------------------------------------------------------

class TestScanPythonFiles:
    def test_detects_pickle_load(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "infer.py", "model = pickle.load(open('m.pkl', 'rb'))\n")
        findings = scan_python_files(tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL
        assert "pickle" in findings[0].title.lower()
        assert findings[0].line_number == 1

    def test_detects_pickle_loads(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "run.py", "obj = pickle.loads(data)\n")
        findings = scan_python_files(tmp_path)
        assert any("pickle" in f.title.lower() for f in findings)

    def test_detects_dill_load(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "run.py", "m = dill.load(f)\n")
        findings = scan_python_files(tmp_path)
        assert findings[0].severity == Severity.HIGH
        assert "dill" in findings[0].title.lower()

    def test_detects_cloudpickle_loads(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "run.py", "obj = cloudpickle.loads(blob)\n")
        findings = scan_python_files(tmp_path)
        assert findings[0].severity == Severity.HIGH

    def test_detects_torch_load(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "model.py", "weights = torch.load('ckpt.pt')\n")
        findings = scan_python_files(tmp_path)
        assert findings[0].severity == Severity.HIGH
        assert "torch" in findings[0].title.lower()

    def test_detects_joblib_load(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "pred.py", "clf = joblib.load('clf.pkl')\n")
        findings = scan_python_files(tmp_path)
        assert findings[0].severity == Severity.MEDIUM

    def test_detects_numpy_allow_pickle(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "load.py", "arr = np.load('data.npy', allow_pickle=True)\n")
        findings = scan_python_files(tmp_path)
        assert findings[0].severity == Severity.MEDIUM
        assert "allow_pickle" in findings[0].title.lower()

    def test_no_findings_for_clean_code(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "safe.py", "import torch\nmodel = torch.nn.Linear(10, 1)\n")
        findings = scan_python_files(tmp_path)
        assert findings == []

    def test_correct_file_path_and_line_number(self, tmp_path: Path) -> None:
        (tmp_path / "deep").mkdir()
        _write_py(tmp_path / "deep", "nested.py", "\n\nmodel = pickle.load(f)\n")
        findings = scan_python_files(tmp_path)
        assert any(f.line_number == 3 for f in findings)
        assert any("deep/nested.py" in (f.file_path or "") for f in findings)

    def test_multiple_patterns_in_one_file(self, tmp_path: Path) -> None:
        content = "pickle.load(f)\njoblib.load('m.pkl')\ntorch.load('ckpt')\n"
        _write_py(tmp_path, "multi.py", content)
        findings = scan_python_files(tmp_path)
        assert len(findings) == 3

    def test_scans_subdirectories_recursively(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub" / "pkg"
        sub.mkdir(parents=True)
        (sub / "model.py").write_text("pickle.load(f)\n", encoding="utf-8")
        findings = scan_python_files(tmp_path)
        assert len(findings) == 1

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "run.py", "pickle.load(f)\n")
        findings = scan_python_files(str(tmp_path))
        assert len(findings) == 1

    def test_skips_venv_directory(self, tmp_path: Path) -> None:
        (tmp_path / "venv" / "lib").mkdir(parents=True)
        (tmp_path / "venv" / "lib" / "site.py").write_text("pickle.load(f)\n", encoding="utf-8")
        assert scan_python_files(tmp_path) == []

    def test_skips_dot_venv_directory(self, tmp_path: Path) -> None:
        (tmp_path / ".venv" / "lib").mkdir(parents=True)
        (tmp_path / ".venv" / "lib" / "site.py").write_text("pickle.load(f)\n", encoding="utf-8")
        assert scan_python_files(tmp_path) == []

    def test_skips_node_modules_directory(self, tmp_path: Path) -> None:
        (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
        (tmp_path / "node_modules" / "pkg" / "script.py").write_text("pickle.load(f)\n", encoding="utf-8")
        assert scan_python_files(tmp_path) == []

    def test_skips_tests_directory(self, tmp_path: Path) -> None:
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "conftest.py").write_text("pickle.load(f)\n", encoding="utf-8")
        assert scan_python_files(tmp_path) == []

    def test_skips_pycache_directory(self, tmp_path: Path) -> None:
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "model.py").write_text("pickle.load(f)\n", encoding="utf-8")
        assert scan_python_files(tmp_path) == []

    def test_skips_test_prefixed_files(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "test_model.py", "pickle.load(f)\n")
        assert scan_python_files(tmp_path) == []

    def test_does_not_skip_non_excluded_dirs(self, tmp_path: Path) -> None:
        # Confirm normal subdirectories are still scanned.
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "loader.py").write_text("pickle.load(f)\n", encoding="utf-8")
        assert len(scan_python_files(tmp_path)) == 1


# ---------------------------------------------------------------------------
# check_huggingface_models — HuggingFace Hub auditing
# ---------------------------------------------------------------------------

class TestCheckHuggingfaceModels:
    def test_no_findings_when_no_pretrained_calls(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "plain.py", "import torch\n")
        result = asyncio.run(
            check_huggingface_models(tmp_path)
        )
        assert result == []

    def test_flags_pickle_only_model(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "inf.py", 'model = AutoModel.from_pretrained("org/mymodel")\n')
        metadata = _hf_model_metadata(
            model_id="org/mymodel",
            author="org",
            downloads=9999,
            file_names=["pytorch_model.bin"],  # no safetensors
        )
        mock_client = _make_hf_client_mock(metadata)
        with patch("backend.scanners.model_scanner.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(
                check_huggingface_models(tmp_path)
            )
        pickle_findings = [f for f in findings if "pickle format" in f.title]
        assert pickle_findings, "Expected a pickle-format finding"
        assert pickle_findings[0].severity == Severity.HIGH

    def test_no_pickle_finding_when_safetensors_present(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "inf.py", 'model = AutoModel.from_pretrained("google/bert")\n')
        metadata = _hf_model_metadata(
            author="google",
            downloads=500000,
            file_names=["model.safetensors", "pytorch_model.bin"],
        )
        mock_client = _make_hf_client_mock(metadata)
        with patch("backend.scanners.model_scanner.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(
                check_huggingface_models(tmp_path)
            )
        assert not any("pickle format" in f.title for f in findings)

    def test_flags_low_download_count(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "inf.py", 'model = AutoModel.from_pretrained("randuser/obscure")\n')
        metadata = _hf_model_metadata(
            model_id="randuser/obscure",
            author="randuser",
            downloads=5,
            file_names=["model.safetensors"],
        )
        mock_client = _make_hf_client_mock(metadata)
        with patch("backend.scanners.model_scanner.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(
                check_huggingface_models(tmp_path)
            )
        low_dl = [f for f in findings if "low-adoption" in f.title.lower()]
        assert low_dl
        assert low_dl[0].severity == Severity.MEDIUM

    def test_no_low_download_finding_above_threshold(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "inf.py", 'model = AutoModel.from_pretrained("google/popular")\n')
        metadata = _hf_model_metadata(author="google", downloads=50000)
        mock_client = _make_hf_client_mock(metadata)
        with patch("backend.scanners.model_scanner.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(
                check_huggingface_models(tmp_path)
            )
        assert not any("low-adoption" in f.title.lower() for f in findings)

    def test_flags_unverified_author(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "inf.py", 'model = AutoModel.from_pretrained("sketchy/model")\n')
        metadata = _hf_model_metadata(
            model_id="sketchy/model",
            author="sketchy",
            downloads=500,
            file_names=["model.safetensors"],
        )
        mock_client = _make_hf_client_mock(metadata)
        with patch("backend.scanners.model_scanner.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(
                check_huggingface_models(tmp_path)
            )
        author_findings = [f for f in findings if "unverified" in f.title.lower()]
        assert author_findings
        assert author_findings[0].severity == Severity.LOW

    def test_trusted_author_no_unverified_finding(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "inf.py", 'model = AutoModel.from_pretrained("google/t5-small")\n')
        metadata = _hf_model_metadata(author="google", downloads=100000)
        mock_client = _make_hf_client_mock(metadata)
        with patch("backend.scanners.model_scanner.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(
                check_huggingface_models(tmp_path)
            )
        assert not any("unverified" in f.title.lower() for f in findings)

    def test_flags_model_not_found_on_hub(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "inf.py", 'model = AutoModel.from_pretrained("ghost/nonexistent")\n')
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("backend.scanners.model_scanner.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(
                check_huggingface_models(tmp_path)
            )
        assert any("not found" in f.title.lower() for f in findings)
        assert findings[0].severity == Severity.MEDIUM

    def test_deduplicates_repeated_model_id(self, tmp_path: Path) -> None:
        content = (
            'AutoModel.from_pretrained("google/bert")\n'
            'AutoTokenizer.from_pretrained("google/bert")\n'
        )
        _write_py(tmp_path, "inf.py", content)
        metadata = _hf_model_metadata(author="google", downloads=100000)
        mock_client = _make_hf_client_mock(metadata)
        with patch("backend.scanners.model_scanner.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(
                check_huggingface_models(tmp_path)
            )
        # Should only query the API once — duplicate model IDs are deduplicated.
        assert mock_client.get.call_count == 1

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "inf.py", 'model = AutoModel.from_pretrained("google/bert")\n')
        metadata = _hf_model_metadata(author="google", downloads=100000)
        mock_client = _make_hf_client_mock(metadata)
        with patch("backend.scanners.model_scanner.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(
                check_huggingface_models(str(tmp_path))
            )
        assert isinstance(findings, list)


# ---------------------------------------------------------------------------
# scan_models — combined entry point
# ---------------------------------------------------------------------------

class TestScanModels:
    def test_combines_static_and_hf_findings(self, tmp_path: Path) -> None:
        # File has both an unsafe load and a from_pretrained call.
        content = (
            "import pickle\n"
            "model_data = pickle.load(open('w.pkl', 'rb'))\n"
            'clf = AutoModel.from_pretrained("sketchy/clf")\n'
        )
        _write_py(tmp_path, "pipeline.py", content)
        metadata = _hf_model_metadata(
            model_id="sketchy/clf",
            author="sketchy",
            downloads=10,
            file_names=["model.bin"],  # pickle only
        )
        mock_client = _make_hf_client_mock(metadata)
        with patch("backend.scanners.model_scanner.httpx.AsyncClient", return_value=mock_client):
            findings = asyncio.run(scan_models(tmp_path))

        severities = {f.severity for f in findings}
        # Expect CRITICAL from pickle.load, HIGH from bin format, etc.
        assert Severity.CRITICAL in severities
        assert Severity.HIGH in severities
        assert len(findings) >= 3

    def test_returns_empty_list_for_clean_project(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "app.py", "import os\nprint('hello')\n")
        findings = asyncio.run(scan_models(tmp_path))
        assert findings == []

    def test_all_findings_have_required_fields(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "run.py", "torch.load('ckpt.pt')\n")
        findings = asyncio.run(scan_models(tmp_path))
        for f in findings:
            assert isinstance(f, Finding)
            assert f.title
            assert f.description
            assert f.severity in Severity
            assert f.scanner_name == "model_scanner"
