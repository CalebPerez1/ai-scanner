"""Unit tests for the config analyzer scanner module.

Covers:
- scan_for_secrets: detects OpenAI, HuggingFace, AWS, GitHub, Slack, postgres,
  mysql patterns; skips binary extensions, venv dirs, .ipynb files; reports
  correct file path and line number; clean files produce no findings.
- scan_for_misconfigs: detects DEBUG=True, app.debug=True, permissive CORS,
  unauthenticated model endpoints; clean files produce no findings.
- scan_jupyter_notebooks: detects secrets in stream, display_data, execute_result,
  and error outputs; skips non-notebook files; handles malformed JSON gracefully.
- scan_configs: combines all three scanners.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core.models import Finding, Severity
from backend.scanners.config_analyzer import (
    scan_configs,
    scan_for_misconfigs,
    scan_for_secrets,
    scan_jupyter_notebooks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    """Write *content* to *tmp_path/name*, creating parent dirs as needed."""
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _make_notebook(cells: list[dict]) -> str:
    """Serialize a minimal .ipynb structure to a JSON string."""
    return json.dumps({"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {}})


def _code_cell(source: str = "", outputs: list[dict] | None = None) -> dict:
    """Return a minimal notebook code cell dict."""
    return {"cell_type": "code", "source": source, "outputs": outputs or [], "metadata": {}}


def _stream_output(text: str | list[str]) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": text}


def _display_output(text: str) -> dict:
    return {"output_type": "display_data", "data": {"text/plain": text}, "metadata": {}}


def _execute_output(text: str) -> dict:
    return {"output_type": "execute_result", "data": {"text/plain": text}, "metadata": {}, "execution_count": 1}


def _error_output(traceback: list[str]) -> dict:
    return {"output_type": "error", "ename": "ValueError", "evalue": "oops", "traceback": traceback}


# ---------------------------------------------------------------------------
# scan_for_secrets
# ---------------------------------------------------------------------------

class TestScanForSecrets:
    def test_detects_openai_key(self, tmp_path: Path) -> None:
        key = "sk-" + "a" * 48
        _write(tmp_path, "config.py", f'OPENAI_KEY = "{key}"\n')
        findings = scan_for_secrets(tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL
        assert "OpenAI" in findings[0].title

    def test_detects_huggingface_token(self, tmp_path: Path) -> None:
        token = "hf_" + "B" * 34
        _write(tmp_path, "train.py", f"token = '{token}'\n")
        findings = scan_for_secrets(tmp_path)
        assert findings[0].severity == Severity.CRITICAL
        assert "HuggingFace" in findings[0].title

    def test_detects_aws_access_key(self, tmp_path: Path) -> None:
        # AWS access key IDs are exactly AKIA + 16 uppercase alphanumeric chars.
        _write(tmp_path, "deploy.sh", "AWS_KEY=AKIAIOSFODNN7EXAMPLE\n")
        findings = scan_for_secrets(tmp_path)
        assert any("AWS" in f.title for f in findings)
        assert all(f.severity == Severity.CRITICAL for f in findings if "AWS" in f.title)

    def test_detects_github_pat(self, tmp_path: Path) -> None:
        token = "ghp_" + "x" * 36
        _write(tmp_path, "ci.yml", f"token: {token}\n")
        findings = scan_for_secrets(tmp_path)
        assert any("GitHub" in f.title for f in findings)

    def test_detects_slack_webhook(self, tmp_path: Path) -> None:
        url = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        _write(tmp_path, "notify.py", f'WEBHOOK = "{url}"\n')
        findings = scan_for_secrets(tmp_path)
        assert any("Slack" in f.title for f in findings)
        assert any(f.severity == Severity.HIGH for f in findings)

    def test_detects_postgres_url(self, tmp_path: Path) -> None:
        _write(tmp_path, "settings.py", 'DB = "postgres://admin:s3cr3t@db.example.com/mydb"\n')
        findings = scan_for_secrets(tmp_path)
        assert any("PostgreSQL" in f.title for f in findings)
        assert any(f.severity == Severity.HIGH for f in findings)

    def test_detects_mysql_url(self, tmp_path: Path) -> None:
        _write(tmp_path, "settings.py", 'DB = "mysql://root:password@localhost/app"\n')
        findings = scan_for_secrets(tmp_path)
        assert any("MySQL" in f.title for f in findings)

    def test_no_findings_for_clean_file(self, tmp_path: Path) -> None:
        _write(tmp_path, "utils.py", "def add(a, b):\n    return a + b\n")
        assert scan_for_secrets(tmp_path) == []

    def test_reports_correct_line_number(self, tmp_path: Path) -> None:
        key = "sk-" + "z" * 48
        _write(tmp_path, "cfg.py", "# config\nimport os\nKEY = '" + key + "'\n")
        findings = scan_for_secrets(tmp_path)
        assert findings[0].line_number == 3

    def test_reports_relative_file_path(self, tmp_path: Path) -> None:
        key = "sk-" + "r" * 48
        _write(tmp_path, "sub/cfg.py", f'KEY = "{key}"\n')
        findings = scan_for_secrets(tmp_path)
        assert findings[0].file_path == "sub/cfg.py"

    def test_skips_venv_directory(self, tmp_path: Path) -> None:
        key = "sk-" + "v" * 48
        _write(tmp_path, "venv/lib/site.py", f'KEY = "{key}"\n')
        assert scan_for_secrets(tmp_path) == []

    def test_skips_node_modules_directory(self, tmp_path: Path) -> None:
        key = "sk-" + "n" * 48
        _write(tmp_path, "node_modules/pkg/index.js", f'const key = "{key}";')
        assert scan_for_secrets(tmp_path) == []

    def test_skips_binary_extensions(self, tmp_path: Path) -> None:
        # Write a .pyc file — it should never be read.
        (tmp_path / "model.pyc").write_bytes(b"\x00\x01\x02sk-" + b"a" * 48)
        assert scan_for_secrets(tmp_path) == []

    def test_skips_ipynb_files(self, tmp_path: Path) -> None:
        # .ipynb secrets are the responsibility of scan_jupyter_notebooks.
        key = "sk-" + "i" * 48
        nb = _make_notebook([_code_cell(source=f'KEY = "{key}"')])
        _write(tmp_path, "notebook.ipynb", nb)
        assert scan_for_secrets(tmp_path) == []

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        key = "sk-" + "s" * 48
        _write(tmp_path, "cfg.py", f'K="{key}"\n')
        assert len(scan_for_secrets(str(tmp_path))) == 1

    def test_only_one_finding_per_line(self, tmp_path: Path) -> None:
        # A line with both an OpenAI key and a HF token should yield one finding.
        key = "sk-" + "a" * 48
        token = "hf_" + "B" * 34
        _write(tmp_path, "cfg.py", f'KEYS = "{key} {token}"\n')
        findings = scan_for_secrets(tmp_path)
        assert len(findings) == 1

    def test_skips_tests_directory(self, tmp_path: Path) -> None:
        key = "sk-" + "t" * 48
        _write(tmp_path, "tests/fixtures.py", f'KEY = "{key}"\n')
        assert scan_for_secrets(tmp_path) == []

    def test_skips_test_prefixed_files(self, tmp_path: Path) -> None:
        key = "sk-" + "u" * 48
        _write(tmp_path, "test_settings.py", f'KEY = "{key}"\n')
        assert scan_for_secrets(tmp_path) == []

    def test_skips_dot_venv_directory(self, tmp_path: Path) -> None:
        key = "sk-" + "w" * 48
        _write(tmp_path, ".venv/lib/site.py", f'KEY = "{key}"\n')
        assert scan_for_secrets(tmp_path) == []


# ---------------------------------------------------------------------------
# scan_for_misconfigs
# ---------------------------------------------------------------------------

class TestScanForMisconfigs:
    def test_detects_debug_true(self, tmp_path: Path) -> None:
        _write(tmp_path, "settings.py", "DEBUG = True\n")
        findings = scan_for_misconfigs(tmp_path)
        assert any("DEBUG" in f.title for f in findings)
        assert any(f.severity == Severity.HIGH for f in findings)

    def test_detects_flask_app_debug(self, tmp_path: Path) -> None:
        _write(tmp_path, "app.py", "app.debug = True\n")
        findings = scan_for_misconfigs(tmp_path)
        assert any("Flask" in f.title for f in findings)
        assert any(f.severity == Severity.HIGH for f in findings)

    def test_detects_permissive_cors(self, tmp_path: Path) -> None:
        _write(tmp_path, "main.py", 'app.add_middleware(CORSMiddleware, allow_origins=["*"])\n')
        findings = scan_for_misconfigs(tmp_path)
        assert any("CORS" in f.title for f in findings)
        assert any(f.severity == Severity.MEDIUM for f in findings)

    def test_detects_unauthenticated_predict_endpoint(self, tmp_path: Path) -> None:
        content = (
            '@app.post("/predict")\n'
            'def predict(data: InputData):\n'
            '    return model(data)\n'
        )
        _write(tmp_path, "routes.py", content)
        findings = scan_for_misconfigs(tmp_path)
        assert any("endpoint" in f.title.lower() for f in findings)
        assert any(f.severity == Severity.MEDIUM for f in findings)

    def test_no_finding_for_authenticated_endpoint(self, tmp_path: Path) -> None:
        content = (
            '@app.post("/predict")\n'
            'def predict(data: InputData, user=Depends(get_current_user)):\n'
            '    return model(data)\n'
        )
        _write(tmp_path, "routes.py", content)
        findings = scan_for_misconfigs(tmp_path)
        assert not any("endpoint" in f.title.lower() for f in findings)

    def test_no_findings_for_clean_file(self, tmp_path: Path) -> None:
        _write(tmp_path, "util.py", "def helper():\n    pass\n")
        assert scan_for_misconfigs(tmp_path) == []

    def test_debug_false_not_flagged(self, tmp_path: Path) -> None:
        _write(tmp_path, "settings.py", "DEBUG = False\n")
        findings = scan_for_misconfigs(tmp_path)
        assert not any("DEBUG" in f.title for f in findings)

    def test_reports_correct_line_number(self, tmp_path: Path) -> None:
        _write(tmp_path, "cfg.py", "import os\n\nDEBUG = True\n")
        findings = scan_for_misconfigs(tmp_path)
        debug_findings = [f for f in findings if "DEBUG" in f.title]
        assert debug_findings[0].line_number == 3

    def test_skips_venv_directory(self, tmp_path: Path) -> None:
        _write(tmp_path, "venv/lib/config.py", "DEBUG = True\n")
        assert scan_for_misconfigs(tmp_path) == []

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        _write(tmp_path, "s.py", "DEBUG = True\n")
        assert len(scan_for_misconfigs(str(tmp_path))) >= 1

    def test_skips_tests_directory(self, tmp_path: Path) -> None:
        _write(tmp_path, "tests/test_app.py", "DEBUG = True\n")
        assert scan_for_misconfigs(tmp_path) == []

    def test_skips_test_prefixed_files(self, tmp_path: Path) -> None:
        _write(tmp_path, "test_settings.py", "DEBUG = True\n")
        assert scan_for_misconfigs(tmp_path) == []


# ---------------------------------------------------------------------------
# scan_jupyter_notebooks
# ---------------------------------------------------------------------------

class TestScanJupyterNotebooks:
    def test_detects_secret_in_stream_output(self, tmp_path: Path) -> None:
        key = "sk-" + "a" * 48
        nb = _make_notebook([_code_cell(outputs=[_stream_output(f"API_KEY={key}\n")])])
        _write(tmp_path, "notebook.ipynb", nb)
        findings = scan_jupyter_notebooks(tmp_path)
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL

    def test_detects_secret_in_display_data_output(self, tmp_path: Path) -> None:
        token = "hf_" + "C" * 34
        nb = _make_notebook([_code_cell(outputs=[_display_output(f"token: {token}")])])
        _write(tmp_path, "nb.ipynb", nb)
        findings = scan_jupyter_notebooks(tmp_path)
        assert any("HuggingFace" in f.title for f in findings)

    def test_detects_secret_in_execute_result_output(self, tmp_path: Path) -> None:
        key = "ghp_" + "y" * 36
        nb = _make_notebook([_code_cell(outputs=[_execute_output(f"'{key}'")])])
        _write(tmp_path, "nb.ipynb", nb)
        findings = scan_jupyter_notebooks(tmp_path)
        assert any("GitHub" in f.title for f in findings)

    def test_detects_secret_in_error_traceback(self, tmp_path: Path) -> None:
        key = "sk-" + "e" * 48
        tb = [f"ValueError: bad key {key}"]
        nb = _make_notebook([_code_cell(outputs=[_error_output(tb)])])
        _write(tmp_path, "nb.ipynb", nb)
        findings = scan_jupyter_notebooks(tmp_path)
        assert len(findings) == 1

    def test_no_findings_for_clean_notebook(self, tmp_path: Path) -> None:
        nb = _make_notebook([_code_cell(outputs=[_stream_output("hello world\n")])])
        _write(tmp_path, "clean.ipynb", nb)
        assert scan_jupyter_notebooks(tmp_path) == []

    def test_no_findings_when_no_notebooks_present(self, tmp_path: Path) -> None:
        _write(tmp_path, "main.py", "print('hi')\n")
        assert scan_jupyter_notebooks(tmp_path) == []

    def test_handles_malformed_json_gracefully(self, tmp_path: Path) -> None:
        _write(tmp_path, "bad.ipynb", "not valid json {{{")
        # Should not raise; bad files are silently skipped.
        findings = scan_jupyter_notebooks(tmp_path)
        assert findings == []

    def test_skips_notebooks_in_venv(self, tmp_path: Path) -> None:
        key = "sk-" + "v" * 48
        nb = _make_notebook([_code_cell(outputs=[_stream_output(key)])])
        _write(tmp_path, "venv/notebooks/nb.ipynb", nb)
        assert scan_jupyter_notebooks(tmp_path) == []

    def test_finding_file_path_is_relative(self, tmp_path: Path) -> None:
        key = "sk-" + "p" * 48
        nb = _make_notebook([_code_cell(outputs=[_stream_output(key)])])
        _write(tmp_path, "nbs/analysis.ipynb", nb)
        findings = scan_jupyter_notebooks(tmp_path)
        assert findings[0].file_path == "nbs/analysis.ipynb"

    def test_stream_output_as_list_of_strings(self, tmp_path: Path) -> None:
        key = "sk-" + "L" * 48
        nb = _make_notebook([
            _code_cell(outputs=[_stream_output([f"key={key}\n", "done\n"])])
        ])
        _write(tmp_path, "nb.ipynb", nb)
        findings = scan_jupyter_notebooks(tmp_path)
        assert len(findings) == 1

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        key = "sk-" + "S" * 48
        nb = _make_notebook([_code_cell(outputs=[_stream_output(key)])])
        _write(tmp_path, "nb.ipynb", nb)
        assert len(scan_jupyter_notebooks(str(tmp_path))) == 1


# ---------------------------------------------------------------------------
# scan_configs — combined entry point
# ---------------------------------------------------------------------------

class TestScanConfigs:
    def test_combines_all_three_scanners(self, tmp_path: Path) -> None:
        key = "sk-" + "c" * 48
        _write(tmp_path, "settings.py", f'KEY = "{key}"\nDEBUG = True\n')
        nb = _make_notebook([_code_cell(outputs=[_stream_output(f"token={key}")])])
        _write(tmp_path, "nb.ipynb", nb)

        findings = scan_configs(tmp_path)
        severities = {f.severity for f in findings}
        # Expect at least CRITICAL (secret) and HIGH (DEBUG).
        assert Severity.CRITICAL in severities
        assert Severity.HIGH in severities
        assert len(findings) >= 3

    def test_returns_empty_for_clean_project(self, tmp_path: Path) -> None:
        _write(tmp_path, "app.py", "def main():\n    pass\n")
        assert scan_configs(tmp_path) == []

    def test_all_findings_have_required_fields(self, tmp_path: Path) -> None:
        key = "sk-" + "f" * 48
        _write(tmp_path, "cfg.py", f'KEY = "{key}"\n')
        for f in scan_configs(tmp_path):
            assert isinstance(f, Finding)
            assert f.title
            assert f.description
            assert f.severity in Severity
            assert f.scanner_name == "config_analyzer"
