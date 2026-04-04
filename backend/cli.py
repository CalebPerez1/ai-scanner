"""Command-line interface for AIScan.

Run a security scan against a local project directory or a GitHub repository
and display findings in your choice of format.

Usage examples::

    python -m backend.cli ./my-project
    python -m backend.cli https://github.com/org/repo --llm-endpoint http://localhost:8080/v1/chat/completions
    python -m backend.cli ./my-project --output json --output-file report.json
    python -m backend.cli ./my-project --output markdown --name "My App"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from backend.core.models import Finding, ScanResult, Severity
from backend.core.risk_engine import scan_project


# ---------------------------------------------------------------------------
# ANSI color helpers (no external dependency required)
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"

_SEVERITY_COLORS = {
    Severity.CRITICAL: "\033[91m",   # bright red
    Severity.HIGH:     "\033[31m",   # red
    Severity.MEDIUM:   "\033[33m",   # yellow
    Severity.LOW:      "\033[36m",   # cyan
}

_COLOR_GREEN  = "\033[32m"
_COLOR_BOLD_WHITE = "\033[1;37m"


def _colored(text: str, color: str, *, bold: bool = False) -> str:
    """Wrap *text* in ANSI escape sequences when stdout is a terminal.

    Falls back to plain text when output is being redirected so that piped
    or captured output stays clean.

    Args:
        text: The string to colorize.
        color: An ANSI escape sequence (e.g. ``"\\033[31m"``).
        bold: If True, also apply the bold attribute.

    Returns:
        The text wrapped in color codes, or the original text if not a TTY.
    """
    if not sys.stdout.isatty():
        return text
    prefix = (_BOLD if bold else "") + color
    return f"{prefix}{text}{_RESET}"


def _severity_label(severity: Severity) -> str:
    """Return a left-padded, colorized severity label for terminal output.

    Args:
        severity: The severity level to format.

    Returns:
        An 8-character wide severity name with appropriate ANSI color.
    """
    label = severity.value.upper().ljust(8)
    return _colored(label, _SEVERITY_COLORS[severity])


# ---------------------------------------------------------------------------
# Terminal formatter
# ---------------------------------------------------------------------------

def _format_terminal(result: ScanResult) -> str:
    """Render a ScanResult as a human-readable, colorized terminal report.

    Prints a header, a severity-breakdown table, and a detailed list of
    every finding grouped by severity (CRITICAL first).

    Args:
        result: The ScanResult to render.

    Returns:
        A multi-line string ready to be printed to stdout.
    """
    lines: list[str] = []

    # Header
    lines.append("")
    lines.append(_colored("╔══ AIScan Security Report ══════════════════════════════════╗", _COLOR_BOLD_WHITE, bold=True))
    lines.append(_colored(f"  Project : {result.project_name}", _COLOR_BOLD_WHITE))
    lines.append(_colored(f"  Scanned : {result.scan_date.strftime('%Y-%m-%d %H:%M UTC')}", _COLOR_BOLD_WHITE))
    lines.append(_colored(f"  Findings: {result.total_findings}", _COLOR_BOLD_WHITE))
    lines.append(_colored("╚════════════════════════════════════════════════════════════╝", _COLOR_BOLD_WHITE, bold=True))
    lines.append("")

    # Severity summary table
    lines.append(_colored("  SEVERITY BREAKDOWN", _BOLD))
    lines.append("  " + "─" * 30)
    severity_order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
    for sev in severity_order:
        count = result.by_severity.get(sev, 0)
        bar = "█" * count if count <= 40 else "█" * 40 + f"…+{count - 40}"
        label = sev.value.upper().rjust(8)
        colored_label = _colored(label, _SEVERITY_COLORS[sev])
        colored_bar   = _colored(bar,   _SEVERITY_COLORS[sev]) if count else ""
        lines.append(f"  {colored_label}  {count:>3}  {colored_bar}")
    lines.append("")

    if not result.findings:
        lines.append(_colored("  No findings — clean scan!", _COLOR_GREEN, bold=True))
        lines.append("")
        return "\n".join(lines)

    # Findings detail — sorted by severity then scanner
    sorted_findings = sorted(
        result.findings,
        key=lambda f: (severity_order.index(f.severity), f.scanner_name),
    )

    lines.append(_colored("  FINDINGS", _BOLD))
    lines.append("  " + "─" * 60)

    for idx, finding in enumerate(sorted_findings, start=1):
        sev_label = _severity_label(finding.severity)
        title_line = f"  [{idx:>3}] {sev_label}  {finding.title}"
        lines.append(_colored(title_line, _COLOR_BOLD_WHITE, bold=True) if sys.stdout.isatty() else title_line)

        lines.append(f"         Scanner : {finding.scanner_name}")
        if finding.file_path:
            loc = finding.file_path
            if finding.line_number:
                loc += f":{finding.line_number}"
            lines.append(f"         Location: {loc}")

        # Wrap description at ~72 chars
        desc = finding.description
        indent = "         "
        wrapped = _wrap(desc, width=72, indent=indent)
        lines.append(f"         Detail  : {wrapped.lstrip()}")

        if finding.recommendation:
            wrapped_rec = _wrap(finding.recommendation, width=72, indent=indent)
            lines.append(f"         Fix     : {wrapped_rec.lstrip()}")

        lines.append("")

    return "\n".join(lines)


def _wrap(text: str, width: int, indent: str) -> str:
    """Soft-wrap *text* to *width* characters, indenting continuation lines.

    Args:
        text: The text to wrap.
        width: Maximum line length before wrapping.
        indent: String prepended to each continuation line.

    Returns:
        The wrapped text as a single string with embedded newlines.
    """
    words = text.split()
    if not words:
        return ""
    result_lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            result_lines.append(current)
            current = indent + word
        else:
            current = (current + " " + word) if current else word
    if current:
        result_lines.append(current)
    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

def _format_json(result: ScanResult) -> str:
    """Serialize a ScanResult to a pretty-printed JSON string.

    Severity enum values are serialized as their string values.
    Datetime is serialized as an ISO 8601 string.

    Args:
        result: The ScanResult to serialize.

    Returns:
        A JSON string.
    """
    payload = {
        "project_name": result.project_name,
        "scan_date": result.scan_date.isoformat(),
        "total_findings": result.total_findings,
        "by_severity": {k.value: v for k, v in result.by_severity.items()},
        "findings": [
            {
                "title": f.title,
                "description": f.description,
                "severity": f.severity.value,
                "scanner_name": f.scanner_name,
                "file_path": f.file_path,
                "line_number": f.line_number,
                "recommendation": f.recommendation,
            }
            for f in result.findings
        ],
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------

def _format_markdown(result: ScanResult) -> str:
    """Render a ScanResult as a Markdown report suitable for GitHub or docs.

    Produces a top-level heading, a metadata section, a severity summary
    table, and a detailed findings table.

    Args:
        result: The ScanResult to render.

    Returns:
        A Markdown string.
    """
    lines: list[str] = []

    lines.append(f"# AIScan Security Report — {result.project_name}")
    lines.append("")
    lines.append(f"**Scan date:** {result.scan_date.strftime('%Y-%m-%d %H:%M UTC')}  ")
    lines.append(f"**Total findings:** {result.total_findings}")
    lines.append("")

    # Severity summary
    lines.append("## Severity Summary")
    lines.append("")
    lines.append("| Severity | Count |")
    lines.append("|----------|------:|")
    for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
        count = result.by_severity.get(sev, 0)
        lines.append(f"| {sev.value.capitalize()} | {count} |")
    lines.append("")

    if not result.findings:
        lines.append("*No findings — clean scan!*")
        lines.append("")
        return "\n".join(lines)

    # Findings table
    lines.append("## Findings")
    lines.append("")
    lines.append("| # | Severity | Scanner | Title | Location |")
    lines.append("|---|----------|---------|-------|----------|")

    severity_order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
    sorted_findings = sorted(
        result.findings,
        key=lambda f: (severity_order.index(f.severity), f.scanner_name),
    )

    for idx, f in enumerate(sorted_findings, start=1):
        loc = ""
        if f.file_path:
            loc = f.file_path
            if f.line_number:
                loc += f":{f.line_number}"
        # Escape pipe characters in titles so the table renders correctly.
        safe_title = f.title.replace("|", "\\|")
        lines.append(
            f"| {idx} | {f.severity.value.capitalize()} | {f.scanner_name} | {safe_title} | {loc} |"
        )

    lines.append("")

    # Per-finding detail sections
    lines.append("## Finding Details")
    lines.append("")
    for idx, f in enumerate(sorted_findings, start=1):
        lines.append(f"### {idx}. {f.title}")
        lines.append("")
        lines.append(f"- **Severity:** {f.severity.value.capitalize()}")
        lines.append(f"- **Scanner:** {f.scanner_name}")
        if f.file_path:
            loc = f.file_path + (f":{f.line_number}" if f.line_number else "")
            lines.append(f"- **Location:** `{loc}`")
        lines.append("")
        lines.append(f.description)
        lines.append("")
        if f.recommendation:
            lines.append(f"> **Recommendation:** {f.recommendation}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output dispatch
# ---------------------------------------------------------------------------

_FORMATTERS = {
    "terminal": _format_terminal,
    "json":     _format_json,
    "markdown": _format_markdown,
}


def _write_output(text: str, output_file: Optional[str]) -> None:
    """Write *text* to a file or stdout.

    Args:
        text: The formatted report string.
        output_file: Path to write to, or None to print to stdout.
    """
    if output_file:
        Path(output_file).write_text(text, encoding="utf-8")
        print(f"Report written to: {output_file}", file=sys.stderr)
    else:
        print(text)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser for the AIScan CLI.

    Returns:
        A configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="aiscan",
        description=(
            "AIScan — AI supply chain security scanner.\n\n"
            "Scans a project directory or GitHub repository for dependency "
            "vulnerabilities, unsafe model loading, hardcoded secrets, "
            "misconfigurations, and (optionally) prompt injection vulnerabilities."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m backend.cli ./my-project\n"
            "  python -m backend.cli https://github.com/org/repo\n"
            "  python -m backend.cli ./app --llm-endpoint http://localhost:8080/v1/chat/completions\n"
            "  python -m backend.cli ./app --output json --output-file report.json\n"
            "  python -m backend.cli ./app --output markdown --name 'My App'"
        ),
    )

    parser.add_argument(
        "project",
        metavar="PROJECT",
        help="Local directory path or GitHub repository URL to scan.",
    )
    parser.add_argument(
        "--llm-endpoint",
        metavar="URL",
        default=None,
        help=(
            "URL of an LLM chat/completions endpoint to probe for prompt "
            "injection vulnerabilities (e.g. http://localhost:8080/v1/chat/completions)."
        ),
    )
    parser.add_argument(
        "--output",
        choices=["terminal", "json", "markdown"],
        default="terminal",
        help="Output format. Default: terminal.",
    )
    parser.add_argument(
        "--output-file",
        metavar="PATH",
        default=None,
        help=(
            "Write the report to this file instead of stdout. "
            "Useful with --output json or --output markdown."
        ),
    )
    parser.add_argument(
        "--name",
        metavar="NAME",
        default=None,
        help="Override the project name shown in the report.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, run the scan, format and output the results.

    Exits with code 1 if the scan cannot be completed due to an error.
    Exits with code 2 if any findings are found (useful for CI pipelines).
    Exits with code 0 on a clean scan.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # Inform the user the scan has started (written to stderr so it doesn't
    # pollute --output json or --output markdown captures).
    print(f"Scanning: {args.project}", file=sys.stderr)
    if args.llm_endpoint:
        print(f"LLM endpoint: {args.llm_endpoint}", file=sys.stderr)

    try:
        result: ScanResult = asyncio.run(
            scan_project(
                args.project,
                project_name=args.name,
                llm_endpoint_url=args.llm_endpoint,
            )
        )
    except KeyboardInterrupt:
        print("\nScan interrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: scan failed — {exc}", file=sys.stderr)
        sys.exit(1)

    formatter = _FORMATTERS[args.output]
    report = formatter(result)
    _write_output(report, args.output_file)

    # Non-zero exit when findings exist so CI/CD pipelines can gate on results.
    sys.exit(2 if result.total_findings > 0 else 0)


if __name__ == "__main__":
    main()
