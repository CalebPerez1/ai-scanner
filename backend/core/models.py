"""Core data models for AIScan.

This module defines the Severity enum, the Finding Pydantic model
and the ScanResult Pydantic model used across scanner modules.

Follow project style: type hints and Google-style docstrings.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, root_validator


class Severity(str, Enum):
    """Severity levels for findings.

    Values are chosen to be human-friendly and serializable.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Finding(BaseModel):
    """A security finding discovered by a scanner module.

    Attributes:
        title: Short summary of the finding.
        description: Full description explaining the issue.
        severity: Severity level of the finding.
        scanner_name: Name of the scanner that produced this finding.
        file_path: Path to the affected file within the project, if applicable.
        line_number: Line number in the file where the issue was detected, if applicable.
        recommendation: Short guidance on how to remediate the issue.
    """

    title: str = Field(..., description="Short summary of the finding")
    description: str = Field(..., description="Detailed description of the finding")
    severity: Severity = Field(..., description="Severity level")
    scanner_name: str = Field(..., description="Originating scanner module name")
    file_path: Optional[str] = Field(None, description="Relative file path in project, if applicable")
    line_number: Optional[int] = Field(None, description="Line number in file, if applicable")
    recommendation: Optional[str] = Field(None, description="Remediation guidance")


class ScanResult(BaseModel):
    """Result of scanning a project.

    Attributes:
        project_name: The project identifier/name that was scanned.
        scan_date: UTC timestamp when the scan was run.
        findings: List of `Finding` objects discovered during the scan.
        total_findings: Computed total number of findings.
        by_severity: Mapping from `Severity` to counts.
    """

    project_name: str = Field(..., description="Name or identifier of the scanned project")
    scan_date: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of the scan")
    findings: List[Finding] = Field(default_factory=list, description="Findings discovered by scanners")
    total_findings: int = Field(0, description="Total number of findings (computed)")
    by_severity: Dict[Severity, int] = Field(default_factory=dict, description="Counts of findings by severity")

    @root_validator(pre=False, skip_on_failure=True)
    def _compute_summary(cls, values: Dict) -> Dict:
        """Compute `total_findings` and `by_severity` from `findings`.

        This keeps the model normalized so consumers can rely on summary
        fields being accurate after model initialization.
        """

        findings: List[Finding] = values.get("findings") or []
        total = len(findings)

        # Initialize counts with zero for all severities to ensure consistent keys
        counts: Dict[Severity, int] = {s: 0 for s in Severity}
        for f in findings:
            counts[f.severity] += 1

        values["total_findings"] = total
        values["by_severity"] = counts
        return values


__all__ = ["Severity", "Finding", "ScanResult"]
