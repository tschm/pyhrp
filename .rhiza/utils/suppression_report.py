# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Reporting layer for the suppression audit: render output and run pip-audit.

This module owns the *reporting* and *output* concerns of the suppression audit:

- ANSI colour constants and the ASCII histogram bar.
- Printing the detailed per-file report, the histogram, and the density grade.
- Cross-checking CVE-tagged ``# nosec`` comments against live pip-audit findings
  (the ``--fail-stale-nosec-cve`` gate), including the subprocess invocation.

Parsing concerns live in :mod:`suppression_parse`; orchestration and the CLI
live in :mod:`suppression_audit`.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404
import sys
from collections import Counter
from pathlib import Path

from suppression_parse import Suppression, compute_grade, nosec_cves

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------

_BAR_WIDTH = 24

_GRADE_COLOURS = {
    "A+": "\033[92m",  # bright green
    "A": "\033[32m",  # green
    "B": "\033[32m",  # green
    "C": "\033[33m",  # yellow
    "D": "\033[33m",  # yellow
    "F": "\033[31m",  # red
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_BLUE = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"


def _bar(count: int, max_count: int) -> str:
    """Render a fixed-width ASCII progress bar."""
    if max_count == 0:
        return "░" * _BAR_WIDTH
    filled = round(count / max_count * _BAR_WIDTH)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def print_report(py_files: list[Path], all_suppressions: list[Suppression], total_lines: int) -> None:
    """Print the suppression audit report.

    Args:
        py_files: The Python files that were scanned.
        all_suppressions: Every suppression discovered across those files.
        total_lines: Combined count of non-empty lines, used as the density
            denominator.
    """
    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    print()
    print(f"{_BOLD}{'=' * 62}{_RESET}")
    print(f"{_BOLD}  Suppression Audit Report{_RESET}")
    print(f"{_BOLD}{'=' * 62}{_RESET}")
    print()

    # -----------------------------------------------------------------------
    # Detailed per-file report
    # -----------------------------------------------------------------------
    print(f"{_BOLD}Detailed Report:{_RESET}")
    if all_suppressions:
        for sup in all_suppressions:
            codes_str = f"[{', '.join(sup.codes)}]" if sup.codes else ""
            print(f"  {_YELLOW}{sup.file}{_RESET}:{_GREEN}{sup.line_no}{_RESET}: # {sup.kind}{codes_str}")
    else:
        print(f"  {_GREEN}No inline suppressions found.{_RESET}")
    print()

    # -----------------------------------------------------------------------
    # Histogram by code
    # -----------------------------------------------------------------------
    print(f"{_BOLD}Histogram (by suppression code):{_RESET}")
    code_counter: Counter[str] = Counter()
    for sup in all_suppressions:
        if sup.codes:
            for code in sup.codes:
                code_counter[f"{sup.kind}[{code}]"] += 1
        else:
            code_counter[f"{sup.kind}"] += 1
    if code_counter:
        max_code_count = max(code_counter.values())
        total_code_count = sum(code_counter.values())
        for label, count in code_counter.most_common():
            pct = count / total_code_count * 100
            print(f"  {label:<20} {_BLUE}{_bar(count, max_code_count)}{_RESET}  {count:>3}  ({pct:.0f}%)")
    else:
        print("  (none)")
    print()

    # -----------------------------------------------------------------------
    # Summary + Grade
    # -----------------------------------------------------------------------
    density = (len(all_suppressions) / total_lines * 100) if total_lines > 0 else 0.0
    grade = compute_grade(density)
    grade_colour = _GRADE_COLOURS.get(grade, _RESET)

    print(f"{_BOLD}Summary:{_RESET}")
    print(f"  Files scanned   : {len(py_files)}")
    print(f"  Lines scanned   : {total_lines:,}")
    print(f"  Suppressions    : {len(all_suppressions)}")
    print(f"  Density         : {density:.2f} per 100 lines")
    print()
    print(f"  Grade           : {grade_colour}{_BOLD}{grade}{_RESET}")
    print()


# ---------------------------------------------------------------------------
# Stale CVE gate
# ---------------------------------------------------------------------------


def _active_pip_audit_ids(extra_args: list[str]) -> set[str]:
    """Return vulnerability IDs currently reported by pip-audit."""
    uvx = shutil.which("uvx") or "uvx"
    cmd = [uvx, "pip-audit", "--format", "json", *extra_args]
    proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603  # nosec B603

    if proc.returncode not in {0, 1}:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError("pip-audit execution failed")

    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("pip-audit did not return valid JSON") from exc

    ids: set[str] = set()
    for dep in data.get("dependencies", []):
        for vuln in dep.get("vulns", []):
            vuln_id = vuln.get("id")
            if vuln_id:
                ids.add(str(vuln_id).upper())
            for alias in vuln.get("aliases", []):
                ids.add(str(alias).upper())
    return ids


def check_stale_nosec_cves(suppressions: list[Suppression], pip_audit_args: list[str]) -> int:
    """Validate CVE-tagged # nosec suppressions against active pip-audit findings.

    Args:
        suppressions: Suppression records to inspect for CVE-tagged ``# nosec``
            comments.
        pip_audit_args: Extra arguments forwarded to ``pip-audit``.

    Returns:
        A process exit code: ``0`` when no stale CVE suppressions are found,
        ``1`` when stale suppressions are detected, or ``2`` when pip-audit
        fails to run.
    """
    suppressed_cves = nosec_cves(suppressions)
    if not suppressed_cves:
        print(f"{_GREEN}[OK]{_RESET} No CVE-tagged # nosec suppressions found.")
        return 0

    try:
        active_cves = _active_pip_audit_ids(pip_audit_args)
    except RuntimeError as exc:
        print(f"{_RED}[FAIL]{_RESET} {exc}")
        return 2

    stale = sorted(cve for cve in suppressed_cves if cve not in active_cves)
    if stale:
        print(f"{_RED}[FAIL]{_RESET} Stale # nosec CVE suppressions detected:")
        for cve in stale:
            print(f"  - {cve}")
        return 1

    print(f"{_GREEN}[OK]{_RESET} All CVE-tagged # nosec suppressions match active pip-audit findings.")
    return 0
