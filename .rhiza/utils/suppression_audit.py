# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Suppression audit: scan codebase for inline suppressions of security, coverage, docs, and linting.

Detects and reports on inline suppression comments such as:
- ``# noqa`` / ``# noqa: CODE`` (ruff/flake8 linting suppressions)
- ``# nosec`` / ``# nosec: CODE`` (bandit security suppressions)
- ``# type: ignore`` / ``# type: ignore[CODE]`` (mypy/pyright type-checking suppressions)
- ``# pragma: no cover`` (coverage suppressions)
- ``# noinspection CODE`` (PyCharm/IDE suppressions)

Outputs a detailed per-file report, an ASCII histogram, and a letter grade.

This module is the thin orchestrator and CLI. Parsing lives in
:mod:`suppression_parse` and reporting/output lives in :mod:`suppression_report`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow the parsing/reporting submodules to be imported whether this file is run
# directly (``python .rhiza/utils/suppression_audit.py``) or loaded by path via
# ``importlib`` (tests and fuzz harness), by ensuring its own directory is on the
# import path.
_UTILS_DIR = str(Path(__file__).resolve().parent)
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from suppression_parse import collect_suppressions  # noqa: E402
from suppression_report import check_stale_nosec_cves, print_report  # noqa: E402

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Run the suppression audit and print a structured report.

    Scans the working directory tree for inline suppression comments, prints a
    per-file report, a histogram by code, and an overall density grade. With
    ``--fail-stale-nosec-cve`` it additionally cross-checks CVE-tagged ``# nosec``
    comments against live pip-audit findings and fails on stale ones.

    Args:
        argv: Argument list to parse (excluding the program name). Defaults to
            ``None``, in which case ``sys.argv[1:]`` is used by the module entry
            point. Unrecognised arguments are forwarded to pip-audit.

    Returns:
        A process exit code: ``0`` on success, ``1`` when stale CVE-tagged
        ``# nosec`` suppressions are found, or ``2`` when pip-audit fails to run.

    Example:
        Run via the Makefile (the supported entry point)::

            $ make suppression-audit

        or invoke the module directly, enabling the stale-CVE gate::

            $ python .rhiza/utils/suppression_audit.py --fail-stale-nosec-cve
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--fail-stale-nosec-cve",
        action="store_true",
        help="Fail when # nosec comments reference CVEs that pip-audit no longer reports.",
    )
    args, pip_audit_args = parser.parse_known_args(argv)

    root = Path(".")
    py_files, all_suppressions, total_lines = collect_suppressions(root)
    print_report(py_files, all_suppressions, total_lines)

    if args.fail_stale_nosec_cve:
        return check_stale_nosec_cves(all_suppressions, pip_audit_args)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
