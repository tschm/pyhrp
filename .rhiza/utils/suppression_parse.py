# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Parsing layer for the suppression audit: scan source for inline suppressions.

This module owns the *parsing* concerns of the suppression audit:

- The suppression comment patterns (``# noqa``, ``# nosec``, ``# type: ignore``,
  ``# pragma: no cover``, ``# noinspection``).
- The :class:`Suppression` data model.
- File scanning via Python's ``tokenize`` module so that only real comment
  tokens are inspected.
- Density grading and CVE extraction from ``# nosec`` comments.

Rendering and process-exit logic live in :mod:`suppression_report`; orchestration
and the CLI live in :mod:`suppression_audit`.
"""

from __future__ import annotations

import io
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Suppression patterns
# ---------------------------------------------------------------------------

# Each entry: (kind_label, compiled_regex).
# The first capture group (if any) captures the comma-separated rule codes.
SUPPRESSION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "noqa",
        re.compile(r"#\s*noqa(?:\s*:\s*([A-Z0-9]+(?:\s*,\s*[A-Z0-9]+)*))?", re.IGNORECASE),
    ),
    (
        "nosec",
        re.compile(r"#\s*nosec(?:\s*:?\s*([A-Z0-9]+(?:\s*,\s*[A-Z0-9]+)*))?", re.IGNORECASE),
    ),
    (
        "type:ignore",
        re.compile(r"#\s*type\s*:\s*ignore(?:\[([^\]]+)\])?", re.IGNORECASE),
    ),
    (
        "no cover",
        re.compile(r"#\s*pragma\s*:\s*no\s+cover", re.IGNORECASE),
    ),
    (
        "noinspection",
        re.compile(r"#\s*noinspection\s+(\w+)", re.IGNORECASE),
    ),
]

# Directories to skip during the scan
_SKIP_DIRS = {".venv", ".git", "node_modules", ".tox", "build", "dist", "__pycache__", "tests"}

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d+\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Suppression:
    """A single suppression comment found in the codebase.

    Attributes:
        file: Path to the file containing the suppression.
        line_no: 1-based line number of the comment.
        kind: Suppression family label (e.g. ``"noqa"``, ``"nosec"``,
            ``"type:ignore"``, ``"no cover"``, ``"noinspection"``).
        codes: The specific rule codes named in the comment, if any
            (e.g. ``["E501", "F401"]``); empty for blanket suppressions.
        raw: The verbatim comment text, used for downstream CVE extraction.
    """

    file: str
    line_no: int
    kind: str
    codes: list[str] = field(default_factory=list)
    raw: str = ""


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------


def _should_skip(path: Path) -> bool:
    """Return True if any path component is in the skip-list."""
    return bool(_SKIP_DIRS.intersection(path.parts))


def _is_rhiza_repo(root: Path) -> bool:
    """Return True if *root* is the rhiza framework repo itself.

    Consumer repos have a ``.rhiza/template.yml`` file that records the upstream
    rhiza repository reference. The rhiza repo itself never has this file — its
    absence is the reliable signal that we are running inside the framework repo.
    """
    return not (root / ".rhiza" / "template.yml").exists()


def scan_file(path: Path) -> list[Suppression]:
    """Scan a single Python file and return all suppressions found.

    Uses Python's ``tokenize`` module so that only actual comment tokens are
    inspected — patterns that appear inside string literals or docstrings are
    correctly ignored.

    Args:
        path: Path to the Python file to scan.

    Returns:
        A list of :class:`Suppression` records, one per matching comment line, in
        source order. Empty if the file cannot be read or fails to tokenize.
    """
    suppressions: list[Suppression] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return suppressions

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok_type, tok_string, tok_start, _tok_end, _line in tokens:
            if tok_type != tokenize.COMMENT:
                continue
            line_no = tok_start[0]
            for kind, pattern in SUPPRESSION_PATTERNS:
                match = pattern.search(tok_string)
                if match:
                    codes_raw = match.group(1) if match.lastindex and match.group(1) else ""
                    codes = [c.strip() for c in codes_raw.split(",") if c.strip()] if codes_raw else []
                    suppressions.append(
                        Suppression(
                            file=str(path),
                            line_no=line_no,
                            kind=kind,
                            codes=codes,
                            raw=tok_string.strip(),
                        )
                    )
                    break  # count each comment line once
    except (tokenize.TokenError, IndentationError):
        pass  # skip files with tokenization errors or bad indentation

    return suppressions


def count_non_empty_lines(path: Path) -> int:
    """Count non-empty lines in a file.

    Args:
        path: Path to the file to measure.

    Returns:
        The number of lines that contain non-whitespace characters, or ``0`` if
        the file cannot be read. Used as the denominator for suppression density.
    """
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
    except OSError:
        return 0


def collect_suppressions(root: Path) -> tuple[list[Path], list[Suppression], int]:
    """Collect Python files, suppressions, and non-empty line counts.

    Args:
        root: Directory tree to scan recursively for Python files.

    Returns:
        A tuple ``(py_files, suppressions, total_lines)`` where ``py_files`` is
        the sorted list of scanned files, ``suppressions`` aggregates every
        :class:`Suppression` found, and ``total_lines`` is the combined count of
        non-empty lines across those files.
    """
    in_rhiza_repo = _is_rhiza_repo(root)

    def _include(p: Path) -> bool:
        """Return True when a Python file should be scanned for suppressions."""
        if _should_skip(p):
            return False
        # In consumer repos, skip the .rhiza/ framework directory entirely
        return not (not in_rhiza_repo and ".rhiza" in p.parts)

    py_files = sorted(p for p in root.rglob("*.py") if _include(p))

    all_suppressions: list[Suppression] = []
    total_lines = 0
    for py_file in py_files:
        all_suppressions.extend(scan_file(py_file))
        total_lines += count_non_empty_lines(py_file)

    return py_files, all_suppressions, total_lines


def nosec_cves(suppressions: list[Suppression]) -> set[str]:
    """Extract CVE identifiers referenced by # nosec suppressions.

    Args:
        suppressions: Suppression records to inspect.

    Returns:
        The set of upper-cased CVE identifiers found in the raw text of
        ``# nosec`` comments. Non-``nosec`` suppressions are ignored.
    """
    cves: set[str] = set()
    for sup in suppressions:
        if sup.kind != "nosec":
            continue
        cves.update(match.upper() for match in _CVE_RE.findall(sup.raw))
    return cves


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

# Grade thresholds: suppressions per 100 lines of code
_GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (0.0, "A+"),
    (0.5, "A"),
    (1.0, "B"),
    (2.0, "C"),
    (3.0, "D"),
]


def compute_grade(density: float) -> str:
    """Return a letter grade based on suppression density.

    Args:
        density: Number of suppressions per 100 non-empty lines of code.

    Returns:
        A letter grade from ``"A+"`` (zero suppressions) down to ``"F"``,
        derived from :data:`_GRADE_THRESHOLDS`.
    """
    grade = "F"
    for threshold, letter in _GRADE_THRESHOLDS:
        if density <= threshold:
            grade = letter
            break
    return grade
