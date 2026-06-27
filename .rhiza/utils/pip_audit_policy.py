"""Run pip-audit with a tiered vulnerability policy.

Fails the build for vulnerabilities in runtime dependencies.
Warns (without failing) for tooling packages: pip, setuptools, wheel, distribute.
Any extra arguments are forwarded to pip-audit (e.g. ``--ignore-vuln CVE-XXXX-YYYY``).
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404
import sys

_RESET = "\033[0m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"

# Packages treated as build tooling — CVEs warn but do not fail CI.
_TOOLING: frozenset[str] = frozenset({"pip", "setuptools", "wheel", "distribute"})


def _vuln_ids(vuln: dict) -> str:  # type: ignore[type-arg]
    """Return a human-readable string of all IDs for a vulnerability entry.

    Args:
        vuln: A single vulnerability object from pip-audit's JSON output. Must
            contain an ``id`` key and may contain an ``aliases`` list.

    Returns:
        A comma-separated string starting with the primary ``id`` followed by any
        distinct aliases (e.g. ``"PYSEC-2024-1, CVE-2024-1234"``).
    """
    ids = [vuln["id"]] + [a for a in vuln.get("aliases", []) if a != vuln["id"]]
    return ", ".join(ids)


def main() -> int:
    """Run pip-audit and apply the tiered vulnerability policy.

    Invokes ``uvx pip-audit`` with JSON output, forwarding any extra
    command-line arguments. Vulnerabilities in build tooling (see ``_TOOLING``)
    are reported as warnings; vulnerabilities in any other (runtime) dependency
    are reported as failures.

    Returns:
        A process exit code: ``0`` when no vulnerabilities affect runtime
        dependencies (clean run, tooling-only findings, or unparseable output is
        passed through with pip-audit's own code), or ``1`` when at least one
        runtime dependency is vulnerable.

    Example:
        Run via the Makefile (the supported entry point)::

            $ make security

        or call pip-audit through this policy directly, forwarding flags::

            $ python .rhiza/utils/pip_audit_policy.py --ignore-vuln CVE-2024-1234
    """
    uvx = shutil.which("uvx") or "uvx"
    cmd = [uvx, "pip-audit", "--format", "json", *sys.argv[1:]]
    proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603  # nosec B603

    if proc.returncode == 0:
        print(f"{_GREEN}[OK] pip-audit: no vulnerabilities found{_RESET}")
        return 0

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode

    deps = data.get("dependencies", [])
    tooling_vulns = [d for d in deps if d.get("vulns") and d["name"].lower() in _TOOLING]
    runtime_vulns = [d for d in deps if d.get("vulns") and d["name"].lower() not in _TOOLING]

    for dep in tooling_vulns:
        for v in dep["vulns"]:
            print(
                f"{_YELLOW}[WARN] {dep['name']}=={dep['version']}: {_vuln_ids(v)} (tooling — not failing build){_RESET}"
            )

    if not runtime_vulns:
        return 0

    for dep in runtime_vulns:
        for v in dep["vulns"]:
            print(f"{_RED}[FAIL] {dep['name']}=={dep['version']}: {_vuln_ids(v)}{_RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
