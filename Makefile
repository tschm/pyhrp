## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

# Override template default: include mkdocstrings plugin for API docs
MKDOCS_EXTRA_PACKAGES = --with 'mkdocstrings[python]'

# This repo keeps its Marimo notebooks in book/marimo, not the template default
# of docs/notebooks. book.mk exports them to docs/notebooks/*.html, which is what
# mkdocs.yml's nav references.
MARIMO_FOLDER = book/marimo

# CLAUDE.md documents 100% coverage as a hard invariant; the template default is 90.
COVERAGE_FAIL_UNDER = 100

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: developer-local extensions (not committed)
-include local.mk
