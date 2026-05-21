# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Placeholder for upcoming changes. Update this section as part of each release.

## [2.0.0] - 2026-05-20

### Changed

- Migrated the codebase from pandas/matplotlib to polars/plotly (`refactor`).
- Updated notebook runtime requirements and compatibility details for marimo workflows.

### Fixed

- Restored and stabilized the `Portfolio.weights` API behavior.
- Improved handling of NaN values in return processing.

## [1.6.4] - 2026-05-20

### Added

- Added `cvx-linalg`-based portfolio variance computation (`feat`).

### Changed

- Synced repository automation and documentation templates with recent Rhiza updates.
- Updated dependency sets through multiple Dependabot maintenance releases.

[Unreleased]: https://github.com/tschm/pyhrp/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/tschm/pyhrp/compare/v1.6.4...v2.0.0
[1.6.4]: https://github.com/tschm/pyhrp/compare/v1.6.3...v1.6.4
