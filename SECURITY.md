# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

**Do NOT report security vulnerabilities through public GitHub issues.**

Please use one of the following methods:

1. **GitHub Security Advisories** (preferred)
   - Go to the [Security Advisories](https://github.com/tschm/pyhrp/security/advisories) page
   - Click "New draft security advisory" and fill in the details

2. **Email**
   - Contact the maintainer at `thomas.schmelzer@gmail.com`
   - Include "SECURITY" in the subject line

### What to include

- **Description**: A clear description of the vulnerability
- **Impact**: Potential impact
- **Steps to Reproduce**: Detailed reproduction steps
- **Affected Versions**: Which versions are affected
- **Suggested Fix**: If you have one (optional)

### What to expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution**: Critical issues within 30 days
- **Credit**: Reporters credited in the security advisory unless anonymity is requested

## Security Measures

### CI Security Scanning

- **Security scans**: `make security` runs on every push and pull request
- **License compliance**: `pip-licenses` verifies all dependency licenses
- **Dependency checks**: `deptry` validates declared vs. actual dependencies
- **Pre-commit hooks**: Enforced on all commits via CI

### Supply Chain Security

- **Locked dependencies**: `uv.lock` ensures reproducible builds
- **SBOM**: CycloneDX Software Bill of Materials (JSON and XML) generated and attached to every release
- **SBOM attestations**: Stored on GitHub for supply chain transparency (public repo)

### Release Security

- **OIDC publishing**: PyPI trusted publishing — no stored PyPI credentials
- **SLSA provenance**: Build attestations generated for all release artifacts (public repo)
- **Tag validation**: Version in `pyproject.toml` must match the release tag before publishing proceeds

## Scope

This policy covers:

- The `pyhrp` Python library (`src/pyhrp/`)
- GitHub Actions workflows in `.github/workflows/`

**Out of scope:**

- Vulnerabilities in upstream dependencies (report these to their respective projects)
- Issues requiring physical machine access
- Social engineering
- Denial of service attacks requiring significant resources
