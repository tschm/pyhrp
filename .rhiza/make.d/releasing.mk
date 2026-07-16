## .rhiza/make.d/releasing.mk - Releasing and Versioning
# This file provides changelog and release-status helpers.
#
# Version bumping and release tagging are NOT make targets. Releasing is driven
# by the rhiza-claude `/release` command, which derives the next version, bumps
# pyproject.toml, regenerates CHANGELOG.md, and creates the tag locally. Pushing
# that tag triggers the release workflow (rhiza_release.yml).
# See https://github.com/Jebel-Quant/rhiza-claude.

# Declare phony targets (they don't produce files)
.PHONY: release-status changelog

##@ Releasing and Versioning
release-status: ## show release workflow status and latest release information
ifeq ($(FORGE_TYPE),github)
	@{ $(MAKE) --no-print-directory workflow-status; printf "\n"; $(MAKE) --no-print-directory latest-release; } 2>&1 | $${PAGER:-less -R}
else ifeq ($(FORGE_TYPE),gitlab)
	@printf "${YELLOW}[WARN] GitLab detected — release-status is not yet supported for GitLab repositories.${RESET}\n"
	@printf "${BLUE}[INFO] Please check your pipeline status in the GitLab UI.${RESET}\n"
else
	@printf "${RED}[ERROR] Could not detect forge type (.github/workflows/ or .gitlab-ci.yml not found)${RESET}\n"
endif

changelog: install-uv ## generate/update CHANGELOG.md from git history using git-cliff (config: cliff.toml)
	@printf "${BLUE}[INFO] Generating CHANGELOG.md with git-cliff...${RESET}\n"
	@${UVX_BIN} git-cliff --output CHANGELOG.md
	@printf "${GREEN}[OK] CHANGELOG.md updated.${RESET}\n"
