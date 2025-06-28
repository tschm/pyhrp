# Colors for pretty output
BLUE := \033[36m
BOLD := \033[1m
RESET := \033[0m

.DEFAULT_GOAL := help

.PHONY: help verify install fmt test marimo clean

##@ Development Setup

venv:
	@printf "$(BLUE)Creating virtual environment...$(RESET)\n"
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@uv venv --python 3.12

install: venv ## Install all dependencies using uv
	@printf "$(BLUE)Installing dependencies...$(RESET)\n"
	@uv sync --all-extras --frozen

##@ Code Quality

fmt: venv ## Run code formatting and linting
	@printf "$(BLUE)Running formatters and linters...$(RESET)\n"
	@uvx pre-commit run --all-files

##@ Testing

test: install ## Run all tests
	@printf "$(BLUE)Running tests...$(RESET)\n"
	@uv pip install pytest
	@uv run pytest src/tests

##@ Cleanup

clean: ## Clean generated files and directories
	@printf "$(BLUE)Cleaning project...$(RESET)\n"
	@git clean -d -X -f

##@ Marimo

marimo_hrp: install ## Start a Marimo server
	@printf "$(BLUE)Start Marimo server...$(RESET)\n"
	#@uv pip install marimo
	@uvx marimo edit --sandbox book/marimo/hrp.py

marimo_1_over_N: install ## Start a Marimo server
	@printf "$(BLUE)Start Marimo server...$(RESET)\n"
	#@uv pip install marimo
	@uvx marimo edit --sandbox book/marimo/1_over_N.py

##@ Help

help: ## Display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
