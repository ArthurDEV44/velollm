.PHONY: help build test clean install check fmt clippy doc run-detect run-benchmark

help: ## Show this help message
	@echo "VeloLLM Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build all crates in release mode
	cargo build --release

build-dev: ## Build all crates in debug mode
	cargo build

test: ## Run all tests
	cargo test --all

test-verbose: ## Run all tests with output
	cargo test --all -- --nocapture

clean: ## Clean build artifacts
	cargo clean
	rm -rf target/

install: ## Install velollm CLI to ~/.cargo/bin
	cargo install --path velollm-cli

check: ## Check code without building
	cargo check --all

fmt: ## Format code with rustfmt
	cargo fmt --all

fmt-check: ## Check code formatting
	cargo fmt --all -- --check

clippy: ## Run clippy linter
	cargo clippy --all -- -D warnings

doc: ## Generate documentation
	cargo doc --all --no-deps --open

doc-build: ## Build documentation without opening
	cargo doc --all --no-deps

# Development shortcuts
run-detect: build-dev ## Run: velollm detect
	./target/debug/velollm detect

run-benchmark: build-dev ## Run: velollm benchmark
	./target/debug/velollm benchmark

run-optimize: build-dev ## Run: velollm optimize --dry-run
	./target/debug/velollm optimize --dry-run

# CI/Testing
ci: fmt-check clippy test ## Run all CI checks

# Watch mode (requires cargo-watch)
watch: ## Watch for changes and rebuild
	cargo watch -x build

watch-test: ## Watch for changes and run tests
	cargo watch -x test
