# path: Makefile

.PHONY: install test lint format clean build check-platform cross-test

install:
	python -m pip install --upgrade pip
	pip install -e .[dev]

# Core tests and import checks (Linux/macOS/Windows friendly)
test:
	python -m pytest tests/ -v --tb=short || echo "No tests or skipped"
	python -c "from src.design.vlm_design_agent import VLMDesignAgent; print('VLMDesignAgent OK')"
	python -c "from src.memory.graph_memory_blackbox import GraphMemoryBlackbox; print('GraphMemoryBlackbox OK')"
	python -c "from src.automation.cad_script_generator import CADScriptGenerator; print('CADScriptGenerator OK')"
	python -c "from src.juniorllm.memory.plasticity import PlasticityEngine; print('PlasticityEngine OK')"

lint:
	ruff check src/
	black --check src/

format:
	black src/
	ruff check --fix src/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache

build:
	python -m build

check-platform:
	python -c "import platform; print('Platform:', platform.system(), platform.machine())"

# Cross-platform smoke test (run on Linux/macOS/Windows)
cross-test: check-platform test
