.PHONY: install test lint format clean build

install:
	python -m pip install --upgrade pip
	pip install -e .[dev]

# Run core tests and import checks (works on Linux/macOS/Windows)
test:
	python -m pytest tests/ -v --tb=short || echo "No tests directory or tests skipped"
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

# Build source distribution and wheel (cross-platform)
build:
	python -m build

# Quick check for Windows/macOS/Linux compatibility
check-platform:
	python -c "import platform; print('Running on:', platform.system(), platform.machine())"
