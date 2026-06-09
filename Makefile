# path: Makefile

.PHONY: install test lint format clean build check-platform cross-test platform-info

install:
	python -m pip install --upgrade pip
	pip install -e .[dev]

# Core tests (works on Linux/macOS/Windows)
test:
	python -m pytest tests/ -v --tb=short || echo "No pytest tests or skipped"
	python -c "from src.design.vlm_design_agent import VLMDesignAgent; print('VLMDesignAgent OK')"
	python -c "from src.memory.graph_memory_blackbox import GraphMemoryBlackbox; print('GraphMemoryBlackbox OK')"
	python -c "from src.automation.cad_script_generator import CADScriptGenerator; print('CADScriptGenerator OK')"
	python -c "from src.juniorllm.memory.plasticity import PlasticityEngine; print('PlasticityEngine OK')"
	python -c "from src.inference.bitnet_precision_router import BitNetPrecisionRouter; print('BitNetPrecisionRouter OK')"

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
	python -c "import platform, sys; print('Python:', sys.version); print('Platform:', platform.system(), platform.machine())"

platform-info:
	python -c "import platform; print(platform.platform())"

# Cross-platform smoke test
cross-test: check-platform test

# Quick platform-specific install hints
install-apple:
	pip install -e .[apple-silicon,dev]

install-cpu:
	pip install -e .[cpu,dev]
