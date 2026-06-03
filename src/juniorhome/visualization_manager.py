# path: src/juniorhome/visualization_manager.py
#!/usr/bin/env python3
"""
Visualization Manager

Modular visualization system for JuniorHome.
Supports Three.js-based manifold and data visualizations that can be
controlled in real-time by local LLMs (via SmartLLMRouter).

Designed to be backend-agnostic and extensible.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .smart_llm_router import SmartLLMRouter
    HAS_ROUTER = True
except ImportError:
    HAS_ROUTER = False
    SmartLLMRouter = None

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class VisualizationManager:
    """
    Manages modular visualizations (Three.js focused) that can be
    updated in real-time by LLMs.
    """

    def __init__(self, output_dir: str = "visualizations", llm_router: Optional[Any] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if HAS_ROUTER and llm_router is None:
            self.llm_router = SmartLLMRouter()
        else:
            self.llm_router = llm_router

        self.active_visualizations: Dict[str, Dict[str, Any]] = {}
        logging.info("VisualizationManager initialized")

    def create_threejs_scene(self, name: str, data: Dict[str, Any]) -> str:
        """
        Generates a basic Three.js HTML scene from data.
        Data should contain points, lines, or meshes.
        """
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{name}</title>
    <style>body {{ margin: 0; }} canvas {{ display: block; }}</style>
</head>
<body>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // Add basic lighting
        const light = new THREE.AmbientLight(0x404040);
        scene.add(light);

        // Placeholder geometry from data
        const geometry = new THREE.BoxGeometry();
        const material = new THREE.MeshPhongMaterial({{ color: 0x00ff00 }});
        const cube = new THREE.Mesh(geometry, material);
        scene.add(cube);

        camera.position.z = 5;

        function animate() {{
            requestAnimationFrame(animate);
            cube.rotation.x += 0.01;
            cube.rotation.y += 0.01;
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>"""

        file_path = self.output_dir / f"{name}.html"
        file_path.write_text(html_content)

        self.active_visualizations[name] = {
            "type": "threejs",
            "path": str(file_path),
            "data": data,
        }

        logging.info(f"Created Three.js visualization: {name}")
        return str(file_path)

    def update_visualization(self, name: str, new_data: Dict[str, Any]) -> bool:
        if name not in self.active_visualizations:
            logging.warning(f"Visualization {name} not found")
            return False

        # In a full implementation, this would regenerate the HTML/JS
        # with new data and possibly notify a frontend
        self.active_visualizations[name]["data"] = new_data
        logging.info(f"Updated visualization: {name}")
        return True

    def generate_manifold_visualization(self, name: str, points: List[List[float]]) -> str:
        data = {"points": points, "type": "manifold"}
        return self.create_threejs_scene(name, data)

    def list_visualizations(self) -> List[str]:
        return list(self.active_visualizations.keys())

    def get_visualization_path(self, name: str) -> Optional[str]:
        vis = self.active_visualizations.get(name)
        return vis["path"] if vis else None
