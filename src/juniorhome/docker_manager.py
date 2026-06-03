# path: src/juniorhome/docker_manager.py
#!/usr/bin/env python3
"""
Docker Manager

Basic container orchestration and deployment tooling for JuniorHome.
Provides helpers for building, running, and managing Docker containers.
Useful for production deployment and development workflows.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class DockerManager:
    """
    Simple Docker orchestration helper.
    """

    def __init__(self, project_name: str = "juniorhome"):
        self.project_name = project_name
        logging.info(f"DockerManager initialized for project: {project_name}")

    def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {
                "success": True,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "stdout": e.stdout.strip() if e.stdout else "",
                "stderr": e.stderr.strip() if e.stderr else "",
                "returncode": e.returncode,
            }

    def build_image(self, dockerfile: str = "Dockerfile", tag: Optional[str] = None) -> Dict[str, Any]:
        tag = tag or f"{self.project_name}:latest"
        cmd = ["docker", "build", "-t", tag, "-f", dockerfile, "."]
        logging.info(f"Building Docker image: {tag}")
        return self._run_command(cmd)

    def run_container(
        self,
        image: str,
        name: Optional[str] = None,
        ports: Optional[Dict[str, str]] = None,
        env: Optional[Dict[str, str]] = None,
        detach: bool = True,
    ) -> Dict[str, Any]:
        cmd = ["docker", "run"]

        if detach:
            cmd.append("-d")

        if name:
            cmd.extend(["--name", name])

        if ports:
            for host_port, container_port in ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])

        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])

        cmd.append(image)

        logging.info(f"Running container from image: {image}")
        return self._run_command(cmd)

    def stop_container(self, name: str) -> Dict[str, Any]:
        logging.info(f"Stopping container: {name}")
        return self._run_command(["docker", "stop", name])

    def remove_container(self, name: str, force: bool = False) -> Dict[str, Any]:
        cmd = ["docker", "rm"]
        if force:
            cmd.append("-f")
        cmd.append(name)
        logging.info(f"Removing container: {name}")
        return self._run_command(cmd)

    def list_containers(self, all: bool = False) -> Dict[str, Any]:
        cmd = ["docker", "ps"]
        if all:
            cmd.append("-a")
        return self._run_command(cmd)

    def compose_up(self, compose_file: str = "docker-compose.yml") -> Dict[str, Any]:
        logging.info(f"Starting docker-compose: {compose_file}")
        return self._run_command(["docker-compose", "-f", compose_file, "up", "-d"])

    def compose_down(self, compose_file: str = "docker-compose.yml") -> Dict[str, Any]:
        logging.info(f"Stopping docker-compose: {compose_file}")
        return self._run_command(["docker-compose", "-f", compose_file, "down"])
