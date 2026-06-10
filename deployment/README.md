# Self-Host & Deployment Guide

## Overview
Production-grade deployment for the JuniorHome / BitNet sovereign ecosystem.

## Quick Start (Docker)

```bash
docker compose up -d
```

## Components
- JuniorHome (Swift client + orchestration)
- BitNet-Intel (agentic layer)
- MCP Server
- JuniorDrive (robotics/sim)

## Hardware Support
- Apple Silicon (MLX native)
- Jetson Orin
- Raspberry Pi 5
- Snapdragon

## Next
- k3s / Kubernetes manifests
- Monitoring (Prometheus + Grafana)
- Automated CI/CD