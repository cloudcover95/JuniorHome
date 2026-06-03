# path: src/juniorhome/architecture.py
#!/usr/bin/env python3
"""
Architecture Overview

Defines the high-level layered architecture of JuniorHome.
This serves as both documentation and a guide for future development.
"""

ARCHITECTURE_LAYERS = {
    "Core Infrastructure": [
        "ConfigManager",
        "ProductionSetup",
        "Application (bootstrap hub)",
    ],
    "Observability & Monitoring": [
        "HealthCheck",
        "MetricsCollector",
        "TracingContext",
        "EventBus",
        "ObservabilityManager",
    ],
    "Security & Safety": [
        "SecretsManager",
        "LLMSecurityGuard",
        "SandboxExecutor",
        "PolicyEngine",
        "ActionAuditor",
        "SecurityMiddleware",
    ],
    "Data & Storage": [
        "DataLakeManager",
        "DataLakeIntegration",
        "RateLimitedFetcher",
    ],
    "Deployment & Operations": [
        "DockerManager",
        "TaskQueue",
        "Scheduler",
    ],
    "Intelligence & Agents": [
        "SmartLLMRouter",
        "AutonomousAgent",
        "WorkflowEngine",
        "QuantizedModelManager",
        "TDAReasoner (via JuniorLLM)",
    ],
    "Knowledge Processing": [
        "ResilientKnowledgePipeline",
        "KnowledgeService",
        "ObsidianVaultProcessor",
    ],
    "Presentation & Integration": [
        "Dashboard / EnhancedDashboard",
        "APIServer",
        "WebSocketServer",
        "CLI",
    ],
}


def print_architecture():
    print("\n=== JuniorHome Architecture Layers ===\n")
    for layer, components in ARCHITECTURE_LAYERS.items():
        print(f"{layer}:")
        for comp in components:
            print(f"  - {comp}")
        print()


if __name__ == "__main__":
    print_architecture()
