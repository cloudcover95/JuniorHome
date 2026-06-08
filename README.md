# JuniorHome

**Typed Deliverables + Coordination Layer Added**

GraphMemoryBlackbox now supports:
- `post_deliverable()` with provenance (produced_by, deliverable_type, version)
- `get_deliverables()` and `get_latest_deliverable()` for clean handoff between components
- Light coordination via `notify_deliverable_ready()`

VLMDesignAgent can now run with `parallel_paths` and posts typed deliverables to GraphMemory.

This enables clean, efficient agent-style workflows entirely inside the BitNet layer
without heavy external frameworks.

Example flow:
VLMDesignAgent posts design deliverable → CADScriptGenerator consumes it and posts artifacts.