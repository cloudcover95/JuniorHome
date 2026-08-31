# JuniorNavMesh

Sovereign offline navigation layer inside JuniorClimbs (2026-08-30).
Implements GMaps / Apple Maps / onX / iOverlander / MP / KAYA *capabilities* without linking those apps.

See `cloudcover95/JuniorClimbs`:
- docs/JUNIOR_NAVMESH.md
- backend/navmesh_engine.py
- backend/routers/navmesh.py
- backend/models_navmesh.py

Key: `/nav/status`, `/nav/iot/nmea`, `/nav/goto/node/{id}`, GPX import/export, land tenure, overland POIs.
`JUNIOR_OFFLINE=1` by default. Vendor links: none.
