# JuniorStoneField (JuniorClimbs)

Community outdoor boulder layer added 2026-08-30. Additive — does not replace gym POS / coaching.

Lives in `cloudcover95/JuniorClimbs`:
- `docs/JUNIOR_STONEFIELD.md`
- `backend/routers/stonefield.py`
- `backend/models_stonefield.py`
- `backend/schemas/stonefield.py`
- `backend/seed_red_feather.py`
- `alembic/versions/002_junior_stonefield.py`

Capabilities mirrored from public guidebook *patterns* (GPS tree, user submits, photos, discussion) for Red Feather Lakes and any future field.

Endpoints: `/stonefield/*` and `/stonefield/red-feather`.

Ties later into SpatialTernaryAutomata + Enhanced TDA + BitNet IoT ascent logs.
