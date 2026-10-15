# Tuner hardware

Local stubs for load cell, encoder, tach. JuniorHome. Additive only. No flash. No stock ECU.

## Stubs
- load_cell(raw, scale, offset) → N
- encoder(ticks, ppr, dt) → RPM
- tach(pulses, dt) → RPM

## Correction
SAE J1349 factor from local temp/pressure/humidity. Default 1.0. Clamped 0.8–1.2.

## Runs
Reads ~/.juniorhome/gaia_mesh/tuner.jsonl. Compares power_hp between two lines.

## No-flash
Stubs do not open a serial port. They do not write. They do not claim J2534 write.

## Status
Live. Pair with docs/TUNER_MANAGER.md.
