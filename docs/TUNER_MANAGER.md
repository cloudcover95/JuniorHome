# Tuner manager

Ties channels, dyno, road, ticks, dash, autotune. JuniorHome. Additive only. No flash. No stock ECU.

## Modes
- load: force × arm → torque → power
- ticks: encoder timestamps → RPM + alpha
- road: GPS speed delta → power

## Files
- web3node/tuner/ticks.py
- web3node/tuner/road.py
- web3node/tuner/dash.py
- web3node/tuner/j2534.py (read-only stub)
- web3node/tuner/manager.py
- scripts/tuner_manager_prod.py

## No-flash
J2534 path is read-only. No write. No KTuner. No HondaReflash.

## Status
Live. Pair with docs/TUNER.md and docs/DYNO.md.
