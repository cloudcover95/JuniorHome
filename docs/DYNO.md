# Dyno

Local RPM + load-cell force → torque → power. JuniorHome. Additive only. No flash. No stock ECU.

## Input
- RPM: tach pulse count or direct.
- Force: load-cell N (or kgf, converted).
- Arm: lever m (load-cell dyno) or roller inertia kg·m² (inertia dyno).
- Optional: OBD-style channels from a local file. No live J2534 write.

## Calc
Torque = force × arm.
Power = torque × RPM × 2π / 60.
Correction: optional SAE J1349 factor from local temp/pressure/humidity. Default 1.0.

## Pack
Same as tuner: Winsor → trit → I2_S. Rate 0.05. Close rule sha256(prev || pack || text)[:16].
Logged to ~/.juniorhome/gaia_mesh/dyno.jsonl.

## Modes
- Load-cell: force × arm.
- Inertia: roller I × α. α from timestamped ticks.
- Road: GPS speed delta. No hardware.

## No-flash
Does not write to any ECU. Does not claim factory Accord compatibility.

## Status
Live. Pair with docs/TUNER.md. Update by adding a line.
