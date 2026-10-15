# Tuner

Local channel pack, table, and autotune. JuniorHome. Additive only. No flash. No stock ECU.

## Channels
RPM, AFR, TPS, MAP, ECT, IAT, VSS, gear, analog.
Each sample: Winsor-P95 → trit {-1,0,1} → I2_S.
Rate 0.05. Close rule sha256(prev || pack || text)[:16].
Logged to ~/.juniorhome/gaia_mesh/tuner.jsonl.

## Table
VE / ignition / AFR tables as local JSON. Cells are trit.
Scale, smooth, interpolate stay in-process.
Schema: {name, axes: {x, y}, cells: [[trit]], units}.

## Autotune
AFR error → recommended cell delta.
Authority limit default 0.05.
Transient filter: ignore |dAFR| > 0.2 in one sample.
Does not write the table until the user confirms.

## No-flash
This module does not write to any ECU.
It does not speak J2534 write, KTuner, or HondaReflash.
Input is a local sample or a file. Output is a mesh line.

## Status
Live. Does not override docs/BOT_FIRST.md or docs/DUAL_AUTONOMY.md.
