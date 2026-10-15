# Tuner hardware profiles

Community stacks mapped to local channels. JuniorHome. Additive only. No flash. No stock ECU.

## Profiles
- speeduino: Speeduino / LibreTune INI. RPM, AFR, TPS, MAP, CLT, IAT, VSS.
- opendyno: OpenDyno GPS road. speed_mps, rpm.
- diydyno: DIY-Dyno / DeeEmm. tach, load_cell.
- hondash: HonDash K-Pro / S300. RPM, AFR, TPS, MAP, ECT, IAT, VSS.

## Use
map_sample(profile, raw) → local channel dict. Pass to channels.pack_sample.

## No-flash
Profiles do not open a serial port. They do not write. They do not claim factory ECU compatibility.

## Status
Live. Pair with docs/TUNER_HARDWARE.md.
