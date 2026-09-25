# OSai vs shop IoT

OSai wire today: `goldend-osai-omega/1` notes + Flagstaff + jsonl goldens.
Jobs: dash-viewport, gaia-spine, terrain-obj, agi-capsule.
download false, ue5_launch false, fire false.

That is **not** Matter/Zigbee/Thread/MQTT. Those are device fabrics.

| Fabric | Role | In Home |
|--------|------|---------|
| Matter / Thread | IP smart-home | operator later |
| Zigbee | 2.4 GHz mesh | dongle later |
| MQTT | pub/sub | not a broker here |
| IPP/CUPS | paper | spool `.pdf` |
| Moonraker HTTP | Klipper | spool `.gcode` |
| GRBL serial | laser/CNC | spool `.nc` |
| ROS 2 | robots | loopback name only |
| OSai handshake | note + vote | live |

Bridge: golden note names a device id; spool writes a file; firmware does motion.
