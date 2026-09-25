# OSai × IoT

Home already has: Flagstaff vote, I2_S note pack, jsonl ledger, lab spool files.
That is not MQTT.

| Wire | Use in the wild | Home |
|------|-----------------|------|
| File drop | OctoPrint watch folder, GRBL send from disk | **on** (`lab_spool`) |
| HTTP/JSON | Moonraker, IPP | catalog ports only |
| MQTT | HA, sensors | off — broker is a service |
| CoAP | constrained UDP | off |
| Matter | consumer fabric | off |
| Zigbee/Z-Wave | radios | off |
| Modbus | PLCs | off |
| OPC-UA | plants | off |
| ROS 2 DDS | robots | named, not talking |

I2_S hex is a note key, not a bus topic. Do not publish trit on MQTT.
Automations may queue a file; they may not fire a laser.
