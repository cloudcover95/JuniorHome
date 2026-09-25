"""Which IoT wires Home will not speak yet."""

PROTO = {
    "file-spool": True,
    "loopback-http-note": False,
    "mqtt": False,
    "coap": False,
    "matter": False,
    "zigbee": False,
    "modbus": False,
    "opcua": False,
    "ros2-dds": False,
    "ipp-cups": False,
    "moonraker-post": False,
}


def status() -> dict:
    return {"bind": "127.0.0.1", "fire": False, "proto": PROTO, "trit_is_not_topic": True}
