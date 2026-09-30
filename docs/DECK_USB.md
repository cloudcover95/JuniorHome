# JuniorDeck USB-C digitizer layer

Physical: USB-C receptacle on the chassis (SS pairs + D+/D- + CC).
Software: `web3node/deck_usb.py` probes `/dev/snd/*`, `/dev/hidraw*`, `/dev/bus/usb`.

Mount order:
1. analog_ok(CV, Z)
2. Flagstaff 6 AND
3. class device present
Then `digitizer` may become true. `live` stays false until Ardour/Hydrogen exist.

No pyusb required. No gadget configfs write from this script.
Pi gadget (`/sys/kernel/config/usb_gadget`) is detected only.
I2_S hex is a note. Not a USB serial number crypto key.
