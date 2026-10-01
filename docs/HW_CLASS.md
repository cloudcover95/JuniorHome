# Hardware class language

Name the class. Do not name a vendor box as a Junior core.

| Class | Cross-sys | Junior owner |
|-------|-----------|--------------|
| USB sound | ALSA / class 01 | JuniorDeck |
| USB HID | hidraw | JuniorDeck |
| USB-C | SS + D+/D- | JuniorDeck port |
| SBC | arm64, ≤12 W | JuniorOS T4 |
| SFF host | x86_64 or arm64 | JuniorHome T0 |
| tun | `/sys/class/net/tun*` | operator probe only |
| spool | file drop | lab_spool |

A travel router, a spare AP, and a wireless mic are classes, not cores.
Home does not flash router firmware and does not pair a radio.
Bind 127.0.0.1. No 0.0.0.0.
