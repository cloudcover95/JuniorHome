# ports/ index

One line per module in `ports/`, read from the source at tip f70a59f.
Home binds 127.0.0.1 only. Nothing here opens a socket. Nothing fires.

| Module | What it does | Bind / fire |
|---|---|---|
| `blender_omega.py` | Omega terrain to OBJ (`JOB = "terrain-obj"`). `bpy` optional. UE5 off. | no socket |
| `lab_catalog.py` | `CATALOG` of OSS device classes (fdm, laser, paper, sbc, robot, calc). | data only |
| `lab_floor.py` | Optional shop floor. Writes `~/.juniorhome/lab/floor.jsonl` and `companion.html`. | `bind: 127.0.0.1` |
| `lab_iot.py` | `map()`: osai `goldend-osai-omega/1`, spool on; mqtt, matter, fire off. | `fire: False` |
| `lab_proto.py` | `PROTO`: only `file-spool` is on. `status()` reports bind and fire. | `bind: 127.0.0.1`, `fire: False` |
| `lab_spool.py` | `queue()` drops a job file under `~/.juniorhome/lab/spool`, checked against `CATALOG`. | `bind: 127.0.0.1`, fire off |
| `pointers.py` | Staged pointers to sibling repos. `runtime_active` false until the JuniorLLM handshake. | no socket |

## Device ports named in `lab_catalog.py`

These are upstream defaults for the device software, listed for reference.
Home does not listen on them and does not post to them (`lab_proto.py`: `ipp-cups` and `moonraker-post` are False).

| id | port |
|---|---|
| klipper-moonraker | 7125 |
| octoprint | 5000 |
| cups-ipp | 631 |

Every other catalog entry has `port: None`.

## Imports to note

`blender_omega.py` imports `ports.gaia_proto` and `ports.trit_gates`.
Those modules are not in this repo's `ports/`. `pointers.py` names the handshake as `JuniorLLM.ports.gaia_proto.handshake`, so they come from JuniorLLM.

See also: `docs/ECOSYSTEM_PORTS.md`, `docs/LLM_PORTS.md`, `docs/LAB_FLOOR.md`, `docs/LANES.md`.
