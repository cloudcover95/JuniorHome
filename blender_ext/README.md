# JuniorCloud Omega (Blender extension)

Blender 4.2+ extension. Imports `~/.juniorhome/omega/patch.obj` written by `ports.blender_omega.harness`.
Does not launch UE5. Does not download. bpy stays inside Blender.

Install: Edit → Preferences → Extensions → Install from Disk → this folder
(or zip `juniorcloud_omega/`).

Generate the OBJ from Home first:

```bash
python scripts/blender_omega_prod.py
```
