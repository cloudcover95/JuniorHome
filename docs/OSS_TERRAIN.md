# Open topography without USGS lock-in

Public options the Gemini note named:
- Open-Elevation — self-host SRTM JSON. Operator machine only. Not a Home docker service.
- Nextzen / AWS Open Data terrain tiles — download once, drop JSON grid at `~/.juniorhome/tiles/flagstaff.json` as `{"grid": [[z...], ...]}`.

Home / Gaia never set `usgs_fetch`. `ports/terrain_spine.fetch_oss_flagstaff_mesh` reads that file or synthesizes a 2100 m Flagstaff prior.

Do not touch empty `llama.plan` files and call the node ready. `llama_ready` is GGUF + binary on disk.

```bash
python scripts/gaia_stack_prod.py gaia they terrain
# writes ~/.juniorhome/gaia_mesh/flagstaff_48x48.obj + gaia_ue5.json
```
