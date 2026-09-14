# Gaia wire — schema × 1.58

Live envelope is JuniorLLM `system()` → `handshake()`, not Gemini `ProtocolEnvelope`.

```
{
  "system": "JuniorGaia",
  "who": {"name", "pronouns"},          // ~/.juniorhome/gaia.json
  "view": {"orient", "px", "scale"},
  "note": {
    "area", "port",
    "gamma": AbsMean scale,
    "trit": [-1|0|1, ...],             // winsor pack of the note
    "i2s_hex": two bits per trit
  },
  "identity": {"issuer": "local-gaia.json", "verified": <named+loopback>},
  "ue5_launch": false,
  "download": false
}
```

Math (already in junior_bitnet.winsor, not a second packer in proto):

    γ = mean(|x|) + ε
    t  = clip(round(x / γ), -1, 1)
    I2_S packs t as {0,1,2} two bits each.

Gemini drop reused handshake(who=str), verified:true, tomli, numpy-in-proto,
unreal.* inside a file named blender, 01_Legal, git init under ~/Developer.
None of that is on main. `note.trit` is now actually on the envelope.
