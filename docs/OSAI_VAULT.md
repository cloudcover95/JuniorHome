# OSai vault

Findings live in JuniorLLM at junior_osai/vault/findings/. Local only. Not synced.

## Current
- 2026-10-15_1930_ternary-matvec-baseline.md (close 368f9370a796cfab)
  Memory: 16x smaller than fp32. Viable.
  Speed: numpy ternary 15-465x slower. Not viable without native kernel.
  Error: ~0.5 on post-hoc ternarised float weights.
- 2026-10-15_2020_lut-kernel-next.md (close 7a1c2e4b9d0f3a6c)
  Next: LUT kernel on synthetic true ternary weights. No GGUF.

## Next build
i2sd userspace kernel. LUT or bit-plane on already-ternary weights. Target: beat fp32 at <=1% error, or document that it does not.

## Status
Live. Additive. Pair with JuniorLLM docs/ROADMAP_I2SD.md.
