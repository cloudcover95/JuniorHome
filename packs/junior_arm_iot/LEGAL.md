# LEGAL — junior_arm_iot

JuniorCloud LLC. MIT.

This pack is **not a Nintendo product**. It is not Yuzu, Ryujinx, Dolphin,
mGBA, PCSX2, or any vendor console core. It does not ship BIOS, keys,
title keys, firmware, or game images.

## What ships

- TritARM: an original 16-register load/store ISA. ARM-inspired only in
  the public RISC sense (PC in r15, LR in r14, SP in r13). Encoding is
  original. Not an Arm Holdings product.
- Guest *envelopes* (RAM size, watt hint, MMIO map). Names like
  `arm7_class` / `a57_class` are thermal and memory budgets, not copies
  of those pipelines.
- Original demo images assembled from text in `images/`.

## What must never land in this tree

- Console ROMs, NAND dumps, prod.keys, titlekeys.
- Vendor boot ROM disassembly.
- Decrypt, anti-DRM, or "how to run X game" paths.
- Nintendo roster, stages, or tells. FrameForge legal floor stays
  Vesper / Quill / Relay / Forge.

## Guest images

TritARM `.tas` / raw payloads **execute**.

`.xci .nsp .nca .cia .3ds .nds .gba .z64` are **classified**. Public
header bytes may be read. Encrypted containers are not decrypted. No
prod.keys, no titlekeys, no NCA body walk. Classification feeds FieldCore
and the MemSys SIS palace. That is processing, not a vendor CPU.

Do not commit copyrighted images to this tree.

## Games hook

BitNet / FieldCore scores **CPU intent only**. Python / canvas sim owns
knockback and stocks. UnrealEditor does not boot on the 45 W home node.
