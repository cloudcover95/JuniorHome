# TritARM ISA

Original 32-bit load/store machine. ARM-inspired register names only.
Not an Arm Holdings product. Not a Nintendo product.

## Registers

r0-r12 GPR. r13=sp. r14=lr. r15=pc. After fetch PC is +4.
Flags N/Z. Cond AL EQ NE LT GE GT LE NV.

Ops: MOV ADD SUB AND ORR EOR LSL LSR LDR STR B BL CMP SVC LDRB STRB.
MMIO base 0x40000000. See pack LEGAL.md.
