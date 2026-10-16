# Layer stack

Picks the kernel by size. No bench. JuniorHome pointer.

## Layers
- T0: n < 512*512. Python trit.
- T1: n < 1024*1024. Bit-plane numpy.
- T2: n >= 1024*1024. Native C if present, else T1.

## Where
JuniorLLM junior_osai/kernels/stack.py.

## Status
Live. Additive. Pair with docs/OSAI_VAULT.md.
