# junior_trit_stack

Compose pack on the 45 W home node.

- junior_arm_iot — TritARM ISA + IoT envelopes (already in this tree)
- junior_lif — ternary LIF
- FrameForge trit_stack — LIF → FFBN → TritARM MMIO → JuniorLLM envelope

BitNet / TritARM score CPU intent only. Python sim owns knockback.
Not a Nintendo product. Not an Arm Holdings product. Not a vendor console core.

```
python3 -c "from packs.junior_trit_stack.compose import eval_features; print(eval_features([0.6,-0.2,0,0,1]))"
```

If FrameForge is not on PYTHONPATH, compose falls back to pad-sign trit.
