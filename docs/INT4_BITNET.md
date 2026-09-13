# INT4 vs 1.58

BitNet train dynamics we implement: refresh γ from a batch. No STE tape.
INT4 store: signed nibble pack. Not GGUF Q4_K.
On n=64 this box: float32 256 B, INT4 36 B, trit 20 B.
