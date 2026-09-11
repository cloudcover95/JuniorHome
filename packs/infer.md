# Local infer

Order: llama.cpp/bitnet.cpp if `JUNIOR_LLAMA`+`JUNIOR_GGUF`, else i2sd on 127.0.0.1:8767.
Fused AbsMean-BitLinear: JuniorLLM `junior_bitnet.fusion` (CPU contract; Triton only if importable).
Flagstaff context: `scripts/junioros_prod.py ./vault`
