SCHEMA = {
    "general.architecture": {"type": "string", "const": "juniorosai"},
    "general.quantization_version": {"type": "string", "const": "i2s-trit-jtr1"},
    "juniorosai.gamma": {"type": "float64"},
    "tensors": [{"name": "blk.0.weight", "layout": "2bit-trit"}],
}
def validate(kv):
    return kv.get("general.architecture") == "juniorosai"
def dump():
    return {"gguf_v": 3, "kv": SCHEMA, "llama_required": False}
