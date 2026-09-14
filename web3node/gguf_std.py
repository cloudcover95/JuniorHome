VALUE_TYPES = {8: "string", 12: "float64"}
OURS = ("general.architecture", "general.quantization_version", "juniorosai.gamma")
def report():
    return {"ours": OURS, "avoid": ("llama.block_count", "tokenizer.ggml.tokens"), "triton": "trit-on"}
