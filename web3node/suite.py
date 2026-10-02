"""Suite status. Training stays at zero without a device."""
from train_step import step
def suite():
    return {"train": step("/no/such/model.gguf", False),
            "ports": ["JuniorLLM", "AGI_SDK", "Gaia"],
            "bind": "127.0.0.1", "model_pull": False}
