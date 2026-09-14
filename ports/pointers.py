"""Staged pointers. runtime_active is false until Home imports JuniorLLM handshake."""
POINTERS = {
    "coach": "cloudcover95/JuniorCoach",
    "pithon": "cloudcover95/JuniorPiThon",
    "sol": "cloudcover95/JuniorSOL",
    "solana": "cloudcover95/JuniorSolana",
    "agi_sdk": "cloudcover95/AGI_SDK",
    "bitnet_mlx": "cloudcover95/BitNet-mlx",
    "web3node": "cloudcover95/web3node",
    "frameforge": "cloudcover95/FrameForge",
    "crispy": "cloudcover95/crispy-mouse",
}


def list_pointers() -> dict:
    return {
        "protocol": "goldend-osai-omega/1",
        "runtime_active": False,
        "handshake": "JuniorLLM.ports.gaia_proto.handshake",
        "items": [{"name": k, "repo": v} for k, v in POINTERS.items()],
    }
