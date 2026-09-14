import os
from voice_face import turn
def status():
    return {"plugin": "livekit-plugins-xai", "license": "Apache-2.0",
            "repo": "livekit/agents", "key": bool(os.environ.get("XAI_API_KEY")),
            "imported": False, "local_face": turn("juniorosai field voice")}
