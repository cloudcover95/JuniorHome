from agent_stack import run as agents
from stack_bench import main as stack
from fieldcore_bridge import pick_juniorllm_port
FUSE = ("mean|W|", "clip-round", "absmax-x", "dot-scale", "trit5")
def weekly():
    st = stack()
    return {"fuse_steps": list(FUSE), "sandbox_ok": st.get("sandbox_ok"),
            "fuse_us": st.get("fuse64", {}).get("us"), "agents": agents("juniorosai weekly fuse").get("ok"),
            "port": pick_juniorllm_port("juniorosai weekly"), "trt": False}
