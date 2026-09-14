from iot_runtime import pick
PROVIDERS = {
    "CPUExecutionProvider": "OSS T0 maybe",
    "CUDAExecutionProvider": "T1",
    "AzureExecutionProvider": "CRUTCH",
}
TRITON = {"license": "BSD-3", "junioros": False, "decentralized": False}
def report():
    return {"osai": "fused-list", "ort_on_t4": False, "providers": PROVIDERS,
            "triton": TRITON, "iot": pick()}
