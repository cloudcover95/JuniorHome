from ternary_kernel import kernel_list
def hook(x, w):
    k = kernel_list(x, w)
    return {"sis": False, "palace": False, "kernel": k.get("path"), "y": (k.get("layer") or {}).get("y")}
