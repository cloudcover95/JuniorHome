from pathlib import Path
from fuse_kernel import fused
from junior_gamma import quant_j
from sandbox_suite import run as sandbox
from second_brain import write_note
IR = ("absmean_w", "absmax_x", "ternary_w", "dot", "scale_y")
def lower(x, w, gamma="absmean"):
    row = fused(x, w)
    row["gamma_kind"] = gamma
    if gamma == "junior":
        row["gamma_j"] = quant_j(w)[1]
    row["ir"] = list(IR)
    row["backend"] = "fused-list"
    return row
def build():
    return {"sandbox_ok": sandbox().get("ok"), "ir": lower([0.2,-0.1,0.3],[0.4,0.0,-0.2]),
            "compiler": "trit_ir.lower",
            "brain": str(write_note(Path(__file__).resolve().parent/"vault", "# trit ir\n"))}
