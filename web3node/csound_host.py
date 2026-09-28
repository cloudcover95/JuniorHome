import shutil
from pathlib import Path
from junior_gamma import gamma_j
def host():
    g = gamma_j([0.2,0.8,0.1,0.9])
    dest = Path(__file__).resolve().parent/"vault"/"deck.csd"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(f"<CsoundSynthesizer>\n<CsOptions>-odac</CsOptions>\n<CsInstruments>\nsr=48000\ninstr 1\na1 oscili {max(0.01,min(g,1.0))}, 220\nouts a1,a1\nendin\n</CsInstruments>\n<CsScore>\ni1 0 0.5\ne\n</CsScore>\n</CsoundSynthesizer>\n", encoding="utf-8")
    return {"csound": shutil.which("csound"), "csd": str(dest), "gamma_j": g, "ran": False}
