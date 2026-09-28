import math, struct
from pathlib import Path
from junior_gamma import gamma_j
def table(n=64):
    return [math.sin(2*math.pi*i/n)+0.3*math.sin(4*math.pi*i/n) for i in range(n)]
def render(sr=8000, sec=0.2):
    tab, g, n = table(), None, int(sr*sec)
    g = gamma_j(tab); pcm=bytearray(); phase=0.0; step=64*220/sr
    for _ in range(n):
        s = tab[int(phase)%64]*min(g,1.0)*0.4
        pcm += struct.pack("<h", int(max(-1,min(1,s))*16000)); phase += step
    dest = Path(__file__).resolve().parent/"vault"/"deck_wt.wav"
    dest.parent.mkdir(exist_ok=True)
    dest.write_bytes(b"RIFF"+struct.pack("<I",36+len(pcm))+b"WAVEfmt "+struct.pack("<IHHIIHH",16,1,1,sr,sr*2,2,16)+b"data"+struct.pack("<I",len(pcm))+pcm)
    return {"n":64,"gamma_j":g,"wav_B":dest.stat().st_size,"engine":"lookup"}
