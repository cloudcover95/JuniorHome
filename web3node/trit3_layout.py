from trit3 import read, write
def details():
    raw = write([[0.2, -0.4, 0.1], [0.0, 0.3, -0.2]])
    return {"layout": "tag,rows,gamma,cols,trit5", "header": 20, "bytes": len(raw), **read(raw)}
