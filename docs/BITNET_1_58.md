# 1.58-bit

log2(3)≈1.585. Store 2 bits/trit (pack2) or 5 trits/byte (pack5).
Hot path: winsor, AbsMean W once, AbsMax X, integer dot, × gamma/127.
C packer: JuniorLLM rails/linux/absmean.c. Stdlib twin: web3node/tritquant.py.
`python3 scripts/tritquant_prod.py`
