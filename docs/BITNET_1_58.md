# 1.58-bit

log2(3)≈1.585. We store 2 bits/trit (≈26% pad). Five trits per byte is denser; 2-bit is simpler.
Hot path: AbsMean W once, AbsMax X, integer dot, × Δw/127. Skip pack until the ticket leaves the node.
`python3 web3node/bitnet_opt.py`
