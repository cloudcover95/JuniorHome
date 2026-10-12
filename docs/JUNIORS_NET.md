# Juniors network envelope

Kinds: hello, block, tip. Bind 127.0.0.1. listen false. peers 0.
This is the message shape a later node would send. It does not open a socket.
layer_one stays false until gossip, consensus, and ML-DSA verify exist.

```bash
python3 scripts/juniors_net_prod.py
```
