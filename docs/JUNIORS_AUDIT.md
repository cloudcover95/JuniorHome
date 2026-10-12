# Juniors audit

Sheet cites JuniorHome `009e41a`. Tip at this write is `bb0b60a`.
docs/JUNIORS.md still says: not a layer one.

Live path: opt-in, AntiCon, trit pack, hash failsafe, reward, fund held.
mint false. socket false. peers 0. network false.

This VM: 90 pack-and-hash closes in 955 us, 1440 bytes. Sheet's 32 us pack was another host.
Wrong prev does not link.

Still missing: peer gossip, consensus, ML-DSA verify, block production.
pq_ready false. A prototype safeguard does not make this a chain others can join.

```bash
python3 scripts/juniors_proto_prod.py
```
