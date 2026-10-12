# Juniors — quant context sheet

Local trit-pack hash chain. Not a network. JuniorHome `009e41a0`.

## What it is

A close is a counter of hits, packed once to trits, hashed with the previous hash.
Rate is 5% until a poll releases the pool. The poll does not set the rate.
Signature scheme available: ML-DSA-65, beside the pack, not inside it.
Hash failsafe accepts order with lattice off.

## Flags that stay false

| Flag | Value | Why |
|---|---|---|
| mint | false | no token created |
| socket | false | no listener |
| peers | 0 | no other node |
| network | false | no wire |
| authority | false | recompute proves order |
| firewall | false | AntiCon is a count gate |

## Path

```
opt-in → AntiCon → trit pack → hash failsafe → reward → fund held
```

AntiCon denies: empty actor (bot), n > 32 (burst), duplicate box.
Trit bytes are 0, 1, 2. Anything else is `not_trit`.
Hash is `sha256(prev || pack)[:16]`.

## Reward in the pack

```
body = f"{path_hash}|{rate}|{juniors}"
block = sha256(prev || pack(body))
```

Change the rate or the cut and the tip moves.

## Modules

| File | Job |
|---|---|
| web3node/validate.py | pack, make, valid |
| web3node/anticon.py | fold before pack |
| web3node/rails.py | burst, multi-box, bot |
| web3node/reward.py | 5% of close, fixed |
| web3node/fund.py | poll releases pool, rate stays |
| web3node/failsafe.py | hash recompute, lattice off |
| web3node/path.py | the one path |
| web3node/record_block.py | rate and juniors in the pack |
| web3node/record_chain.py | two packs, second prev is first |
| web3node/handshake.py | name bound to ML-DSA-65 key |
| web3node/offline.py | 90 closes, 1440 bytes, 0 signs |

## Measured on this host

| Piece | Number |
|---|---|
| Trit pack | ~32 µs |
| Path bench | 158 µs |
| Three-close bench | 209 µs |
| ML-DSA-65 sign | ~90 ms, once, not per hit |
| 90 offline closes | 1440 bytes |

## Not this

Not compared to Bitcoin, Ethereum, or Solana.
Not SLH-DSA. Not Falcon. Not a listener.
A right name without the enrolled key fails the handshake.
A right handshake has nowhere to land until something accepts.
