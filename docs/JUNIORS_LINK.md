# Juniors trit link

`web3node/trit_link_guard.py` links two AntiCon closes into one block.
Close a is `dev/gaia`. Close b is `home/prompt`. One trit pack each.
The block is made from both close hashes and must pass `valid`.

The guard holds only when both closes report:

- rate 0.05
- released 0
- mint false
- socket false

Pool held. layer_one false.
