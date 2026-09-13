# Compute layers

Solar is free watts for T4. Spark is powerful and still a T1 latch.

| Layer | Node | Train weights | Solar |
|---|---|---|---|
| infer_edge | T4_ship 12W | no | yes |
| infer_home | T0_m6 | no | no |
| train_data | T4_ship | no — collect tickets | yes |
| train_dev | T1_spark | yes — fine-tune only | no |

T4 collects. T0 labels. T1 may fine-tune. No 70B pretrain.
`python3 web3node/compute_layers.py`
