# JuniorCloud repo map

Home holds titles and direction. Engines stay in their trees. Do not vendor.

Layout: `~/JuniorCloud/<name>` siblings.

```
python catalog/workspace.py            # print PYTHONPATH + present/missing
python catalog/workspace.py --ensure   # clone *missing only*
python catalog/workspace.py --pull     # git pull existing
```

Prove / vault (no extra clone if siblings exist):

```
export PYTHONPATH=$HOME/JuniorCloud/JuniorLLM:$HOME/JuniorCloud/JuniorEngrTools:$HOME/JuniorCloud/JuniorOmega:$HOME/JuniorCloud/JuniorMemSys-Suite
python $HOME/JuniorCloud/JuniorLLM/scripts/home_sync.py $HOME/JuniorCloud/JuniorHome/vault
python $HOME/JuniorCloud/JuniorLLM/scripts/layer_prod.py
```

| name | role |
|------|------|
| JuniorHome | index |
| JuniorLLM | ternary engine, Teqp, ports |
| BitNet-mlx | MLX 1.58 kernels |
| JuniorMemSys-Suite | palace / SIS optional |
| JuniorEngrTools | desk + vault writer |
| JuniorOmega | CAD after compile |
| JuniorStock / JuniorQuant / stocksnode | book |
| AGI_SDK / JuniorAGI_SDK | agents |
| JuniorClimbs | gym / StoneField |
| crispy-mouse | HMI |
| web3node / JuniorSOL / JuniorSolana | chain |
