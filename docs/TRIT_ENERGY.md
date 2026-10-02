# Trit mesh energy

energy = 1 - zeros/n on the AbsMean row from trit_mesh.py.
Not joules. measured false. No 1024 SVD.

| band | bound |
|------|--------|
| fail | energy < 0.40 |
| pass | 0.40 to 0.85 |
| dense | energy > 0.85 |

python3 scripts/trit_energy_prod.py
Hourly automation reads ~/.juniorhome/os/trit_energy.json. ok is false only on fail.
