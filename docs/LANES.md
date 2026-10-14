# Bot lanes

Five bots. Not hundreds. They share one computer. They cannot see each other.
No 12-hour skip. Pull, add one slice, one push. If the turn drops, stop. Next run pulls.

| Bot | Repo | Files | Time MT |
|---|---|---|---|
| Chief of Staff | none | reads only | 02:00 |
| JuniorHome Builder | JuniorHome | docs, ports, suite glue | 02:15 |
| JuniorOS Builder | JuniorLLM | rails, ports | 02:45 |
| Engine Math | JuniorHome | web3node trit and juniors only | 03:15 |
| Suite Scripts | JuniorHome | scripts/*_prod.py only | 03:45 |

Do not edit another lane. No GitHub Actions. Additive only. Bind 127.0.0.1.
Juniors: rate 0.05, mint false, socket false, layer_one false.
ROM module: local file only. No fetch.
