# Eight-point agent post vs Home

1. Linux — host. JuniorOS is overlay + loopback, not an ISO.
2. Tailscale — out. Home binds 127.0.0.1. No overlay mesh from automation.
3. llama.cpp — T3 if GGUF on disk. Not Ollama day-one, no pull.
4. Hermes/Telegram — out. Agent is Flagstaff + handshake + jsonl.
5. tmux — operator. Not a repo service.
6. Termius/mosh — operator. Phone is Swift rail spec only.
7. Git — cloudcover95 remotes + local commits. No self-hosted Gitea from Home.
8. Scripts — `scripts/*_prod.py` is the habit we kept.

Decorative here: rented GPU nodes, tensor-parallel over a cable, public IP.
