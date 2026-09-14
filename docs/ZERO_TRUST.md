# Zero-trust review 2026-09-13

## Pass
- Bind 127.0.0.1 only on hook/UI/i2sd
- No docker.sock, no PAT, no device-login in juniorctl T12 tests
- No force-push, no deletes this session
- xAI / credit board: call=false until a key is opted in
- GGUF not downloaded by bots (>8GB rule)
- FieldCore does not scrape Mountain Project / KAYA
- FrameForge2D tick: no fetch
- os_route.launch is false (UE5 does not boot from Home)

## Fail / do not merge if seen again
- FastAPI host 0.0.0.0:8080
- docker-compose network_mode: host
- git config user.email rewrite in deploy scripts
- 4B / 1.5TB model pulls
- Parquet+pyarrow as a required path for a Home tick

## Legacy files (keep, do not delete)
web3node/community_saas.py and saas_*.py exist. Treat as unused. Live path is trit_tick + os_route + JuniorLLM ports.

## Identity
GitHub connector = cloudcover95. Automations share that login. Not a second bot user.
