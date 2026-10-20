# JuniorOS agent rail

JuniorLLM ports/junioros_agent.py writes:

```
~/.juniorhome/os/junioros_agent.json
```

Home workflows read it via scripts/home_rlvr.py.
End user: python scripts/rlvr_cli.py --show

Local only. No model pull.
