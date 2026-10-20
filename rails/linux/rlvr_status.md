# RLVR status

JuniorLLM ports/rlvr.py appends each agent cycle to:

```
~/.juniorhome/os/rlvr.jsonl
```

Hardware probe writes:

```
~/.juniorhome/os/hardware_probe.json
```

Home rails can surface the latest envelope layer and hardware without running the agent.
Local only. No model pull.
