# Rails — JuniorOS overlay

bitnet.cpp runs **if** `JUNIOR_BITNET_CPP` and `JUNIOR_GGUF` exist. Else placeholder on 127.0.0.1:8765.
Asahi: stock kernel + MLX userspace. No AGX firmware flash.
Home catalog is the map; engines stay siblings under ~/JuniorCloud.

```
PYTHONPATH=../JuniorLLM python ../JuniorLLM/scripts/os_probe.py
```
