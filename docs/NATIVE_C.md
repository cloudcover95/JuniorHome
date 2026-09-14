# Optional C core

`rails/linux/absmean.c` — mean(|W|)
`rails/linux/winsor.c` — p95 clip then mean, same as Python wire
`rails/linux/i2s_pack.c` — two bits per trit

Handshake still Python winsor. Compile optional:

```bash
cc -O2 -shared -fPIC rails/linux/winsor.c -lm -o rails/linux/libjunior_winsor.so
```

Not taken from Gemini: Metal C++, 32 µs SLA, FrameForge physics port, ESP-IDF, tau passed in instead of computed, unused NEON.
