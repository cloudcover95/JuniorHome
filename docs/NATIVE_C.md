# Optional C core

Source: JuniorLLM `rails/linux/absmean.c` (same formula as Python absmean).
Live handshake still uses winsor p95 in Python.

```bash
cc -O2 -shared -fPIC rails/linux/absmean.c -lm -o rails/linux/libjunior_absmean.so
```

Gemini dump: NEON header unused, centered-absmean ≠ our wire, ~/Developer path rejected.
