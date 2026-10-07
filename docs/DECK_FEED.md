# Channel map feed

```bash
python3 scripts/deck_feed_prod.py
python3 scripts/deck_engine_prod.py 8
```

Feed writes ~/.juniorhome/os/deck_feed.json. Engine reads that path's rows via the same call.
source is name-vector. adc false until a node is present.
