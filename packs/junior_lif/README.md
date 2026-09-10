# junior_lif

Ternary LIF pack for the 45 W home node. Same math as FrameForge `lif_ternary`.

Spikes are events for memory / intent. Not a physics clock.
Optional MLX stays in BitNet-mlx. This pack is stdlib.

```
from packs.junior_lif.lif_ternary import demo_layer, encode_features
print(encode_features([1.0]+[0.0]*15, 8, demo_layer()))
```
