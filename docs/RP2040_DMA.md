# RP2040 DMA

12 channels. Base 0x50000000. Stride 0x40. Size 8, 16, or 32 bits.
A channel moves when its DREQ fires. ADC DREQ is 36. PIO0 RX0 is 4. PIO0 TX0 is 0.
Geode: one channel paced by ADC for GPIO 26-29, one paced by PIO for the matrix. Not chained on this host.
This box has no RP2040. dma_present false.
