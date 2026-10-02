# RP2040 mux

Two muxes.

FUNCSEL is per pad. Reset 31 NULL. Classes: SPI, UART, I2C, PWM, SIO, PIO0, PIO1. SIO is the matrix. PIO is the scan engine.

AINSEL is the ADC mux. One channel at a time. 0-3 are GPIO 26-29. 4 is the temperature sensor. 500 ksps shared, 2 us.

Geode: GPIO 0-23 are SIO or PIO, not ADC. GPIO 26-29 are AINSEL only, FUNCSEL NULL while sampling. GPIO 24-25 free. Display is DSI. Not flashed.
