"""ADC 26 is RP2040 ADC0. This host has no pin."""
import time
BITS = 12
FULL = (1 << BITS) - 1
def gamma(code):
    code = 0 if code < 0 else FULL if code > FULL else code
    return round(code / FULL, 4)
def bench():
    t0 = time.perf_counter()
    rows = [gamma(c) for c in (0, 1024, 2048, 3072, 4095)]
    return {"pin": 26, "ainsel": 0, "bits": BITS, "ksps": 500, "us_per_sample": 2,
            "rows": rows, "us": round((time.perf_counter() - t0) * 1e6, 1),
            "adc_present": False, "flashed": False, "model_pull": False}
