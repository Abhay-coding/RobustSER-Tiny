# Software-only completion scope

This project will not use physical hardware. It can report reproducible
desktop-side model size, parameter count, conversion artifacts, and desktop
latency. It cannot claim measured ESP32 flash/RAM use, energy consumption,
microphone behavior, DSP latency, or end-to-end device latency.

Remaining software experiments are cross-corpus RAVDESS evaluation, calibrated
confidence/rejection, and final report generation. RAVDESS is external-only:
it must never be merged into CREMA-D model training for the primary benchmark.
