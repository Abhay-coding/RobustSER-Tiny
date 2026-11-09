# Software-only export note

## Scope decision

This project deliberately has **no physical-hardware phase**. It does not
require an ESP32 board, microphone module, firmware, cloud service, or device
profiling. Consequently, this repository makes no RAM, flash, energy, or
end-to-end latency claim for any embedded device.

## What is retained

The file `results/deployment/e4_esp32s3.onnx` is retained only as a software
interchange check. It demonstrates that the compact E4 model can be exported
to a fixed-shape ONNX graph; it is not a microcontroller binary and must not
be described as an embedded deployment result.

## Export verification — 2026-11-09

`scripts/export_onnx.py` exported the canonical E4 checkpoint to
`results/deployment/e4_esp32s3.onnx`. ONNX validation passed. The graph is
6,581 bytes with fixed input `[1, 1, 32, 31]`, fixed output `[1, 6]`, opset 17,
and operators `Conv`, `Relu`, `MaxPool`, `GlobalAveragePool`, `Flatten`, and
`Gemm`. This is a compatibility checkpoint only and cannot support any
embedded memory or latency claim.

The college demonstration is the local Streamlit application in `demo/app.py`.
See [the demo guide](demo_guide.md) for the supported presentation workflow.
