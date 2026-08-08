# Kimi K3 IQ1_S TP3 Distributed Inference

Distributed tensor-parallel inference of Kimi K3 (IQ1_S quantization, ~313 GB) across 3×DGX Spark nodes using AllExpertsFfnWidth sharding.

## Results

| Context | Decode tok/s | Status |
|---------|-------------|--------|
| 2K | 6.51 | ✅ |
| 8K | 7.20 | ✅ (max safe context) |

See [REPORT.md](REPORT.md) for full details.

## Contents

-  — 12 patches for SparkInfer (vcruz305 AllExpertsFfnWidth TP3 recipe)
-  — Modified generate with multi-prompt + KV reset + logits debug
-  — Forward pass with logit dump instrumentation
-  — Protocol with reset sentinel (-2) support
-  — Full benchmark report

## Build

See REPORT.md for build instructions.
