# Kimi K3 IQ1_S TP3 on 3x DGX Spark

Distributed tensor-parallel inference for Kimi K3 (IQ1_S quantization) across 3 NVIDIA DGX Spark nodes using the AllExpertsFfnWidth sharding strategy.

## Results

- **7.20 tok/s decode** at 8K context (max safe context on Spark)
- 3-node TP3 with NCCL, ~110 GB GPU per rank
- All 12 patches from vcruz305 recipe applied to SparkInfer base

See [results/benchmark.md](results/benchmark.md) for full details.

## Contents

- `patches/` - 12 git patches (0001-0012) for SparkInfer base commit 7a9b77a
- `scripts/kimi_k3_dist_generate.cpp` - Modified binary with multi-prompt support (--prompts-file)
- `scripts/run_humaneval.sh` - HumanEval benchmark orchestrator
- `results/benchmark.md` - Throughput and memory measurements

## Quick Start

```bash
git clone https://github.com/nousresearch/sparkinfer.git
cd sparkinfer && git checkout 7a9b77a
git am /path/to/patches/00*.patch
mkdir build && cd build
cmake .. -DSPARKINFER_TP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target kimi_k3_dist_generate -j$(nproc)
```

## Hardware Requirements

- 3x DGX Spark (121 GiB unified memory each)
- 100 Gbps RoCE network with jumbo frames
- Model: K3 IQ1_S GGUF (313 GB total, local NVMe on each node)
