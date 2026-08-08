# K3 IQ1_S TP3 Benchmark Results - 3x DGX Spark

Date: 2026-08-08
Hardware: 3x NVIDIA DGX Spark (GB10, 121 GiB unified memory each)
Network: 100 Gbps RoCE, jumbo frames (MTU 9000)
Model: Kimi K3 IQ1_S GGUF (313 GB, 9 shards)
Recipe: vcruz305 AllExpertsFfnWidth TP3
Patches: 0001-0012 on SparkInfer base 7a9b77a

## Throughput Results

- Context 2048: decode 6.51 tok/s (clean)
- Context 8192: prefill 1.27 tok/s, decode 7.20 tok/s (clean, ~1GB free)
- Context 12284: OOM during loading
- Context 16384: OOM during loading

## Memory Usage (8K context)

- R0 (10.10.10.13): 110 GB GPU, 114 GB RAM, ~1 GB free
- R1 (10.10.10.12): 103 GB GPU, 107 GB RAM, ~4 GB free
- R2 (10.10.10.14):  98 GB GPU, 102 GB RAM, ~9 GB free

## Key Findings

1. 8K is the maximum context on 3x DGX Spark for K3 IQ1_S TP3
2. 7.20 tok/s decode at 8K context
3. Per-rank MoE budget: ~88.6 GB (512 experts per rank, 896 total)
4. Loading time: ~31 min for all 3 ranks
5. Patches 0010-0012 fixed finish deadlock + added timing
