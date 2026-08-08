# Kimi K3 IQ1_S Distributed TP3 Inference on 3×DGX Spark

## Hardware

| Component | Spec |
|-----------|------|
| Nodes | 3× NVIDIA DGX Spark (GB10 Superchip) |
| Memory per node | 121 GiB unified (CPU+GPU) |
| GPU per node | NVIDIA GB10 (Blackwell, 121 GiB unified) |
| Interconnect | 100 Gbps RoCE v2, MTU 9000 (jumbo frames) |
| NVMe | Local SSD per node |

## Model

| Property | Value |
|----------|-------|
| Model | Kimi K3 (MoonshotAI) |
| Quantization | IQ1_S (1.5 bpw) |
| Parameters | ~1T (MoE) |
| GGUF size | ~313 GB (9 shards × ~37 GB) |
| Architecture | 93 layers, 163840 vocab, 896 MoE experts, 7168 hidden |
| Max context (arch) | 1,048,576 tokens |

## Sharding Strategy

**Recipe**: vcruz305 AllExpertsFfnWidth (TP3)

| Rank | Node | MoE Expert Range | FFN Width | MoE Budget | Attention |
|------|------|-----------------|-----------|------------|-----------|
| 0 (coordinator) | 10.10.10.13 | [0, 512) | 512 | ~88.6 GB | ~11.5 GB |
| 1 | 10.10.10.12 | [512, 1024) | 512 | ~88.6 GB | ~11.5 GB |
| 2 | 10.10.10.14 | [1024, 1536) | 512 | ~88.6 GB | ~11.5 GB |

- **Embedding + LM head**: Rank 0 only
- **Attention**: Full replicated on all ranks
- **MoE FFN**: Sharded 3 ways via AllExpertsFfnWidth geometry

## Patches Applied

All 12 patches from vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe applied to SparkInfer base commit `7a9b77a`:

| Patch | Description |
|-------|-------------|
| 0001-0007 | Core AllExpertsFfnWidth TP3 sharding |
| 0008 | LoadReady fix |
| 0009 | F16 embed support |
| 0010 | Forward stall instrumentation |
| 0011 | **Finish deadlock fix** + decode tok/s measurement |
| 0012 | **Worker teardown hang fix** (finish ack after wait_token) |

## Additional Modifications

### Multi-prompt mode (`--prompts-file`)
- Loads model once, runs N prompts sequentially
- Saves ~31 min per prompt (avoids reload)
- KV cache reset between prompts via reset sentinel (token -2)

### Chat template tokenization
- K3 is a chat model requiring `<|open|>message role="user"...` format
- Prompts tokenized via `tokenizer.apply_chat_template()`

### Logit debug instrumentation
- Top-10 logit dump after each forward pass
- Logit statistics (min/max/mean/nan/inf counts)

## Benchmark Results

### Decode Throughput

| Context Size | Decode tok/s | Prefill tok/s | Status |
|-------------|-------------|---------------|--------|
| 2048 | 6.51 | 1.31 | ✅ Clean |
| 8192 | **7.20** | 1.27 | ✅ Clean (tight: ~1 GB free) |
| 12284 | — | — | ❌ OOM during loading |
| 16384 | — | — | ❌ OOM during loading |

### Memory Usage at 8K Context

| Rank | GPU Used | RAM Used | Free |
|------|----------|----------|------|
| R0 (.13) | 110 GiB | 114 GiB | ~1 GiB |
| R1 (.12) | 103 GiB | 107 GiB | ~4 GiB |
| R2 (.14) | 98 GiB | 102 GiB | ~9 GiB |

### Loading Time

All 3 ranks: ~31 minutes to upload weights from NVMe to GPU.

### Generation Quality

⚠️ **Known issue**: Model generates repeated single tokens (e.g., "atos" or "Neg" repeated). This occurs regardless of prompt format (raw IDs or chat template). The forward pass produces valid logits with correct statistics, but the argmax sampling selects the same token repeatedly.

**Hypothesis**: The issue may be in:
1. The lm_head projection (`k3_proj_f32`) not correctly computing logits from the hidden state
2. NCCL allreduce producing incorrect `x_next` values before the lm_head projection
3. A missing normalization step in the distributed forward path

**Evidence**:
- Logits have valid range (not NaN/Inf)
- Forward pass completes in ~10ms per step (93 layers)
- All 3 ranks synchronize correctly on every step
- The issue is **not** in prompt formatting — occurs with both raw IDs and properly chat-formatted prompts

## Maximum Safe Context

**8K tokens** is the confirmed maximum for 3×DGX Spark with IQ1_S quantization.

The per-rank memory budget is:
- ~88.6 GB for MoE FFN weights
- ~11.5 GB for attention weights
- ~21 GB remaining for KV cache + overhead
- 8K KV cache fits; 12K does not

Higher contexts would require:
- More nodes (TP4/TP6)
- More aggressive quantization
- KV cache quantization (not yet implemented)

## Build Instructions

```bash
git clone https://github.com/chishiki37/sparkinfer-k3-tp3.git
cd sparkinfer-k3-tp3
git checkout 7a9b77a043596157d77e4af376cf9f29f68ce368  # SparkInfer base
git am patches/0001-*.patch  # Apply all 12 patches

mkdir build && cd build
cmake .. -DSPARKINFER_TP=ON -DCMAKE_CUDA_ARCHITECTURES="89;90;100;120;121"
cmake --build . --target kimi_k3_dist_generate -j$(nproc)
```

## Running

```bash
# Rank 0 (coordinator)
kimi_k3_dist_generate \
  --rank 0 --world 3 \
  --listen 0.0.0.0:41231 \
  --model /path/to/k3-iq1s-00001-of-00009.gguf \
  --prompts-file prompts.txt \
  --n-predict 512 --max-ctx 8192

# Rank 1
kimi_k3_dist_generate \
  --rank 1 --world 3 \
  --coord 10.10.10.13:41231 \
  --model /path/to/k3-iq1s-00001-of-00009.gguf

# Rank 2
kimi_k3_dist_generate \
  --rank 2 --world 3 \
  --coord 10.10.10.13:41231 \
  --model /path/to/k3-iq1s-00001-of-00009.gguf
```

## Known Issues

1. **Generation quality**: Model produces repeated tokens. Needs investigation of lm_head computation in distributed path.
2. **Context ceiling**: 8K max on 121 GiB Sparks with IQ1_S.
3. **Loading time**: ~31 min per cold start (weights uploaded via cudaMemcpy H2D).
4. **Finish protocol**: Patches 0011-0012 fix the deadlock — must be applied.

## References

- [vcruz305 Recipe](https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe)
- [SparkInfer](https://github.com/chishiki37/sparkinfer-k3-tp3)
- [Kimi K3 GGUF](https://huggingface.co/mradermacher/Kimi-K3-GGUF)
