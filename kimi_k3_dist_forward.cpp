#include "sparkinfer/models/kimi_k3_dist_forward.h"

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/models/kimi_k3.h"

#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace k3k = sparkinfer::kernels::k3;

namespace sparkinfer {
namespace {


void dist_trace(const char* msg) {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
    std::fprintf(stderr, "[k3-dist][fwd +%lldms] %s\n", (long long)ms, msg);
    std::fflush(stderr);
}

void set_err(std::string* error, const std::string& v) {
    if (error) *error = v;
}

bool embed_token(const KimiK3Weights& w, const KimiK3Config& cfg, int token_id,
                 float* x, cudaStream_t stream) {
    if (!w.token_embd.ok()) return false;
    // Mirror single-GPU / TP embed: F32, F16 (Neuron release), Q8_0.
    long row_bytes = 0;
    if (w.token_embd.type == 0)
        row_bytes = (long)cfg.hidden * sizeof(float);
    else if (w.token_embd.type == 1)
        row_bytes = (long)cfg.hidden * sizeof(uint16_t);
    else if (w.token_embd.type == 8)
        row_bytes = (long)(cfg.hidden / 32) * 34;
    else
        return false;
    const char* base = (const char*)w.token_embd.data + (size_t)token_id * (size_t)row_bytes;
    return k3k::dequant_f32_by_type(x, base, cfg.hidden, w.token_embd.type, stream);
}

bool ensure_logits(KimiK3DistRank& rank) {
    if (rank.logits) return true;
    if (rank.plan.spec.rank != 0) return true;  // workers skip head
    if (cudaMalloc(&rank.logits, (size_t)rank.cfg.vocab * sizeof(float)) != cudaSuccess)
        return false;
    return true;
}

bool allreduce_partial(KimiK3Forward& fwd, int layer, K3LayerPhase phase,
                       tp::dist::RankCollective& coll, cudaStream_t stream,
                       std::string* error) {
    int n = 0;
    float* buf = kimi_k3_partial_buffer(fwd, layer, phase, &n);
    if (!buf || n <= 0) {
        set_err(error, "partial buffer missing for allreduce");
        return false;
    }
    if (!coll.allreduce_f32(buf, (std::size_t)n,
                            (tp::dist::RankStreamHandle)(std::uintptr_t)stream)) {
        set_err(error, "RankCollective::allreduce_f32 failed");
        return false;
    }
    return true;
}

}  // namespace

bool kimi_k3_dist_rank_init_nccl(KimiK3DistRank* rank,
                                 const tp::dist::NcclUniqueId& id,
                                 std::unique_ptr<tp::dist::RankCollective>* out,
                                 std::string* error) {
    if (!rank || !out || !rank->loaded) {
        set_err(error, "dist_rank_init_nccl: bad args / not loaded");
        return false;
    }
    auto coll = tp::dist::make_nccl_rank_collective(
        rank->plan.spec.world_size, rank->plan.spec.rank, rank->device, id, error);
    if (!coll) return false;
    *out = std::move(coll);
    return true;
}

bool kimi_k3_dist_forward_token(KimiK3DistRank& rank,
                                tp::dist::RankCollective& coll,
                                int token_id,
                                float* out_logits_host,
                                std::string* error) {
    if (!rank.loaded || !rank.x || !rank.x_next || !rank.stream) {
        set_err(error, "dist_forward_token: rank not loaded");
        return false;
    }
    if (coll.world_size() != rank.plan.spec.world_size ||
        coll.rank() != rank.plan.spec.rank) {
        set_err(error, "dist_forward_token: collective rank/world mismatch");
        return false;
    }
    if (cudaSetDevice(rank.device) != cudaSuccess) {
        set_err(error, "cudaSetDevice failed");
        return false;
    }

    const KimiK3Config& cfg = rank.cfg;
    const int H = cfg.hidden;
    const int tp_size = rank.plan.spec.world_size;
    const int my_rank = rank.plan.spec.rank;
    KimiK3Weights& w = rank.weights;
    KimiK3Forward& fwd = rank.fwd;

    // Sentinel: token_id == -2 means "reset KV cache for next prompt".
    if (token_id == -2) {
        kimi_k3_reset_state(rank.state);
        return true;
    }
    rank.state.n_ckpt = 0;

    {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "rank%d embed_token enter tok=%d", my_rank, token_id);
        dist_trace(buf);
    }
    if (!embed_token(w, cfg, token_id, rank.x, rank.stream)) {
        set_err(error, "embed_token failed");
        return false;
    }
    {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "rank%d embed_token ok tok=%d", my_rank, token_id);
        dist_trace(buf);
    }

    long n_coll = 0;
    for (int layer = 0; layer < cfg.n_layers; ++layer) {
        const bool is_moe = layer >= cfg.leading_dense;
        const bool kda_reduce =
            tp_size > 1 && cfg.is_kda_layer(layer) && KimiK3Weights::shards_kda(w.policy);
        const bool mla_reduce =
            tp_size > 1 && !cfg.is_kda_layer(layer) && KimiK3Weights::shards_mla(w.policy);
        const bool attn_reduce = kda_reduce || mla_reduce;

        if (!kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::Attn, rank.x, rank.x_next)) {
            set_err(error, "Attn phase failed at layer " + std::to_string(layer));
            return false;
        }

        if (attn_reduce) {
            if (layer == 0) {
                char buf[96];
                std::snprintf(buf, sizeof(buf), "rank%d first attn allreduce enter L0", my_rank);
                dist_trace(buf);
            }
            if (!allreduce_partial(fwd, layer, K3LayerPhase::Attn, coll, rank.stream, error)) {
                if (error && error->empty())
                    set_err(error, "attn allreduce failed layer " + std::to_string(layer));
                return false;
            }
            if (layer == 0) dist_trace("first attn allreduce ok L0");
            ++n_coll;
            if (!kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::FfnPartial, rank.x,
                                             rank.x_next)) {
                set_err(error, "FfnPartial after attn-reduce failed layer " +
                                   std::to_string(layer));
                return false;
            }
        } else {
            if (!kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::FfnPartial, rank.x,
                                             rank.x_next)) {
                set_err(error, "FfnPartial failed layer " + std::to_string(layer));
                return false;
            }
        }

        // MoE width/expert partial must be summed before FfnFinish (rms_norm not linear).
        if (is_moe && tp_size > 1) {
            if (!allreduce_partial(fwd, layer, K3LayerPhase::FfnPartial, coll, rank.stream,
                                   error)) {
                if (error && error->empty())
                    set_err(error, "moe allreduce failed layer " + std::to_string(layer));
                return false;
            }
            ++n_coll;
        }

        if (!kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::FfnFinish, rank.x,
                                         rank.x_next)) {
            set_err(error, "FfnFinish failed layer " + std::to_string(layer));
            return false;
        }
        std::swap(rank.x, rank.x_next);
        if (layer == 0 || layer + 1 == cfg.n_layers || ((layer + 1) % 16) == 0) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "rank%d layer %d/%d done coll=%ld", my_rank, layer + 1,
                          cfg.n_layers, n_coll);
            dist_trace(buf);
        }
    }

    // Head: only rank 0 holds output.weight under dist load policy.
    if (my_rank == 0) {
        if (!ensure_logits(rank)) {
            set_err(error, "logits cudaMalloc failed");
            return false;
        }
        if (cfg.attn_res_block_size > 0) {
            if (!w.has_output_res_score || !w.output_res_score.ok()) {
                set_err(error, "missing output_res_score");
                return false;
            }
            k3k::attn_res_mix_f32(rank.x_next, rank.state.res_bank, rank.x,
                                  (const float*)w.output_res_score.data, H, rank.state.n_ckpt,
                                  cfg.rms_eps, rank.stream);
            std::swap(rank.x, rank.x_next);
        }
        if (!w.output_norm.ok() || !w.output.ok()) {
            set_err(error, "missing output_norm/output");
            return false;
        }
        k3k::rms_norm_f32(rank.x_next, rank.x, (const float*)w.output_norm.data, H,
                          cfg.rms_eps, rank.stream);
        if (!k3k::k3_proj_f32(rank.logits, rank.x_next, w.output.data, w.output.type,
                              cfg.vocab, H, rank.stream)) {
            set_err(error, "lm head proj failed");
            return false;
        }
        if (out_logits_host) {
            if (cudaStreamSynchronize(rank.stream) != cudaSuccess) {
                set_err(error, "stream sync before logits D2H failed");
                return false;
            }
            if (cudaMemcpy(out_logits_host, rank.logits,
                           (size_t)cfg.vocab * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                set_err(error, "logits D2H failed");
                return false;
            }
            // DEBUG: dump top-10 logits
            {
                static int logit_dump_count = 0;
                if (logit_dump_count < 5) {
                    struct IdxVal { int idx; float val; };
                    IdxVal top[10] = {};
                    for (int i = 0; i < 10; i++) { top[i] = {0, -1e30f}; }
                    for (int i = 0; i < cfg.vocab; i++) {
                        float v = out_logits_host[i];
                        if (v > top[9].val) {
                            top[9] = {i, v};
                            for (int j = 8; j >= 0; j--) {
                                if (top[j+1].val > top[j].val) std::swap(top[j], top[j+1]);
                                else break;
                            }
                        }
                    }
                    std::fprintf(stderr, "[logits-dump #%d] ", logit_dump_count);
                    for (int i = 0; i < 10; i++) {
                        std::fprintf(stderr, "tok=%d val=%.3f ", top[i].idx, top[i].val);
                    }
                    std::fprintf(stderr, "\n");
                    // dump first 10 logits
                    std::fprintf(stderr, "[logits-first10] ");
                    for (int i = 0; i < 10; i++) {
                        std::fprintf(stderr, "%.3f ", out_logits_host[i]);
                    }
                    std::fprintf(stderr, "\n");
                    // dump stats
                    float min_v = 1e30f, max_v = -1e30f, sum = 0;
                    int nan_count = 0, inf_count = 0;
                    for (int i = 0; i < cfg.vocab; i++) {
                        float v = out_logits_host[i];
                        if (v != v) nan_count++;
                        else if (v == 1e30f || v == -1e30f) inf_count++;
                        else { min_v = std::min(min_v, v); max_v = std::max(max_v, v); sum += v; }
                    }
                    std::fprintf(stderr, "[logits-stats] min=%.3f max=%.3f mean=%.3f nan=%d inf=%d\n",
                                 min_v, max_v, sum/(cfg.vocab-nan_count-inf_count), nan_count, inf_count);
                    logit_dump_count++;
                }
            }
        }
    } else {
        // Workers still need stream-ordered progress and matching KDA/KV state.
        if (cudaStreamSynchronize(rank.stream) != cudaSuccess) {
            set_err(error, "worker stream sync failed");
            return false;
        }
    }

    // Advance device position (KV / KDA index) and host mirror together.
    if (!kimi_k3_set_position(rank.state, rank.state.position + 1)) {
        set_err(error, "kimi_k3_set_position failed");
        return false;
    }

    (void)n_coll;
    return true;
}

bool kimi_k3_dist_forward_prompt(KimiK3DistRank& rank,
                                 tp::dist::RankCollective& coll,
                                 const int* ids,
                                 int n_ids,
                                 float* out_logits_host,
                                 std::string* error) {
    if (!ids || n_ids <= 0) {
        set_err(error, "empty prompt");
        return false;
    }
    for (int i = 0; i < n_ids; ++i) {
        float* logits = (i + 1 == n_ids) ? out_logits_host : nullptr;
        if (!kimi_k3_dist_forward_token(rank, coll, ids[i], logits, error)) return false;
    }
    return true;
}

int kimi_k3_dist_argmax(const float* logits, int n_vocab) {
    if (!logits || n_vocab <= 0) return 0;
    int best = 0;
    float v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > v) {
            v = logits[i];
            best = i;
        }
    }
    return best;
}

}  // namespace sparkinfer
