// kimi_k3_dist_generate — multi-Spark (one process per rank) greedy decode.
// Modified for multi-prompt HumanEval benchmark.
//
//   Rank 0: TCP coordinator + sample + begin_step + local forward
//   Ranks 1..W-1: workers (NCCL InitRank + load + wait_token loop)
//
// Build with -DSPARKINFER_TP=ON (NCCL). Protocol-only: --dry-protocol.
//
// Usage:
//   # rank 0
//   kimi_k3_dist_generate --rank 0 --world 3 --listen 0.0.0.0:29500 \
//       --model /path/k3-00001-of-00009.gguf --prompt-ids 1,2,3 --n-predict 8
//   # rank r
//   kimi_k3_dist_generate --rank r --world 3 --coord HOST:29500 \
//       --model /path/k3-00001-of-00009.gguf

#include "sparkinfer/models/kimi_k3_dist_forward.h"
#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_dist_rank.h"
#include "sparkinfer/tp/rank_protocol.h"
#include "sparkinfer/tp/rank_transport.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace sparkinfer;
using namespace sparkinfer::tp::dist;

namespace {

void gen_trace(const char* msg) {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
    std::fprintf(stderr, "[k3-dist][gen +%lldms] %s\n", (long long)ms, msg);
    std::fflush(stderr);
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --rank R --world 3|4 (--listen HOST:PORT | --coord HOST:PORT)\n"
                 "          --model FIRST.gguf [--prompt-ids 1,2,3] [--prompt-ids-file F]\n"
                 "          [--prompts-file F] [--n-predict N] [--max-ctx N] [--device 0]\n"
                 "          [--session-id N] [--dry-protocol]\n",
                 argv0);
}

bool parse_host_port(const std::string& s, std::string* host, uint16_t* port) {
    const auto pos = s.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= s.size()) return false;
    *host = s.substr(0, pos);
    *port = (uint16_t)std::atoi(s.c_str() + pos + 1);
    return *port != 0;
}

std::vector<int> parse_ids_csv(const std::string& csv) {
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::atoi(item.c_str()));
    }
    return out;
}

std::vector<int> load_ids_file(const std::string& path) {
    std::ifstream in(path);
    std::vector<int> out;
    int v;
    while (in >> v) out.push_back(v);
    return out;
}

// Load multiple prompts from a file. Each line is comma-separated token IDs.
std::vector<std::vector<int>> load_multi_prompts(const std::string& path) {
    std::vector<std::vector<int>> all;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        all.push_back(parse_ids_csv(line));
    }
    return all;
}

void fill_digests(RankSpec* spec) {
    const std::string mid = "Kimi-K3-Neuron-IQ1S-GGUF";
    const std::string pid = plan_identity_string(spec->world_size);
    for (size_t i = 0; i < mid.size() && i < spec->model_digest.size(); ++i)
        spec->model_digest[i] = (unsigned char)mid[i];
    for (size_t i = 0; i < pid.size() && i < spec->plan_digest.size(); ++i)
        spec->plan_digest[i] = (unsigned char)pid[i];
    if (!spec->model_digest[0]) spec->model_digest[0] = 0xA5;
    if (!spec->plan_digest[0]) spec->plan_digest[0] = 0x5A;
}

}  // namespace

int main(int argc, char** argv) {
    int rank = -1, world = 3, device = 0, n_predict = 8, max_ctx = 8192;
    std::uint64_t session_id = 1;
    std::string model, listen, coord, prompt_csv, prompt_file, prompts_file;
    bool dry_protocol = false;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--rank")) rank = std::atoi(need("--rank"));
        else if (!std::strcmp(argv[i], "--world")) world = std::atoi(need("--world"));
        else if (!std::strcmp(argv[i], "--device")) device = std::atoi(need("--device"));
        else if (!std::strcmp(argv[i], "--model")) model = need("--model");
        else if (!std::strcmp(argv[i], "--listen")) listen = need("--listen");
        else if (!std::strcmp(argv[i], "--coord")) coord = need("--coord");
        else if (!std::strcmp(argv[i], "--prompt-ids")) prompt_csv = need("--prompt-ids");
        else if (!std::strcmp(argv[i], "--prompt-ids-file"))
            prompt_file = need("--prompt-ids-file");
        else if (!std::strcmp(argv[i], "--prompts-file"))
            prompts_file = need("--prompts-file");
        else if (!std::strcmp(argv[i], "--n-predict")) n_predict = std::atoi(need("--n-predict"));
        else if (!std::strcmp(argv[i], "--max-ctx")) max_ctx = std::atoi(need("--max-ctx"));
        else if (!std::strcmp(argv[i], "--session-id"))
            session_id = (std::uint64_t)std::strtoull(need("--session-id"), nullptr, 10);
        else if (!std::strcmp(argv[i], "--dry-protocol")) dry_protocol = true;
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (rank < 0 || (world != 3 && world != 4) || rank >= world) {
        usage(argv[0]);
        return 2;
    }
    if (!dry_protocol && model.empty()) {
        std::fprintf(stderr, "--model required unless --dry-protocol\n");
        return 2;
    }
    if (rank == 0 && listen.empty()) {
        std::fprintf(stderr, "rank 0 requires --listen HOST:PORT\n");
        return 2;
    }
    if (rank != 0 && coord.empty()) {
        std::fprintf(stderr, "workers require --coord HOST:PORT\n");
        return 2;
    }

    RankSpec spec;
    spec.session_id = session_id ? session_id : 1;
    spec.world_size = world;
    spec.rank = rank;
    spec.local_device = device;
    spec.n_experts = 896;
    spec.moe_ffn = 1536;
    spec.expert_block_elems = 256;
    spec.vocab = 163840;
    fill_digests(&spec);

    TransportConfig tcfg;
    tcfg.connect_timeout = std::chrono::milliseconds(120000);
    tcfg.recv_timeout = std::chrono::milliseconds(5400000);   // 90 min
    tcfg.phase_timeout = std::chrono::milliseconds(5400000);  // 90 min

    std::string err;

    if (dry_protocol) {
        if (rank != 0) {
            std::fprintf(stderr, "--dry-protocol is rank0-only (localhost multi-thread)\n");
            return 2;
        }
        LocalhostDryRunConfig dcfg;
        dcfg.world_size = world;
        dcfg.tokens = {1, 2, 3, 4};
        dcfg.session_id = spec.session_id;
        dcfg.vocab = spec.vocab;
        dcfg.model_digest = spec.model_digest;
        dcfg.plan_digest = spec.plan_digest;
        auto result = run_localhost_protocol_dry_run(dcfg);
        if (!result.ok) {
            std::fprintf(stderr, "dry-protocol failed: %s\n", result.error.c_str());
            return 1;
        }
        std::printf("dry-protocol OK world=%d steps=%zu\n", world, dcfg.tokens.size());
        return 0;
    }

    KimiK3DistRank dist;
    KimiK3DistRankLoadOptions lopt;
    lopt.max_ctx = max_ctx;
    lopt.verbose = true;
    std::unique_ptr<RankCollective> coll;

    auto do_load = [&](std::string* e) -> bool {
        return kimi_k3_dist_rank_load(model, spec, lopt, &dist, e);
    };

    if (rank == 0) {
        std::string host;
        uint16_t port = 0;
        if (!parse_host_port(listen, &host, &port)) {
            std::fprintf(stderr, "bad --listen %s\n", listen.c_str());
            return 2;
        }
        tcfg.host = host;
        tcfg.port = port;

        CoordinatorTransport ctr(spec, tcfg);
        if (!ctr.host(&err)) {
            std::fprintf(stderr, "listen failed: %s\n", err.c_str());
            return 1;
        }
        std::printf("[rank0] listening %s:%u — waiting for %d workers\n", host.c_str(),
                    (unsigned)ctr.port(), world - 1);

        NcclUniqueId nccl_id{};
        std::thread nccl_thr;
        std::string nccl_err;
        std::atomic<bool> nccl_ok{false};

        if (!ctr.bootstrap(
                [&](NcclUniqueId* id, std::string* e) {
                    if (!make_nccl_unique_id_bytes(id, e)) return false;
                    nccl_id = *id;
                    nccl_thr = std::thread([&] {
                        auto c = make_nccl_rank_collective(world, 0, device, nccl_id, &nccl_err);
                        if (c) {
                            coll = std::move(c);
                            nccl_ok = true;
                        }
                    });
                    return true;
                },
                [&](std::string* e) {
                    if (nccl_thr.joinable()) nccl_thr.join();
                    if (!nccl_ok || !coll) {
                        if (e) *e = nccl_err.empty() ? "rank0 NCCL init failed" : nccl_err;
                        return false;
                    }
                    std::printf("[rank0] NCCL ready — loading weights...\n");
                    return do_load(e);
                },
                &err)) {
            if (nccl_thr.joinable()) nccl_thr.join();
            std::fprintf(stderr, "coord bootstrap failed: %s\n", err.c_str());
            return 1;
        }
        if (nccl_thr.joinable()) nccl_thr.join();
        if (!nccl_ok || !coll) {
            std::fprintf(stderr, "rank0 NCCL init failed: %s\n", nccl_err.c_str());
            return 1;
        }
        gen_trace("rank0 bootstrap+load complete; starting prompt loop");

        // Load prompts: multi-prompt file has priority
        std::vector<std::vector<int>> all_prompts;
        if (!prompts_file.empty()) {
            all_prompts = load_multi_prompts(prompts_file);
            if (all_prompts.empty()) {
                std::fprintf(stderr, "no prompts in %s\n", prompts_file.c_str());
                return 2;
            }
            std::fprintf(stderr, "[k3-dist] loaded %zu prompts from %s\n",
                         all_prompts.size(), prompts_file.c_str());
        } else {
            // Single prompt mode (original behavior)
            std::vector<int> prompt;
            if (!prompt_file.empty()) prompt = load_ids_file(prompt_file);
            else if (!prompt_csv.empty()) prompt = parse_ids_csv(prompt_csv);
            else prompt = {1};
            if (prompt.empty()) {
                std::fprintf(stderr, "empty prompt ids\n");
                return 2;
            }
            all_prompts.push_back(std::move(prompt));
        }

        std::vector<float> logits((size_t)dist.cfg.vocab);

        auto one_step = [&](int tok, float* logits_out) -> bool {
            {
                char buf[96];
                std::snprintf(buf, sizeof(buf), "rank0 begin_step enter tok=%d", tok);
                gen_trace(buf);
            }
            if (!ctr.begin_step(tok, &err)) {
                std::fprintf(stderr, "begin_step: %s\n", err.c_str());
                std::fflush(stderr);
                return false;
            }
            gen_trace("rank0 begin_step ok");
            if (tok == -2) {
                // Reset sentinel: reset KV cache instead of forward
                kimi_k3_reset_state(dist.state);
                gen_trace("rank0 reset_state done (sentinel -2)");
            } else {
                gen_trace("rank0 forward enter");
                if (!kimi_k3_dist_forward_token(dist, *coll, tok, logits_out, &err)) {
                    std::fprintf(stderr, "forward: %s\n", err.c_str());
                    std::fflush(stderr);
                    ctr.complete_local_step(false, 1, err, nullptr);
                    ctr.fail(err, nullptr);
                    return false;
                }
                gen_trace("rank0 forward ok");
            }
            if (!ctr.complete_local_step(true, 0, "", &err)) {
                std::fprintf(stderr, "complete_local_step: %s\n", err.c_str());
                std::fflush(stderr);
                return false;
            }
            gen_trace("rank0 pump wait workers StepDone");
            if (!ctr.pump(std::chrono::milliseconds(600000), &err)) {
                std::fprintf(stderr, "pump: %s\n", err.c_str());
                std::fflush(stderr);
                return false;
            }
            gen_trace("rank0 step complete");
            return true;
        };

        using clock = std::chrono::steady_clock;
        double total_prefill_s = 0, total_gen_s = 0;
        int total_gen_tokens = 0;

        for (size_t pi = 0; pi < all_prompts.size(); ++pi) {
            const auto& prompt = all_prompts[pi];

            std::fprintf(stderr, "[k3-dist] === prompt %zu/%zu (%zu tokens) ===\n",
                         pi + 1, all_prompts.size(), prompt.size());
            std::fflush(stderr);
            // Reset KV cache between prompts (sentinel token -2)
            if (pi > 0) {
                gen_trace("rank0 sending reset sentinel");
                if (!one_step(-2, nullptr)) return 1;
                gen_trace("rank0 reset sentinel done");
            }
            std::printf("=== PROMPT %zu ===\n", pi);
            std::fflush(stdout);

            // Prefill
            const auto t_prefill0 = clock::now();
            for (size_t i = 0; i < prompt.size(); ++i) {
                float* lp = (i + 1 == prompt.size()) ? logits.data() : nullptr;
                if (!one_step(prompt[i], lp)) return 1;
            }
            const auto t_prefill1 = clock::now();
            const double prefill_s =
                std::chrono::duration<double>(t_prefill1 - t_prefill0).count();
            total_prefill_s += prefill_s;
            std::printf("prefill done (%zu toks) in %.3fs\n", prompt.size(), prefill_s);
            std::fflush(stdout);
            std::fprintf(stderr,
                         "[k3-dist][tps] prompt=%zu prefill_tokens=%zu prefill_s=%.6f prefill_tok_s=%.4f\n",
                         pi, prompt.size(), prefill_s,
                         prefill_s > 0 ? (double)prompt.size() / prefill_s : 0.0);
            std::fflush(stderr);

            // Generate
            std::vector<int> generated;
            const auto t_gen0 = clock::now();
            for (int n = 0; n < n_predict; ++n) {
                const int next = kimi_k3_dist_argmax(logits.data(), dist.cfg.vocab);
                generated.push_back(next);
                std::printf(" gen[%d]=%d\n", n, next);
                std::fflush(stdout);
                {
                    char buf[96];
                    std::snprintf(buf, sizeof(buf), "rank0 prompt=%zu gen[%d]=%d", pi, n, next);
                    gen_trace(buf);
                }
                if (!one_step(next, logits.data())) return 1;
            }
            const auto t_gen1 = clock::now();
            const double gen_s = std::chrono::duration<double>(t_gen1 - t_gen0).count();
            const double decode_tps = gen_s > 0 ? (double)n_predict / gen_s : 0.0;
            total_gen_s += gen_s;
            total_gen_tokens += n_predict;

            std::printf("generated %zu tokens in %.3fs (decode_tok_s=%.4f)\n",
                        generated.size(), gen_s, decode_tps);
            std::fflush(stdout);
            std::fprintf(stderr,
                         "[k3-dist][tps] prompt=%zu decode_tokens=%d decode_s=%.6f decode_tok_s=%.4f\n",
                         pi, n_predict, gen_s, decode_tps);
            std::fprintf(stderr, "[k3-dist][tps] prompt=%zu generated_ids=", pi);
            for (size_t i = 0; i < generated.size(); ++i) {
                std::fprintf(stderr, "%s%d", i ? "," : "", generated[i]);
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }

        // Summary
        const double total_decode_tps = total_gen_s > 0 ? (double)total_gen_tokens / total_gen_s : 0.0;
        std::printf("\n=== SUMMARY ===\n");
        std::printf("prompts=%zu total_prefill_s=%.3f total_gen_s=%.3f\n",
                    all_prompts.size(), total_prefill_s, total_gen_s);
        std::printf("total_gen_tokens=%d avg_decode_tok_s=%.4f\n",
                    total_gen_tokens, total_decode_tps);
        std::fflush(stdout);
        std::fprintf(stderr, "[k3-dist][tps] summary prompts=%zu total_prefill_s=%.6f total_gen_s=%.6f total_gen_tokens=%d avg_decode_tok_s=%.4f\n",
                     all_prompts.size(), total_prefill_s, total_gen_s, total_gen_tokens, total_decode_tps);
        std::fflush(stderr);

        gen_trace("rank0 finish enter");
        if (!ctr.finish(&err)) {
            std::fprintf(stderr, "finish: %s\n", err.c_str());
            std::fflush(stderr);
            return 1;
        }
        gen_trace("rank0 finish ok");
        std::printf("OK finished clean\n");
        std::fflush(stdout);
        kimi_k3_dist_rank_free(&dist);
        return 0;
    }

    // ---- workers ----
    std::string host;
    uint16_t port = 0;
    if (!parse_host_port(coord, &host, &port)) {
        std::fprintf(stderr, "bad --coord\n");
        return 2;
    }
    tcfg.host = host;
    tcfg.port = port;
    RankTransport wtr(spec, tcfg);
    if (!wtr.bootstrap(
            [&](const NcclUniqueId& id, std::string* e) {
                auto c = make_nccl_rank_collective(world, rank, device, id, e);
                if (!c) return false;
                coll = std::move(c);
                return true;
            },
            [&](std::string* e) { return do_load(e); }, &err)) {
        std::fprintf(stderr, "worker bootstrap: %s\n", err.c_str());
        std::fflush(stderr);
        return 1;
    }
    {
        char buf[80];
        std::snprintf(buf, sizeof(buf), "rank%d bootstrap+load complete; wait_token loop", rank);
        gen_trace(buf);
    }

    for (;;) {
        int tok = -1;
        {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "rank%d wait_token enter", rank);
            gen_trace(buf);
        }
        if (!wtr.wait_token(&tok, &err)) {
            if (wtr.state() == RankState::Finished || wtr.state() == RankState::FinishPending) {
                wtr.wait_finished(nullptr);
                std::printf("[rank %d] finished\n", rank);
                std::fflush(stdout);
                kimi_k3_dist_rank_free(&dist);
                return 0;
            }
            std::fprintf(stderr, "wait_token: %s state=%d\n", err.c_str(), (int)wtr.state());
            std::fflush(stderr);
            return 1;
        }
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "rank%d wait_token got tok=%d; forward enter", rank, tok);
            gen_trace(buf);
        }
        if (tok == -2) {
            // Reset sentinel: reset KV cache instead of forward
            kimi_k3_reset_state(dist.state);
            {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "rank%d reset_state done (sentinel -2)", rank);
                gen_trace(buf);
            }
        } else {
            if (!kimi_k3_dist_forward_token(dist, *coll, tok, nullptr, &err)) {
                wtr.report_step_done(false, 1, err, nullptr);
                std::fprintf(stderr, "forward: %s\n", err.c_str());
                std::fflush(stderr);
                return 1;
            }
            {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "rank%d forward ok; StepDone", rank);
                gen_trace(buf);
            }
        }
        if (!wtr.report_step_done(true, 0, "", &err)) {
            std::fprintf(stderr, "step_done: %s\n", err.c_str());
            std::fflush(stderr);
            return 1;
        }
    }
}
