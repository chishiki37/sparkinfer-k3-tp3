#include "sparkinfer/tp/rank_protocol.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace sparkinfer {
namespace tp {
namespace dist {

namespace {

constexpr std::uint32_t kMagic = 0x5052334bU;  // little-endian bytes: "K3RP"

void set_error(std::string* error, const std::string& value) {
    if (error) *error = value;
}

template <std::size_t N>
bool any_nonzero(const std::array<std::uint8_t, N>& bytes) {
    for (std::uint8_t v : bytes) if (v != 0) return true;
    return false;
}

bool valid_type(std::uint16_t v) {
    return v >= static_cast<std::uint16_t>(MessageType::Hello) &&
           v <= static_cast<std::uint16_t>(MessageType::FinishAck);
}

bool valid_phase(std::uint16_t v) {
    return v <= static_cast<std::uint16_t>(Phase::Shutdown);
}

void put_u16(std::vector<std::uint8_t>& out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v));
    out.push_back(static_cast<std::uint8_t>(v >> 8));
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

void put_i32(std::vector<std::uint8_t>& out, int v) {
    put_u32(out, static_cast<std::uint32_t>(static_cast<std::int32_t>(v)));
}

struct Reader {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::size_t at = 0;

    bool u16(std::uint16_t* out) {
        if (!out || at + 2 > size) return false;
        *out = static_cast<std::uint16_t>(data[at]) |
               (static_cast<std::uint16_t>(data[at + 1]) << 8);
        at += 2;
        return true;
    }
    bool u32(std::uint32_t* out) {
        if (!out || at + 4 > size) return false;
        *out = 0;
        for (int i = 0; i < 4; ++i)
            *out |= static_cast<std::uint32_t>(data[at + i]) << (8 * i);
        at += 4;
        return true;
    }
    bool u64(std::uint64_t* out) {
        if (!out || at + 8 > size) return false;
        *out = 0;
        for (int i = 0; i < 8; ++i)
            *out |= static_cast<std::uint64_t>(data[at + i]) << (8 * i);
        at += 8;
        return true;
    }
    bool i32(int* out) {
        std::uint32_t v = 0;
        if (!u32(&v) || !out) return false;
        // Avoid implementation-defined unsigned-to-signed conversion for wire
        // values with the high bit set (negative status codes).
        *out = v <= 0x7fffffffU
            ? static_cast<int>(v)
            : -1 - static_cast<int>(0xffffffffU - v);
        return true;
    }
    bool bytes(std::uint8_t* out, std::size_t n) {
        if (!out || at + n > size) return false;
        std::memcpy(out, data + at, n);
        at += n;
        return true;
    }
};

std::uint32_t crc32(const std::uint8_t* data, std::size_t size) {
    std::uint32_t crc = 0xffffffffU;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return ~crc;
}

Message from_spec(const RankSpec& spec, MessageType type, int rank) {
    Message m;
    m.type = type;
    m.session_id = spec.session_id;
    m.rank = rank;
    m.world_size = spec.world_size;
    m.local_device = spec.local_device;
    m.n_experts = spec.n_experts;
    m.moe_ffn = spec.moe_ffn;
    m.expert_block_elems = spec.expert_block_elems;
    m.vocab = spec.vocab;
    m.model_digest = spec.model_digest;
    m.plan_digest = spec.plan_digest;
    return m;
}

bool all_seen(const std::vector<unsigned char>& seen) {
    return !seen.empty() &&
           std::all_of(seen.begin(), seen.end(), [](unsigned char v) { return v != 0; });
}

Phase coordinator_phase(CoordinatorState state) {
    switch (state) {
        case CoordinatorState::CollectingHello:
        case CoordinatorState::AwaitingNcclUniqueId: return Phase::Bootstrap;
        case CoordinatorState::WaitingNcclReady: return Phase::NcclInit;
        case CoordinatorState::WaitingLoadReady: return Phase::Load;
        case CoordinatorState::Ready:
        case CoordinatorState::StepInFlight: return Phase::Step;
        case CoordinatorState::WaitingFinishAck:
        case CoordinatorState::Finished:
        case CoordinatorState::Failed: return Phase::Shutdown;
    }
    return Phase::None;
}

}  // namespace

ShardError validate_rank_spec(const RankSpec& spec, ShardDims* out) {
    if (!out) return ShardError{"validate_rank_spec: null out"};
    if (spec.session_id == 0) return ShardError{"session_id must be nonzero"};
    if (!supported_k3_dist_world_size(spec.world_size))
        return ShardError{"supported world_size values are 3 (TP3 width) and 4 (TP4 2-D)"};
    if (spec.rank < 0 || spec.rank >= spec.world_size)
        return ShardError{"rank outside [0, world_size)"};
    if (spec.local_device != 0)
        return ShardError{"one-process-per-Spark milestone requires local_device=0"};
    if (spec.n_experts != 896)
        return ShardError{"this K3 plan requires exactly 896 routed experts"};
    if (spec.moe_ffn != 1536)
        return ShardError{"this pruned K3 plan requires moe_ffn=1536"};
    if (spec.expert_block_elems != 256)
        return ShardError{"this IQ1_S K3 plan requires 256-element expert blocks"};
    if (spec.vocab <= 0) return ShardError{"vocab must be > 0"};
    if (!any_nonzero(spec.model_digest)) return ShardError{"model_digest is unset"};
    if (!any_nonzero(spec.plan_digest)) return ShardError{"plan_digest is unset"};

    ShardDims d;
    d.hidden = 7168;
    d.tp_size = spec.world_size;
    d.rank = spec.rank;
    if (spec.world_size == kK3Tp3WorldSize) {
        // 896 % 3 != 0 → keep all expert identities; split FFN width 1536→512.
        const ShardError e = moe_all_expert_width_dims(
            spec.n_experts, spec.moe_ffn, spec.world_size, spec.rank,
            spec.expert_block_elems, &d);
        if (!e.ok()) return e;
    } else {
        // world=4: default K3 2-D plan eg=2, fs=2 → 448 experts × 768 FFN.
        // 896%4==0 also admits WholeExperts, but the audited memory plan and
        // executor path for four Sparks is ExpertFfn2D.
        const int expert_groups = k3_default_moe_expert_groups(spec.world_size);
        const ShardError e = moe_2d_dims(
            spec.n_experts, spec.moe_ffn, spec.world_size, spec.rank,
            expert_groups, spec.expert_block_elems, &d);
        if (!e.ok()) return e;
        if (!d.moe_2d || d.moe_shard_mode != MoeShardMode::ExpertFfn2D) {
            return ShardError{"TP4 validate expected ExpertFfn2D geometry"};
        }
    }
    *out = d;
    return ShardError{};
}

bool encode_message(const Message& m, std::vector<std::uint8_t>* frame,
                    std::string* error) {
    if (!frame) { set_error(error, "encode_message: null output"); return false; }
    const std::uint16_t type = static_cast<std::uint16_t>(m.type);
    const std::uint16_t phase = static_cast<std::uint16_t>(m.phase);
    if (!valid_type(type)) { set_error(error, "unknown message type"); return false; }
    if (!valid_phase(phase)) { set_error(error, "unknown protocol phase"); return false; }
    if (m.detail.size() > kMaxDetailBytes) {
        set_error(error, "diagnostic detail exceeds protocol bound");
        return false;
    }

    std::vector<std::uint8_t> payload;
    payload.reserve(kWireFixedPayloadBytes + m.detail.size());
    put_u64(payload, m.session_id);
    put_u64(payload, m.sequence);
    put_i32(payload, m.rank);
    put_i32(payload, m.world_size);
    put_i32(payload, m.local_device);
    put_i32(payload, m.n_experts);
    put_i32(payload, m.moe_ffn);
    put_i32(payload, m.expert_block_elems);
    put_i32(payload, m.vocab);
    put_i32(payload, m.token_id);
    put_i32(payload, m.status);
    put_u16(payload, phase);
    put_u16(payload, 0);  // reserved; must decode as zero
    payload.insert(payload.end(), m.model_digest.begin(), m.model_digest.end());
    payload.insert(payload.end(), m.plan_digest.begin(), m.plan_digest.end());
    payload.insert(payload.end(), m.nccl_id.begin(), m.nccl_id.end());
    put_u16(payload, static_cast<std::uint16_t>(m.detail.size()));
    payload.insert(payload.end(), m.detail.begin(), m.detail.end());

    frame->clear();
    if (payload.size() < kWireFixedPayloadBytes || payload.size() > kMaxPayloadBytes) {
        set_error(error, "internal payload length outside protocol bounds");
        return false;
    }
    frame->reserve(kWireHeaderBytes + payload.size());
    put_u32(*frame, kMagic);
    put_u16(*frame, kProtocolVersion);
    put_u16(*frame, type);
    put_u32(*frame, static_cast<std::uint32_t>(payload.size()));
    put_u32(*frame, crc32(payload.data(), payload.size()));
    frame->insert(frame->end(), payload.begin(), payload.end());
    return true;
}

bool decode_message(const std::uint8_t* frame, std::size_t size, Message* message,
                    std::string* error) {
    if (!frame || !message) { set_error(error, "decode_message: null input/output"); return false; }
    if (size < kWireHeaderBytes) { set_error(error, "truncated frame header"); return false; }
    Reader h{frame, kWireHeaderBytes, 0};
    std::uint32_t magic = 0, payload_size = 0, checksum = 0;
    std::uint16_t version = 0, type = 0;
    if (!h.u32(&magic) || !h.u16(&version) || !h.u16(&type) ||
        !h.u32(&payload_size) || !h.u32(&checksum)) {
        set_error(error, "invalid frame header"); return false;
    }
    if (magic != kMagic) { set_error(error, "bad frame magic"); return false; }
    if (version != kProtocolVersion) { set_error(error, "unsupported protocol version"); return false; }
    if (!valid_type(type)) { set_error(error, "unknown message type"); return false; }
    if (payload_size < kWireFixedPayloadBytes || payload_size > kMaxPayloadBytes) {
        set_error(error, "payload length outside protocol bounds");
        return false;
    }
    if (size != kWireHeaderBytes + static_cast<std::size_t>(payload_size)) {
        set_error(error, "frame length mismatch");
        return false;
    }
    const std::uint8_t* payload = frame + kWireHeaderBytes;
    if (crc32(payload, payload_size) != checksum) { set_error(error, "payload CRC mismatch"); return false; }

    Reader r{payload, payload_size, 0};
    Message m;
    m.type = static_cast<MessageType>(type);
    std::uint16_t phase = 0, reserved = 0, detail_size = 0;
    if (!r.u64(&m.session_id) || !r.u64(&m.sequence) ||
        !r.i32(&m.rank) || !r.i32(&m.world_size) || !r.i32(&m.local_device) ||
        !r.i32(&m.n_experts) || !r.i32(&m.moe_ffn) ||
        !r.i32(&m.expert_block_elems) || !r.i32(&m.vocab) ||
        !r.i32(&m.token_id) || !r.i32(&m.status) ||
        !r.u16(&phase) || !r.u16(&reserved) ||
        !r.bytes(m.model_digest.data(), m.model_digest.size()) ||
        !r.bytes(m.plan_digest.data(), m.plan_digest.size()) ||
        !r.bytes(m.nccl_id.data(), m.nccl_id.size()) ||
        !r.u16(&detail_size)) {
        set_error(error, "truncated message payload"); return false;
    }
    if (!valid_phase(phase)) { set_error(error, "unknown protocol phase"); return false; }
    if (reserved != 0) { set_error(error, "nonzero reserved field"); return false; }
    if (detail_size > kMaxDetailBytes || r.at + detail_size != r.size) {
        set_error(error, "invalid diagnostic detail length"); return false;
    }
    m.phase = static_cast<Phase>(phase);
    m.detail.assign(reinterpret_cast<const char*>(r.data + r.at), detail_size);
    r.at += detail_size;
    *message = std::move(m);
    return true;
}

Coordinator::Coordinator(const RankSpec& cluster) : cluster_(cluster) {
    ShardDims d;
    const ShardError e = validate_rank_spec(cluster_, &d);
    if (!e.ok()) {
        state_ = CoordinatorState::Failed;
        failure_ = e.message;
        return;
    }
    if (cluster_.rank != 0) {
        state_ = CoordinatorState::Failed;
        failure_ = "coordinator must use global rank 0";
        return;
    }
    hello_.assign(cluster_.world_size, 0);
    nccl_ready_.assign(cluster_.world_size, 0);
    load_ready_.assign(cluster_.world_size, 0);
    step_done_.assign(cluster_.world_size, 0);
    finish_ack_.assign(cluster_.world_size, 0);
    peer_connection_.assign(cluster_.world_size, 0);
}

bool Coordinator::matches_cluster(const Message& m, std::string* why) const {
    if (m.session_id != cluster_.session_id) { set_error(why, "session mismatch"); return false; }
    if (m.world_size != cluster_.world_size) { set_error(why, "world-size mismatch"); return false; }
    if (m.rank < 0 || m.rank >= cluster_.world_size) { set_error(why, "rank out of range"); return false; }
    if (m.local_device != 0) { set_error(why, "rank-local device must be 0"); return false; }
    if (m.n_experts != cluster_.n_experts || m.moe_ffn != cluster_.moe_ffn ||
        m.expert_block_elems != cluster_.expert_block_elems || m.vocab != cluster_.vocab) {
        set_error(why, "model/plan shape mismatch"); return false;
    }
    if (m.model_digest != cluster_.model_digest) { set_error(why, "model digest mismatch"); return false; }
    if (m.plan_digest != cluster_.plan_digest) { set_error(why, "plan digest mismatch"); return false; }
    return true;
}

Message Coordinator::control(MessageType type, int target) const {
    Message m = from_spec(cluster_, type, target);
    m.sequence = next_sequence_;
    return m;
}

bool Coordinator::abort_cluster(Phase phase, int status, const std::string& detail,
                                std::vector<Message>* outbound, std::string* error) {
    if (state_ == CoordinatorState::Failed) {
        set_error(error, failure_);
        return false;
    }
    if (state_ == CoordinatorState::Finished) {
        set_error(error, "cluster already finished");
        return false;
    }
    if (!outbound) {
        set_error(error, "cluster abort requires an outbound sink");
        return false;
    }
    failure_ = detail.empty() ? "cluster aborted" : detail;
    state_ = CoordinatorState::Failed;
    for (int rank = 0; rank < cluster_.world_size; ++rank) {
        Message m = control(MessageType::Abort, rank);
        m.phase = phase;
        m.status = status == 0 ? -1 : status;
        m.detail = failure_.substr(0, kMaxDetailBytes);
        outbound->push_back(std::move(m));
    }
    set_error(error, failure_);
    return false;
}

bool Coordinator::accept(const AuthenticatedPeer& peer, const Message& m,
                         std::vector<Message>* outbound,
                         std::string* error) {
    if (state_ == CoordinatorState::Failed) { set_error(error, failure_); return false; }
    if (state_ == CoordinatorState::Finished) {
        set_error(error, "cluster already finished");
        return false;
    }
    if (!outbound) {
        set_error(error, "Coordinator::accept requires an outbound sink");
        return false;
    }
    if (peer.rank < 0 || peer.rank >= cluster_.world_size)
        return abort_cluster(coordinator_phase(state_), -17,
                             "authenticated peer rank out of range", outbound, error);
    if (peer.connection_id == 0)
        return abort_cluster(coordinator_phase(state_), -23,
                             "authenticated connection id is unset", outbound, error);
    if (m.rank != peer.rank)
        return abort_cluster(coordinator_phase(state_), -18,
                             "transport peer rank does not match Message::rank",
                             outbound, error);
    std::string why;
    if (!matches_cluster(m, &why))
        return abort_cluster(coordinator_phase(state_), -2, why, outbound, error);

    const std::uint64_t bound_connection = peer_connection_[peer.rank];
    if (bound_connection != 0 && bound_connection != peer.connection_id)
        return abort_cluster(coordinator_phase(state_), -24,
                             "duplicate connection for authenticated rank",
                             outbound, error);
    if (m.type != MessageType::Hello && bound_connection == 0)
        return abort_cluster(coordinator_phase(state_), -25,
                             "rank message received before bound Hello",
                             outbound, error);

    if (m.type == MessageType::Failure) {
        if (m.status == 0 || m.phase == Phase::None)
            return abort_cluster(coordinator_phase(state_), -26,
                                 "invalid Failure report", outbound, error);
        return abort_cluster(m.phase, m.status, m.detail, outbound, error);
    }

    switch (m.type) {
        case MessageType::Hello:
            if (state_ != CoordinatorState::CollectingHello || m.sequence != 0 ||
                m.phase != Phase::Bootstrap)
                return abort_cluster(Phase::Bootstrap, -3, "unexpected Hello", outbound, error);
            if (hello_[m.rank])
                return abort_cluster(Phase::Bootstrap, -4, "duplicate Hello", outbound, error);
            peer_connection_[m.rank] = peer.connection_id;
            hello_[m.rank] = 1;
            if (all_seen(hello_)) state_ = CoordinatorState::AwaitingNcclUniqueId;
            return true;

        case MessageType::NcclReady:
            if (state_ != CoordinatorState::WaitingNcclReady || m.sequence != 0 ||
                m.status != 0 || m.phase != Phase::NcclInit)
                return abort_cluster(Phase::NcclInit, -5, "unexpected NcclReady", outbound, error);
            if (nccl_ready_[m.rank])
                return abort_cluster(Phase::NcclInit, -6, "duplicate NcclReady", outbound, error);
            nccl_ready_[m.rank] = 1;
            if (all_seen(nccl_ready_)) {
                state_ = CoordinatorState::WaitingLoadReady;
                for (int rank = 0; rank < cluster_.world_size; ++rank) {
                    Message begin = control(MessageType::BeginLoad, rank);
                    begin.phase = Phase::Load;
                    outbound->push_back(std::move(begin));
                }
            }
            return true;

        case MessageType::LoadReady:
            if (state_ != CoordinatorState::WaitingLoadReady || m.sequence != 0 ||
                m.status != 0 || m.phase != Phase::Load)
                return abort_cluster(Phase::Load, -7, "unexpected LoadReady", outbound, error);
            if (load_ready_[m.rank])
                return abort_cluster(Phase::Load, -8, "duplicate LoadReady", outbound, error);
            load_ready_[m.rank] = 1;
            if (all_seen(load_ready_)) state_ = CoordinatorState::Ready;
            return true;

        case MessageType::StepDone:
            if (state_ != CoordinatorState::StepInFlight ||
                m.sequence != next_sequence_ || m.status != 0 || m.phase != Phase::Step)
                return abort_cluster(Phase::Step, -9, "unexpected StepDone", outbound, error);
            if (step_done_[m.rank])
                return abort_cluster(Phase::Step, -10, "duplicate StepDone", outbound, error);
            step_done_[m.rank] = 1;
            if (all_seen(step_done_)) {
                ++next_sequence_;
                state_ = CoordinatorState::Ready;
            }
            return true;

        case MessageType::FinishAck:
            if (state_ != CoordinatorState::WaitingFinishAck ||
                m.sequence != next_sequence_ || m.status != 0 ||
                m.phase != Phase::Shutdown)
                return abort_cluster(Phase::Shutdown, -19,
                                     "unexpected FinishAck", outbound, error);
            if (finish_ack_[m.rank])
                return abort_cluster(Phase::Shutdown, -20,
                                     "duplicate FinishAck", outbound, error);
            finish_ack_[m.rank] = 1;
            if (all_seen(finish_ack_)) state_ = CoordinatorState::Finished;
            return true;

        default:
            return abort_cluster(coordinator_phase(state_), -11,
                                 "unexpected rank message type", outbound, error);
    }
}

bool Coordinator::install_nccl_unique_id(const NcclUniqueId& id,
                                         std::vector<Message>* outbound,
                                         std::string* error) {
    if (state_ != CoordinatorState::AwaitingNcclUniqueId) {
        if (state_ == CoordinatorState::Failed) { set_error(error, failure_); return false; }
        return abort_cluster(Phase::Bootstrap, -12, "NCCL id installed out of order",
                             outbound, error);
    }
    if (!outbound) return abort_cluster(Phase::Bootstrap, -13,
                                        "NCCL id has no outbound sink", outbound, error);
    if (!any_nonzero(id))
        return abort_cluster(Phase::Bootstrap, -27,
                             "NCCL unique id is unset", outbound, error);
    for (int rank = 0; rank < cluster_.world_size; ++rank) {
        Message m = control(MessageType::NcclUniqueId, rank);
        m.phase = Phase::NcclInit;
        m.nccl_id = id;
        outbound->push_back(std::move(m));
    }
    state_ = CoordinatorState::WaitingNcclReady;
    return true;
}

bool Coordinator::begin_step(int token_id, std::vector<Message>* outbound,
                             std::string* error) {
    if (state_ != CoordinatorState::Ready) {
        if (state_ == CoordinatorState::Failed) { set_error(error, failure_); return false; }
        if (state_ == CoordinatorState::Finished) {
            set_error(error, "cluster already finished");
            return false;
        }
        return abort_cluster(Phase::Step, -14, "token started before cluster Ready",
                             outbound, error);
    }
    if (!outbound) return abort_cluster(Phase::Step, -15,
                                        "token broadcast has no outbound sink", outbound, error);
    if ((token_id < -2) || (token_id >= cluster_.vocab) || (token_id == -1))
        return abort_cluster(Phase::Step, -16, "token id outside vocabulary", outbound, error);
    std::fill(step_done_.begin(), step_done_.end(), 0);
    for (int rank = 0; rank < cluster_.world_size; ++rank) {
        Message m = control(MessageType::Token, rank);
        m.phase = Phase::Step;
        m.token_id = token_id;
        outbound->push_back(std::move(m));
    }
    state_ = CoordinatorState::StepInFlight;
    return true;
}

bool Coordinator::finish(std::vector<Message>* outbound, std::string* error) {
    if (state_ != CoordinatorState::Ready) {
        if (state_ == CoordinatorState::Failed) { set_error(error, failure_); return false; }
        if (state_ == CoordinatorState::Finished) {
            set_error(error, "cluster already finished");
            return false;
        }
        return abort_cluster(Phase::Shutdown, -21,
                             "Finish sent before cluster Ready", outbound, error);
    }
    if (!outbound)
        return abort_cluster(Phase::Shutdown, -22,
                             "Finish has no outbound sink", outbound, error);
    std::fill(finish_ack_.begin(), finish_ack_.end(), 0);
    for (int rank = 0; rank < cluster_.world_size; ++rank) {
        Message m = control(MessageType::Finish, rank);
        m.phase = Phase::Shutdown;
        outbound->push_back(std::move(m));
    }
    state_ = CoordinatorState::WaitingFinishAck;
    return true;
}

bool Coordinator::fail(Phase phase, int status, const std::string& detail,
                       std::vector<Message>* outbound, std::string* error) {
    if (state_ == CoordinatorState::Finished) {
        set_error(error, "cluster already finished");
        return false;
    }
    return abort_cluster(phase, status, detail, outbound, error);
}

RankSession::RankSession(const RankSpec& spec) : spec_(spec) {
    ShardDims d;
    const ShardError e = validate_rank_spec(spec_, &d);
    if (!e.ok()) {
        state_ = RankState::Failed;
        failure_ = e.message;
    }
}

Message RankSession::reply(MessageType type, Phase phase) const {
    Message m = from_spec(spec_, type, spec_.rank);
    m.sequence = next_sequence_;
    m.phase = phase;
    return m;
}

bool RankSession::reject(const std::string& detail, std::string* error) {
    if (failure_.empty()) failure_ = detail;
    state_ = RankState::Failed;
    set_error(error, failure_);
    return false;
}

bool RankSession::hello(Message* outbound, std::string* error) {
    if (!outbound) return reject("Hello has no outbound sink", error);
    if (state_ != RankState::Created) return reject("Hello sent out of order", error);
    *outbound = reply(MessageType::Hello, Phase::Bootstrap);
    outbound->sequence = 0;
    state_ = RankState::HelloSent;
    return true;
}

bool RankSession::accept_control(const Message& m, int* token_id, std::string* error) {
    if (m.session_id != spec_.session_id || m.world_size != spec_.world_size ||
        m.rank != spec_.rank || m.local_device != spec_.local_device ||
        m.n_experts != spec_.n_experts || m.moe_ffn != spec_.moe_ffn ||
        m.expert_block_elems != spec_.expert_block_elems || m.vocab != spec_.vocab ||
        m.model_digest != spec_.model_digest || m.plan_digest != spec_.plan_digest)
        return reject("control message cluster identity mismatch", error);

    // Abort is deliberately idempotent and must remain consumable after a local
    // failure so the dedicated control thread can still call ncclCommAbort. Keep
    // the more specific local failure text when one already exists.
    if (m.type == MessageType::Abort) {
        if (m.status == 0 || m.phase == Phase::None)
            return reject("malformed coordinator Abort", error);
        if (failure_.empty())
            failure_ = m.detail.empty() ? "coordinator aborted cluster" : m.detail;
        state_ = RankState::Failed;
        return true;
    }
    if (state_ == RankState::Failed) { set_error(error, failure_); return false; }
    if (state_ == RankState::Finished) {
        set_error(error, "rank already finished");
        return false;
    }

    switch (m.type) {
        case MessageType::NcclUniqueId:
            if (state_ != RankState::HelloSent || m.sequence != 0 ||
                m.phase != Phase::NcclInit || m.status != 0)
                return reject("NCCL id received out of order", error);
            if (!any_nonzero(m.nccl_id))
                return reject("NCCL unique id is unset", error);
            nccl_id_ = m.nccl_id;
            state_ = RankState::NeedNcclInit;
            return true;

        case MessageType::BeginLoad:
            if (state_ != RankState::NcclReadySent || m.sequence != 0 ||
                m.phase != Phase::Load || m.status != 0)
                return reject("BeginLoad received before NCCL ready", error);
            state_ = RankState::Loading;
            return true;

        case MessageType::Token:
            if (state_ != RankState::WaitingToken)
                return reject("Token received before rank ready or while step running", error);
            if (m.sequence != next_sequence_)
                return reject("Token sequence gap/replay", error);
            if (m.phase != Phase::Step)
                return reject("Token carries the wrong protocol phase", error);
            if (m.status != 0)
                return reject("Token carries a nonzero status", error);
            if ((m.token_id < -2) || (m.token_id >= spec_.vocab) || (m.token_id == -1))
                return reject("Token id outside vocabulary", error);
            if (!token_id) return reject("Token has no output sink", error);
            *token_id = m.token_id;
            state_ = RankState::StepRunning;
            return true;

        case MessageType::Finish:
            if (state_ != RankState::WaitingToken ||
                m.sequence != next_sequence_ || m.phase != Phase::Shutdown ||
                m.status != 0)
                return reject("Finish received out of order", error);
            state_ = RankState::FinishPending;
            return true;

        default:
            return reject("unexpected coordinator message type", error);
    }
}

bool RankSession::report_nccl_ready(bool success, int status, const std::string& detail,
                                    Message* outbound, std::string* error) {
    if (!outbound) return reject("NCCL result has no outbound sink", error);
    if (state_ != RankState::NeedNcclInit)
        return reject("NCCL result reported out of order", error);
    *outbound = reply(success ? MessageType::NcclReady : MessageType::Failure,
                      Phase::NcclInit);
    outbound->sequence = 0;
    outbound->status = success ? 0 : (status == 0 ? -1 : status);
    outbound->detail = success
        ? detail.substr(0, kMaxDetailBytes)
        : (detail.empty() ? "NCCL initialization failed" : detail).substr(0, kMaxDetailBytes);
    if (success) state_ = RankState::NcclReadySent;
    else { state_ = RankState::Failed; failure_ = outbound->detail; }
    return true;
}

bool RankSession::report_load_ready(bool success, int status, const std::string& detail,
                                    Message* outbound, std::string* error) {
    if (!outbound) return reject("load result has no outbound sink", error);
    if (state_ != RankState::Loading)
        return reject("load result reported out of order", error);
    *outbound = reply(success ? MessageType::LoadReady : MessageType::Failure, Phase::Load);
    outbound->sequence = 0;
    outbound->status = success ? 0 : (status == 0 ? -1 : status);
    outbound->detail = success
        ? detail.substr(0, kMaxDetailBytes)
        : (detail.empty() ? "rank-local model load failed" : detail).substr(0, kMaxDetailBytes);
    if (success) state_ = RankState::WaitingToken;
    else { state_ = RankState::Failed; failure_ = outbound->detail; }
    return true;
}

bool RankSession::report_step_done(bool success, int status, const std::string& detail,
                                   Message* outbound, std::string* error) {
    if (!outbound) return reject("step result has no outbound sink", error);
    if (state_ != RankState::StepRunning)
        return reject("step result reported out of order", error);
    *outbound = reply(success ? MessageType::StepDone : MessageType::Failure, Phase::Step);
    outbound->status = success ? 0 : (status == 0 ? -1 : status);
    outbound->detail = success
        ? detail.substr(0, kMaxDetailBytes)
        : (detail.empty() ? "rank-local forward step failed" : detail).substr(0, kMaxDetailBytes);
    if (success) {
        ++next_sequence_;
        state_ = RankState::WaitingToken;
    } else {
        state_ = RankState::Failed;
        failure_ = outbound->detail;
    }
    return true;
}

bool RankSession::report_finish_ack(Message* outbound, std::string* error) {
    if (!outbound) return reject("FinishAck has no outbound sink", error);
    if (state_ != RankState::FinishPending)
        return reject("FinishAck reported out of order", error);
    *outbound = reply(MessageType::FinishAck, Phase::Shutdown);
    outbound->status = 0;
    state_ = RankState::Finished;
    return true;
}

bool RankSession::failure_message(Phase phase, int status, Message* outbound,
                                  std::string* error) const {
    if (!outbound) { set_error(error, "failure message has no outbound sink"); return false; }
    if (state_ != RankState::Failed || failure_.empty()) {
        set_error(error, "rank has no sticky failure to report");
        return false;
    }
    *outbound = reply(MessageType::Failure, phase);
    outbound->status = status == 0 ? -1 : status;
    outbound->detail = failure_.substr(0, kMaxDetailBytes);
    return true;
}

}  // namespace dist
}  // namespace tp
}  // namespace sparkinfer
