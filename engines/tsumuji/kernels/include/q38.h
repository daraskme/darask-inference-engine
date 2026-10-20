// Qwen3.8-Flash-Next (model_type qwen4_exp) shapes as compile-time constants.
// Source: Qwen/Qwen3.8-Flash-Next config.json @ de4b8e4 and transformers qwen4_exp.
// Mirrored in crates/tsumuji-model/src/shape.rs; a unit test checks the two agree.
#pragma once

namespace q38 {
constexpr int H = 2560;             // hidden_size
constexpr int HC = 4;               // hc_count: residual streams
constexpr int HCW = H * HC;         // 10240, width of the hyper-connection residual
constexpr int GR_RANK = 320;        // hc_lowrank
constexpr int L = 48;               // decoder layers; QSA on layers with (i + 1) % 4 == 0
constexpr int VOCAB = 248320;

constexpr int E = 512;              // routed experts
constexpr int TOPK = 10;
constexpr int FF = 640;             // moe_intermediate_size == shared_expert_intermediate_size

constexpr int GDN_QK = 16;          // linear_num_key_heads
constexpr int GDN_V = 48;           // linear_num_value_heads; k head = v head / 3
constexpr int GDN_D = 128;          // key and value head dim
constexpr int GDN_QKV = 2 * GDN_QK * GDN_D + GDN_V * GDN_D;  // 10240, conv channels
constexpr int GDN_CONV = 4;

constexpr int ATT_Q = 24;
constexpr int ATT_KV = 2;
constexpr int ATT_D = 256;
constexpr int ROPE_D = 64;          // partial_rotary_factor 0.25, NeoX halves, theta 1e7
constexpr int IDX_H = 4;            // indexer query heads (one key head)
constexpr int IDX_D = 128;
constexpr int IDX_R = 4;            // tokens averaged per compressed block
constexpr int IDX_TOPB = 512;       // indexer_budget 2048 / IDX_R
constexpr int QSA_MAX_KEYS = IDX_TOPB * IDX_R + IDX_R - 1;  // 2051 incl. the incomplete tail

constexpr int PLE_LAYER = 1;        // 0-based; config ple_layer_ids = [2] is one-based
constexpr int PLE_HEADS = 16;       // (ngram 2, 3) x 8 hashes
constexpr int PLE_ROW = 160;        // ple_embed_dim 2560 / 16
constexpr int PLE_CONV = 4;         // depthwise conv over HCW channels, dilation 3
constexpr int PLE_DILATION = 3;

constexpr int EOS = 248044;
constexpr float RMS_EPS = 1e-6f;
}  // namespace q38
