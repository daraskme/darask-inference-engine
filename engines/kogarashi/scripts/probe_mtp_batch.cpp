// Compare batched CUDA operators with single-column views of the same inputs.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void fill(ggml_tensor * t, int seed) {
    std::vector<float> values(ggml_nelements(t));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(float((i * 37 + seed * 13) % 10007) * 0.017f);
    }
    std::vector<char> bytes(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, values.data(), bytes.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
}

static bool check(ggml_backend_t backend, ggml_context * ctx, ggml_tensor * batch,
                  const std::vector<ggml_tensor *> & singles, const char * kind, int n, int kv = 0) {
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, batch);
    for (auto * single : singles) {
        ggml_build_forward_expand(graph, single);
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual(ggml_nelements(batch));
    ggml_backend_tensor_get(batch, actual.data(), 0, ggml_nbytes(batch));
    size_t different = 0;
    float max_abs = 0;
    for (size_t col = 0; col < singles.size(); ++col) {
        auto * single = singles[col];
        std::vector<float> expected(ggml_nelements(single));
        ggml_backend_tensor_get(single, expected.data(), 0, ggml_nbytes(single));
        for (size_t i = 0; i < expected.size(); ++i) {
            const float a = actual[col * expected.size() + i], b = expected[i];
            different += std::memcmp(&a, &b, sizeof(float)) != 0 || !std::isfinite(a) || !std::isfinite(b);
            max_abs = std::max(max_abs, std::abs(a - b));
        }
    }
    std::printf("%s n=%d kv=%d bit_differences=%zu max_abs=%g\n", kind, n, kv, different, max_abs);
    return different == 0;
}

static bool matmul(ggml_backend_t backend, int n, bool moe, bool broadcast, ggml_type type) {
    auto * ctx = ggml_init({16*1024*1024, nullptr, true});
    auto * a = ggml_new_tensor_3d(ctx, type, 3072, moe ? 1024 : 256, moe ? 16 : 1);
    auto * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3072, moe ? (broadcast ? 1 : 8) : n, moe ? n : 1);
    auto * ids = moe ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, n) : nullptr;
    auto * out = moe ? ggml_mul_mat_id(ctx, a, b, ids) : ggml_mul_mat(ctx, a, b);
    std::vector<ggml_tensor *> singles;
    for (int col = 0; col < n; ++col) {
        auto * bv = moe ? ggml_view_3d(ctx, b, b->ne[0], b->ne[1], 1, b->nb[1], b->nb[2], col*b->nb[2])
                       : ggml_view_2d(ctx, b, b->ne[0], 1, b->nb[1], col*b->nb[1]);
        auto * iv = moe ? ggml_view_2d(ctx, ids, 8, 1, ids->nb[1], col*ids->nb[1]) : nullptr;
        singles.push_back(moe ? ggml_mul_mat_id(ctx, a, bv, iv) : ggml_mul_mat(ctx, a, bv));
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    fill(a, 1);
    fill(b, 2);
    if (ids) {
        std::vector<int32_t> values(8*n);
        for (int i = 0; i < 8*n; ++i) {
            values[i] = (i*3 + i/8) % 16;
        }
        ggml_backend_tensor_set(ids, values.data(), 0, values.size()*sizeof(int32_t));
    }
    const bool ok = check(backend, ctx, out, singles, moe ? (broadcast ? "moe-broadcast" : "moe") : "f32", n);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static bool attention(ggml_backend_t backend, int n, int kv) {
    auto * ctx = ggml_init({16*1024*1024, nullptr, true});
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, n, 32);
    auto * k = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, 256, kv, 2);
    auto * v = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, 256, kv, 2);
    auto * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, n);
    auto * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/16, 0, 0);
    std::vector<ggml_tensor *> singles;
    for (int col = 0; col < n; ++col) {
        auto * qv = ggml_view_3d(ctx, q, 256, 1, 32, q->nb[1], q->nb[2], col*q->nb[1]);
        auto * mv = ggml_view_2d(ctx, m, kv, 1, m->nb[1], col*m->nb[1]);
        singles.push_back(ggml_flash_attn_ext(ctx, qv, k, v, mv, 1.0f/16, 0, 0));
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    fill(q, 3);
    fill(k, 4);
    fill(v, 5);
    std::vector<ggml_fp16_t> mask(kv*n);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < kv; ++row) {
            mask[col*kv+row] = ggml_fp32_to_fp16(row <= kv-8+col ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(m, mask.data(), 0, mask.size()*sizeof(ggml_fp16_t));
    const bool ok = check(backend, ctx, out, singles, "q8-fa", n, kv);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    auto * backend = ggml_backend_cuda_init(0);
    GGML_ASSERT(backend);
    bool ok = true;
    for (int n : {2, 3, 4}) {
        for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
            std::printf("type=%s\n", ggml_type_name(type));
            ok = matmul(backend, n, true, true, type) && ok;
            ok = matmul(backend, n, true, false, type) && ok;
        }
        ok = matmul(backend, n, false, false, GGML_TYPE_F32) && ok;
        for (int kv : {256, 512, 2048, 8192}) {
            ok = attention(backend, n, kv) && ok;
        }
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
