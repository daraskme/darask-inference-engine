// MTP (draft) layer input and bookkeeping (docs/m3.md §5). The MTP row for position j fuses the
// main model's residual streams h_j (after its last layer, before the final mixer) with the
// embedding of token j + 1:
//   R_i = fc_emb(norm_emb(embed(x_{j+1}))) + fc_hidden(norm_hidden(h_j)_i)    (i < HC streams)
// with (1 + w) RMSNorms, norm_hidden over all HC * H values together (sglang qwen4_exp_mtp.py,
// `_fuse_residual_linear_shared`). Then one QSA + MoE layer and the MTP's own final mixer run on R.
#include "decode.cuh"

// en[n] = norm_emb(embed[tok[n]]) [H];  hn[n][i] = norm_hidden(hidden[n])_i [HC][H], i.e. rows
// n * HC + i of a [N * HC][H] matrix for the shared fc_hidden GEMV. hidden rows are `ld` floats
// apart. grid = N blocks of 512 threads.
extern "C" __global__ void __launch_bounds__(512) mtp_fuse_norm(const __nv_bfloat16* __restrict__ embed,
                                                               const int* __restrict__ tok,
                                                               const float* __restrict__ hidden, int ld,
                                                               const float* __restrict__ w_emb,
                                                               const float* __restrict__ w_hidden,
                                                               float* __restrict__ en, float* __restrict__ hn, int H,
                                                               int HC, float eps) {
  TSJ_PDL_ENTRY;
  __shared__ float red[32];
  const int n = blockIdx.x;
  const __nv_bfloat16* e = embed + size_t(tok[n]) * H;
  const float* x = hidden + size_t(n) * ld;
  const int W = HC * H;
  float se = 0.f, sx = 0.f;
  for (int k = threadIdx.x; k < H; k += blockDim.x) {
    const float v = bf16_to_f(e[k]);
    se = fmaf(v, v, se);
  }
  for (int k = threadIdx.x; k < W; k += blockDim.x) sx = fmaf(x[k], x[k], sx);
  const float re = rsqrtf(block_sum(se, red) / float(H) + eps);
  const float rx = rsqrtf(block_sum(sx, red) / float(W) + eps);
  for (int k = threadIdx.x; k < H; k += blockDim.x) en[size_t(n) * H + k] = bf16_to_f(e[k]) * re * (1.f + w_emb[k]);
  for (int k = threadIdx.x; k < W; k += blockDim.x) hn[size_t(n) * W + k] = x[k] * rx * (1.f + w_hidden[k]);
}

// R[n][i][k] = ye[n][k] + yh[n * HC + i][k], plus the per-stream sum-of-squares partials
// ss[n][block][HC] for the next GR read (as embed_streams). grid = (ceil(H / 256), N), 256 threads.
extern "C" __global__ void __launch_bounds__(256) mtp_fuse_out(const float* __restrict__ ye, const float* __restrict__ yh,
                                                              float* __restrict__ R, float* __restrict__ ss, int H,
                                                              int HC) {
  TSJ_PDL_ENTRY;
  __shared__ float red[32];
  const int n = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x;
  float vals[4] = {};
  if (k < H) {
    const float e = ye[size_t(n) * H + k];
    for (int i = 0; i < HC; ++i) {
      vals[i] = e + yh[(size_t(n) * HC + i) * H + k];
      R[(size_t(n) * HC + i) * H + k] = vals[i];
    }
  }
  for (int i = 0; i < HC; ++i) {
    const float sq = block_sum(vals[i] * vals[i], red);
    if (threadIdx.x == 0) ss[(size_t(n) * gridDim.x + blockIdx.x) * HC + i] = sq;
  }
}

// Copies residual row `row` (row = *napply - 1 when napply is set, else 0: the last committed
// token of a verified window) and its statistics: dst [HC * H], dst_ss [rows][HC].
// grid = ceil(HC * H / 256), 256 threads.
extern "C" __global__ void __launch_bounds__(256) mtp_select(const float* __restrict__ R, const float* __restrict__ ss,
                                                            const int* __restrict__ napply, int W, int rows, int HC,
                                                            float* __restrict__ dst, float* __restrict__ dst_ss) {
  TSJ_PDL_ENTRY;
  const int row = napply ? *napply - 1 : 0;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < W) dst[k] = R[size_t(row) * W + k];
  if (blockIdx.x == 0 && threadIdx.x < rows * HC) dst_ss[threadIdx.x] = ss[size_t(row) * rows * HC + threadIdx.x];
}
