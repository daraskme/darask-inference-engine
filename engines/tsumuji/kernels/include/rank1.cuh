// Runtime rank-1 adapters (LoRA rank 1: heretic / abliteration edits, crates/tsumuji-engine/src/adapter.rs):
// a matrix's output gets y += s (a . x) b, x being the matrix's input. s = *scale is read on the
// device (0: off), so an adapter can be switched on or off without re-capturing a graph.
#pragma once

struct Rank1 {
  const float* a;      // [in] (routed experts: [E][in]); null: no adapter on this matrix
  const float* b;      // [out] (routed experts: [E][out])
  const float* scale;  // device scalar
};
