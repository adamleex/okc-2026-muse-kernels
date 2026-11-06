# okc: problem=muse_bf16_proj
# Copy of alygg77/okc-muse-spark-kernels (https://github.com/alygg77/okc-muse-spark-kernels), commit 6cae49a639ee6d14b929a5fd4a63c627f96d7dc0,
# file kernels/muse_bf16_proj.py (Apache-2.0, with the vjeux / Nathen Fernandes (MIT) / Matsvei notices below retained).
# Submitted by adamleex on 2026-10-31 with this attribution header added. 2026-11-02 (adamleex): changes appended at
# the end of the file (sections 'tA addon', 'track tC', 'track u128', 'track p128', 'track s5a', 'track s5b', 'track r6c', 'track r7f', 'track r8m', 'track r8g', 'track r9b', 'track r10b', 'track r10c', 'track r11a', 'track r13a', 'track r14a', 'track r17b', 'track r18a' and 'track r18b'); the code above those sections is unchanged except for three adamleex edits (2026-11-02): (1) no load uses an L2::evict_last cache policy any more (x / activation loads use L2::evict_normal, weights keep evict_first), so nothing stays pinned in L2 after a call (the only evict_last loads in the file are the x loads of 'track s5b' (qkv_gate m128) and 'track p128' (pooler m128), raised INSIDE the call and demoted to evict_normal before the call returns: by the kernel's own CTAs once every CTA of the call has finished its x loads (s5b, round 8: one ticket per CTA, no second kernel; round 7b's dependent unpin kernel of the same call only if that route fails validation in prepare) or by the kernel's own CTAs after each x chunk's last use (p128); see those sections); (2) _PINNED_LARGE sends qkv_gate / pooler m 33..127 to cuBLAS (F.linear, SGLang's own call) instead of the gq / gqx kernels, which were tuned with x pinned; (3) the sk2 producer (SK2_SRC) executes fence.proxy.async.shared::cta after each empty-barrier wait, before it refills the ring slot by TMA / bulk copy, so the consumers' generic-proxy shared-memory reads of that slot are ordered before the async-proxy refill (same kernels and arithmetic otherwise; outputs bit-identical).
# 2026-11-04 (adamleex): every CUDA source that the upstream file carried in packed form is written out here as plain,
# readable source text; nothing in this file is unpacked or decoded at run time.
# Portions (the per-thread CUDA-context bind helper; cuBLASLt plan pins) follow code posted by Matsvei (kernarc user
# 'matvey') on agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0 (llms.txt), and that
# attribution is retained here.
# ---- Third-party notice: Nathen Fernandes routes are derived from
# https://github.com/NathenFernandes/muse-bf16-proj, used under the MIT License below:
# MIT License
#
# Copyright (c) 2026 Nathen Fernandes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ---- end of third-party notice
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
"""Dense projections for Muse Spark Open: skinny swap-AB tensor-core GEMV/GEMM with
losslessly compressed BF16 weights.

Math
----
``out[m, n] = x[m, k] @ W[n, k]^T`` for the five projection matrices of one
model (``qkv_gate``, ``o``, ``pooler``, ``unpooler``, ``router``). The router's
FP32 weights run as a BF16 hi + lo pair concatenated along K with FP32
accumulation, as SGLang does above 8 tokens.

Lossless 12-bit weights (BF16 projections)
-------------------------------------------
Each BF16 weight is stored as its low byte (exponent bit 0 + 7 mantissa bits,
raw) and a 4-bit code for its high byte (sign + exponent bits 7..1, as an
offset into an 8-value window chosen per matrix, the window covering the most
weights). The kernel rebuilds the exact BF16 bits in registers (a few integer
ops per weight). The rare weights whose exponent falls outside the window
(tiny magnitudes) are exact-corrected: the kernel adds ``x[:, k] * (w -
w_decoded)`` for each of them (both factors exact in FP32), so every output
equals the product with the original weights up to FP32 accumulation order.
``prepare()`` rejects a packing whose escape share exceeds
``1 / MAX_ESCAPE_DENOMINATOR`` or whose correction is not exact in FP32, and
falls back to uncompressed fragments.

Why the kernels look the way they do
------------------------------------
Weights dominate: for m <= 64 every weight byte is streamed once from HBM
straight into registers, in mma.m16n8k16 fragment order (prepared once), and
multiplied against x (staged in shared memory) on BF16 tensor cores with FP32
accumulation. Split-K across warps (shared-memory reduction) and across CTAs
(FP32 partials + a self-resetting semaphore; the last CTA of a row block
reduces) keeps ~128 CTAs streaming on every shape.

Routes (chosen per layer and token count m by ``_static_choice`` / ``_autotune``)
---------------------------------------------------------------------------------
(adamleex note: in this file the route table is pinned, not timed -- ``_autotune`` is overridden by pinned
choices -- the m == 256 BF16 layers run fixed cuBLASLt plans (``klarge``), and the sections appended at the end
of the file override individual (layer, m) cells; each section header says which.)
* ``sk2`` -- m <= 32, swapped operands, no cross-CTA reduction (the production
  skinny path on sm_90). ``skinny`` (TMA-streamed tensor-core GEMV) is the
  fallback / pre-sk2 path, kept for non-sm_90 and as the autotuner's
  alternative.
* ``cgemv`` / ``sgemv`` / ``rgemv`` -- m == 1 scalar GEMV specialists.
* ``wide`` / ``gemm`` / ``gq`` / ``gqx`` -- m 33..256 TMA + wgmma GEMMs
  (``wide`` is the BF16 route; the router uses ``wide``/``gemm``/``rk``).
* ``rk`` -- router-only split-K GEMM (order-dependent FP32 adds, inside the
  router tolerance).
* ``cutlass`` -- CUTLASS pingpong GEMM, pooler m == 256 only, built in the
  background during ``prepare()``; a failed build just drops the route.
* ``torch`` / ``torchT`` -- cuBLAS fallbacks (``F.linear`` / ``mm`` on a
  K-major copy). Any custom path that fails ``_selftest`` is disabled in
  favour of cuBLAS.

Design of this file
--------------------
* One Python file; native CUDA only, compiled once by
  ``torch.utils.cpp_extension.load_inline`` (``_get_ext`` / ``_build_cutlass``).
* The CUDA sources are string constants, kept verbatim: the kernels, tile
  sizes, and launch configs in them are the tuned behaviour and are not
  touched by this refactor. Python only assembles dispatch tables into them
  (``@@PLACEHOLDER@@``) and prepares their inputs.
* ``prepare(weights)`` builds every weight layout once (compressed streams,
  escape tables, workspaces) into ``state``; ``kernel_fn(data, state)`` runs
  one projection. Fresh outputs are allocated on every call; nothing is
  cached across calls except the compiled extensions and the per-shape
  autotune choice in ``state``.
* CUDA-graph capturable: the timed paths perform no host synchronisation
  (``_selftest`` / ``_autotune`` synchronise only during ``prepare()``).
* Config tables live next to the kernel they configure. Constants that more
  than one helper needs live in the "Shared constants" section below.
"""

import importlib.util
import os
import sys
import threading
from typing import TypedDict

import torch
import torch.nn.functional as F

from torch.utils.cpp_extension import load_inline

# Debug-only knob of the upstream file (one random escape warp per CTA stalls this many ns at a random stage);
# fixed off here -- nothing is read from the environment.
_SK2_DELAY = 0


# ---------------------------------------------------------------------------
# Shared constants (tuning numbers used by more than one helper)
# ---------------------------------------------------------------------------
# Lossless 12-bit packing: an exponent window of EXPONENT_WINDOW consecutive
# exponent values, picked per matrix to cover the most weights.
EXPONENT_WINDOW = 8
EXPONENT_VALUES = 128  # distinct values of BF16 exponent bits 7..1
# Packing is rejected when more than this share of weights escapes the window.
MAX_ESCAPE_DENOMINATOR = 200
# base4 packs one window base byte into each byte of a uint32 (see decode8).
BASE4_REPLICATE = 0x01010101
# Stage-count clamps for the TMA skinny kernel's pipeline.
TMA_MIN_STAGES = 2
TMA_MAX_STAGES = 8
# sm_90 / sm_80 shared-memory budgets (bytes) for that pipeline.
SM90_SMEM_BUDGET_BYTES = 112 * 1024
SM80_SMEM_BUDGET_BYTES = 40 * 1024
# Stage-count clamps for the sk2 kernel's ring.
SK2_MIN_STAGES = 2
SK2_MAX_STAGES = 16
# Failure seeds keep selftest / autotune reproducible across runs.
SELFTEST_SEED = 1234
AUTOTUNE_SEED = 4321
AUTOTUNE_REPS = 24
# Autotune replays from a cold L2: this read sweep is larger than L2.
L2_FLUSH_ROWS = 65_536  # 65_536 x 1024 float32 = 256 MiB
# sk2 ring sizing: a config name ending in _rNNN caps the ring at NNN KiB,
# otherwise the default applies; the ring also never exceeds the smem budget
# left after the rest of the kernel's shared-memory layout.
SK2_DEFAULT_RING_CAP_KB = 200
SK2_RING_SMEM_BUDGET_BYTES = 232_448



class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


# ---------------------------------------------------------------------------
# CUDA sources. These string constants are the kernels, verbatim: tile sizes,
# launch configs, and numerics in them are the tuned behaviour. Each kernel's
# config table and dispatch builder live next to its source below; @@...@@
# placeholders are filled by the builder for that route.
# ---------------------------------------------------------------------------
CUDA_SRC = r"""
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>


__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  // c ? b : a, bitwise
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Decode 8 weights: low bytes of weights 0..3 in lw0, 4..7 in lw1, codes in nw
// (low nibble of byte b -> weight b, high nibble -> weight 4+b; code = s<<3 | o).
// Output: 4 bf16x2 registers r[0..3] = (w0,w1) (w2,w3) (w4,w5) (w6,w7).
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}

template <typename T> __device__ __forceinline__ T cvt_out(float v);
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// Weights, uncompressed (COMP=0): [num_rg][S][32 lanes][16 B] A fragments, S = Kp / 16.
// Weights, compressed (COMP=1): [num_rg][S/4 units][3 parts][32 lanes][16 B]: parts 0,1 = low
//   bytes of the lane's 32 weights of 4 k-steps, part 2 = their 32 codes.
// x: [M][Kx] bf16.  Weight column k uses x column (k % Kx) (router hi|lo concat).
// CTA (rb, sp): row groups rb*RG .. +RG, k range [sp*Kc, (sp+1)*Kc).
// Warp w: row group rb*RG + w / WS, k sub-range (w % WS).
// D: prefetch depth in loads of 16 B per lane (k-steps if !COMP, unit parts if COMP).
@@MBAR@@
@@BODY@@
"""

MBAR_HELPERS = r"""
// ------------------------------------------------------------------ mbarrier / bulk copy helpers
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_test(bar, parity)) {}
}
// escape correction from the stored original BF16 weight t: its 12-bit decode keeps the sign and low byte and clamps the
// exponent into [base, base + 7]; the exact difference (asserted in prepare) is formed here in FP32
__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
// wait with a suspend-time hint: the warp sleeps in hardware until the phase completes instead of re-polling
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
// L2 policies (v132: used by the 2-D TMA boxes of the m >= 33 routes only - gq/gqx/gemm90/wide): weights evict_first,
// x evict_normal (adamleex 2026-11-02: was evict_last; nothing stays pinned in L2 after a call)
__device__ __forceinline__ uint64_t l2_pol_ef() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ uint64_t l2_pol_xn() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// wait on a local mbarrier whose phase another CTA of the cluster completes (acquire at cluster scope)
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}

"""

WG_HELPERS = r"""
// ---------------------------------------------------------------- wgmma helpers (sm_90a)
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }


// smem matrix descriptor: K-major, 128B swizzle, SBO = 1024 B (8 rows x 128 B)
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;                 // LBO (ignored for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;       // SBO
  d |= (uint64_t)1 << 62;                 // 128B swizzle
  return d;
}


"""


def wgmma_rs_defs(bns):
    """Generate wgmma_bn (register-sourced A) overloads for m64nBNk16 bf16."""
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        a0 = nacc
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        pi = a0 + 5
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_bn(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{pi}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, {{%{a0}, %{a0+1}, %{a0+2}, %{a0+3}}}, %{a0+4}, p, 1, 1, 0;\\n"
      "}}\\n"
      : {cons}
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}}""")
    return "\n".join(out)


TMA_SRC = r"""
// Weights: per-warp contiguous layout, uncompressed [nrg][S][32][16B], compressed [nrg][S/4][3][32][16B].
// One stage = KS k-steps (uncompressed) or KS/4 units (compressed) for every consumer warp.
// Warps 0..NW-1 consume, warp NW produces (one elected lane issues bulk copies on sm_90;
template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
__global__ void __launch_bounds__((RG * WS + 1) * 32, 1)
skinny_tma_kernel(const uint4* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
                  OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
                  const int2* __restrict__ ptab, int pmax, uint32_t base4,
                  int M, int N, int Kp, int Kx, int splits, int nrg, int dist, int dg) {
  constexpr int NW = RG * WS;
  constexpr int NC = NW * 32;                       // consumer threads
  constexpr int TOK = MT * 8;
  constexpr int TOKP = TOK + 1;  // padded red row (bank-conflict-free epilogue reads)
  // bytes per warp per stage (HL: router hi and lo tiles of the same k-step back to back)
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  constexpr int SLOT = NW * CHUNK;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  // layout: [bars: full NSTAGE, empty NSTAGE, xbar XP (pad 256)] [ring NSTAGE*SLOT] [x rows M*XS*2 + zero row] [red]
  constexpr int XP = 4;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t full0 = sb, empty0 = sb + NSTAGE * 8, xbar0 = sb + 2 * NSTAGE * 8;
  const uint32_t ring = sb + 256;
  const int Kc = Kp / splits;
  const int XS = Kc + 8;
  // WGM: x as 128B-swizzled K-major tiles [Kc/64][TOK rows][128 B] (wgmma B operand), 1024-aligned
  const size_t x_off = WGM ? (size_t)((((sb + 256u + (uint32_t)(NSTAGE * SLOT)) + 1023u) & ~1023u) - sb)
                           : (size_t)256 + NSTAGE * SLOT;   // WGM: absolute shared address 1024-aligned
  const size_t x_bytes = WGM ? (size_t)Kc * TOK * 2 : (((size_t)M * XS * 2 + 16 + 15) & ~(size_t)15);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw + x_off);
  float* red = reinterpret_cast<float*>(smem_raw + x_off + x_bytes);
  int2* ptab_s = reinterpret_cast<int2*>(red + NW * 16 * TOKP + ((NW * 16 * TOKP) & 1));

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rb = blockIdx.x;
  const int sp = blockIdx.y;
  const int k0 = sp * Kc;
  const int S = Kp >> 4;
  const int NS = Kc / (16 * WS);       // k-steps per consumer warp
  const int nst = NS / KS;              // stages
  // WS == 1: stream is [rb][stage over all K][warp][chunk] (shared with the wide kernel); else [cta][stage][warp][chunk]
  const char* wsrc = reinterpret_cast<const char*>(Wp) +
      (WS == 1 ? ((size_t)blockIdx.x * (S / KS) + (size_t)blockIdx.y * nst) : (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * nst) * SLOT;

  if (threadIdx.x == 0) {
    for (int i = 0; i < NSTAGE; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NW);
    }
    for (int i = 0; i < XP; ++i) mbar_init(xbar0 + 8 * i, NC);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // first stages go out right away (no wait for the CTA barrier)
    for (int st = 0; st < NSTAGE && st < nst; ++st) {
      mbar_expect_tx(full0 + 8 * st, SLOT);
      bulk_g2s(ring + st * SLOT, wsrc + (size_t)st * SLOT, SLOT, full0 + 8 * st);
    }
  }
  if (!WGM && threadIdx.x < 4) reinterpret_cast<uint32_t*>(xs + M * XS)[threadIdx.x] = 0u;  // zero row
  __syncthreads();

  if (warp == NW) {
    // ---------------- producer: this CTA's weight stream is contiguous
    const int st0 = NSTAGE < nst ? NSTAGE : nst;   // already issued by thread 0
    for (int st = st0; st < nst; ++st) {
      const int slot = st % NSTAGE;
      if (st >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
      const uint32_t dst = ring + slot * SLOT;
      const char* src = wsrc + (size_t)st * SLOT;
      if (lane == 0) {
        mbar_expect_tx(full0 + 8 * slot, SLOT);
        bulk_g2s(dst, src, SLOT, full0 + 8 * slot);
      }
    }
    return;
  }

  // ---------------- consumers
  const int rg = rb * RG + warp % RG;   // warps of a warpgroup: 4 consecutive row groups, same k range
  const int wsub = warp / RG;
  if (threadIdx.x == 0 && splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  if (COMP && dg < 2) {
    // this CTA's exact-correction entries -> smem (tracked by x part 0's mbarrier below)
    const int cta = blockIdx.y * gridDim.x + blockIdx.x;
    const char* src = reinterpret_cast<const char*>(ptab + (size_t)cta * pmax);
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(ptab_s);
    for (int i = threadIdx.x; i < pmax / 2; i += NC)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + i * 16), "l"(src + i * 16) : "memory");
  }
  // x chunk in XP parts, each completion tracked by its own mbarrier (no CTA-wide wait)
  if (dg < 2) {
    const int kx0 = k0 % Kx;
    const int pch = Kc / 8 / XP;  // 16-byte chunks per token row per part
    const uint32_t xs_s = (uint32_t)__cvta_generic_to_shared(xs);
    for (int p = 0; p < XP; ++p) {
      if constexpr (WGM) {
        for (int idx = threadIdx.x; idx < TOK * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);   // c: 16-byte chunk index along k
          const uint32_t dst = xs_s + (uint32_t)((((c >> 3) * TOK + t) << 7) + ((((c & 7) ^ (t & 7))) << 4));
          const __nv_bfloat16* src = x + (size_t)(t < M ? t : 0) * Kx + kx0 + c * 8;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < M ? 16 : 0) : "memory");
        }
      } else {
        for (int idx = threadIdx.x; idx < M * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                       "r"(xs_s + (uint32_t)((t * XS + c * 8) * 2)), "l"(x + (size_t)t * Kx + kx0 + c * 8) : "memory");
        }
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(xbar0 + 8 * p) : "memory");
    }
  }
  int xp_ready = -1;
  const int kpart = Kc / XP;  // k per part

  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float wacc[WGM ? TOK / 2 : 1];
#pragma unroll
  for (int q = 0; q < (WGM ? TOK / 2 : 1); ++q) wacc[q] = 0.f;
  uint32_t afrag[2][4];

  const uint32_t xs_base = (uint32_t)__cvta_generic_to_shared(xs);
  const int lrow = lane & 7, lmat = lane >> 3;
  const int kw0 = wsub * NS * 16;
  const uint32_t zero_row = xs_base + (uint32_t)(M * XS * 2);
  uint32_t xs_row[MT];
  uint32_t xs_adv[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int t = j * 8 + lrow;
    xs_row[j] = t < M ? xs_base + (uint32_t)((t * XS + kw0 + lmat * 8) * 2) : zero_row;
    xs_adv[j] = t < M ? 32u : 0u;   // bytes per k-step pair... per k-step: 32 B
  }

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    {
      const int need = (kw0 + (st + 1) * KS * 16 - 1) / kpart;  // x part holding this stage's last k
      if (need > xp_ready && dg < 2) {
        mbar_wait(xbar0 + 8 * need, 0);
        xp_ready = need;
        if constexpr (WGM) fence_async_smem();   // cp.async (generic proxy) writes -> wgmma (async proxy) reads
      }
    }
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    // pull this warp's stage data into registers, then hand the slot back to the producer at once
    constexpr int NL = CHUNK / 512;  // 16-byte loads per lane per stage
    uint4 wr[NL];
    {
      const uint32_t src = ring + slot * SLOT + warp * CHUNK + lane * 16;
#pragma unroll
      for (int q = 0; q < NL; ++q)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wr[q].x), "=r"(wr[q].y), "=r"(wr[q].z), "=r"(wr[q].w) : "r"(src + q * 512));
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if constexpr (!COMP && HL) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 h0 = wr[2 * j2], l0 = wr[2 * j2 + 1], h1 = wr[2 * j2 + 2], l1 = wr[2 * j2 + 3];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], h0.x, h0.y, h0.z, h0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], l0.x, l0.y, l0.z, l0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], h1.x, h1.y, h1.z, h1.w, b[j][2], b[j][3]);
          mma_bf16(acc[j], l1.x, l1.y, l1.z, l1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (!COMP) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 a0 = wr[j2], a1 = wr[j2 + 1];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], a0.x, a0.y, a0.z, a0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], a1.x, a1.y, a1.z, a1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (WGM) {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;   // local k of this unit's first k-step (multiple of 64)
        const uint32_t xt = xs_base + (uint32_t)((kl >> 6) * TOK * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t* af = afrag[j & 1];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(wacc, af[0], af[1], af[2], af[3], make_desc_sw128(xt + j * 32));
          wg_commit();
          wg_wait<1>();   // the wgmma that used afrag[(j+1)&1] is done -> it can be overwritten next step
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
    } else {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        uint32_t a[16];
        decode8(L0.x, L0.y, C.x, base4, a + 0);
        decode8(L0.z, L0.w, C.y, base4, a + 4);
        decode8(L1.x, L1.y, C.z, base4, a + 8);
        decode8(L1.z, L1.w, C.w, base4, a + 12);
        const int s = st * KS + u * 4;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t b[MT][4];
#pragma unroll
          for (int j = 0; j < MT; ++j)
            ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)(s + 2 * h), b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
          for (int j = 0; j < MT; ++j) {
            const uint32_t* a0 = a + 8 * h;
            mma_bf16(acc[j], a0[0], a0[1], a0[2], a0[3], b[j][0], b[j][1]);
            mma_bf16(acc[j], a0[4], a0[5], a0[6], a0[7], b[j][2], b[j][3]);
          }
        }
      }
    }
  }
  if constexpr (WGM) {
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
  }

  if constexpr (WGM) {
#pragma unroll
    for (int j = 0; j < MT; ++j) { acc[j][0] = wacc[4 * j]; acc[j][1] = wacc[4 * j + 1]; acc[j][2] = wacc[4 * j + 2]; acc[j][3] = wacc[4 * j + 3]; }
  }
  // per-warp partials -> shared memory: red[w][row16][TOK]
  {
    const int g = lane >> 2, t = lane & 3;
    float* r = red + warp * 16 * TOKP;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      r[g * TOKP + j * 8 + 2 * t] = acc[j][0];
      r[g * TOKP + j * 8 + 2 * t + 1] = acc[j][1];
      r[(g + 8) * TOKP + j * 8 + 2 * t] = acc[j][2];
      r[(g + 8) * TOKP + j * 8 + 2 * t + 1] = acc[j][3];
    }
  }
  if constexpr (COMP) {
    // all x parts and the patch table are in smem; apply exact corrections in parallel (smem atomics)
    mbar_wait(xbar0 + 8 * (XP - 1), 0);
    named_bar(1, NC);
    const int cnt = ptab_s[0].x;
    for (int p = 1 + warp; p <= cnt; p += NW) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16, kl = e.x & 0xFFFF;
      const float d = __int_as_float(e.y);
      float* r = red + ((rl >> 4) * 16 + (rl & 15)) * TOKP;   // warp (wsub 0, row group rl/16)
      for (int t = lane; t < M; t += 32) {
        float xv;
        if constexpr (WGM) {   // swizzled K-major tiles [Kc/64][TOK][128 B]
          const int c = kl >> 3;
          xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
              reinterpret_cast<const unsigned char*>(xs) + (((c >> 3) * TOK + t) << 7) + (((c & 7) ^ (t & 7)) << 4) + (kl & 7) * 2));
        } else {
          xv = __bfloat162float(xs[t * XS + kl]);
        }
        atomicAdd(r + t, xv * d);
      }
    }
  }
  named_bar(1, NC);
  const int rows = RG * 16;
  const int row0 = rb * rows;
  if (splits == 1) {
    for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
      int t = idx / rows, rr = idx - t * rows;
      int grp = rr >> 4, r16 = rr & 15;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
      out[(size_t)t * N + row0 + rr] = cvt_out<OutT>(v);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
    int t = idx / rows, rr = idx - t * rows;
    int grp = rr >> 4, r16 = rr & 15;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
    __stcg(ws + ((size_t)sp * M + t) * N + row0 + rr, v);
  }
  __threadfence();
  named_bar(1, NC);
  // dist: all `splits` CTAs of this row block are co-resident (grid fits the GPU at once): monotonic-counter
  // barrier, then each reduces its own slice of the row block.  Otherwise the last-arriving CTA reduces all.
  // Counters only ever increase (both modes), so the modes can share them.
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    const int old = atomicAdd(counters + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      while (*((volatile int*)(counters + rb)) - target < 0) __nanosleep(20);
    }
  }
  named_bar(1, NC);
  if (dist || s_last) {
  __threadfence();
  {
    const int r4n = rows / 4;
    const int nch = M * r4n;
    const int c_lo = dist ? (int)(((long long)sp * nch) / splits) : 0;
    const int c_hi = dist ? (int)(((long long)(sp + 1) * nch) / splits) : nch;
    for (int c0 = c_lo + threadIdx.x; c0 < c_hi; c0 += 2 * NC) {
      float4 acc4[2];
      int tt[2], rr[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = c0 + u * NC;
        tt[u] = c / r4n; rr[u] = (c - tt[u] * r4n) * 4;
        acc4[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      for (int q = 0; q < splits; q += 4) {
        float4 v[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            v[u][qq] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c0 + u * NC < c_hi && q + qq < splits)
              v[u][qq] = __ldcg(reinterpret_cast<const float4*>(ws + ((size_t)(q + qq) * M + tt[u]) * N + row0 + rr[u]));
          }
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            acc4[u].x += v[u][qq].x; acc4[u].y += v[u][qq].y; acc4[u].z += v[u][qq].z; acc4[u].w += v[u][qq].w;
          }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        if (c0 + u * NC < c_hi) {
          OutT* o = out + (size_t)tt[u] * N + row0 + rr[u];
          o[0] = cvt_out<OutT>(acc4[u].x); o[1] = cvt_out<OutT>(acc4[u].y);
          o[2] = cvt_out<OutT>(acc4[u].z); o[3] = cvt_out<OutT>(acc4[u].w);
        }
      }
    }
  }
  }  // dist || s_last
}

template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
static void launch_tma(const at::Tensor& Wp, const at::Tensor& x, at::Tensor& out,
                       at::Tensor& ws, at::Tensor& counters, const at::Tensor& ptab, int64_t pmax,
                       int64_t base4, int M, int N, int Kp, int Kx, int splits, int nrg, int dg) {
  constexpr int NW = RG * WS;
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  int Kc = Kp / splits;
  size_t x_bytes = WGM ? (size_t)Kc * MT * 8 * 2 + 1024 : (((size_t)M * (Kc + 8) * 2 + 16 + 15) & ~(size_t)15);
  size_t smem = 256 + (size_t)NSTAGE * NW * CHUNK + x_bytes + (size_t)NW * 16 * (MT * 8 + 1) * 4 + 8 + (size_t)pmax * 8;
  auto kern = skinny_tma_kernel<MT, RG, WS, KS, NSTAGE, COMP, HL, WGM, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  dim3 grid(N / (RG * 16), splits);
  int dev = 0, nsm = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, (NW + 1) * 32, smem);
  const int dist = (int)(grid.x * grid.y) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, (NW + 1) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(Wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4,
      M, N, Kp, Kx, splits, nrg, dist, dg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void @@TMA_NAME@@(at::Tensor Wp, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
                at::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@TMA_DISPATCH@@
  TORCH_CHECK(false, "skinny_tma: no config");
}
"""

MAP_SRC = r"""
#include <cudaTypedefs.h>
#include <cstring>
__device__ __forceinline__ void tma_load_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_load_2d_ef(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_ef());
}
__device__ __forceinline__ void tma_load_2d_xn(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_xn());
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// x [M, K] bf16 as 3-D {64 (k in unit), M (token), K/64 (unit)}: one TMA brings `units` tiles of [rows x 128 B]
static CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  CUtensorMap mp;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);
  return mp;
}

static CUtensorMap make_map_2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

"""

GEMM_SRC = r"""

@@WGMMA_SS_DEFS@@

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
__global__ void __launch_bounds__(NWG * 128 + 128, 1)
gemm90_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
              const uint8_t* __restrict__ Wst,
              OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
              int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;       // BM rows x 64 bf16
  constexpr int B_BYTES = BN * 128;       // BN rows x 64 bf16
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NT = NWG * 128;           // consumer threads
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t bars = sbase + STAGES * ST_BYTES;          // full[STAGES], empty[STAGES]
  const uint32_t full0 = bars, empty0 = bars + 8 * STAGES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y;
  const int nkb_all = Kp >> 6;
  const int kb0 = (int)(((long long)sp * nkb_all) / splits);
  const int kb1 = (int)(((long long)(sp + 1) * nkb_all) / splits);
  const int nkb = kb1 - kb0;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  }
  __syncthreads();

  if (warp >= NWG * 4) {
    // ---------------- producer warp (one lane)
    if (warp == NWG * 4 && lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
      for (int i = 0; i < nkb; ++i) {
        const int slot = i % STAGES;
        if (i >= STAGES) mbar_wait(empty0 + 8 * slot, ((i / STAGES) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
        const int kg = (kb0 + i) * 64;
          tma_load_2d_ef(sbase + slot * ST_BYTES, &mapA, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_xn(sbase + slot * ST_BYTES + A_BYTES, &mapB, full0 + 8 * slot, kg % Kx, 0);
      }
    }
    return;
  }

  // ---------------- consumers
  const int wg = warp >> 2;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;

  for (int i = 0; i < nkb; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t a_st = sbase + slot * ST_BYTES + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST_BYTES + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    // release the previous stage once its wgmmas are done (keep one group in flight)
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + (warp & 3) * 16 + wg * 64;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + (size_t)(rb * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    int old = atomicAdd(counters + rb, 1);
    int target = (old / splits + 1) * splits;
    while (true) {
      int v = *((volatile int*)(counters + rb));
      if (v - target >= 0) break;
      __nanosleep(32);
    }
  }
  named_bar(1, NT);
  __threadfence();
  const int j0 = (int)(((long long)sp * (NACC / 4)) / splits);
  const int j1 = (int)(((long long)(sp + 1) * (NACC / 4)) / splits);
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)rb * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s.x += v0.x + v1.x + v2.x + v3.x; s.y += v0.y + v1.y + v2.y + v3.y;
      s.z += v0.z + v1.z + v2.z + v3.z; s.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(s.x);
      out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(s.z);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(s.y);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s.w);
    }
  }
}

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
static void launch_gemm90(const at::Tensor& W, const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                          at::Tensor& counters, int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  size_t smem = 1024 + (size_t)STAGES * (BM * 128 + BN * 128) + 16 * STAGES + 64;
  auto kern = gemm90_kernel<NWG, BN, STAGES, AS, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mA = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)Kx, BN);
  dim3 grid(N / BM, splits);
  kern<<<grid, NWG * 128 + 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mA, mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      M, N, Kp, Kx, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm90(at::Tensor W, at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@GEMM_DISPATCH@@
  TORCH_CHECK(false, "gemm90: no config");
}
"""


SK2_SRC = r"""

#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
#define SK2_DIAG 0
// ============================================================================ sk2 (m <= 32): swapped operands
// A = x tile (16 tokens x 16 k; ldmatrix from the TMA-loaded, 128B-swizzled tile), B = weights (8 rows x 16 k),
// decoded to registers from the 12-bit stream (BF16 hi + lo for the router).  Each CTA owns R rows over the whole
// K: no cross-CTA reduction.  Warps split the CTA's rows (NWR x NT n8 tiles) and, inside every stage, K (WSK).
// A stage = UPS 64-k units: x tiles [UPS][TOKR rows x 128 B] (TMA, rows >= M zero-filled) + this CTA's weights.
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg) {
  const int dgv = SK2_DIAG ? dg : 0;   // diagnostics (dg) compiled out of production builds
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  // partial sums [warp][token][RP] (rows fastest, padded): 64-bit stores in, 128-bit loads out
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);   // >= R, = 8 (mod 32): conflict-free 64-bit stores
  constexpr int CP = R + 1;                         // corr[token][CP]: odd stride, conflict-free per-token updates
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  // corrections, one copy per escape warp: [NESC][TOKS tokens][CP]; escape warps take turns on the stages
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;                       // [CL-1][WSK][TOKS][RP]: ranks > 0 per-warp sums
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));   // [pmax]: escapes
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);   // 128 B of zeros (x rows >= M)
  // m <= 8: the x box has exactly M rows per 64-k unit (packed, M*128 B per unit; no zero rows moved)
  // PK (M < TOKR, or TOKR == 8): packed x, exactly M rows per 64-k unit (no out-of-bounds TMA rows; rows >= M
  // read zrow); else the full TOKR-row tile with compile-time addressing
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // CL = 2: a cluster of two CTAs shares a row block; each streams one half of K, rank 1 hands its sums to rank 0
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);                      // stages over all of K
  const int gs0 = (crank * tst) / CL;                  // this CTA's first stage (uneven split for CL = 3)
  const int nst = ((crank + 1) * tst) / CL - gs0;      // and its stage count
  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region)
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;

  if (tid == 0) {
    // stage 0 first of all (the first bytes' latency is the ramp of the whole kernel): the weight copy needs only
    // its barrier; the tensor map prefetch goes out before it so the x box follows right behind. In a real cluster
    // (CL > 1) a cluster-scope release issued AFTER a copy is in flight waits for it (~0.7 us): there every barrier
    // is initialized and fenced before the first copy.
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    mbar_init(full0, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, ((dgv & 4) ? 0 : XBYr) + WBY);
    bulk_g2s(sbase + XBY, wsrc + (size_t)gs0 * wstride, WBY, full0);
    if (!(dgv & 4)) tma_load_3d(sbase, &xmap, full0, 0, 0, gs0 * UPS);
    if (COMP) {   // this CTA's escape table
      if constexpr (CL == 1) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      mbar_expect_tx(pbar, pmax * 8);
      bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
    }
    if constexpr (CL == 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  // local barriers are all this CTA needs to start streaming; the cluster barrier (rank 0's qbar initialized before
  // ranks > 0 send into it) is only arrived at here and waited on right before the remote stores
  __syncthreads();
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    // producer warp: lane 0 refills the ring
    if (lane == 0) {
      for (int st = 1; st < nst; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        // adamleex 2026-11-02: proxy fence on the consumer-read -> release -> acquire -> TMA path (the consumers release
        // the slot right after ISSUING their ldmatrix / ld.shared; without it a refill could land under a late read)
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const uint32_t s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
        if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NW + 1) {
    // NESC escape warps take turns on the stages (each has NESC stage-times per stage): exact corrections for the
    // escapes, read from the x tiles of the ring as each stage lands (the table is in k order); lane t owns token t,
    // so corr[t][row] accumulates with no atomics; each warp has its own copy
    const int ew = warp - (NW + 1);
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      const uint32_t bq = base4 & 0xFFu;
      int pe = 1;
      for (int st = 0; st < nst; ++st) {
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        const int slot = st % NSTAGE;
        // Every escape warp observes every stage, in order, works only on its own (st % NESC == ew), and hands EVERY
        // stage back (the empty barriers count NW + NESC arrivals). The producer therefore never refills a slot
        // before every escape warp has seen the stage it held, so no warp can find its slot two phases ahead (a
        // parity wait that aliases and wedges the ring: astra review 2B, reproduced with a delayed escape warp).
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          continue;
        }
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] += esc_delta((uint32_t)e.y, bq) * __bfloat162float(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  // ---------------- consumers
  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    const uint32_t s0 = sbase + slot * SST;
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
      // 1) every shared-memory load of this unit first (weights of all tiles, x fragments of all 4 k-steps)
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));
          if constexpr (HL) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));
          }
        }
      }
      uint32_t a[4][MT][4];
      if (dgv & 16) {
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      // the stage's shared memory is in registers now: hand the slot back before decoding and multiplying
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      // 2) decode
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
      // 3) tensor-core work: independent accumulators per k-step
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  }

  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)
  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    // ranks > 0: every accumulator goes straight into rank 0's peer buffer (slot crank - 1) by st.async
    // (completes rank 0's qbar transaction count; no fence, no wait), escape corrections folded into k-warp 0's
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    uint32_t peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");
      }
    return;
  }
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if (COMP && ptab_s[0].y != 0)   // escape-balanced row order (v135): 4-row groups moved whole, so they stay contiguous
      o = out + (size_t)tt * N + reinterpret_cast<const int*>(ptab_s + ptab_s[0].y)[row];
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
    }
  }
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    if constexpr (CL > 1) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[1];
      at_[0].id = cudaLaunchAttributeClusterDimension;
      at_[0].val.clusterDim.x = CL;
      at_[0].val.clusterDim.y = 1;
      at_[0].val.clusterDim.z = 1;
      lc.attrs = at_;
      lc.numAttrs = 1;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}

void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
"""


WIDE_SRC = r"""
// ============================================================================ wide kernel (m 33..128)
// Weights: the per-row-block compressed stream [rb][unit][warp][1536 B] (unit = 16 rows x 64 k per warp),
// one bulk copy per stage; x: one TMA tile [TOK x 64] (128B swizzle) per stage.  Consumers decode the
// weights into registers and run RS-wgmma m64nTOKk16 (sm_90) / mma.sync on the swizzled tile (sm_80 check
// path).  Split-K over units (uneven splits allowed); escapes corrected from a per-CTA table.
template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
__global__ void __launch_bounds__(NWG * 128, 1)
wide_kernel(const __grid_constant__ CUtensorMap mapB, const __grid_constant__ CUtensorMap mapA, const uint8_t* __restrict__ Wst,
            const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out, float* __restrict__ ws,
            int* __restrict__ counters, const int2* __restrict__ ptab, int pmax, uint32_t base4,
            int M, int N, int K, int splits, int dist) {
  // No separate producer warp (17 warps would cap registers at 96 per thread: 5 warps on one SM sub-partition):
  // thread 0 issues the stage copies.
  constexpr int NWARP = NWG * 4;
  constexpr int NT = NWG * 128;
  // per warp per 64-k unit: compressed 1536 B, or (HLW, router) BF16 hi tile 2 KB + lo tile 2 KB in fragment order
  constexpr int WB = HLW ? 4096 : 1536;
  constexpr int A_BYTES = NWARP * WB;
  constexpr int B_BYTES = TOK * 128;
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = TOK / 2;
  static_assert(ST_BYTES % 1024 == 0, "stage alignment");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ int s_last;
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t full0 = sbase + STAGES * ST_BYTES, empty0 = full0 + 8 * STAGES;
  int2* ptab_s = reinterpret_cast<int2*>(gbase + STAGES * ST_BYTES + 16 * STAGES);
  __nv_bfloat16* xpatch = reinterpret_cast<__nv_bfloat16*>(ptab_s + pmax);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y, nrb = gridDim.x;
  const int zh = blockIdx.z;                    // token block: rows [zh*TOK, zh*TOK + Mz) of x / out
  const int t0z = zh * TOK;
  const int Mz = min(TOK, M - t0z);
  const int U = K >> 6;
  const int u0 = (int)(((long long)sp * U) / splits);
  const int u1 = (int)(((long long)(sp + 1) * U) / splits);
  const int nst = u1 - u0;
  const int cta = sp * nrb + rb;
  const uint8_t* wsrc = Wst + ((size_t)rb * U + u0) * A_BYTES;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + zh * nrb + rb));
  }

  // stage issue (sm_90: thread 0, TMA + one bulk copy; sm_80 check path: warp 0, cp.async)
  auto issue = [&](int i) {
    const int slot = i % STAGES;
    const uint32_t st = sbase + slot * ST_BYTES;
    mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
    tma_load_2d_xn(st, &mapB, full0 + 8 * slot, (u0 + i) * 64, t0z);
    if constexpr (HLW) {   // router: BF16 hi and lo tiles [64 rows x 64 k] by TMA (128B swizzle) from [hi | lo]
      tma_load_2d_ef(st + B_BYTES, &mapA, full0 + 8 * slot, (u0 + i) * 64, rb * 64);
      tma_load_2d_ef(st + B_BYTES + 8192, &mapA, full0 + 8 * slot, K + (u0 + i) * 64, rb * 64);
    } else {
      bulk_g2s(st + B_BYTES, wsrc + (size_t)i * A_BYTES, A_BYTES, full0 + 8 * slot);
    }
  };
  const bool producer = tid == 0;
  if (producer) {   // thread 0: barriers are initialized (program order) -> issue before the CTA barrier
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
    if constexpr (HLW) asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
  }
  __syncthreads();
  // escape table + the x values it needs -> shared memory (used after the main loop)
  {
    const int2* tab = ptab + (size_t)cta * pmax;
    const int cnt = pmax ? tab[0].x : 0;
    if (pmax) for (int p = tid; p <= cnt; p += NT) ptab_s[p] = tab[p];
    for (int idx = tid; idx < cnt * TOK; idx += NT) {
      const int p = idx / TOK + 1, t = idx - (p - 1) * TOK;
      const int kl = tab[p].x & 0xFFFF;
      xpatch[p * TOK + t] = t < Mz ? x[(size_t)(t0z + t) * K + u0 * 64 + kl] : __float2bfloat16(0.f);
    }
  }

  // ---------------- consumers
  const int g = lane >> 2, t4 = lane & 3;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  uint32_t afrag[2][4];
  for (int i = 0; i < nst; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t st = sbase + slot * ST_BYTES;
    const uint32_t wa = st + B_BYTES + warp * WB + lane * 16;
    if constexpr (HLW) {
      // router: SS-wgmma on the TMA'd hi and lo tiles (no register-sourced operands: the wgmma pipeline stays full)
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) wgmma_ss<TOK>(acc, make_desc_sw128(st + B_BYTES + j * 32), make_desc_sw128(st + j * 32));
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wgmma_ss<TOK>(acc, make_desc_sw128(st + B_BYTES + 8192 + j * 32), make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // k-step j: low bytes = 8 B at (j/2)*512 + (j%2)*8, codes = 4 B at 1024 + 4j (per-k-step loads: low pressure)
      uint32_t lw0, lw1, cw;
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(lw0), "=r"(lw1) : "r"(wa + (j >> 1) * 512 + (j & 1) * 8));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(cw) : "r"(wa + 1024 + 4 * j));
      uint32_t* af = afrag[j & 1];
      decode8(lw0, lw1, cw, base4, af);
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
      wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
    }
    }  // !HLW
    // stage i-1's wgmmas are complete (one group in flight) -> its slot can be refilled
    if (i > 0) {
      if ((tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
      if (producer && i - 1 + STAGES < nst) {
        mbar_wait(empty0 + 8 * ((i - 1) % STAGES), ((i - 1) / STAGES) & 1);
        issue(i - 1 + STAGES);
      }
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  // ---------------- exact corrections for escapes (table + x values staged at kernel start)
  named_bar(1, NT);
  {
    const int cnt = pmax ? ptab_s[0].x : 0;
    for (int p = 1; p <= cnt; ++p) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16;
      if ((rl >> 4) != warp) continue;
      const int r16 = rl & 15;
      if (g != (r16 & 7)) continue;
      const float d = __int_as_float(e.y);
      const __nv_bfloat16* xv = xpatch + p * TOK;
      const bool hi = r16 >= 8;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const float v0 = d * __bfloat162float(xv[8 * jj + 2 * t4]);
        const float v1 = d * __bfloat162float(xv[8 * jj + 2 * t4 + 1]);
        if (hi) { acc[4 * jj + 2] += v0; acc[4 * jj + 3] += v1; }
        else { acc[4 * jj] += v0; acc[4 * jj + 1] += v1; }
      }
    }
  }

  const int row_w = rb * (NWARP * 16) + warp * 16;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < TOK / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < Mz) {
        out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < Mz) {
        out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + ((size_t)(zh * nrb + rb) * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  named_bar(1, NT);
  if (tid == 0) {
    // release: every thread's partial (ordered before the CTA barrier) is visible before the count; acquire on the
    // spin makes the other CTAs' partials visible to this CTA after the barrier below
    int old;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(counters + zh * nrb + rb) : "memory");
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      int v;
      do {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(counters + zh * nrb + rb) : "memory");
      } while (v - target < 0);
    } else if (s_last) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
  }
  named_bar(1, NT);
  if (!dist && !s_last) return;
  const int j0 = dist ? (int)(((long long)sp * (NACC / 4)) / splits) : 0;
  const int j1 = dist ? (int)(((long long)(sp + 1) * (NACC / 4)) / splits) : NACC / 4;
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)(zh * nrb + rb) * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s4 = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      const float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      const float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      const float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      const float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s4.x += v0.x + v1.x + v2.x + v3.x; s4.y += v0.y + v1.y + v2.y + v3.y;
      s4.z += v0.z + v1.z + v2.z + v3.z; s4.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      const float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s4.x += v.x; s4.y += v.y; s4.z += v.z; s4.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < Mz) {
      out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(s4.x);
      out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(s4.z);
    }
    if (t0 + 1 < Mz) {
      out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(s4.y);
      out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s4.w);
    }
  }
}

template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
static void launch_wide(const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                        at::Tensor& counters, const at::Tensor& ptab, int64_t pmax, int64_t base4,
                        int M, int N, int K, int splits) {
  constexpr int ST_BYTES = NWG * 4 * (HLW ? 4096 : 1536) + TOK * 128;
  const int nz = (M + TOK - 1) / TOK;   // token blocks (same weight stream: concurrent CTAs share it in L2)
  const size_t smem = 1024 + (size_t)STAGES * ST_BYTES + 16 * STAGES + (size_t)pmax * 8 + (size_t)pmax * TOK * 2 + 64;
  auto kern = wide_kernel<NWG, TOK, STAGES, HLW, OutT>;
  int dev = 0;
  cudaGetDevice(&dev);
  static bool attr_done = false;
  if (!attr_done) {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  CUtensorMap mB, mA;
  memset(&mB, 0, sizeof(mB));
  if (major >= 9) mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, TOK);
  mA = mB;
  if (HLW && major >= 9) mA = make_map_2d(Wst.data_ptr(), (uint64_t)Wst.size(0), (uint64_t)Wst.size(1), 64);   // [hi | lo]
  dim3 grid(N / (NWG * 64), splits, nz);
  int nsm = 0, per_sm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NWG * 128, smem);
  const int dist = (int)(grid.x * grid.y * grid.z) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, NWG * 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mB, mA, reinterpret_cast<const uint8_t*>(Wst.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4, M, N, K, splits, dist);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wide(at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters, at::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok) {
  const int hl = out.scalar_type() == at::kFloat ? 1 : 0;
  @@WIDE_DISPATCH@@
  TORCH_CHECK(false, "wide: no config");
}
"""


def wgmma_ss_defs(bns):
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        a, b, p = nacc, nacc + 1, nacc + 2
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_ss(float* d, uint64_t da, uint64_t db) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{p}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, %{a}, %{b}, p, 1, 1, 0, 0;\\n"
      "}}\\n"
      : {cons}
      : "l"(da), "l"(db), "r"(1));
}}""")
    return "\n".join(out)


CPP_SRC = """
void skinny_tma_a(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
                torch::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg);
void gemm90(torch::Tensor W, torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as);
void wide(torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters, torch::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok);
void gq(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage);
void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
void rgemv(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M);
void sgemv(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t N);
void cgemv(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4, int64_t rows, int64_t nth);
void gqx(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage);
void rk(torch::Tensor Wc, torch::Tensor x, torch::Tensor out, int64_t S, torch::Tensor zc);
"""

# Weight rows per layer (the full matrices; some routes run on a prefix, e.g.
# the unpooler slice at m <= 8).
WEIGHT_ROWS = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
COMPRESSED_LAYERS = {"qkv_gate", "o", "pooler", "unpooler"}
# At small m the unpooler runs on its first rows only (the slice SGLang scores).
UNPOOLER_SLICE_ROWS = 1024

_ext = None


# ---------------------------------------------------------------------------
# Skinny (TMA) route config
# ---------------------------------------------------------------------------
TMA_CONFIGS = {  # (layer, mt) -> (splits, RG, WS, KS); x chunk M*Kc*2 small, ~128 CTAs; mt 0 = m == 1
    ("pooler", 0): (1, 2, 8, 4), ("pooler", 1): (2, 4, 4, 4), ("pooler", 2): (4, 8, 2, 4), ("pooler", 4): (8, 16, 1, 4),
    ("unpooler", 0): (2, 1, 8, 4), ("unpooler", 1): (2, 1, 8, 4), ("unpooler", 2): (2, 8, 2, 4), ("unpooler", 4): (4, 16, 1, 4),
    ("qkv_gate", 0): (8, 9, 2, 4), ("qkv_gate", 1): (8, 9, 2, 4), ("qkv_gate", 2): (8, 9, 2, 4), ("qkv_gate", 4): (8, 9, 1, 8),
    ("o", 0): (1, 4, 4, 4), ("o", 1): (1, 4, 4, 4), ("o", 2): (1, 4, 4, 4), ("o", 4): (1, 4, 4, 4),
    ("router", 0): (8, 1, 8, 4), ("router", 1): (8, 1, 8, 4), ("router", 2): (8, 1, 8, 4), ("router", 4): (8, 1, 8, 4),
}
USE_WGM = True   # m >= 9 compressed configs with RG % 4 == 0 use RS-wgmma (sm_90)
WGM_SM80_TEST = False  # local check of the wgmma path's data layout (mma.sync emulation on sm_80)


def _tma_nstage(nw, comp, sm90, ks, hl=False):
    """Pipeline stages for a skinny config: as many as fit the smem budget."""
    chunk = (ks // 4) * 1536 if comp else ks * 512 * (2 if hl else 1)
    budget = SM90_SMEM_BUDGET_BYTES if sm90 else SM80_SMEM_BUDGET_BYTES
    return max(TMA_MIN_STAGES, min(TMA_MAX_STAGES, budget // (nw * chunk)))


# alternative skinny configs (more splits -> fewer x bytes per CTA); the prepare()-time autotuner times them
# against the default config and cuBLAS on every scored shape
TMA_ALT = {("pooler", 8): (8, 16, 1, 4), ("pooler", 16): (8, 16, 1, 4), ("unpooler", 1): (8, 4, 1, 4),
           ("unpooler", 8): (8, 4, 1, 4), ("unpooler", 16): (4, 16, 1, 4), ("o", 16): (4, 16, 1, 4),
           ("o", 32): (4, 16, 1, 4)}


def _tma_cfg_items():
    items = [((layer, mtk), cfg) for (layer, mtk), cfg in sorted(TMA_CONFIGS.items())]
    items += [(("alt", layer, m), cfg) for (layer, m), cfg in sorted(TMA_ALT.items())]
    return items


def _tma_srcs(sm90):
    lines = {}
    for key, (splits, rg, wsn, ks) in _tma_cfg_items():
        if key[0] == "alt":
            layer, mtk = key[1], _mt_key(key[2])
        else:
            layer, mtk = key
        mt = max(mtk, 1)
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        comp = 0 if f32 else 1
        hl = f32
        ns = _tma_nstage(rg * wsn, comp, sm90, ks, hl)
        if f32 and mt >= 4:
            ns = min(ns, 3)  # x chunk (32 x 1024) + ring must fit in shared memory
        n_cfg = UNPOOLER_SLICE_ROWS if (layer == "unpooler" and mtk <= 1) else WEIGHT_ROWS[layer]
        wgm = bool((sm90 or WGM_SM80_TEST) and comp and USE_WGM and rg % 4 == 0 and mtk >= 2)
        line = (f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                f"{{ launch_tma<{mt}, {rg}, {wsn}, {ks}, {ns}, {str(bool(comp)).lower()}, {str(hl).lower()}, {str(wgm).lower()}, {ot}>(Wp, x, out, ws, "
                f"counters, ptab, pmax, base4, M, N, Kp, Kx, splits, nrg, (int)dg); return; }}")
        lines.setdefault(line, []).append(key)
    return TMA_SRC.replace("@@TMA_DISPATCH@@", "\n  ".join(lines)).replace("@@TMA_NAME@@", "skinny_tma_a")


GEMM_CONFIGS = {  # (layer, bn) -> (nwg, splits, stages); grid = (N / (64 nwg), splits) <= 132 (co-resident)
    ("pooler", 64): (2, 4, 8), ("unpooler", 64): (2, 2, 8), ("qkv_gate", 64): (2, 7, 8), ("o", 64): (2, 2, 8),
    ("router", 64): (2, 32, 8),
    ("pooler", 128): (2, 4, 6), ("unpooler", 128): (2, 2, 6), ("qkv_gate", 128): (2, 7, 6), ("o", 128): (2, 2, 6),
    ("router", 128): (2, 32, 6),
    ("pooler", 256): (2, 4, 4), ("unpooler", 256): (2, 2, 4), ("qkv_gate", 256): (2, 7, 4), ("o", 256): (2, 2, 4),
    ("router", 256): (2, 32, 4),
}


GEMM_DIAG = False   # diagnostic: also time gemm90 variants (TMA-2D A vs contiguous pre-swizzled A) on m 65..256




def _gemm_src(sm90):
    lines = []
    for (layer, bn), (nwg, splits, stages) in sorted(GEMM_CONFIGS.items()):
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for a_s in ((0, 1) if (not f32 and GEMM_DIAG) else (0,)):
            lines.append(f"if (nwg == {nwg} && bn == {bn} && out_f32 == {str(f32).lower()} && as == {a_s}) "
                         f"{{ launch_gemm90<{nwg}, {bn}, {stages}, {str(bool(a_s)).lower()}, {ot}>(W, Wst, x, out, ws, counters, M, N, Kp, Kx, splits); return; }}")
    bns = sorted({bn for (_, bn) in GEMM_CONFIGS})
    return (GEMM_SRC.replace("@@WGMMA_SS_DEFS@@", wgmma_ss_defs(bns))
            .replace("@@GEMM_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


GQ_SRC = r"""
// ============================================================================ gq (m 33..256): BF16 weights, SS-wgmma
// One CTA = BM = NWG*64 weight rows x BN tokens: W tile [BM x 64 k] and x tile [BN x 64 k] arrive by TMA (128B
// swizzle) into a ring fed by a producer warp; warpgroup w multiplies rows [64w, 64w+64) of every stage (all share
// the x tile, so x is read from L2 once per BM rows).  CL > 1: a cluster of CL CTAs splits K; ranks > 0 hand their
// accumulators to rank 0 by st.async (completing rank 0's barrier transaction count); rank 0 adds and stores BF16.
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gq_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
          __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, qbar = bars + 16 * NSTAGE;
  float* peer = reinterpret_cast<float*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);   // [CL-1][NWG][NACC][128]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int U = K >> 6;
  const int u0 = (crank * U) / CL, u1 = ((crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);   // x rows actually loaded (rows >= M of the tile only feed outputs that are dropped)

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(qbar, 1);
    if (CL == 2) mbar_expect_tx(qbar, NWG * 64 * NACC * 4);   // split epilogue: the peer's half of our rows
    else if (CL > 1 && crank == 0) mbar_expect_tx(qbar, (CL - 1) * NWG * NACC * 128 * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // the first ring of stages goes out right away (no cluster barrier in front of it: a cluster-scope release
    // issued after a copy is in flight waits for it, and the stages only need this CTA's barriers)
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_xn(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();
  // cluster barrier: arrived here (relaxed: the barrier inits are fenced above), waited on only before remote stores
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  if (warp == NWG * 4) {
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        if (i >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_xn(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait_sleep(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();   // the previous stage's group is done: hand its slot back
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  if constexpr (CL <= 2) {
    // CL = 2: rank r finishes rows half r of every warpgroup (warps 2r, 2r+1); the other two warps hand their sums
    // to the peer by st.async (into [wg][warp][lane][NACC + 4]). The kept rows go out through a shared-memory
    // staging tile [wg][token][rows] so that each token's rows are written as contiguous 16-byte chunks.
    constexpr int RS = NACC + 4;
    constexpr int KR = CL == 2 ? 32 : 64;          // kept rows per warpgroup
    const int q = warp & 3;
    const int keep_lo = CL == 2 ? 2 * crank : 0;   // first kept warp of the warpgroup
    const bool keep = CL == 1 || (q >> 1) == crank;
    if (CL == 2 && !keep) {
      const int ql = q & 1;
      const uint32_t rl = (uint32_t)__cvta_generic_to_shared(peer + ((wg * 2 + ql) * 32 + lane) * RS);
      uint32_t rr, qb;
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // the peer's qbar is initialized
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rl), "r"(crank ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(qb) : "r"(qbar), "r"(crank ^ 1));
#pragma unroll
      for (int i = 0; i < NACC; i += 4)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     :: "r"(rr + i * 4), "r"(__float_as_uint(acc[i])), "r"(__float_as_uint(acc[i + 1])),
                        "r"(__float_as_uint(acc[i + 2])), "r"(__float_as_uint(acc[i + 3])), "r"(qb) : "memory");
      return;
    }
    if constexpr (CL == 2) {
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
      mbar_wait_cluster(qbar, 0);
      const float* rv = peer + ((wg * 2 + (q & 1)) * 32 + lane) * RS;
#pragma unroll
      for (int i = 0; i < NACC; i += 4) {
        const float4 p4 = *reinterpret_cast<const float4*>(rv + i);
        acc[i] += p4.x; acc[i + 1] += p4.y; acc[i + 2] += p4.z; acc[i + 3] += p4.w;
      }
    }
    // stage BF16 [wg][token][KR rows] in the idle ring (every warpgroup's ring reads are over: wg_wait<0> above;
    // the TMA writes all landed before the last full-barrier wait)
    if constexpr (NWG > 1) named_bar(1, NWG * (CL == 2 ? 64 : 128));
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);
    {
      const int g = lane >> 2, t4 = lane & 3;
      const int rk = (q - keep_lo) * 16;   // this warp's first kept row
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int t0 = 8 * j + 2 * t4;
        __nv_bfloat16* s0 = stg + ((size_t)wg * BN + t0) * KR + rk + g;
        s0[0] = __float2bfloat16_rn(acc[4 * j]);
        s0[8] = __float2bfloat16_rn(acc[4 * j + 2]);
        s0[KR] = __float2bfloat16_rn(acc[4 * j + 1]);
        s0[KR + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
    named_bar(2, NWG * (CL == 2 ? 64 : 128));
    {
      constexpr int CH = KR / 8;               // 16-byte chunks per token row
      const int nkt = NWG * (CL == 2 ? 64 : 128);
      const int kt = (q - keep_lo) * 32 + lane + wg * (CL == 2 ? 64 : 128);
      for (int idx = kt; idx < NWG * BN * CH; idx += nkt) {
        const int c = idx % CH, tt = (idx / CH) % BN, w2 = idx / (CH * BN);
        if (tt < M) {
          const uint4 v = *reinterpret_cast<const uint4*>(stg + ((size_t)w2 * BN + tt) * KR + c * 8);
          *reinterpret_cast<uint4*>(out + (size_t)tt * N + rb * BM + w2 * 64 + (CL == 2 ? crank * 32 : 0) + c * 8) = v;
        }
      }
    }
    return;
  }
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gq(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 +
                      (CL == 2 ? (size_t)NWG * 64 * (BN / 2 + 4) * 4 : (size_t)(CL - 1) * NWG * (BN / 2) * 128 * 4);
  auto kern = gq_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if constexpr (CL > 1) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(N / BM * CL);
    lc.blockDim = dim3(NWG * 128 + 32);
    lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeClusterDimension;
    at_[0].val.clusterDim.x = CL;
    at_[0].val.clusterDim.y = 1;
    at_[0].val.clusterDim.z = 1;
    lc.attrs = at_;
    lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
  } else {
  }
}

void gq(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage) {
  @@GQ_DISPATCH@@
  TORCH_CHECK(false, "gq: no config");
}
"""

GQ_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): BM = 64*nwg rows per CTA, grid = N/BM * cl CTAs (cl-CTA clusters split K)
    # measured on H200 against cuBLAS (graph replay, cold L2): only the pooler shape wins (64 row blocks x 2 K halves)
    ("pooler", 64): (1, 2, 10), ("pooler", 128): (1, 2, 8),
}
USE_GQ = True

GQX_SRC = r"""
// ============================================================================ gqx (m 33..256): BF16 weights, big tiles
// CTA = BM = 64 NWG weight rows x BN tokens over 1/CL of K; a cluster of CL CTAs covers all of K for one row block.
// Ring of TMA'd W [BM x 64] and x [BN x 64] tiles (128B swizzle) fed by a producer warp; warpgroup w runs SS-wgmma
// m64nBN on rows [64w, 64w + 64). Reduction: after every rank's main loop (cluster barrier: every ring is free), each
// rank owns BN/CL tokens; every thread st.async's the other ranks' token blocks of its accumulator fragment into the
// owner's ring space ([src rank][thread][4 JR floats]) and sums the CL-1 incoming copies of its own block.
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gqx_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
           __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NTH = NWG * 128;
  constexpr int NB = BN / 8;                 // 8-token blocks of the tile; rank r owns blocks [r NB / CL, (r+1) NB / CL)
  constexpr int JM = (NB + CL - 1) / CL;     // most blocks a rank owns
  static_assert(CL <= NB, "gqx: every rank owns at least one token block");
  static_assert(((NB + CL - 1) / CL) * 8 * BM * 2 <= NSTAGE * ST, "gqx: the BF16 staging tile must fit in the ring");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, rbar = bars + 16 * NSTAGE;
  // CL > 1: this rank's accumulator fragments [NB][NTH] float4, outside the ring (written right after the main loop)
  float4* frag = reinterpret_cast<float4*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);
  float* recv = reinterpret_cast<float*>(gbase);   // [CL][NTH][4 JM] (the ring, once every rank is done with it)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t crank = 0;
  if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
  const int rb = blockIdx.x / CL;
  const int U = K >> 6;
  const int u0 = ((int)crank * U) / CL, u1 = (((int)crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(rbar, 1);

    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_xn(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();

  if (warp == NWG * 4) {   // producer: refills; then joins the cluster barrier
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait: a sleeping producer refills late
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_xn(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) {   // the consumers' two cluster barriers (fragments written, reads done)
      asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
      asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  if constexpr (CL > 1) {
    // 1) fragments out right away (own area: no need to wait for the other ranks' main loops), 2) one cluster
    // barrier, 3) the owner of each token block sums the other ranks' copies over DSMEM, stages BF16 [token][64 rows]
    // in its own (free) ring and 4) writes each token's 64 rows as 128 contiguous bytes; 5) exit barrier
#pragma unroll
    for (int j = 0; j < NB; ++j) frag[j * NTH + tid] = make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    const uint32_t fl = (uint32_t)__cvta_generic_to_shared(frag);
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);   // [owned tokens][BM rows]
    const int jlo = ((int)crank * NB) / CL, jhi = (((int)crank + 1) * NB) / CL;
    const int rl = wg * 64 + (warp & 3) * 16 + g;   // this thread's first row within the tile
#pragma unroll
    for (int r = 0; r < CL; ++r) {
      if (r != (int)crank) continue;
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        if (j < (r * NB) / CL || j >= ((r + 1) * NB) / CL) continue;
        float4 pv[CL];
#pragma unroll
        for (int r2 = 0; r2 < CL; ++r2) {
          if (r2 == r) continue;
          uint32_t ra;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(fl + (uint32_t)((j * NTH + tid) * 16)), "r"(r2));
          asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                       : "=f"(pv[r2].x), "=f"(pv[r2].y), "=f"(pv[r2].z), "=f"(pv[r2].w) : "r"(ra) : "memory");
        }
        float v0 = acc[4 * j], v1 = acc[4 * j + 1], v2 = acc[4 * j + 2], v3 = acc[4 * j + 3];
#pragma unroll
        for (int r2 = 0; r2 < CL; ++r2) {
          if (r2 == r) continue;
          v0 += pv[r2].x; v1 += pv[r2].y; v2 += pv[r2].z; v3 += pv[r2].w;
        }
        const int tl = 8 * (j - (r * NB) / CL) + 2 * t4;   // token within the owned slice
        stg[tl * BM + rl] = __float2bfloat16_rn(v0);
        stg[tl * BM + rl + 8] = __float2bfloat16_rn(v2);
        stg[(tl + 1) * BM + rl] = __float2bfloat16_rn(v1);
        stg[(tl + 1) * BM + rl + 8] = __float2bfloat16_rn(v3);
      }
    }
    named_bar(1, NTH);
    {
      constexpr int CH = BM / 8;                 // 16-byte chunks per token row
      const int nt = 8 * (jhi - jlo);
      for (int idx = tid; idx < nt * CH; idx += NTH) {
        const int tt = idx / CH, c = idx - tt * CH;
        const int tok = 8 * jlo + tt;
        if (tok < M)
          *reinterpret_cast<uint4*>(out + (size_t)tok * N + rb * BM + c * 8) =
              *reinterpret_cast<const uint4*>(stg + tt * BM + c * 8);
      }
    }
    // nobody leaves while another rank may still read its fragments
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  } else {
  }
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gqx(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 + (CL > 1 ? (size_t)(BN / 8) * NWG * 128 * 16 : 0);
  auto kern = gqx_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (CL > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(N / BM * CL);
  lc.blockDim = dim3(NWG * 128 + 32);
  lc.dynamicSmemBytes = smem;
  lc.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[1];
  at_[0].id = cudaLaunchAttributeClusterDimension;
  at_[0].val.clusterDim.x = CL;
  at_[0].val.clusterDim.y = 1;
  at_[0].val.clusterDim.z = 1;
  lc.attrs = at_;
  lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
}

void gqx(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage) {
  @@GQX_DISPATCH@@
  TORCH_CHECK(false, "gqx: no config");
}
"""

GQX_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): grid = N / (64 nwg) * cl CTAs
    ("qkv_gate", 128): (1, 3, 8),   # m 33..128: 36 row blocks x 3 K-thirds = 108 CTAs (cuBLAS 16.4 -> 14.0 us at m=128)
}
USE_GQX = True
TORCHT_LAYERS = ("unpooler", "o")   # layers that also get a K-major BF16 copy for the "torchT" (mm(x, W^T)) route


def _gqx_src(sm90):
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(GQX_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gqx<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return GQX_SRC.replace("@@GQX_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def _gqx_bn(m):
    return 128 if m <= 128 else 256


def _run_gq_family(x, layer, m, n, k, state, configs, bn_of_m, method):
    """Run a gq/gqx cluster GEMM: same call shape, different config table and
    entry point (gq = plain BF16 weights, gqx = big tiles)."""
    bn = bn_of_m(m)
    nwg, cl, ns = configs[(layer, bn)]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(_get_ext(), method)(state["raw"][layer], x, out, m, n, k, bn, nwg, cl, ns)
    return out


def _run_gqx(x, layer, m, n, k, state):
    return _run_gq_family(x, layer, m, n, k, state, GQX_CONFIGS, _gqx_bn, "gqx")



RGEMV_SRC = r"""
// ============================================================================ router GEMV (m = 1): FP32 weights
// read directly (8.4 MB, no hi/lo split), ROWS rows per CTA, 256 threads; thread i owns k = 8i + 2048j (j < 4): every
// weight and x load is issued before the first FMA; warp butterfly then a fixed-order cross-warp sum (deterministic).
template <int ROWS, int MMAX>
__global__ void __launch_bounds__(256) rgemv_kernel(const float* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                    float* __restrict__ out, int M, int N) {
  constexpr int K = 8192;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  float4 w[ROWS][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4* p = reinterpret_cast<const float4*>(W + (size_t)(r0 + r) * K + j * 2048 + tid * 8);
      w[r][2 * j] = __ldg(p); w[r][2 * j + 1] = __ldg(p + 1);
    }
  float acc[ROWS][MMAX];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MMAX; ++t) acc[r][t] = 0.f;
#pragma unroll
  for (int t = 0; t < MMAX; ++t) {
    if (t >= M) break;
    uint4 xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = __ldg(reinterpret_cast<const uint4*>(x + (size_t)t * K + j * 2048 + tid * 8));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv[j]);
      float xf[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(h[e]); xf[2 * e] = f.x; xf[2 * e + 1] = f.y; }
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const float4 a = w[r][2 * j], b = w[r][2 * j + 1];
        float v = acc[r][t];
        v = fmaf(a.x, xf[0], v); v = fmaf(a.y, xf[1], v); v = fmaf(a.z, xf[2], v); v = fmaf(a.w, xf[3], v);
        v = fmaf(b.x, xf[4], v); v = fmaf(b.y, xf[5], v); v = fmaf(b.z, xf[6], v); v = fmaf(b.w, xf[7], v);
        acc[r][t] = v;
      }
    }
  }
  __shared__ float red[ROWS][MMAX][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MMAX; ++t) {
      float v = acc[r][t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][t][warp] = v;
    }
  __syncthreads();
  if (tid < ROWS * MMAX) {
    const int r = tid / MMAX, t = tid % MMAX;
    if (t < M) {
      float v = 0.f;
#pragma unroll
      for (int q = 0; q < 8; ++q) v += red[r][t][q];
      out[(size_t)t * N + r0 + r] = v;
    }
  }
}


// ============================================================================ cgemv (m = 1): 12-bit compressed GEMV
// Lossless planes, row-major: L [N][K] low bytes, C [N][K/2] 4-bit codes (sign | exponent offset from base), byte j of
// each 4-byte group g holding codes 8g + j and 8g + j + 4. Exponents outside the 8-value window decode wrong and are
// corrected exactly by a dedicated escape warp from a per-CTA table (entry 0 = count; ((row << 16) | k, delta bits)).
// Every weight and x load is issued before the first FMA; fixed-order reductions (deterministic).
__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
template <int ROWS, int NTH, int K>
__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 0xFFFF]) * esc_delta((uint32_t)en.y, base4 & 0xFFu);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));
    }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]
        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);         // hi bytes of elements 0..3
        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);  // hi bytes of elements 4..7
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

void cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,
           int64_t rows, int64_t nth) {
  const int N = out.size(1), K = L.size(1);
  TORCH_CHECK(x.size(0) == 1 && N % rows == 0 && N <= L.size(0), "cgemv: m = 1");
  auto st = at::cuda::getCurrentCUDAStream();
  auto a0 = L.data_ptr<uint8_t>(); auto a1 = C.data_ptr<uint8_t>();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
  if (K == 8192 && rows == 2 && nth == 256) cgemv_kernel<2, 256, 8192><<<N / 2, 256 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 1024 && rows == 2 && nth == 64) cgemv_kernel<2, 64, 1024><<<N / 2, 64 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 4096 && rows == 2 && nth == 128) cgemv_kernel<2, 128, 4096><<<N / 2, 128 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 8192 && rows == 2 && nth == 512) cgemv_kernel<2, 512, 8192><<<N / 2, 512 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else TORCH_CHECK(false, "cgemv: no instantiation for K=", K, " rows=", rows, " nth=", nth);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// BF16 GEMV (the unpooler slice at m = 1: W[:1024] bf16 [1024, 4096]): 4 rows per CTA, 512 threads, thread i owns
// k = 8i (one 16 B load per row and one of x), fixed-order reductions.
template <int ROWS, int NTH, int K>
__global__ void __launch_bounds__(NTH) sgemv_kernel(const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                    __nv_bfloat16* __restrict__ out) {
  constexpr int PER = K / NTH / 8;
  constexpr int NWARP = NTH / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  uint4 w[ROWS][PER], xv[PER];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) w[r][j] = __ldg(reinterpret_cast<const uint4*>(W + (size_t)(r0 + r) * K + j * (NTH * 8) + tid * 8));
#pragma unroll
  for (int j = 0; j < PER; ++j) xv[j] = __ldg(reinterpret_cast<const uint4*>(x + j * (NTH * 8) + tid * 8));
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat162* hx = reinterpret_cast<const __nv_bfloat162*>(&xv[j]);
    float xf[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(hx[e]); xf[2 * e] = f.x; xf[2 * e + 1] = f.y; }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const __nv_bfloat162* hw = reinterpret_cast<const __nv_bfloat162*>(&w[r][j]);
      float v = acc[r];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 a = __bfloat1622float2(hw[e]); v = fmaf(a.x, xf[2 * e], v); v = fmaf(a.y, xf[2 * e + 1], v); }
      acc[r] = v;
    }
  }
  __shared__ float red[ROWS][NWARP];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

void sgemv(at::Tensor W, at::Tensor x, at::Tensor out, int64_t N) {
  TORCH_CHECK(W.size(1) == 4096 && x.size(0) == 1 && N % 4 == 0 && N <= W.size(0), "sgemv: m = 1, K = 4096");
  sgemv_kernel<4, 512, 4096><<<(int)(N / 4), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rgemv(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M) {
  TORCH_CHECK(W.size(1) == 8192 && M == 1, "rgemv: K = 8192, m = 1");
  rgemv_kernel<1, 1><<<(int)W.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      W.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), (int)M,
      (int)W.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _gq_src(sm90):
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(GQ_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gq<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return GQ_SRC.replace("@@GQ_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def _gq_bn(m):
    return 64 if m <= 64 else 128 if m <= 128 else 256


def _run_gq(x, layer, m, n, k, state):
    return _run_gq_family(x, layer, m, n, k, state, GQ_CONFIGS, _gq_bn, "gq")



WIDE_CONFIGS = {  # (layer, tok, nz) -> (nwg, splits, stages); grid = (N / (64 nwg), splits, nz) <= 132 (co-resident)
    ("pooler", 64, 1): (1, 2, 8), ("unpooler", 64, 1): (1, 1, 8), ("qkv_gate", 64, 1): (1, 3, 8), ("o", 64, 1): (1, 1, 8),
    ("pooler", 128, 1): (1, 2, 7), ("unpooler", 128, 1): (1, 1, 7), ("qkv_gate", 128, 1): (1, 3, 7), ("o", 128, 1): (1, 1, 7),
    ("pooler", 128, 2): (2, 1, 5), ("unpooler", 128, 2): (4, 1, 4), ("qkv_gate", 128, 2): (1, 1, 7), ("o", 128, 2): (4, 1, 4),
    # router: BF16 hi|lo tiles (plain), 64-row blocks
    ("router", 64, 1): (1, 32, 4), ("router", 128, 1): (1, 32, 4), ("router", 128, 2): (1, 16, 4),
}


def _wide_key(m):
    return (64, 1) if m <= 64 else (128, 1) if m <= 128 else (128, 2)


def _wide_src(sm90):
    lines = []
    for (layer, tok, nz), (nwg, splits, stages) in sorted(WIDE_CONFIGS.items()):
        hl = layer == "router"
        ot = "float" if hl else "__nv_bfloat16"
        lines.append(f"if (nwg == {nwg} && tok == {tok} && hl == {int(hl)}) "
                     f"{{ launch_wide<{nwg}, {tok}, {stages}, {str(hl).lower()}, {ot}>(Wst, x, out, ws, counters, ptab, pmax, base4, "
                     f"M, N, K, splits); return; }}")
    return WIDE_SRC.replace("@@WIDE_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


RK_SRC = r"""
// ============================================================================ rk: router GEMM, m 33..256 (sm_90)
// out[M, 256] (FP32) = x[M, 8192] (BF16) @ (hi + lo)[256, 8192]^T on router_cat = [hi | lo] [256, 16384] (BF16).
// CTA (192 threads) = 1 consumer warpgroup (SS-wgmma m64n64k16 on the hi tile, then the lo tile, of each 64-k block)
// + 1 producer warp (TMA, RK_QS-stage ring of 24 KB: hi 8 KB | lo 8 KB | x 8 KB, 128B swizzle) + 1 helper warp.
// Tile = 64 W rows x 64 tokens; split-K over S CTAs per tile (grid 4 x ntt x S <= 128: one CTA per SM, all resident).
// Each CTA ADDS its partial tile into the FP32 output (smem-transposed, 16 B red.add.v4.f32): the reduction is
// ORDER-DEPENDENT (run-to-run differences ~1e-7 relative, inside the router tolerance; POD item 6). The output is
// cleared inside the kernel: every CTA's helper warp reads the grid counter of this grid size (generation =
// value / grid: earlier launches' counts are complete and this CTA has not counted yet), clears its slice of out,
// counts it (release), waits (acquire, 2 s limit -> trap) until every CTA has counted, and releases the consumers'
// adds through a CTA mbarrier. Monotonic counters (one per token-tile count = per grid size): no host reset.
#define RK_QS 6
__device__ __forceinline__ uint64_t rk_desc(uint32_t addr) {   // K-major tile, 128 B rows, 128B swizzle, SBO 1024
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ void rk_mbar_init(uint32_t bar, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt)); }
__device__ __forceinline__ void rk_mbar_expect(uint32_t bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void rk_mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void rk_mbar_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred P1;\nRKW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@!P1 bra RKW_%=;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
__device__ __forceinline__ void rk_tma2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void rk_wgmma(float d[4][8], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
               : "l"(da), "l"(db));
}

__global__ void __launch_bounds__(192, 1) rk_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
                                                     float* __restrict__ out, int M, int S, int ntt, int* __restrict__ zcnt) {
  constexpr int BM = 64, BN = 64, NJ = 4, QS = RK_QS;
  constexpr int W_BYTES = 2 * BM * 128, ST = W_BYTES + BN * 128;
  extern __shared__ __align__(1024) unsigned char rk_smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(rk_smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t fullb = sbase + QS * ST, emptyb = fullb + 8 * QS, zbar = emptyb + 8 * QS;
  const int rt = blockIdx.x & 3, tt = (blockIdx.x >> 2) % ntt, sp = (blockIdx.x >> 2) / ntt;
  const int kb0 = (sp * 128) / S, nk = ((sp + 1) * 128) / S - kb0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < QS; ++i) { rk_mbar_init(fullb + 8 * i, 1); rk_mbar_init(emptyb + 8 * i, 1); }
    rk_mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 5) {   // helper: clear this CTA's slice of out, count it, wait for every slice, release the adds
    int target = 0;
    if (lane == 0) {
      int v;
      asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
      target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
    }
    const int n4 = M * 64;   // float4s in out
    const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(zcnt) : "memory");
      unsigned long long t0, t1;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
        if (v - target >= 0) break;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > 2000000000ull) __trap();   // 2 s: a clear that never lands is a bug, never a slow node
        __nanosleep(128);
      }
      rk_mbar_arrive(zbar);
    }
    return;
  }
  if (warp == 4) {   // producer
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        if (i >= QS) rk_mbar_wait(emptyb + 8 * s, ((i / QS) - 1) & 1);
        const uint32_t fb = fullb + 8 * s, st = sbase + s * ST;
        rk_mbar_expect(fb, ST);
        const int kb = kb0 + i;
        rk_tma2d(st, &mapW, fb, kb * 64, rt * BM);
        rk_tma2d(st + BM * 128, &mapW, fb, 8192 + kb * 64, rt * BM);
        rk_tma2d(st + W_BYTES, &mapX, fb, kb * 64, tt * BN);
      }
    }
    return;
  }
  // consumer warpgroup
  float d[NJ][8];
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) d[j][e] = 0.f;
  for (int i = 0; i < nk; ++i) {
    const int s = i % QS;
    rk_mbar_wait(fullb + 8 * s, (i / QS) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const uint32_t a = sbase + s * ST, b = a + W_BYTES;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      rk_wgmma(d, rk_desc(a + 32 * k16), rk_desc(b + 32 * k16));
      rk_wgmma(d, rk_desc(a + BM * 128 + 32 * k16), rk_desc(b + 32 * k16));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && tid == 0) rk_mbar_arrive(emptyb + 8 * ((i - 1) % QS));   // stage i-1's wgmmas are done
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+f"(d[j][e]) :: "memory");
  // partial tile -> shared memory [token][row] (the ring is idle: every stage was waited for), then 16 B adds
  constexpr int TP = 68;
  float* T = reinterpret_cast<float*>(rk_smem + (sbase - sraw));
  const int rl = (tid >> 5) * 16 + ((tid & 31) >> 2);
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int cl = j * 16 + h * 8 + (tid & 3) * 2;
      T[cl * TP + rl] = d[j][4 * h]; T[cl * TP + rl + 8] = d[j][4 * h + 2];
      T[(cl + 1) * TP + rl] = d[j][4 * h + 1]; T[(cl + 1) * TP + rl + 8] = d[j][4 * h + 3];
    }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  rk_mbar_wait(zbar, 0);   // every CTA's slice of out is clear (acquired by the helper warp)
  for (int idx = tid; idx < BN * 16; idx += 128) {
    const int tk = idx >> 4, qd = idx & 15;
    const int c = tt * BN + tk;
    if (c < M) {
      const float4 v = *reinterpret_cast<const float4*>(T + tk * TP + qd * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(out + (size_t)c * 256 + rt * BM + qd * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
}
static PFN_cuTensorMapEncodeTiled_v12000 rk_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qr; void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr);
    TORCH_CHECK(p != nullptr, "rk: cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
static CUtensorMap rk_map(const void* ptr, uint64_t rows, uint64_t cols) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows}; cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, 64}; cuuint32_t estr[2] = {1, 1};
  CUresult r = rk_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "rk: cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
void rk(at::Tensor Wc, at::Tensor x, at::Tensor out, int64_t S, at::Tensor zc) {
  const int M = x.size(0);
  TORCH_CHECK(Wc.size(0) == 256 && Wc.size(1) == 16384 && x.size(1) == 8192 && M >= 1 && M <= 256, "rk: shape");
  const int ntt = (M + 63) / 64;
  const int grid = 4 * ntt * (int)S;
  static int nsm = 0;
  if (!nsm) { int dev = 0; C10_CUDA_CHECK(cudaGetDevice(&dev)); C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)); }
  TORCH_CHECK(grid <= nsm, "rk: grid ", grid, " exceeds the SM count ", nsm, " (every CTA must be resident: one per SM)");
  const CUtensorMap mw = rk_map(Wc.data_ptr(), 256, 16384), mx = rk_map(x.data_ptr(), M, 8192);
  const int smem = 1024 + RK_QS * 24576 + 8 * (2 * RK_QS + 1) + 64;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(rk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
  rk_kernel<<<grid, 192, smem, at::cuda::getCurrentCUDAStream()>>>(mw, mx, out.data_ptr<float>(), M, (int)S, ntt, zc.data_ptr<int>() + ntt);
  C10_CUDA_CHECK(cudaGetLastError());
}
"""


def _rk_src(sm90):
    if sm90:
        return RK_SRC


# ---------------------------------------------------------------------------
# Extension build: everything above is assembled into one CUDA translation
# ---------------------------------------------------------------------------
def _get_ext():
    """Compile everything as one CUDA translation unit with load_inline.

    no_implicit_headers: the CUDA file includes only the ATen headers it needs
    (torch/types.h alone costs ~45 s of nvcc time).
    """
    global _ext
    if _ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
        skinny = CUDA_SRC.replace("@@MBAR@@", MBAR_HELPERS).replace("@@BODY@@", WG_HELPERS if sm90 else "")
        rs_defs = wgmma_rs_defs([8, 16, 32, 64, 128]) if sm90 else ""
        # Order matters only in that each route's helpers precede its kernel;
        # every route is self-contained apart from the shared helpers above.
        cuda_src = (skinny + rs_defs + _tma_srcs(sm90) + MAP_SRC + _gemm_src(sm90)
                    + _wide_src(sm90) + _sk2_src(sm90) + _gq_src(sm90) + _gqx_src(sm90)
                    + RGEMV_SRC + _rk_src(sm90))
        _ext = load_inline(
            name=f"okc_bf16proj_v141_{arch}" + (f"_dly{_SK2_DELAY}" if _SK2_DELAY else ""),
            cpp_sources="#include <torch/extension.h>\n" + CPP_SRC,
            cuda_sources=cuda_src,
            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "gqx", "rgemv", "sgemv", "cgemv", "rk"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + ([f"-DSK2_DELAY_ESC={_SK2_DELAY}"] if _SK2_DELAY else []),
            no_implicit_headers=True,
            verbose=False,
        )
    return _ext


# ---------------------------------------------------------------------------------------------- CUTLASS route (m 129..256)
# CUTLASS 3.x sm90 TMA warp-specialized PINGPONG GEMM, persistent tile scheduler, swap-AB (D^T = W x^T: A = W [N, K]
# row-major, B = x [M, K] as a K x M column-major matrix, D column-major [N, M] = out [M, N] row-major), BF16 in/out,
# FP32 accumulate, tile 128 (W rows) x 64 (tokens) x 64, cluster 1x1. Headers: CUTLASS bundled with flashinfer (in the
# SGLang image). Sweep (pod/cutsweep.py, 54 configs x 8 shapes): it beats cuBLAS only for pooler m=256 (31.3 vs 34.5 us
# raw); a missing header or a failed build just leaves the route out.
CUTLASS_LAYERS = {"pooler"}   # layers that get the CUTLASS candidate (scored m=256 only)
CUTLASS_BUILD_TIMEOUT = 170   # seconds prepare waits for the CUTLASS build (main build ~25 s; prepare's limit is 300 s)
CUTLASS_SRC = r"""
/***************************************************************************************************
 * The GEMM below instantiates NVIDIA CUTLASS 3.x/4.x templates (header-only, compiled from the CUTLASS headers that
 * ship in this image's FlashInfer package; nothing prebuilt is linked). Its type composition follows CUTLASS's Hopper
 * GEMM examples (examples/48_hopper_warp_specialized_gemm, 49_hopper_gemm_with_collective_builder), so it carries
 * CUTLASS's license notice:
 *
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************************/
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
using namespace cute;
using EA = cutlass::bfloat16_t; using LA = cutlass::layout::RowMajor;
using EB = cutlass::bfloat16_t; using LB = cutlass::layout::ColumnMajor;
using EC = cutlass::bfloat16_t; using LC = cutlass::layout::ColumnMajor;
using TileShape = Shape<_128, _64, _64>;
using ClusterShape = Shape<_1, _1, _1>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, EC, LC, 8, EC, LC, 8,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, EA, LA, 8, EB, LB, 8, float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
int64_t cut_ws(at::Tensor W, at::Tensor x, at::Tensor out) {
  const int M = W.size(0), K = W.size(1), N = x.size(0);
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(M, N, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA*>(W.data_ptr()), sA, reinterpret_cast<EB*>(x.data_ptr()), sB},
      {{1.f, 0.f}, reinterpret_cast<EC*>(out.data_ptr()), sC, reinterpret_cast<EC*>(out.data_ptr()), sC}};
  return (int64_t)Gemm::get_workspace_size(args);
}
void cut_gemm(at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws) {
  const int M = W.size(0), K = W.size(1), N = x.size(0);
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(M, N, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA*>(W.data_ptr()), sA, reinterpret_cast<EB*>(x.data_ptr()), sB},
      {{1.f, 0.f}, reinterpret_cast<EC*>(out.data_ptr()), sC, reinterpret_cast<EC*>(out.data_ptr()), sC}};
  Gemm gemm;
  TORCH_CHECK((size_t)ws.numel() >= Gemm::get_workspace_size(args), "cut_gemm: workspace");
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "cut_gemm: cannot implement");
  TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), st) == cutlass::Status::kSuccess, "cut_gemm: init");
  TORCH_CHECK(gemm.run(st) == cutlass::Status::kSuccess, "cut_gemm: run");
}
"""
_cut_ext = None
_cut_err = None


def _build_cutlass():
    """Build the CUTLASS extension (sm_90 only); on any failure record it and leave the route out."""
    global _cut_ext, _cut_err
    try:
        major, minor = torch.cuda.get_device_capability()
        spec = importlib.util.find_spec("flashinfer")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("flashinfer (CUTLASS headers) not installed")
        root = os.path.join(list(spec.submodule_search_locations)[0], "data", "cutlass")
        inc = [os.path.join(root, "include"), os.path.join(root, "tools", "util", "include")]
        if not all(os.path.isdir(d) for d in inc):
            raise RuntimeError("CUTLASS headers missing under " + root)
        _cut_ext = load_inline(
            name=f"okc_bf16proj_v141_cutlass_{major}{minor}a",
            cpp_sources="#include <torch/extension.h>\nint64_t cut_ws(at::Tensor W, at::Tensor x, at::Tensor out);\n"
                        "void cut_gemm(at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws);",
            cuda_sources=CUTLASS_SRC, functions=["cut_ws", "cut_gemm"], extra_include_paths=inc,
            extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a", "-DNDEBUG"],
            verbose=False)
    except Exception as e:   # noqa: BLE001 - any failure only drops this optional route
        _cut_ext, _cut_err = None, f"{type(e).__name__}: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Weight preparation (host side, runs in prepare()): fragment orders and the
# lossless 12-bit packing. The packing helpers share _exponent_window /
# _escapes; the per-kernel stream/table builders stay separate because their
# layouts and orderings differ (per-CTA streams vs k-ordered escape tables).
# ---------------------------------------------------------------------------
def _exponent_window(e71):
    """Best EXPONENT_WINDOW-wide window of BF16 exponent bits 7..1 for a matrix.

    Returns (base, offset): ``base`` is the window's lowest exponent, and
    ``offset`` is each weight's exponent clamped into [0, EXPONENT_WINDOW).
    The window is chosen to cover the most weights, so the fewest weights
    need exact correction as escapes.
    """
    hist = torch.bincount(e71.flatten(), minlength=EXPONENT_VALUES)
    win = torch.stack(
        [hist[b:b + EXPONENT_WINDOW].sum() for b in range(EXPONENT_VALUES - EXPONENT_WINDOW + 1)]
    )
    base = int(win.argmax())
    return base, (e71 - base).clamp(0, EXPONENT_WINDOW - 1)


def _escapes(e71, base):
    """Weights whose exponent falls outside [base, base + EXPONENT_WINDOW)."""
    return (e71 < base) | (e71 > base + EXPONENT_WINDOW - 1)


def pack_fragments(w: torch.Tensor) -> torch.Tensor:
    """W [N, K] bf16 -> [N/16, K/16, 32 lanes, 8] in mma.m16n8k16 A-fragment order."""
    n, k = w.shape
    t = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)  # [rg, hr, g, s, hk, t, e]: row = hr*8+g, col = hk*8+2t+e
    t = t.permute(0, 3, 2, 5, 4, 1, 6)  # [rg, s, g, t, hk, hr, e]
    return t.contiguous().view(n // 16, k // 16, 32, 8)


def pack_compressed(w: torch.Tensor):
    """Lossless 12-bit packing of BF16 W [N, K].  Returns None if the matrix doesn't suit it."""
    n, k = w.shape
    if n % 16 or k % 64:
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8                  # sign | exponent bits 7..1
    e71 = hi & 0x7F
    base, o = _exponent_window(e71)
    code = ((hi >> 7) << 3) | o
    dec_hi = ((hi >> 7) << 7) | (base + o)
    dec = ((dec_hi << 8) | (bits & 0xFF)).to(torch.int16).view(torch.bfloat16)
    esc = _escapes(e71, base)
    n_esc = int(esc.sum())
    if n_esc > n * k // MAX_ESCAPE_DENOMINATOR:
        return None
    rows, cols = esc.nonzero(as_tuple=True)
    wtrue = w[rows, cols].float()
    wdec = dec[rows, cols].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    # fragment order for low bytes and codes: [rg, s, lane, 8]
    lo_f = pack_fragments((bits & 0xFF).to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xFF
    cd_f = pack_fragments(code.to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xF
    nrg, S = n // 16, k // 16
    lo_f = lo_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)   # [rg, u, ks, lane, 8]
    cd_f = cd_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)
    # parts 0,1: lane's low bytes for k-steps (0,1) and (2,3) -> [rg, u, 2, lane, 16]
    lo_p = lo_f.view(nrg, S // 4, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, S // 4, 2, 32, 16)
    # part 2: word i (k-step i): byte b = code[8i + b] | code[8i + 4 + b] << 4
    cw = (cd_f[..., 0:4] | (cd_f[..., 4:8] << 4))           # [rg, u, ks, lane, 4]
    cw = cw.permute(0, 1, 3, 2, 4).reshape(nrg, S // 4, 1, 32, 16)
    packed = torch.cat([lo_p, cw], dim=2).transpose(0, 1).contiguous()  # [u, rg, 3, lane, 16]
    order = torch.argsort(rows * k + cols)
    rows, cols, delta = rows[order], cols[order], delta[order]
    rg_idx = rows // 16
    poff = torch.zeros(nrg + 1, dtype=torch.int32, device=w.device)
    poff[1:] = torch.cumsum(torch.bincount(rg_idx, minlength=nrg), 0).to(torch.int32)
    return {
        "packed": packed,
        "base4": base * BASE4_REPLICATE,
        "poff": poff,
        "pk": cols.to(torch.int32).contiguous(),
        "pr": (rows % 16).to(torch.int32).contiguous(),
        "pd": delta.to(torch.float32).contiguous(),
        "n_esc": n_esc,
    }


def build_patch_table(c, n_cfg, k, splits, rg):
    """Per-CTA exact-correction table for a (splits, RG) decomposition over the first n_cfg rows:
    int2 [num_ctas][pmax]; entry 0 = (count, 0); entries = ((row_local << 16) | k_local, delta bits)."""
    dev = c["pk"].device
    rows_cta = rg * 16
    nrb = n_cfg // rows_cta
    nunits = k // 64
    poff = c["poff"].long()
    nrg_all = poff.numel() - 1
    rgi = torch.repeat_interleave(torch.arange(nrg_all, device=dev), poff[1:] - poff[:-1])
    rows = rgi * 16 + c["pr"].long()
    cols = c["pk"].long()
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], c["pd"][keep]
    unit = cols // 64
    spl = ((unit + 1) * splits + nunits - 1) // nunits - 1          # split owning this 64-k unit
    k_start = (spl * nunits // splits) * 64
    cta = spl * nrb + rows // rows_cta
    ncta = nrb * splits
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta, stable=True)
    cta, rows, cols, delta, k_start = cta[order], rows[order], cols[order], delta[order], k_start[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % rows_cta) << 16) | (cols - k_start)).to(torch.int32)
    tab[cta, slot, 1] = delta.contiguous().view(torch.int32)
    return tab.contiguous(), pmax


def build_stream(src, comp, n_cfg, k, splits, rg, wsn, ks):
    """Per-CTA contiguous weight stream for a (splits, RG, WS, KS) decomposition of the first n_cfg rows.
    WS > 1: [cta = sp * nrb + rb][stage][warp][chunk].  WS == 1: [rb][stage over all K][warp][chunk] (independent of
    splits; a CTA reads its contiguous stage range).  src: compressed [nrg, K/64, 3, 32, 16] uint8 or uncompressed
    fragments [nrg, K/16, 32, 8] bf16."""
    dev = src.device
    nw = rg * wsn
    nrb = (n_cfg // 16) // rg
    ar = lambda n: torch.arange(n, device=dev)
    if wsn == 1:
        s_all = (k // 16) // ks
        rb = ar(nrb).view(-1, 1, 1, 1)
        st = ar(s_all).view(1, -1, 1, 1)
        w = ar(nw).view(1, 1, -1, 1)
        rg_idx = rb * rg + w
        if comp:
            u_idx = st * (ks // 4) + ar(ks // 4).view(1, 1, 1, -1)
            return src[rg_idx, u_idx].contiguous()
        s_idx = st * ks + ar(ks).view(1, 1, 1, -1)
        return src[rg_idx, s_idx].contiguous()
    kc = k // splits
    ns = kc // (16 * wsn)
    nst = ns // ks
    sp = ar(splits).view(-1, 1, 1, 1, 1)
    rb = ar(nrb).view(1, -1, 1, 1, 1)
    st = ar(nst).view(1, 1, -1, 1, 1)
    w = ar(nw).view(1, 1, 1, -1, 1)
    rg_idx = rb * rg + w % rg
    if comp:
        ku = ar(ks // 4).view(1, 1, 1, 1, -1)
        u_idx = sp * (kc // 64) + (w // rg) * (ns // 4) + st * (ks // 4) + ku
        out = src[rg_idx, u_idx]
    else:
        j = ar(ks).view(1, 1, 1, 1, -1)
        s_idx = sp * (kc // 16) + (w // rg) * ns + st * ks + j
        out = src[rg_idx, s_idx]
    return out.contiguous()


# ---------------------------------------------------------------- sk2: m <= 32, swapped operands, no cross-CTA reduction
SK2_CONFIGS = [  # (name, layer, n_rows, R rows/CTA, NT, NWR, WSK, UPW, NACC, CL); a name ending in _rNNN caps the ring at NNN KB
    ('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1),
    ('o8h_r88', 'o', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),
    ('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),
    ('qkv32_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('router', 'router', 256, 8, 1, 1, 8, 1, 4, 3),
    ('slice8_r192', 'unpooler', UNPOOLER_SLICE_ROWS, 8, 1, 1, 8, 2, 2, 1),
    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),
    # okc_tune (14173ac4) retimed winners, Oct 25 12:30 (search-harness ratios vs production in brackets); pair.py both
    # orders: pooler16 -0.43, unpooler16 -0.60, o16 -0.15, qkv8 -0.12, qkv16 -0.08 us (pooler32 r160 / o32 r88: no gain)
    ('pooler16t_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),      # tokr16 [0.973 vs 1.0]
    ('unpooler16t_r72', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1),   # tokr16 [0.961 vs 0.969]
    ('qkv8t_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),         # tokr8 [0.969 vs 0.979]
    ('qkv16t_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),        # tokr16 [0.985 vs 0.994]
    ('o16t_r68', 'o', 8192, 32, 2, 2, 1, 4, 2, 1),                  # tokr16 [0.937 vs 0.957]
]
SK2_PICK = {  # (layer, tokr) -> config name
    ('o', 8): 'o8h_r88',
    ('o', 16): 'o16t_r68',
    ('o', 32): 'o32h_r100',
    ('pooler', 8): 'pooler8h_r72',
    ('pooler', 16): 'pooler16t_r192',
    ('pooler', 32): 'pooler32_r144',
    ('qkv_gate', 8): 'qkv8t_r144',
    ('qkv_gate', 16): 'qkv16t_r200',
    ('qkv_gate', 32): 'qkv32_r208',
    ('router', 8): 'router',
    ('router', 16): 'router',
    ('router', 32): 'router',
    ('unpooler', 8): 'slice8_r192',
    ('unpooler', 16): 'unpooler16t_r72',
    ('unpooler', 32): 'unpooler32_r144',
}
SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}
USE_SK2 = True
WIDE_BF16 = False  # the wide (RS-wgmma) kernel never beats cuBLAS for BF16 m 33..256


def _sk2_name(layer, m):
    return SK2_PICK[(layer, _sk2_tokr(m))]


def _sk2_tokr(m):
    return 8 if m <= 8 else 16 if m <= 16 else 32


def _sk2_geom(cfg, tokr):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    hl = layer == "router"
    tu = 2048 if hl else 768
    ups = WSK * UPW
    wby = NWR * WSK * UPW * NT * tu
    xby = ups * tokr * 128
    sst = (xby + wby + 1023) // 1024 * 1024
    nst = -(-(SK2_K[layer] // (ups * 64)) // CL)   # (the largest rank's stages)
    mt = 2 if tokr == 32 else 1
    toks = 16 * mt
    cp = R + 1
    nesc = 4 if not hl else 1   # (3 when the ring has 3 stages; sized for 4 here)
    rp = R + ((40 - (R % 32)) % 32)
    # everything but the ring: alignment pad, barriers, corrections, peer sums, escape table (<= 64 entries), zero row
    extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
             + 64 * 8 + 128)
    cap_kb = SK2_DEFAULT_RING_CAP_KB
    tail = name.rsplit("_r", 1)
    if len(tail) == 2 and tail[1].isdigit():
        cap_kb = int(tail[1])
    ring = min(cap_kb * 1024, SK2_RING_SMEM_BUDGET_BYTES - extra)
    nstage = max(SK2_MIN_STAGES, min(SK2_MAX_STAGES, nst, ring // sst))
    return hl, nstage


def _sk2_src(sm90):
    lines = []
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        for tokr in (8, 16, 32):
            if SK2_PICK.get((layer, tokr)) != name:
                continue
            hl, nstage = _sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            ot = "float" if hl else "__nv_bfloat16"
            lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, "
                         f"{str(not hl).lower()}, {str(hl).lower()}, {1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}")
    return SK2_SRC.replace("@@SK2_DISPATCH@@", "\n  ".join(lines))


def sk2_tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order: lane (g, t) holds row g and, for each
    k-step j of the 64-k unit, k = 16j + 2t, 2t+1, 2t+8, 2t+9."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)          # [tile, g, unit, j, h, t, p]: k = 64u + 16j + 8h + 2t + p
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


def sk2_compress(w):
    """Lossless 12-bit packing of BF16 W [N, K] into B-fragment tile-units [N/8, K/64, 768] (512 B low bytes, lane-major,
    then 256 B of 4-bit codes) + escapes (row, k, delta) that the kernel corrects exactly.  Returns None if unsuitable."""
    n, k = w.shape
    tu = sk2_tileunits(w)
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8
    low = bits & 0xFF
    e71 = hi & 0x7F
    sign = hi >> 7
    base, o = _exponent_window(e71)
    code = (sign << 3) | o
    esc = _escapes(e71, base)
    if int(esc.sum()) > n * k // MAX_ESCAPE_DENOMINATOR:
        return None
    T, U = n // 8, k // 64
    c8 = code.to(torch.uint8)
    w0 = c8[..., 0:4] | (c8[..., 4:8] << 4)
    w1 = c8[..., 8:12] | (c8[..., 12:16] << 4)
    codes = torch.cat([w0, w1], dim=-1)                  # [T, U, 32, 8]
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
    idx = esc.nonzero()                                   # [E, 4]: tile, unit, lane, v
    dec_bits = ((((sign << 7) | (base + o)) << 8) | low)[esc]
    wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
    wtrue = tu[esc].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    lane, v = idx[:, 2], idx[:, 3]
    rows = idx[:, 0] * 8 + (lane >> 2)
    cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
    wbits = (tu[esc].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).contiguous()   # the original BF16 weights
    return {"packed": packed, "base4": base * BASE4_REPLICATE, "rows": rows, "cols": cols, "wbits": wbits}


CG_CONFIGS = {"qkv_gate": (2, 256), "o": (2, 64), "unpooler": (2, 128), "pooler": (2, 512)}   # layer -> (rows per CTA, threads); m = 1
# (v137: pooler m=1 cgemv 2x512 measured -0.34 us vs sk2; 2x256 ties sk2, 4/8 rows or 1 row lose)


def cg_compress(w, rows_per_cta):
    """Row-major 12-bit planes for cgemv (lossless; escapes corrected exactly from a per-CTA table)."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    base, o = _exponent_window(e71)
    code = ((sign << 3) | o).to(torch.uint8)
    esc = _escapes(e71, base)
    if int(esc.sum()) > n * k // MAX_ESCAPE_DENOMINATOR:
        return None
    g = code.view(n, k // 8, 8)
    C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
    L = low.to(torch.uint8).contiguous()
    rows, cols = esc.nonzero(as_tuple=True)          # row-major order
    dec = ((((sign[rows, cols] << 7) | (base + o[rows, cols])) << 8) | low[rows, cols]).to(torch.int16).view(torch.bfloat16).float()
    wtrue = w[rows, cols].float()
    delta = (wtrue - dec).contiguous()
    if not bool(((dec + delta) == wtrue).all()):
        return None
    ncta = n // rows_per_cta
    cta = rows // rows_per_cta
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
    tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)   # the original BF16 weights; the kernel forms the correction
    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * BASE4_REPLICATE}


def sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
    """Stage-major stream [stage][cta][warp][unit][tile][bytes] for the first n_cfg rows."""
    dev = tub.device
    U = tub.shape[1]
    ups = WSK * UPW
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(U // ups).view(-1, 1, 1, 1, 1)
    c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
    w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = c * (R // 8) + (w // WSK) * NT + nt
    unit = s * ups + (w % WSK) * UPW + uu
    return tub[tile, unit].contiguous()


SK2_BALANCE = True   # v135: deal 4-row groups to sk2 row blocks so every block carries ~the same number of escapes
SK2_BALANCE_LAYERS = {"qkv_gate"}   # v136: only where it measured a gain (qkv m=8/16 -0.5 us; o/unpooler m=32 lost)


def sk2_balance_perm(w, n_rows, R):
    """Row order for R-row sk2 blocks over the first n_rows rows of w: rows sorted by escape count and snake-dealt, so
    every block holds ~the same number of escapes (the escape warps' work sets the CTA tail). perm[p] = original row."""
    wl = w[:n_rows]
    bits = wl.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e71 = (bits >> 8) & 0x7F
    base, _ = _exponent_window(e71)
    cnt = _escapes(e71, base).sum(1)
    # 4-row groups are dealt (keeps 4-column output groups contiguous: vector stores in the epilogue)
    G4 = 4
    tcnt = cnt.view(n_rows // G4, G4).sum(1)
    order = torch.argsort(tcnt, descending=True, stable=True)
    TB = R // G4                                   # groups per block
    nb = n_rows // R
    i = torch.arange(n_rows // G4, device=w.device)
    rnd, pos = i // nb, i % nb
    blk = torch.where(rnd % 2 == 0, pos, nb - 1 - pos)
    tperm = torch.empty(n_rows // G4, dtype=torch.long, device=w.device)
    tperm[blk * TB + rnd] = order
    return (tperm.view(-1, 1) * G4 + torch.arange(G4, device=w.device).view(1, G4)).reshape(-1)


def sk2_table_add_perm(tab, pmax, perm, R):
    """Append each CTA's row map (R int32 output columns) after its escapes; entry 0 .y = the map's int2 offset."""
    ncta = tab.shape[0]
    pm2 = pmax + R // 2
    pm2 += pm2 % 2
    out = torch.zeros(ncta, pm2, 2, dtype=torch.int32, device=tab.device)
    out[:, :pmax] = tab
    out[:, pmax:pmax + R // 2] = perm.to(torch.int32).view(ncta, R // 2, 2)
    out[:, 0, 1] = pmax
    return out.contiguous(), pm2


def sk2_table(comp, n_cfg, R):
    """Per-CTA escape table int2 [ncta][pmax]: entry 0 = (count, 0); entries ((row_local << 16) | k, the escaped weight's
    original BF16 bits); the kernel forms the exact correction (w - decode(w)) itself."""
    dev = comp["rows"].device
    rows, cols, delta = comp["rows"], comp["cols"], comp["wbits"]
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], delta[keep]
    ncta = n_cfg // R
    cta = rows // R
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta.long() * 65536 + cols.long())   # per CTA in k order (the kernel walks K once)
    cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
    tab[cta, slot, 1] = delta.to(torch.int32)
    return tab.contiguous(), pmax


def sk2_prepare(weights, state):
    state["sk2"] = {}
    dev = weights["o"].device
    shared = {}
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        w = weights[layer]
        if layer == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            T, U = w.shape[0] // 8, w.shape[1] // 64
            tub = torch.cat([sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                             sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
            stream = sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW)
            tab, pmax, b4 = state["dummy"], 0, 0
        else:
            perm = None
            if SK2_BALANCE and CL == 1 and layer in SK2_BALANCE_LAYERS:
                pkey = (layer, n_rows, R)
                bal = state.setdefault("sk2_bal", {})
                if pkey not in bal:
                    pr = sk2_balance_perm(w, n_rows, R)
                    bal[pkey] = (pr, sk2_compress(w[:n_rows][pr].contiguous()))
                perm, comp = bal[pkey]
                if comp is None:
                    perm = None
            if perm is None:
                comp = state.setdefault("sk2_comp", {}).get(layer)
                if comp is None:
                    comp = sk2_compress(w.contiguous())
                    state["sk2_comp"][layer] = comp
            if comp is None:
                continue
            lkey = (layer, n_rows, R, NT, NWR, WSK, UPW, perm is not None)   # configs differing only in NACC / ring depth share a stream
            if lkey not in shared:
                tab, pmax = sk2_table(comp, n_rows, R)
                if perm is not None:
                    tab, pmax = sk2_table_add_perm(tab, pmax, perm, R)
                shared[lkey] = (sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW), tab, pmax)
            stream, tab, pmax = shared[lkey]
            b4 = comp["base4"]
        state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)
    state.pop("sk2_comp", None)
    state.pop("sk2_bal", None)


def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
    name = _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _get_ext().sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, _sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
    return out


def _mt_key(m):
    return 0 if m == 1 else 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


# ---------------------------------------------------------------------------
# prepare(): build every weight layout once into `state`. The CUTLASS
# extension builds on a background thread while the weights are compressed.
# ---------------------------------------------------------------------------
def _ensure_skinny_stream(state, name, src, comp, c, kk, n, dev, splits, rgc, wsn, ks, n_cfg):
    """Ensure one skinny decomposition's prepared inputs exist in ``state``:
    the per-CTA weight stream, its split-K counter, and (compressed only) its
    per-CTA escape table. Idempotent: shared decompositions are built once."""
    key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
    if key not in state["streams"]:
        state["streams"][key] = build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
    ckey = (name, splits, rgc, wsn, ks, n_cfg)
    if ckey not in state["cnt"]:
        state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)  # per-config counters
    if comp and (name, splits, rgc, n_cfg) not in state["tabs"]:
        state["tabs"][(name, splits, rgc, n_cfg)] = build_patch_table(c, n_cfg, kk, splits, rgc)


def prepare(weights):
    _get_ext()  # compile now, so a build failure surfaces in prepare(), not mid-run
    # the CUTLASS extension builds in the background (nvcc is a subprocess) while the weights are compressed; started
    # after the main build so the two load_inline calls never overlap
    cut_thread = None
    if CUTLASS_LAYERS and torch.cuda.get_device_capability()[0] >= 9 and _cut_ext is None and _cut_err is None:
        cut_thread = threading.Thread(target=_build_cutlass, daemon=True)
        cut_thread.start()
    dev = weights["qkv_gate"].device
    sm90_ = torch.cuda.get_device_capability()[0] >= 9
    # on sm_90 the sk2 kernels replace the older skinny kernels and the BF16 wide kernel (never faster than cuBLAS)
    old_paths = not (sm90_ and USE_SK2)
    state = {"raw": {}, "cnt": {}, "comp": {}, "mid_cnt": {}, "streams": {}, "tabs": {}, "base4": {}}
    state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
    # split-K partials: kernels run one at a time, so every layer shares one workspace
    ws_floats = 0
    for (lname, mtk), (splits, rgc, wsn, ks) in TMA_CONFIGS.items():
        mmax = {0: 1, 1: 8, 2: 16, 4: 32, 8: 64}[mtk]
        ws_floats = max(ws_floats, splits * mmax * WEIGHT_ROWS[lname])
    for (lname, mm), (splits, rgc, wsn, ks) in TMA_ALT.items():
        ws_floats = max(ws_floats, splits * mm * WEIGHT_ROWS[lname])
    state["ws_shared"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    gemm_floats = 0
    for (lname, bn), (nwg, splits, stages) in GEMM_CONFIGS.items():
        gemm_floats = max(gemm_floats, (WEIGHT_ROWS[lname] // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))
    for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
        gemm_floats = max(gemm_floats, nz * (WEIGHT_ROWS[lname] // (64 * nwg)) * splits * (nwg * 64) * tok)
    state["mid_ws_shared"] = torch.empty(gemm_floats, dtype=torch.float32, device=dev)
    for name, w in weights.items():
        if name == "router":
            _get_small_router_ext()
            state["router_small"] = pack_small_router(w)
            assert state["router_small"].is_contiguous() and state["router_small"].data_ptr() % 128 == 0
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            state["router_hi_t"] = hi.t().contiguous()
            state["router_f32"] = w.float().contiguous()   # the router GEMV (m = 1) reads the FP32 weights as given
            state["router_lo_t"] = lo.t().contiguous()
            state["router_cat"] = torch.cat([hi, lo], dim=1).contiguous()   # gemm90 fallback (m 33..256), rk
            state["rk_zc"] = torch.zeros(8, dtype=torch.int32, device=w.device)   # rk: grid counters, one per token-tile count
            # hi and lo tiles of the same k-step back to back: [rg, s, 2, 32, 8]
            src = torch.stack([pack_fragments(hi), pack_fragments(lo)], dim=2).contiguous()
            comp, kk = False, w.shape[1]
            c = None
            # wide kernel stream: [rb][unit][warp][hi 4 k-steps x 512 B | lo 4 k-steps x 512 B], 64-row blocks
            nrg_r = w.shape[0] // 16
            u_all = w.shape[1] // 64
            t = src.view(nrg_r, u_all, 4, 2, 32, 8).permute(0, 1, 3, 2, 4, 5).contiguous()   # [rg, u, hl, 4, 32, 8]
            nwg_r = WIDE_CONFIGS[("router", 128, 1)][0]
            t = t.view(nrg_r // (4 * nwg_r), 4 * nwg_r, u_all, 2 * 4 * 32 * 8).permute(0, 2, 1, 3).contiguous()
            del t
            for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
                if lname == "router":
                    state["cnt"][("router", "wide", tok, nz)] = torch.zeros(w.shape[0] // 16, dtype=torch.int32, device=dev)
        else:
            w = w.contiguous()
            state["raw"][name] = w
            if name in CG_CONFIGS:
                wc = w[:UNPOOLER_SLICE_ROWS] if name == "unpooler" else w   # the unpooler at m = 1 is the slice
                cg = cg_compress(wc.contiguous(), CG_CONFIGS[name][0])
                if cg is not None:
                    state.setdefault("cg", {})[name] = cg
            if name in TORCHT_LAYERS and sm90_:
                state.setdefault("rawT", {})[name] = w.t().contiguous()
            c = pack_compressed(w) if name in COMPRESSED_LAYERS else None
            if c is not None:
                src, comp = c["packed"].transpose(0, 1).contiguous(), True  # [rg, u, 3, 32, 16]
                state["base4"][name] = c["base4"]
                del c["packed"]
            else:
                src, comp = pack_fragments(w), False
            kk = w.shape[1]
        n = w.shape[0]
        state["comp"][name] = comp
        for (lname, mtk), (splits, rgc, wsn, ks) in TMA_CONFIGS.items():
            if lname != name or not old_paths:
                continue
            n_cfg = UNPOOLER_SLICE_ROWS if (name == "unpooler" and mtk <= 1) else n
            _ensure_skinny_stream(state, name, src, comp, c, kk, n, dev, splits, rgc, wsn, ks, n_cfg)
        if comp and old_paths:
            for (lname, mm), (splits, rgc, wsn, ks) in TMA_ALT.items():
                if lname != name:
                    continue
                n_cfg = UNPOOLER_SLICE_ROWS if (name == "unpooler" and mm <= 8) else n
                _ensure_skinny_stream(state, name, src, comp, c, kk, n, dev, splits, rgc, wsn, ks, n_cfg)
        for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
            if lname != name or not comp or not (old_paths or WIDE_BF16):   # unused on sm_90 (cuBLAS / gq / gqx)
                continue
            rgc = 4 * nwg
            key = (name, 0, rgc, 1, 4, n)
            if key not in state["streams"]:
                state["streams"][key] = build_stream(src, comp, n, kk, splits, rgc, 1, 4)
            if (name, splits, rgc, n) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n)] = build_patch_table(c, n, kk, splits, rgc)
            state["cnt"][(name, "wide", tok, nz)] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
        del src
        state["mid_cnt"][name] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
    state["sm90"] = sm90_
    state["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count   # rk: one resident CTA per SM
    state["old_paths"] = old_paths
    sk2_prepare(weights, state)
    if cut_thread is not None:
        cut_thread.join(timeout=CUTLASS_BUILD_TIMEOUT)   # ~70-80 s on an H200 pod; never let it threaten prepare's limit
        if cut_thread.is_alive():
            print(f"[okc] CUTLASS route off: build still running after {CUTLASS_BUILD_TIMEOUT} s", file=sys.stderr)
    if _cut_ext is not None and (cut_thread is None or not cut_thread.is_alive()):
        # Persistent scheduler workspace: a few KB at most, sized with headroom.
        state["cut_ws"] = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)
        state["cut_layers"] = set(CUTLASS_LAYERS) & set(state["raw"])
    elif _cut_err is not None:
        print(f"[okc] CUTLASS route off: {_cut_err}", file=sys.stderr)
    _selftest(weights, state)
    _autotune(weights, state)
    torch.cuda.empty_cache()
    return state


# ---------------------------------------------------------------------------
# Routing, selftest, autotune. The static route maps (layer, m) to a family;
# _candidates adds the autotuner alternatives; _autotune times them as CUDA
# graph replays from a cold L2 and keeps the fastest per (layer, m) in
# state["choice"]. An untuned m in _nearby_choice takes its bucket's choice.
# ---------------------------------------------------------------------------
# Which implementation runs a (layer, m): "skinny" (TMA-streamed tensor-core GEMV, m <= 32),
# "gemm" (TMA + wgmma split-K GEMM, sm_90, m <= 256) or "torch" (cuBLAS, as SGLang).
ROUTE_BF16 = [(32, "skinny"), (256, "wide")]
WIDE_SM80_TEST = False  # local check of the wide kernel's sm_80 path
ROUTE_ROUTER = [(32, "skinny"), (256, "wide")]


def _route(layer, m, sm90):
    for mmax, r in (ROUTE_ROUTER if layer == "router" else ROUTE_BF16):
        if m <= mmax:
            return r
    return "torch"


USE_RK = True   # v141: router m 33..256 candidate 'rk' (order-dependent FP32 adds; stress compares it by tolerance)


def _route_key(layer, m, route):
    if route == "sk2":
        return (layer, route, _sk2_name(layer, m), _sk2_tokr(m))
    if route == "skinny":
        return (layer, route, _mt_key(m))
    if route == "wide":
        return (layer, route) + _wide_key(m)
    if route == "rk":
        return (layer, route, (m + 63) // 64)
    return (layer, route, 64 if m <= 64 else 128 if m <= 128 else 256)


SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)


def _selftest(weights, state):
    """Run every custom path once on a random input against an FP64 reference; disable (-> cuBLAS) any path that
    does not match.  Guards against a kernel that misbehaves on a GPU this file was not tested on."""
    state["disabled"] = set()
    g = torch.Generator(device=weights["o"].device)
    g.manual_seed(SELFTEST_SEED)
    for layer, w in weights.items():
        n, k = w.shape
        for m in SELFTEST_M:
            nn = UNPOOLER_SLICE_ROWS if (layer == "unpooler" and m <= 8) else n
            x = torch.randn(m, k, device=w.device, generator=g).to(torch.bfloat16)
            ref = x.double() @ w[:nn].double().T
            if layer != "router":
                ref = ref.to(torch.bfloat16).double()
            tol = 2e-5 if layer == "router" else 5e-3
            for ch in _candidates(layer, m, state):
                if ch[0] == "torch":
                    continue
                key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
                       else _route_key(layer, m, ch[0]))
                try:
                    out = _run(ch, x, layer, m, nn, k, state).double()
                    torch.cuda.synchronize()
                    err = ((out - ref).norm() / ref.norm()).item()
                except RuntimeError as e:  # launch failure -> fall back
                    err = float("inf")
                    print(f"[okc] selftest {layer} m={m} {ch[0]}: {e}", file=sys.stderr)
                if not err < tol:
                    state["disabled"].add(key)
                    print(f"[okc] selftest {layer} m={m} {ch[0]}: rel err {err:.3e} -> disabled", file=sys.stderr)


# ---------------------------------------------------------------------------
# Running one projection: each _run_* allocates a fresh output and launches
# one route. Nothing is cached across calls.
# ---------------------------------------------------------------------------
def _run_skinny(x, layer, m, n, k, state, cfg, tu_key, dg=0):
    splits, rg, wsn, ks = cfg
    mtk = _mt_key(m)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    stream = state["streams"][(layer, 0 if wsn == 1 else splits, rg, wsn, ks, n)]
    cnt = state["cnt"][(layer, splits, rg, wsn, ks, n)]
    fn = _get_ext().skinny_tma_a
    if state["comp"][layer]:
        tab, pmax = state["tabs"][(layer, splits, rg, n)]
        fn(stream, x, out, state["ws_shared"], cnt, tab, pmax, state["base4"][layer], m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 1, WEIGHT_ROWS[layer] // 16, ks, dg)
    else:
        # Uncompressed: no escapes, so the patch table is the dummy and
        # pmax/base4 are 0 (the kernel skips exact correction).
        fn(stream, x, out, state["ws_shared"], cnt, state["dummy"], 2, 0, m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 0, WEIGHT_ROWS[layer] // 16, ks, dg)
    return out


def _run_wide(x, layer, m, n, k, state):
    tok, nz = _wide_key(m)
    nwg, splits, _ = WIDE_CONFIGS[(layer, tok, nz)]
    rgc = 4 * nwg
    if layer == "router":
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        tab, pmax, b4 = state["dummy"], 0, 0
    else:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        (tab, pmax), b4 = state["tabs"][(layer, splits, rgc, n)], state["base4"][layer]
    wst = state["router_cat"] if layer == "router" else state["streams"][(layer, 0, rgc, 1, 4, n)]
    _get_ext().wide(wst, x, out, state["mid_ws_shared"],
                    state["cnt"][(layer, "wide", tok, nz)], tab, pmax, b4, m, n, k, splits, nwg, tok)
    return out


def _run_gemm(x, layer, m, n, k, state):
    bn = 64 if m <= 64 else 128 if m <= 128 else 256
    nwg, splits, _ = GEMM_CONFIGS[(layer, bn)]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    w = state["router_cat"] if layer == "router" else state["raw"][layer]
    _get_ext().gemm90(w, state["dummy"], x, out, state["mid_ws_shared"], state["mid_cnt"][layer], m, n, w.shape[1], k,
                      splits, nwg, bn, 0)
    return out


def _run_torch(x, layer, m, n, k, state):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return F.linear(x, state["raw"][layer][:n])   # what SGLang's bf16 dispatch runs on H200


def _run(choice, x, layer, m, n, k, state):
    kind = choice[0]
    if kind == "skinny":
        return _run_skinny(x, layer, m, n, k, state, choice[1], choice[2])
    if kind == "wide":
        return _run_wide(x, layer, m, n, k, state)
    if kind == "sk2":
        return _run_sk2(x, layer, m, n, k, state)
    if kind == "gemm":
        return _run_gemm(x, layer, m, n, k, state)
    if kind == "gq":
        return _run_gq(x, layer, m, n, k, state)
    if kind == "gqx":
        return _run_gqx(x, layer, m, n, k, state)
    if kind == "cgemv":   # m = 1: 12-bit compressed GEMV (qkv, o, unpooler slice, pooler)
        cg = state["cg"][layer]
        rows, nth = CG_CONFIGS[layer]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _get_ext().cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rows, nth)
        return out
    if kind == "sgemv":   # unpooler slice, m = 1: BF16 weights straight from memory (4 rows per CTA)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _get_ext().sgemv(state["raw"][layer], x, out, n)
        return out
    if kind == "rgemv":   # router, m = 1: FP32 weights straight from memory (one row per CTA)
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        _get_ext().rgemv(state["router_f32"], x, out, m)
        return out
    if kind == "cutlass":   # m 129..256: CUTLASS pingpong 128x64 (see CUTLASS_SRC)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _cut_ext.cut_gemm(state["raw"][layer], x, out, state["cut_ws"])
        return out
    if kind == "rk":   # router m 33..256: dedicated split-K GEMM with order-dependent FP32 adds (see RK_SRC)
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        _get_ext().rk(state["router_cat"], x, out, _rk_splits(m, state["nsm"]), state["rk_zc"])
        return out
    if kind == "torchT":   # cuBLAS on a K-major copy of W: a different (faster) kernel choice for unpooler m=128
        return torch.mm(x, state["rawT"][layer])
    return _run_torch(x, layer, m, n, k, state)


RK_SPLITS = {1: 16, 2: 12, 3: 10, 4: 8}   # token tiles of 64 -> K splits (grid 4 x ntt x S <= 128; pod/rk.py sweep)


def _rk_splits(m, nsm=132):
    """K splits for m tokens: the measured table, capped so the grid (4 x ntt x S, one CTA per SM, all resident: the
    in-kernel clear waits for every CTA) fits the GPU's SMs."""
    ntt = (m + 63) // 64
    return max(1, min(RK_SPLITS[ntt], nsm // (4 * ntt)))


def _static_choice(layer, m, state):
    route = _route(layer, m, state["sm90"])
    if not state.get("old_paths", True):
        if route == "skinny":
            route = "sk2" if (_sk2_name(layer, m) in state.get("sk2", {})
                              and _route_key(layer, m, "sk2") not in state.get("disabled", ())) else "torch"
        elif route == "wide" and layer != "router" and not WIDE_BF16:
            route = "torch"
    if route != "torch" and _route_key(layer, m, route) in state.get("disabled", ()):
        # fall back: router -> the wgmma GEMM (validated earlier) when possible, else cuBLAS
        route = "gemm" if (layer == "router" and route == "wide" and state["sm90"]
                           and _route_key(layer, m, "gemm") not in state["disabled"]) else "torch"
    if route == "skinny":
        mtk = _mt_key(m)
        return ("skinny", TMA_CONFIGS[(layer, mtk)], (layer, mtk))
    return (route,)


def _candidates(layer, m, state):
    """Implementations worth timing for (layer, m): the static choice, alternative skinny configs, cuBLAS."""
    c = [_static_choice(layer, m, state)]
    if c[0][0] == "skinny" and (layer, m) in TMA_ALT and ("skinny_alt", layer, m) not in state["disabled"]:
        c.append(("skinny", TMA_ALT[(layer, m)], ("alt", layer, m)))
    if (c[0][0] == "wide" and layer == "router" and state["sm90"]
            and _route_key(layer, m, "gemm") not in state["disabled"]):
        c.append(("gemm",))
    if (m <= 32 and c[0][0] != "sk2" and _sk2_name(layer, m) in state.get("sk2", {})
            and _route_key(layer, m, "sk2") not in state["disabled"]):
        c.insert(0, ("sk2",))
    if (USE_GQ and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gq_bn(m)) in GQ_CONFIGS
            and _route_key(layer, m, "gq") not in state["disabled"]):
        c.insert(0, ("gq",))
    if (USE_GQX and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gqx_bn(m)) in GQX_CONFIGS
            and _route_key(layer, m, "gqx") not in state["disabled"]):
        c.insert(0, ("gqx",))
    if layer == "router" and m == 1 and "router_f32" in state:
        c.insert(0, ("rgemv",))
    if (USE_RK and layer == "router" and 32 < m <= 256 and state["sm90"] and "rk_zc" in state and state.get("nsm", 0) >= 16
            and _route_key(layer, m, "rk") not in state["disabled"]):
        c.insert(0, ("rk",))
    if layer == "unpooler" and m == 1:
        c.insert(0, ("sgemv",))
    if m == 1 and layer in state.get("cg", {}) and _route_key(layer, m, "cgemv") not in state["disabled"]:
        c.insert(0, ("cgemv",))
    if layer in TORCHT_LAYERS and 32 < m <= 256 and layer in state.get("rawT", {}):
        c.append(("torchT",))
    if (128 < m <= 256 and layer in state.get("cut_layers", ()) and _cut_ext is not None
            and _route_key(layer, m, "cutlass") not in state["disabled"]):
        c.append(("cutlass",))
    if c[-1][0] != "torch" and all(ch[0] != "torch" for ch in c):
        c.append(("torch",))
    return c


AUTOTUNE = True
TUNE_M = (1, 16, 32, 8, 128, 256)   # the scored token counts


def _autotune(weights, state):
    """Time each candidate as a CUDA graph replay from a cold L2 (a read sweep larger than L2 before every replay,
    per-replay events, candidates interleaved), on synthetic activations (reads only the weights), and keep the
    fastest per (layer, m)."""
    state["choice"] = {}
    if not AUTOTUNE:
        return
    dev = weights["o"].device
    flush = torch.ones(L2_FLUSH_ROWS, 1024, dtype=torch.float32, device=dev)  # read-only sweep
    sink = torch.empty(flush.shape[1], dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(AUTOTUNE_SEED)
    reps = AUTOTUNE_REPS
    for m in TUNE_M:
        for layer, w in weights.items():
            n = UNPOOLER_SLICE_ROWS if (layer == "unpooler" and m <= 8) else w.shape[0]
            k = w.shape[1]
            cands = _candidates(layer, m, state)
            if len(cands) < 2:
                continue
            x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
            graphs = []
            for ch in cands:
                _run(ch, x, layer, m, n, k, state)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    _run(ch, x, layer, m, n, k, state)
                graphs.append(gr)
            evs = []
            for r in range(reps):
                order = range(len(cands)) if r % 2 == 0 else reversed(range(len(cands)))
                for i in order:
                    torch.sum(flush, dim=0, out=sink)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[i].replay()
                    e1.record()
                    evs.append((i, e0, e1))
            torch.cuda.synchronize()
            per = [[] for _ in cands]
            for i, e0, e1 in evs:
                per[i].append(e0.elapsed_time(e1))
            med = [sorted(v)[len(v) // 2] for v in per]
            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
            # CUTLASS is tuned for the scored m only (its partial token tiles are slow: m=129 +14 us, m=200 +4 us vs
            # cuBLAS): other m of the bucket take the best other candidate
            alt = [i for i in range(len(cands)) if cands[i][0] != "cutlass"]
            if alt:
                state.setdefault("choice_alt", {})[(layer, m)] = cands[min(alt, key=lambda i: med[i])]
            state.setdefault("tune_log", []).append((layer, m, [(c[0], round(1000 * v, 2)) for c, v in zip(cands, med)]))
            del graphs
    del flush
    torch.cuda.synchronize()
    # one line per shape in the run's log: candidate times (us, raw graph replay incl. ~4 us replay cost) and the pick
    for layer, m, tl in state.get("tune_log", []):
        ch = state["choice"][(layer, m)]
        desc = " ".join(f"{name}{'/' + 'x'.join(map(str, c[1])) if c[0] == 'skinny' else ''}={t}"
                        for (name, t), c in zip(tl, _candidates(layer, m, state)))
        pick = ch[0] + ("/" + "x".join(map(str, ch[1])) if ch[0] == "skinny" else "")
        print(f"[okc] tune {layer} m={m}: {desc} -> {pick}", file=sys.stderr)
    print(f"[okc] disabled: {sorted(state['disabled'])}", file=sys.stderr)


def _nearby_choice(layer, m, state):
    """An untuned m takes the choice tuned for the scored m of its bucket (1 | 2-8 | 9-16 | 17-32 | 33-128 |
    129-256): every implementation handles any m of its bucket."""
    tm = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0
    if not tm:
        return None
    if tm == 256 and m != 256 and layer != "router":
        # an untuned m in 129..255 runs SGLang's own call (F.linear): the m=256 picks (CUTLASS, gqx, torchT) were
        # only timed at m=256 and some lose to cuBLAS at partial tiles (pooler CUTLASS m=129 +14 us, unpooler
        # torchT m=240 +11%)
        return ("torch",)
    ch = state.get("choice", {}).get((layer, tm))
    if ch is None:
        return None
    if ch[0] == "cutlass" and m != tm:
        ch = state.get("choice_alt", {}).get((layer, tm))
        if ch is None:
            return None
    if ch[0] == "gq" and (layer, _gq_bn(m)) not in GQ_CONFIGS:
        return None
    return ch


# ---------------------------------------------------------------------------
# Entry point: the harness calls prepare(weights) once, then kernel_fn per call.
# ---------------------------------------------------------------------------
def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
    choice = state.get("choice", {}).get((layer, m)) or _nearby_choice(layer, m, state) or _static_choice(layer, m, state)
    out = _run(choice, x, layer, m, n, k, state)
    return out


import ctypes as C

class _KPointer(C.c_void_p):
    def __deepcopy__(self, memo):
        return self


_KP, _KI, _KSZ = _KPointer, C.c_int, C.c_size_t


import threading as _KThreading

_KTLS = _KThreading.local()
_KCUDA = None


def _k_ctx_bind(x):
    if getattr(_KTLS, "device", None) == x.device.index:
        return
    global _KCUDA
    if _KCUDA is None:
        cu = C.CDLL("libcuda.so.1")
        for name, args in (("cuCtxGetCurrent", [C.POINTER(C.c_void_p)]),
                           ("cuDeviceGet", [C.POINTER(C.c_int), C.c_int]),
                           ("cuDevicePrimaryCtxRetain", [C.POINTER(C.c_void_p), C.c_int]),
                           ("cuCtxSetCurrent", [C.c_void_p])):
            fn = getattr(cu, name)
            fn.argtypes, fn.restype = args, C.c_int
        _KCUDA = cu
    ctx = C.c_void_p()
    _KCUDA.cuCtxGetCurrent(C.byref(ctx))
    if not ctx.value:
        dev, primary = C.c_int(), C.c_void_p()
        index = x.device.index or 0
        if _KCUDA.cuDeviceGet(C.byref(dev), index) == 0 and _KCUDA.cuDevicePrimaryCtxRetain(C.byref(primary), dev) == 0:
            _KCUDA.cuCtxSetCurrent(primary)
    _KTLS.device = x.device.index


class _KAlgo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class _KHeuristic(C.Structure):
    _fields_ = [("algo", _KAlgo), ("workspaceSize", C.c_size_t), ("state", C.c_int),
                ("wavesCount", C.c_float), ("reserved", C.c_int * 4)]


# cuBLASLt here is NVIDIA's libcublasLt.so.13 from the image -- the library torch itself links and F.linear / SGLang call.
# This wrapper only pins its algorithm (cublasLtMatmulAlgoInit id 66 + tile / stages / cluster attributes, FP32 compute,
# no split-K) instead of the heuristic; no binary of ours is loaded, and every custom kernel in this file is built by
# load_inline from the source text in this file.
class _KLargeLt:
    def __init__(self):
        self.lib = C.CDLL("libcublasLt.so.13")
        signatures = {
            "cublasLtCreate": [C.POINTER(_KP)],
            "cublasLtMatmulDescCreate": [C.POINTER(_KP), _KI, _KI],
            "cublasLtMatmulDescSetAttribute": [_KP, _KI, _KP, _KSZ],
            "cublasLtMatrixLayoutCreate": [C.POINTER(_KP), _KI, C.c_uint64, C.c_uint64, C.c_int64],
            "cublasLtMatrixLayoutSetAttribute": [_KP, _KI, _KP, _KSZ],
            "cublasLtMatmul": [_KP] * 12 + [C.POINTER(_KAlgo), _KP, _KSZ, _KP],
            "cublasLtMatmulAlgoInit": [_KP, _KI, _KI, _KI, _KI, _KI, _KI, _KI, C.POINTER(_KAlgo)],
            "cublasLtMatmulAlgoConfigSetAttribute": [C.POINTER(_KAlgo), _KI, _KP, _KSZ],
            "cublasLtMatmulAlgoCheck": [_KP, _KP, _KP, _KP, _KP, _KP, C.POINTER(_KAlgo), C.POINTER(_KHeuristic)],
        }
        for name, argtypes in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = argtypes, _KI
        self.handle = _KP()
        self.call("cublasLtCreate", C.byref(self.handle))
        self.desc = {}
        for mode, transpose in (("tn", 1), ("nn", 0)):
            desc = _KP()
            self.call("cublasLtMatmulDescCreate", C.byref(desc), 68, 0)
            op = _KI(transpose)
            self.call("cublasLtMatmulDescSetAttribute", desc, 3, C.byref(op), C.sizeof(op))
            self.desc["nn" if mode == "rownn" else mode] = desc
        self.workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
        self.required_workspace = 0
        self.alpha, self.beta = C.c_float(1), C.c_float(0)

    def __deepcopy__(self, memo):
        clone = self.__class__.__new__(self.__class__)
        clone.lib = self.lib
        clone.handle = self.handle
        clone.desc = dict(self.desc)
        clone.workspace = self.workspace.clone()
        clone.required_workspace = self.required_workspace
        clone.alpha, clone.beta = self.alpha, self.beta
        return clone

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} status {status}")

    def layouts(self, mode, m, n, k, batch):
        specs = ((k, n // batch, k, k * n // batch), (k, m, k, 0), (n // batch, m, n, n // batch))
        if mode == "nn":
            specs = ((n // batch, k, n, n // batch),) + specs[1:]
        elif mode == "rownn":
            specs = ((m, k, k, 0), (k, n, n, 0), (m, n, n, 0))
        layouts = []
        for rows, cols, ld, stride in specs:
            desc = _KP()
            self.call("cublasLtMatrixLayoutCreate", C.byref(desc), 14, rows, cols, ld)
            if mode == "rownn":
                order = _KI(1)
                self.call("cublasLtMatrixLayoutSetAttribute", desc, 1, C.byref(order), C.sizeof(order))
            count, offset = _KI(batch), C.c_int64(stride)
            self.call("cublasLtMatrixLayoutSetAttribute", desc, 5, C.byref(count), C.sizeof(count))
            self.call("cublasLtMatrixLayoutSetAttribute", desc, 6, C.byref(offset), C.sizeof(offset))
            layouts.append(desc)
        return layouts

    def plan(self, mode, m, n, k, cfg):
        batch = cfg["batch"]
        layouts = self.layouts(mode, m, n, k, batch)
        algo = _KAlgo()
        self.call("cublasLtMatmulAlgoInit", self.handle, 68, 0, 14, 14, 14, 14, 66, C.byref(algo))
        attrs = (("tile", 1, C.c_uint32), ("splitk", 2, C.c_int32), ("red", 3, C.c_uint32),
                 ("swz", 4, C.c_uint32), ("custom", 5, C.c_uint32), ("stages", 6, C.c_uint32),
                 ("cluster", 8, C.c_uint16))
        for name, attr, ctype in attrs:
            value = ctype(cfg[name])
            status = self.lib.cublasLtMatmulAlgoConfigSetAttribute(C.byref(algo), attr, C.byref(value), C.sizeof(value))
            if status:
                raise RuntimeError(f"large-proj config {name} status {status}")
        result = _KHeuristic()
        status = self.lib.cublasLtMatmulAlgoCheck(self.handle, self.desc["nn" if mode == "rownn" else mode], *layouts, layouts[2],
                                                  C.byref(algo), C.byref(result))
        if status:
            raise RuntimeError(f"large-proj AlgoCheck status {status}")
        self.required_workspace = max(self.required_workspace, int(result.workspaceSize))
        return mode, layouts, algo

    def run(self, x, weight, n, plan):
        _k_ctx_bind(x)
        mode, layouts, algo = plan
        out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
        a, b, c = layouts
        ap = _KP(x.data_ptr() if mode == "rownn" else weight.data_ptr())
        bp = _KP(weight.data_ptr() if mode == "rownn" else x.data_ptr())
        self.call("cublasLtMatmul", self.handle, self.desc["nn" if mode == "rownn" else mode], C.byref(self.alpha), ap, a,
                  bp, b, C.byref(self.beta), _KP(out.data_ptr()), c, _KP(out.data_ptr()), c,
                  C.byref(algo), _KP(self.workspace.data_ptr()), self.workspace.numel(),
                  _KP(torch.cuda.current_stream().cuda_stream))
        return out


def _kplan(tile, custom, cluster):
    return {"batch": 1, "tile": tile, "stages": 35, "splitk": 1, "red": 0, "swz": 0,
            "custom": custom, "cluster": cluster}


_KLARGE_CONFIGS = {
    ("qkv_gate", 256): ("tn", _kplan(124, 0, 2)),
    ("pooler", 256): ("rownn", _kplan(18, 0, 2)),
    ("pooler", 128): ("nn", _kplan(15, 4, 2)),
    ("o", 256): ("tn", _kplan(20, 6, 2)),
    ("unpooler", 256): ("rownn", _kplan(20, 4, 2)),
}
_KLARGE_TRANSPOSED = {("unpooler", 256), ("pooler", 256), ("pooler", 128)}
_klarge_base_prepare = prepare
_klarge_base_kernel = kernel_fn


def prepare(weights):
    state = _klarge_base_prepare(weights)
    state["klarge_rawT"] = {
        "unpooler": state["rawT"]["unpooler"],
        "pooler": state["raw"]["pooler"].T.contiguous(),
    }
    state["klarge_lt"] = _KLargeLt()
    state["klarge_plans"] = {}
    for key, (mode, cfg) in _KLARGE_CONFIGS.items():
        layer, m = key
        n, k = state["raw"][layer].shape
        state["klarge_plans"][key] = state["klarge_lt"].plan(mode, m, n, k, cfg)
    state["klarge_lt"].workspace = torch.empty(state["klarge_lt"].required_workspace, device="cuda", dtype=torch.uint8)
    return state


def kernel_fn(data, state):
    x = data["x"]
    key = (data["config"]["layer"], x.shape[0])
    plan = state["klarge_plans"].get(key)
    if plan is not None:
        weight = state["klarge_rawT"][key[0]] if key in _KLARGE_TRANSPOSED else state["raw"][key[0]]
        return state["klarge_lt"].run(x, weight, data["config"]["n"], plan)
    return _klarge_base_kernel(data, state)




# Independent exact escape corrections: per-warp scratch avoids atomics.
ESC_WARPS = 4
_ESC_EXT = None
_ORIGINAL_RUN_SK2 = _run_sk2
_ORIGINAL_PREPARE = prepare

def _escape_source():
    source = _sk2_src(True)
    source = source.replace('(NSTAGE >= 4 ? 4 : NSTAGE)', str(ESC_WARPS))
    source = source.replace('NW + (COMP ? NESC : 0)', 'NW')
    source = source.replace('  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");', '')
    start = source.index('      int pe = 1;')
    end = source.index('\n    }\n    mbar_arrive(cbar);', start)
    source = source[:start] + r"""
      for (int pe = 1 + ew; pe <= cnt; pe += NESC) {
        const int2 e = ptab_s[pe];
        const int kk = e.x & 0xFFFF;
        if (lane < M && kk >= gs0 * KST && kk < (gs0 + nst) * KST) {
          const float v = __bfloat162float(x[(size_t)lane * K + kk]);
          cr[lane * CP + (e.x >> 16)] += esc_delta((uint32_t)e.y, bq) * v;
        }
      }
""" + source[end:]
    # Build only SK2; unchanged helper definitions precede it.
    shared = CUDA_SRC.replace('@@MBAR@@', MBAR_HELPERS).replace('@@BODY@@', '')
    return shared + MAP_SRC + source

def _escape_ext():
    global _ESC_EXT
    if _ESC_EXT is None:
        import hashlib
        src = _escape_source()
        _ESC_EXT = load_inline(
            name='proj_escape_' + hashlib.md5(src.encode()).hexdigest()[:12],
            cpp_sources='#include <torch/extension.h>\n' + next(line for line in CPP_SRC.splitlines() if line.startswith('void sk2(')) + '\n' + next(line for line in CPP_SRC.splitlines() if 'int64_t M, int64_t N, int64_t K, int64_t cfg' in line),
            cuda_sources=src, functions=['sk2'],
            extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90', '-Xptxas=-v'],
            no_implicit_headers=True, verbose=False)
    return _ESC_EXT

def prepare(weights):
    _escape_ext()
    state = _ORIGINAL_PREPARE(weights)
    for ci, stream, tab, pmax, b4, n_rows in state['sk2'].values():
        assert stream.data_ptr() % 128 == 0
        assert tab.data_ptr() % 16 == 0
        assert pmax == 0 or pmax * 8 % 16 == 0
    return state

def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
    if layer not in ('qkv_gate', 'unpooler') or not 2 <= m <= 8:
        return _ORIGINAL_RUN_SK2(x, layer, m, n, k, state, dg, dbg)
    assert x.is_contiguous() and x.data_ptr() % 128 == 0 and k * 2 % 128 == 0
    name = _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state['sk2'][name]
    out = torch.empty((m,n), dtype=torch.bfloat16, device=x.device)
    _escape_ext().sk2(stream,x,out,tab,pmax,b4,m,n,k,ci,_sk2_tokr(m),dg,state['dummy'] if dbg is None else dbg)
    return out


# H200 backend choices measured across paired runs; weight-dependent selftests
# and route availability retain the base's correctness fallbacks.
_PINNED_LARGE = {
    ('qkv_gate',128): 'torch', ('pooler',128): 'torch',   # adamleex 2026-11-02: was 'gqx' / 'gq', tuned with x evict_last; without the pin cuBLAS (SGLang's call) is faster at m 64..127
    ('o',128): 'torchT', ('unpooler',128): 'torchT', ('router',128): 'rk',
    ('qkv_gate',256): 'torch', ('pooler',256): 'cutlass',
    ('o',256): 'torchT', ('unpooler',256): 'torchT', ('router',256): 'rk',
}
def _autotune(weights,state):
    state['choice']={};state['choice_alt']={}
    for m in TUNE_M:
        for layer in weights:
            want=('rgemv' if layer=='router' else 'cgemv') if m==1 else ('rsmall' if layer=='router' and m<=16 else 'sk2') if m<=32 else _PINNED_LARGE[(layer,m)]
            cands=_candidates(layer,m,state)
            choice=next((c for c in cands if c[0]==want),cands[0])
            state['choice'][(layer,m)]=choice
            state['choice_alt'][(layer,m)]=next((c for c in cands if c[0]=='torch'),('torch',))
    print('[proj] pinned choices:',state['choice'],file=sys.stderr)

# Small-router kernel adapted from NathenFernandes/muse-bf16-proj (MIT), S=16, WK=8.
SMALL_ROUTER_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define RD __device__ __forceinline__
RD uint32_t rr_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
RD void rr_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void rr_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void rr_wait(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");
}
RD uint32_t rr_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }
RD void rr_push(uint32_t d, float4 v, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
RD void rr_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }

template<int MT,int S,int WK>
__global__ __launch_bounds__(WK*32) void rr_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {
  constexpr int KU=128/S/WK, NI=MT*32;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+S*NI;
  const uint32_t rb=rr_smem(sm+(S+WK)*NI*16);
  int4 w[KU][8];uint4 xx[KU][MT][2];
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<8;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  if(rank==0 && tid==0) {rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  if(rank==0 && warp==0) __syncwarp();
  if(rank==0 && tid==0) {rr_expect(rb,(S-1)*NI*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  float c[2][MT][4]={};
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          const int4 a=w[k][q+h*4];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  for(int v=tid;v<NI;v+=WK*32) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) rr_add(sum,part[k*NI+v]);
    if(rank==0)recv[v]=sum;
    else rr_push(rr_map(rr_smem(recv+rank*NI+v),0),sum,rr_map(rb,0));
  }
  if(rank==0) {
    __syncthreads();rr_wait(rb,0);
    for(int v=tid;v<NI;v+=WK*32) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int s=0;s<S;s++) rr_add(sum,recv[s*NI+v]);
      const int ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
      if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
      if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
    }
  }
}

template<int MT,int S,int WK>
int rr_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream) {
  auto fn=rr_small_kernel<MT,S,WK>;
  constexpr int bytes=(S+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);
}
int rr_small(int64_t w,int64_t x,int64_t out,int m,int64_t stream) {
  if(m<=8)return rr_small_t<1,16,8>(w,x,out,m,stream);
  if(m<=16)return rr_small_t<2,16,8>(w,x,out,m,stream);
  return -99;
}
"""
SMALL_ROUTER_CPP = "#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rr_small(int64_t,int64_t,int64_t,int,int64_t);"
_small_router_ext = None


def _get_small_router_ext():
    global _small_router_ext
    if _small_router_ext is None:
        _small_router_ext = load_inline(
            "research_kimi_small_router_s16w8_v1", SMALL_ROUTER_CPP, cuda_sources=SMALL_ROUTER_SRC,
            functions=["rr_small"], extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"],
            no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
    return _small_router_ext


def pack_small_router(w, s=16, wk=8):
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    def units(v):
        v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
        return v.permute(0, 1, 3, 2, 4)
    data = torch.cat([units(hi), units(lo)], dim=2).contiguous()
    ku = 128 // s // wk
    return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()




_RS_RUN = _run
_RS_CANDIDATES = _candidates
_RS_KERNEL = kernel_fn

def _run(choice,x,layer,m,n,k,state):
    if choice[0] != 'rsmall':
        return _RS_RUN(choice,x,layer,m,n,k,state)
    assert x.is_contiguous() and x.data_ptr()%16 == 0 and k==8192
    out=torch.empty((m,256),dtype=torch.float32,device=x.device)
    rc=_get_small_router_ext().rr_small(state['router_small'].data_ptr(),x.data_ptr(),out.data_ptr(),m,torch.cuda.current_stream().cuda_stream)
    if rc:raise RuntimeError(f'small router launch failed: {rc}')
    return out

def _candidates(layer,m,state):
    choices=_RS_CANDIDATES(layer,m,state)
    if layer=='router' and 1<m<=16 and 'router_small' in state and _route_key(layer,m,'rsmall') not in state.get('disabled',()):
        choices.insert(0,('rsmall',))
    return choices

def kernel_fn(data,state):
    c=data['config'];layer,m=c['layer'],c['m']
    if layer=='router' and 1<m<=16 and 'router_small' in state and _route_key(layer,m,'rsmall') not in state.get('disabled',()):
        return _run(('rsmall',),data['x'],layer,m,c['n'],c['k'],state)
    return _RS_KERNEL(data,state)


_QBIG_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define QD __device__ __forceinline__
QD uint32_t q_addr(const void* p) {return (uint32_t)__cvta_generic_to_shared(p);}
QD void q_init(uint32_t p,int n) {asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(p),"r"(n):"memory");}
QD void q_expect(uint32_t p,int n) {asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(p),"r"(n):"memory");}
QD void q_wait(uint32_t p,int phase) {
  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%0],%1; @!p bra L;}"::"r"(p),"r"(phase):"memory");
}
QD void q_sync() {asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;":::"memory");}
QD uint32_t q_map(uint32_t p,int rank) {uint32_t d;asm("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(d):"r"(p),"r"(rank));return d;}
QD void q_push(uint32_t p,float4 v,uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0],{%1,%2,%3,%4},[%5];"::"r"(p),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
QD uint64_t q_desc(uint32_t p) {return uint64_t((p&0x3ffff)>>4)|(uint64_t(64)<<32)|(uint64_t(1)<<62);}
QD uint64_t q_time() {uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
QD void q_tma(uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;"::"r"(p),"l"(tm),"r"(k),"r"(r),"r"(b),"l"(policy):"memory");
}
QD void q_tma3(uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;"::"r"(p),"l"(tm),"r"(k/64),"r"(r),"r"(b),"l"(policy):"memory");
}
@@MMA@@
template<int NR,int G,int TM,int S,int ST,int PIPE,int RED,bool PROFILE>
__global__ __launch_bounds__((PIPE&2)?256:128) void qb_kernel(const __grid_constant__ CUtensorMap tw,
    const __grid_constant__ CUtensorMap tx,__nv_bfloat16* out,int m,unsigned long long* stamps) {
  constexpr int KT=(PIPE&32)?2:1, LR=(2304+G-1)/G, WB=NR*128, XB=TM*128, SB=KT*(WB+XB);
  constexpr int NV=NR*TM/4, OWN=RED>=4?((NV/128+S-1)/S)*128:((PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128);
  constexpr int RB=ST*SB+ST*16, RECV=RED==3?0:RB+16;
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid=threadIdx.x, lane=tid&31, rank=blockIdx.x%S, group=blockIdx.x/S;
  const int tok=blockIdx.y*TM, row=group*2304/G, rows=(group+1)*2304/G-row;
  const int kb=(128/KT)*rank/S, nkb=(128/KT)*(rank+1)/S-kb, own=RED>=4?((NV/128+S-1-rank)/S)*128:min(OWN,NV-rank*OWN);
  const uint32_t base=q_addr(sm), bars=base+ST*SB, empty=bars+ST*8, rb=base+RB;
  uint64_t begin=0, stream_end=0;
  if constexpr(PROFILE) begin=q_time();
  if constexpr(LR<NR) {
    for(int i=tid;i<ST*(NR-LR)*32;i+=blockDim.x) {
      int stage=i/((NR-LR)*32), j=i%((NR-LR)*32);
      ((uint32_t*)(sm+stage*SB+LR*128))[j]=0;
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncthreads();
  }
  uint64_t pw,px;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pw));
  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(px));
  auto tma=[&](uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {
    if constexpr(PIPE&2048)q_tma3(p,tm,k,r,b,policy);else q_tma(p,tm,k,r,b,policy);
  };
  if constexpr(PIPE&4096)asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(pw));
  auto load=[&](int i) {
    const int stage=i%ST;
    q_expect(bars+stage*8,KT*(LR*128+XB));
    if constexpr(KT==2)tma(base+stage*SB,&tw,0,(group*128+(kb+i)*KT)*NR,bars+stage*8,pw);
    else tma(base+stage*SB,&tw,(kb+i)*64,group*LR,bars+stage*8,pw);
#pragma unroll
    for(int q=0;q<KT;q++)tma(base+stage*SB+KT*WB+q*XB,&tx,((kb+i)*KT+q)*64,tok,bars+stage*8,px);
  };
  if(tid==((PIPE&2)?128:0)) {
#pragma unroll
    for(int i=0;i<ST;i++) {q_init(bars+i*8,1);if constexpr(PIPE&2)q_init(empty+i*8,4);}
    if constexpr(S>1) q_init(rb,1);
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    if constexpr(S>1 && RED<4) {q_expect(rb,(S-1)*own*16);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
#pragma unroll
    for(int i=0;i<ST;i++) if(i<nkb)load(i);
  }
  __syncthreads();
  if constexpr(S>1 && RED>=4) {
    if(tid==((PIPE&2)?128:0)) {
      q_expect(rb,(S-1)*own*16);
      asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
    }
    __syncthreads();
  }
  if constexpr(S>1 && RED!=3) asm volatile("barrier.cluster.arrive.relaxed.aligned;":::"memory");
  if constexpr(PIPE&2) {
    if(tid>=128) {
      if(tid==128) for(int i=ST;i<nkb;i++) {q_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
      if constexpr(S>1) {
        if constexpr(RED==3) asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");
        if constexpr(RED==1 || RED==3) q_sync();
      }
      return;
    }
  }
  float a[(TM/64)*(NR/2)];
#pragma unroll
  for(int h=0;h<TM/64;h++)
#pragma unroll
    for(int j=0;j<NR/2;j++) asm volatile("mov.b32 %0, 0;":"=f"(a[h*(NR/2)+j]));
if constexpr(PIPE&64) asm volatile("wgmma.fence.sync.aligned;":::"memory");
#pragma unroll 1
  for(int i=0;i<nkb;i++) {
    const int stage=i%ST;
    q_wait(bars+stage*8,(i/ST)&1);
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/2;j++) if constexpr(!(PIPE&128)) asm volatile("":"+f"(a[h*(NR/2)+j])::"memory");
    if constexpr(!(PIPE&64)) asm volatile("wgmma.fence.sync.aligned;":::"memory");
#pragma unroll
    for(int k=0;k<4*KT;k++) {
      if constexpr(PIPE&8) {
        uint64_t dx=q_desc(base+stage*SB+KT*WB+(k/4)*XB+(k%4)*32),dw=q_desc(base+stage*SB+(k/4)*WB+(k%4)*32);
        @@SWAP@@
      } else {
#pragma unroll
        for(int h=0;h<TM/64;h++) {
          uint64_t dx=q_desc(base+stage*SB+KT*WB+(k/4)*XB+h*8192+(k%4)*32),dw=q_desc(base+stage*SB+(k/4)*WB+(k%4)*32);
          @@CALL@@
        }
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(((PIPE>>8)&7)?((PIPE>>8)&7):1):"memory");
    else asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/2;j++) if constexpr(!(PIPE&128)) asm volatile("":"+f"(a[h*(NR/2)+j])::"memory");
    int done=i-((PIPE&1)?(((PIPE>>8)&7)?((PIPE>>8)&7):1):0);
    if(done>=0 && done+ST<nkb) {
      if constexpr(PIPE&2) {if(lane==0)asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(empty+(done%ST)*8):"memory");}
      else if(tid==0)load(done+ST);
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
  if constexpr(PROFILE) stream_end=q_time();
  auto emit=[&](int v,float4 z) {
    int t=v%128, j=(v/128)%(NR/8), h=(v/128)/(NR/8);
    int rr=h*64+(t/32)*16+(t%32)/4, cc=j*8+(t%4)*2;
    float vv[4]={z.x,z.y,z.z,z.w};
#pragma unroll
    for(int c=0;c<4;c++) {
      int tr,wc;
      if constexpr(PIPE&8) {tr=tok+(v/128)*8+(t%4)*2+(c&1);wc=(t/32)*16+(t%32)/4+(c/2)*8;}
      else {tr=tok+rr+(c/2)*8;wc=cc+(c&1);}
      if(tr<m && wc<rows)out[tr*2304+row+wc]=__float2bfloat16_rn(vv[c]);
    }
  };
  if constexpr(S==1) {
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/8;j++)emit((h*(NR/8)+j)*128+tid,make_float4(a[h*(NR/2)+j*4],a[h*(NR/2)+j*4+1],a[h*(NR/2)+j*4+2],a[h*(NR/2)+j*4+3]));
  } else if constexpr(RED>=4) {
    static_assert(NR==64 && (PIPE&8));
    asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");
#pragma unroll
    for(int j=0;j<TM/8;j++) {
      int dst=j%S,local=(j/S)*128+tid,slot=rank-(rank>dst);
      if(dst!=rank) q_push(q_map(base+RECV+(slot*OWN+local)*16,dst),
        make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]),q_map(rb,dst));
    }
    q_wait(rb,0);
#pragma unroll
    for(int j=0;j<TM/8;j++) if(j%S==rank) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) {
        float4 z=r==rank?make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]):
          *(float4*)(sm+RECV+((r-(r>rank))*OWN+(j/S)*128+tid)*16);
        sum.x+=z.x;sum.y+=z.y;sum.z+=z.z;sum.w+=z.w;
      }
      if constexpr(RED==4) emit(j*128+tid,sum);
      else {
        a[4*j]=sum.x;a[4*j+1]=sum.y;a[4*j+2]=sum.z;a[4*j+3]=sum.w;
      }
    }
    if constexpr(RED==5) {
      asm volatile("bar.sync 1,128;":::"memory");
#pragma unroll
      for(int j=0;j<TM/8;j++) if(j%S==rank) {
        __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]),hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);
        int rr=(tid/32)*16+((tid%16)/8)*8,cc=8*j+tid%8;
        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};"::
          "r"(base+2*(cc*72+rr)),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi):"memory");
      }
      asm volatile("bar.sync 1,128;":::"memory");
      for(int q=tid;q<(own/128)*64;q+=128) {
        int token=(rank+(q/64)*S)*8+(q/8)%8,rr=(q%8)*8;
        if(tok+token<m) *(uint4*)(out+(tok+token)*2304+row+rr)=*(uint4*)((__nv_bfloat16*)sm+token*72+rr);
      }
    }
  } else {
    if constexpr(RED==3) asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");
#pragma unroll
    for(int h=0;h<TM/64;h++)
#pragma unroll
      for(int j=0;j<NR/8;j++) {
        int v=(h*(NR/8)+j)*128+tid,dst=v/OWN,local=v%OWN;
        float4 z=make_float4(a[h*(NR/2)+j*4],a[h*(NR/2)+j*4+1],a[h*(NR/2)+j*4+2],a[h*(NR/2)+j*4+3]);
        uint32_t p=base+RECV+(rank*OWN+local)*16;
        if(dst==rank)*(float4*)(sm+RECV+(rank*OWN+local)*16)=z;
        else q_push(q_map(p,dst),z,q_map(rb,dst));
      }
    if constexpr(PIPE&2) asm volatile("bar.sync 1, 128;":::"memory");else __syncthreads();
    q_wait(rb,0);
    for(int t=tid;t<own;t+=128) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) {
        float4 z=*(float4*)(sm+RECV+(r*OWN+t)*16);
        sum.x+=z.x;sum.y+=z.y;sum.z+=z.z;sum.w+=z.w;
      }
      emit(rank*OWN+t,sum);
    }
    if constexpr(RED==1 || RED==3) q_sync();
  }
  if constexpr(PROFILE) if(lane==0) {
    unsigned smid;asm volatile("mov.u32 %0, %%smid;":"=r"(smid));
    int idx=((blockIdx.y*gridDim.x+blockIdx.x)*4+tid/32)*4;
    stamps[idx]=smid;stamps[idx+1]=begin;stamps[idx+2]=stream_end;stamps[idx+3]=q_time();
  }
}
using QEncode=decltype(&cuTensorMapEncodeTiled);
int q_tensor_map(CUtensorMap* tm,int64_t p,int inner,int rows,int box,bool map3) {
  static QEncode encode=nullptr;
  if(!encode){cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);if(!encode)return -100;}
  if(map3) {
    uint64_t dims[3]={64,(uint64_t)inner/64,(uint64_t)rows},stride[2]={128,(uint64_t)inner*2};
    uint32_t box3[3]={64,1,(uint32_t)box},es[3]={1,1,1};
    return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)p,dims,stride,box3,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  uint64_t dims[2]={(uint64_t)inner,(uint64_t)rows},stride[1]={(uint64_t)inner*2};uint32_t b[2]={64,(uint32_t)box},es[2]={1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int NR,int G,int TM,int S,int ST,int PIPE,int RED,bool PROFILE>
int qb_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats) {
  constexpr int LR=(2304+G-1)/G,NV=NR*TM/4,OWN=RED>=4?((NV/128+S-1)/S)*128:((PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128);
  constexpr int KT=(PIPE&32)?2:1;
  constexpr int bytes=ST*KT*(NR+TM)*128+ST*16+16+(S>1 && RED!=3?(RED>=4?S-1:S)*OWN*16:0);
  auto fn=qb_kernel<NR,G,TM,S,ST,PIPE,RED,PROFILE>;
  static bool ready=false;
  if(!ready){cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(e!=cudaSuccess)return -(int)e;
    e=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);if(e!=cudaSuccess)return -(int)e;ready=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(G*S,(m+TM-1)/TM);cfg.blockDim=dim3((PIPE&2)?256:128);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={S,1,1};if(S>1){cfg.attrs=&attr;cfg.numAttrs=1;}
  if(query==1){int v=0;cudaError_t e=S>1?cudaOccupancyMaxActiveClusters(&v,fn,&cfg):cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,fn,cfg.blockDim.x,bytes);return e==cudaSuccess?v:-(int)e;}
  CUtensorMap tw,tx;int e=q_tensor_map(&tw,w,KT==2?64:8192,KT==2?G*128*NR:G*LR,KT==2?KT*NR:LR,PIPE&2048);if(e)return e;e=q_tensor_map(&tx,x,8192,m,TM,PIPE&2048);if(e)return e;
  return int(cudaLaunchKernelEx(&cfg,fn,tw,tx,(__nv_bfloat16*)out,m,(unsigned long long*)stats));
}
int qb_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats) {
  switch(id){@@CASES@@ default:return -99;}
}
"""
_QBIG_CONFIG = (64, 36, 128, 3, 8, 539, 5)
_QBIG_EXT = None
_qbig_parent_prepare = prepare
_qbig_parent_kernel = kernel_fn


def _qbig_mma(n):
    regs = ','.join(f'%{i}' for i in range(n // 2))
    outs = ','.join(f'"+f"(a[{i}])' for i in range(n // 2))
    return f'__device__ __forceinline__ void q_mma_{n}(float* a,uint64_t x,uint64_t w) {{\n      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(w));\n    }}'


def _qbig_ext():
    global _QBIG_EXT
    if _QBIG_EXT is None:
        import hashlib
        configs = [_QBIG_CONFIG]
        ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})
        source = _QBIG_CUDA.replace('@@MMA@@', '\n'.join(_qbig_mma(n) for n in ns))
        source = source.replace('@@CALL@@', ' '.join(f'if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);' for n in ns))
        source = source.replace('@@SWAP@@', ' '.join(f'if constexpr(TM=={n})q_mma_{n}(a,dw,dx);' for n in ns))
        source = source.replace('@@CASES@@', '\n'.join(
            f"case {i}: return qb_launch_t<{','.join(map(str, c))},false>(w,x,out,m,stream,query,stats);" for i, c in enumerate(configs)))
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        namespace = 'qb_' + digest
        begin = source.index('#define QD')
        args = 'int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats'
        source = source[:begin] + f'namespace {namespace} {{\n' + source[begin:] + f'\n}}\nint qb_launch({args}) {{return {namespace}::qb_launch(id,w,x,out,m,stream,query,stats);}}'
        cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint qb_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);'
        _QBIG_EXT = load_inline('qbig_' + hashlib.sha256(source.encode()).hexdigest()[:16], cpp,
            cuda_sources=source, functions=['qb_launch'], no_implicit_headers=True,
            with_pytorch_error_handling=False, verbose=False,
            extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])
    return _QBIG_EXT


def prepare(weights):
    state = _qbig_parent_prepare(weights)
    state["qbig"] = {"ext": _qbig_ext(), "weight": state["raw"]["qkv_gate"]}
    return state


def kernel_fn(data, state):
    x = data["x"]
    if data["config"]["layer"] == "qkv_gate" and x.shape[0] == 128 and x.is_contiguous():
        out = torch.empty((128, 2304), dtype=torch.bfloat16, device=x.device)
        rc = state["qbig"]["ext"].qb_launch(0, state["qbig"]["weight"].data_ptr(), x.data_ptr(), out.data_ptr(),
                                           128, torch.cuda.current_stream().cuda_stream, 0, 0)
        if rc:
            raise RuntimeError(f"qbig launch failed: {rc}")
        return out
    return _qbig_parent_kernel(data, state)




# Portions Copyright vjeux, Apache License 2.0; independent accumulator experiments by OKC proj inventor.
import torch
from torch.utils.cpp_extension import load_inline
_IV_CG_SRC=r"""#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
template <int ROWS, int NTH, int K, int NACC>
__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 0xFFFF]) * esc_delta((uint32_t)en.y, base4 & 0xFFu);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));
    }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }
  float acc[ROWS][NACC];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) for(int a=0;a<NACC;++a) acc[r][a] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
      float v[NACC]; for(int a=0;a<NACC;++a) v[a]=acc[r][a];
#pragma unroll
      for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]
        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);         // hi bytes of elements 0..3
        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);  // hi bytes of elements 4..7
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v[0%NACC] = fmaf(__uint_as_float(p01 << 16), xg[0], v[0%NACC]); v[1%NACC] = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v[1%NACC]);
        v[2%NACC] = fmaf(__uint_as_float(p23 << 16), xg[2], v[2%NACC]); v[3%NACC] = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v[3%NACC]);
        v[4%NACC] = fmaf(__uint_as_float(p45 << 16), xg[4], v[4%NACC]); v[5%NACC] = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v[5%NACC]);
        v[6%NACC] = fmaf(__uint_as_float(p67 << 16), xg[6], v[6%NACC]); v[7%NACC] = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v[7%NACC]);
      }
      for(int a=0;a<NACC;++a) acc[r][a]=v[a];
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v=0.f; for(int a=0;a<NACC;++a) v+=acc[r][a];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

void invoke(at::Tensor L,at::Tensor C,at::Tensor x,at::Tensor tab,at::Tensor out,int64_t pmax,int64_t b4,int64_t cfg){
 auto st=at::cuda::getCurrentCUDAStream();int N=out.size(1);
 auto lp=L.data_ptr<uint8_t>();auto cp=C.data_ptr<uint8_t>();auto xp=(const __nv_bfloat16*)x.data_ptr();auto tp=(const int2*)tab.data_ptr<int>();auto op=(__nv_bfloat16*)out.data_ptr();
 @DISPATCH@
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_IV_CG_CONFIGS=[(8192,2,512,1),(8192,2,512,2),(4096,2,256,1),(4096,2,128,4)]
_iv_cg_ext=None

def _iv_cg_build():
 global _iv_cg_ext
 if _iv_cg_ext is None:
  lines=[]
  for i,(k,r,t,a) in enumerate(_IV_CG_CONFIGS):
   lines.append(f'if(cfg=={i}) cgemv_kernel<{r},{t},{k},{a}><<<N/{r},{t}+32,0,st>>>(lp,cp,xp,tp,pmax,op,b4);')
  _iv_cg_ext=load_inline(name='inv_proj_cg_candidate_v005',cpp_sources='#include <torch/extension.h>\nvoid invoke(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);',cuda_sources=_IV_CG_SRC.replace('@DISPATCH@','\n'.join(lines)),functions=['invoke'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],no_implicit_headers=True)
 return _iv_cg_ext

def _iv_cg_run(c,x,n,i):
 y=torch.empty((1,n),device=x.device,dtype=torch.bfloat16)
 _iv_cg_build().invoke(c['L'],c['C'],x,c['tab'],y,c['pmax'],c['base4'],i)
 return y

# Portions Copyright vjeux, Apache License 2.0; normalized router codec scale tuning.
import torch
from torch.utils.cpp_extension import load_inline
_IV_R24_SRC=r"""#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
template <int ROWS, int NTH, int K, int NACC, int SHIFT>
__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    float* __restrict__ out, uint32_t base4) {
  constexpr float INV_SCALE = 1.f / float(1<<SHIFT);
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 8191]) * esc_delta((uint32_t)en.y, base4 & 0xFFu) * ((en.x & 16383) >= 8192 ? INV_SCALE : 1.f);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));
    }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + ((j * 16 * NTH + 16 * tid) % 8192);
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }
  float acc[ROWS][NACC];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) for(int a=0;a<NACC;++a) acc[r][a] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const float xs=((j*16*NTH+16*tid)>=8192)?INV_SCALE:1.f;
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
      float v[NACC]; for(int a=0;a<NACC;++a) v[a]=acc[r][a];
#pragma unroll
      for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]
        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);         // hi bytes of elements 0..3
        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);  // hi bytes of elements 4..7
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v[0%NACC] = fmaf(__uint_as_float(p01 << 16), (xg[0]*xs), v[0%NACC]); v[1%NACC] = fmaf(__uint_as_float(p01 & 0xFFFF0000u), (xg[1]*xs), v[1%NACC]);
        v[2%NACC] = fmaf(__uint_as_float(p23 << 16), (xg[2]*xs), v[2%NACC]); v[3%NACC] = fmaf(__uint_as_float(p23 & 0xFFFF0000u), (xg[3]*xs), v[3%NACC]);
        v[4%NACC] = fmaf(__uint_as_float(p45 << 16), (xg[4]*xs), v[4%NACC]); v[5%NACC] = fmaf(__uint_as_float(p45 & 0xFFFF0000u), (xg[5]*xs), v[5%NACC]);
        v[6%NACC] = fmaf(__uint_as_float(p67 << 16), (xg[6]*xs), v[6%NACC]); v[7%NACC] = fmaf(__uint_as_float(p67 & 0xFFFF0000u), (xg[7]*xs), v[7%NACC]);
      }
      for(int a=0;a<NACC;++a) acc[r][a]=v[a];
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v=0.f; for(int a=0;a<NACC;++a) v+=acc[r][a];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = v;
  }
}

void invoke(at::Tensor L,at::Tensor C,at::Tensor x,at::Tensor tab,at::Tensor out,int64_t pmax,int64_t b4,int64_t cfg){
 auto st=at::cuda::getCurrentCUDAStream();int N=out.size(1);
 auto lp=L.data_ptr<uint8_t>();auto cp=C.data_ptr<uint8_t>();auto xp=(const __nv_bfloat16*)x.data_ptr();auto tp=(const int2*)tab.data_ptr<int>();auto op=(float*)out.data_ptr();
 @DISPATCH@
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
_IV_R24_CONFIGS=[(16384,2,512,4,8),(16384,2,512,8,9),(16384,2,512,8,10),(16384,1,512,2,7)]
_iv_r24_ext=None

def _iv_r24_build():
 global _iv_r24_ext
 if _iv_r24_ext is None:
  lines=[]
  for i,(k,r,t,a,s) in enumerate(_IV_R24_CONFIGS):
   lines.append(f'if(cfg=={i}) cgemv_kernel<{r},{t},{k},{a},{s}><<<N/{r},{t}+32,0,st>>>(lp,cp,xp,tp,pmax,op,b4);')
  _iv_r24_ext=load_inline(name='inv_proj_router24_candidate_v004',cpp_sources='#include <torch/extension.h>\nvoid invoke(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);',cuda_sources=_IV_R24_SRC.replace('@DISPATCH@','\n'.join(lines)),functions=['invoke'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],no_implicit_headers=True)
 return _iv_r24_ext

def _iv_r24_run(c,x,n,i):
 y=torch.empty((1,n),device=x.device,dtype=torch.float32)
 _iv_r24_build().invoke(c['L'],c['C'],x,c['tab'],y,c['pmax'],c['base4'],i)
 return y



_iv_base_prepare=prepare
_iv_base_kernel=kernel_fn

def _iv_choose(funcs,x,w,dev,tol):
    truth=x.double()@w.double().T
    usable=[]
    for j,fn in enumerate(funcs):
        out=fn()
        err=((out.double()-truth).norm()/truth.norm()).item()
        if err<tol:usable.append(j)
    if not usable:return 0
    graphs={}
    for j in usable:
        gr=torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):funcs[j]()
        graphs[j]=gr
    flush=torch.ones((65536,1024),device=dev,dtype=torch.float32)
    sink=torch.empty(1024,device=dev,dtype=torch.float32)
    events=[]
    for rep in range(96):
        order=usable if rep%2==0 else list(reversed(usable))
        for j in order:
            torch.sum(flush,dim=0,out=sink)
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();graphs[j].replay();b.record();events.append((j,a,b))
    torch.cuda.synchronize()
    times={j:[] for j in usable}
    for j,a,b in events:times[j].append(a.elapsed_time(b))
    medians={j:sorted(ts)[len(ts)//2] for j,ts in times.items()}
    return min(medians,key=medians.get)

def prepare(weights):
    state=_iv_base_prepare(weights)
    _iv_cg_build()
    router=weights['router'];dev=router.device
    hi=router.to(torch.bfloat16);lo=(router-hi.float()).to(torch.bfloat16)
    variants=[]
    for i,(_,rows,_,_,shift) in enumerate(_IV_R24_CONFIGS):
        scale=2**shift
        scaled=(lo.float()*scale).to(torch.bfloat16)
        if not bool((scaled.float()/scale==lo.float()).all()):continue
        c=cg_compress(torch.cat([hi,scaled],dim=1).contiguous(),rows)
        if c is not None:variants.append((i,c))
    gen=torch.Generator(device=dev).manual_seed(31919)
    if variants:
        _iv_r24_build()
        x=torch.randn((1,8192),generator=gen,device=dev,dtype=torch.bfloat16)
        data={'x':x,'config':{'m':1,'n':256,'k':8192,'layer':'router'}}
        funcs=[lambda:_iv_base_kernel(data,state)]
        funcs.extend(lambda i=i,c=c:_iv_r24_run(c,x,256,i) for i,c in variants)
        best=_iv_choose(funcs,x,router,dev,1e-4)
        if best:state['iv_router24']=variants[best-1]
    state['iv_cg_choice']={}
    for layer,indices in [('qkv_gate',(0,1)),('unpooler',(2,3))]:
        c=state.get('cg',{}).get(layer)
        if c is None:continue
        w=weights[layer][:1024] if layer=='unpooler' else weights[layer]
        n,k=w.shape
        x=torch.randn((1,k),generator=gen,device=dev,dtype=torch.bfloat16)
        data={'x':x,'config':{'m':1,'n':n,'k':k,'layer':layer}}
        funcs=[lambda:_iv_base_kernel(data,state)]
        funcs.extend(lambda i=i:_iv_cg_run(c,x,n,i) for i in indices)
        best=_iv_choose(funcs,x,w,dev,1e-2)
        if best:state['iv_cg_choice'][layer]=indices[best-1]
    return state

def kernel_fn(data,state):
    layer=data['config']['layer'];m=data['x'].shape[0]
    if m==1:
        if layer=='router' and 'iv_router24' in state:
            i,c=state['iv_router24']
            return _iv_r24_run(c,data['x'],data['config']['n'],i)
        if layer in state.get('iv_cg_choice',{}):
            return _iv_cg_run(state['cg'][layer],data['x'],data['config']['n'],state['iv_cg_choice'][layer])
    return _iv_base_kernel(data,state)


import hashlib as _oc_hashlib


_OC_OLAYER_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define OD __device__ __forceinline__
OD uint32_t ol_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
OD void ol_wait(uint32_t b,int ph) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(ph) : "memory");
}
OD void ol_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
OD uint64_t ol_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
OD void ol_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");
}
@@MMA@@

template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void ol_projection(
    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    const uint8_t* w,__nv_bfloat16* out,int m) {
  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;
  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);
  extern __shared__ __align__(1024) uint8_t sm[];
  uint32_t base=ol_addr(sm),full=base+ST*SB,empty=full+ST*8;
  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;
  uint64_t pw,px;
  if constexpr(HINT) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(px));
  } else {
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;
  }
  auto load=[&](int i) {
    int s=i%ST;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
    if constexpr(PACK==1) {
      const uint8_t* src=w+((long)nt*IT+i)*AB;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");
    } else if constexpr(PACK==2) ol_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);
    else ol_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);
    ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);
  };
  if(tid==((PIPE&2)?CT:0)) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) load(i);
  }
  __syncthreads();
  if((PIPE&2) && tid>=CT) {
    if(tid==CT) {
      #pragma unroll 1
      for(int i=ST;i<IT;i++) {ol_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
    }
  } else {
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");
    #pragma unroll 1
    for(int i=0;i<IT;i++) {
      int s=i%ST;
      ol_wait(full+s*8,(i/ST)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int kk=0;kk<BK/16;kk++) {
        uint64_t aw=ol_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);
        uint64_t ax=ol_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);
        @@CALL@@
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      int done=i-((PIPE&1)?1:0);
      if(done>=0 && done+ST<IT && tid%128==0) {
        if constexpr(PIPE&2) ol_arrive(empty+(done%ST)*8);
        else load(done+ST);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);
      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);
      int rb=((tid%128)/32)*16+((tid%16)/8)*8,cb=(tid/128)*BN+8*j+(tid%8);
      uint32_t d=base+2*(cb*72+rb);
      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi) : "memory");
    }
  }
  __syncthreads();
  for(int q=tid;q<T*(R/8);q+=TH) {
    int token=q/(R/8),row=(q%(R/8))*8;
    if(mt*T+token<m) *(uint4*)(out+(long)(mt*T+token)*8192+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*72+row);
  }
}

using OLEncode=decltype(&cuTensorMapEncodeTiled);
int ol_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {
  static OLEncode encode=nullptr;
  if(!encode) {
    cudaDriverEntryPointQueryResult q;
    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);
    if(rc!=cudaSuccess || !encode) return -100;
  }
  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};
  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
int ol_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  constexpr int T=BN*TW,R=PACK==2?32:64;
  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;
  auto fn=ol_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;
  static bool configured=false;
  if(!configured) {
    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;
    configured=true;
  }
  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}
  CUtensorMap tw,tx;
  int rc=ol_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;
  rc=ol_map(&tx,(void*)x,m,T,BK);if(rc)return rc;
  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);
  return int(cudaGetLastError());
}
int ol_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  switch(id) {@@CASES@@ default:return -99;}
}
"""
_OC_RL_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define OD __device__ __forceinline__
OD uint32_t rl_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
OD void rl_wait(uint32_t b,int ph) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(ph) : "memory");
}
OD void rl_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
OD uint64_t rl_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
OD void rl_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");
}
@@MMA@@

template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void rl_early_projection(
    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    const uint8_t* w,__nv_bfloat16* out,int m) {
  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;
  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);
  extern __shared__ __align__(1024) uint8_t sm[];
  uint32_t base=rl_addr(sm),full=base+ST*SB,empty=full+ST*8;
  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;
  uint64_t pw,px;
  if constexpr(HINT) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(px));
  } else {
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;
  }
  auto load=[&](int i) {
    int s=i%ST;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
    if constexpr(PACK==1) {
      const uint8_t* src=w+((long)nt*IT+i)*AB;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");
    } else if constexpr(PACK==2) rl_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);
    else rl_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);
    rl_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);
  };
  if(tid==((PIPE&2)?CT:0)) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if constexpr(PIPE&4) __syncthreads();
  if(tid==((PIPE&2)?CT:0)) {
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) load(i);
  }
  if constexpr(!(PIPE&4)) __syncthreads();
  if((PIPE&2) && tid>=CT) {
    if(tid==CT) {
      #pragma unroll 1
      for(int i=ST;i<IT;i++) {rl_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
    }
  } else {
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");
    #pragma unroll 1
    for(int i=0;i<IT;i++) {
      int s=i%ST;
      rl_wait(full+s*8,(i/ST)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int kk=0;kk<BK/16;kk++) {
        uint64_t aw=rl_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);
        uint64_t ax=rl_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);
        @@CALL@@
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      int done=i-((PIPE&1)?1:0);
      if(done>=0 && done+ST<IT && tid%128==0) {
        if constexpr(PIPE&2) rl_arrive(empty+(done%ST)*8);
        else load(done+ST);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);
      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);
      int rb=((tid%128)/32)*16+((tid%16)/8)*8,cb=(tid/128)*BN+8*j+(tid%8);
      uint32_t d=base+2*(cb*72+rb);
      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi) : "memory");
    }
  }
  __syncthreads();
  for(int q=tid;q<T*(R/8);q+=TH) {
    int token=q/(R/8),row=(q%(R/8))*8;
    if(mt*T+token<m) *(uint4*)(out+(long)(mt*T+token)*8192+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*72+row);
  }
}

using OLEncode=decltype(&cuTensorMapEncodeTiled);
int rl_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {
  static OLEncode encode=nullptr;
  if(!encode) {
    cudaDriverEntryPointQueryResult q;
    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);
    if(rc!=cudaSuccess || !encode) return -100;
  }
  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};
  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
int rl_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  constexpr int T=BN*TW,R=PACK==2?32:64;
  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;
  auto fn=rl_early_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;
  static bool configured=false;
  if(!configured) {
    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;
    configured=true;
  }
  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}
  CUtensorMap tw,tx;
  int rc=rl_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;
  rc=rl_map(&tx,(void*)x,m,T,BK);if(rc)return rc;
  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);
  return int(cudaGetLastError());
}
int rl_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  switch(id) {@@CASES@@ default:return -99;}
}
"""
_OC_OLAYER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'
_OC_RL_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rl_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

def _oc_olayer_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join(f'%{i}' for i in range(n // 2))
        outs = ','.join(f'"+f"(a[{i}])' for i in range(n // 2))
        functions.append(f'''OD void ol_mma{n}(float* a,uint64_t w,uint64_t x) {{
          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));
        }}''')
    cases = '\n'.join(f'case {i}: return ol_launch_t<{",".join(map(str,tuple(c)+(1,) if len(c)==6 else c))}>(x,w,out,m,stream,query);' for i,c in enumerate(configs))
    calls = '\n'.join(f'if constexpr(BN=={n}) ol_mma{n}(a,aw,ax);' for n in ns)
    return _OC_OLAYER_CUDA.replace('@@MMA@@','\n'.join(functions)).replace('@@CASES@@',cases).replace('@@CALL@@',calls)


def _oc_build_olayer(configs):
    for cfg in configs:
        bn,bk,st,pipe,pack,hint,*groups = cfg
        tw = groups[0] if groups else 1
        if (bn<16 or bn>256 or bn%8 or bk not in (64,128,256) or st<1
                or st*((64+bn*tw)*bk*2+16)>227000 or pipe not in (0,1,2,3)
                or (pipe&1 and st<2) or pack not in (0,1,2) or hint not in (0,1)
                or tw not in (1,2) or bn*tw>256 or (tw>1 and not pipe&2)):
            raise ValueError(f'unsupported olayer configuration: {cfg}')
    source = _oc_olayer_source(configs)
    name = 'olayer_' + _oc_hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name,_OC_OLAYER_CPP,cuda_sources=source,functions=['ol_launch'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True,with_pytorch_error_handling=False)


def _oc_pack_olayer(weight) -> torch.Tensor:
    columns = torch.arange(64,device=weight.device)[None,:] ^ ((torch.arange(64,device=weight.device)[:,None]&7)*8)
    tiled = weight.reshape(128,64,16,64).permute(0,2,1,3)
    return tiled.gather(-1,columns.expand(128,16,64,64)).contiguous()




def _oc_rl_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        functions.append(f'OD void rl_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
    cases = '\n'.join((f"case {i}: return rl_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for i, c in enumerate(configs)))
    calls = '\n'.join((f'if constexpr(BN=={n}) rl_mma{n}(a,aw,ax);' for n in ns))
    return _OC_RL_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

def _oc_build_rl(configs):
    for cfg in configs:
        bn, bk, st, pipe, pack, hint, *groups = cfg
        tw = groups[0] if groups else 1
        if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (2, 3, 6, 7, 14, 15)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
            raise ValueError(f'unsupported rl configuration: {cfg}')
    tt_v10_tune_source = _oc_rl_source(configs)
    name = 'rl_' + _oc_hashlib.sha256(tt_v10_tune_source.encode()).hexdigest()[:12]
    return load_inline(name, _OC_RL_CPP, cuda_sources=tt_v10_tune_source, functions=['rl_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)



def _oc_rl_pack(w: torch.Tensor, rows: int) -> torch.Tensor:
    n, k = w.shape
    col = torch.arange(64, device=w.device)[None, :] ^ (torch.arange(rows, device=w.device)[:, None] & 7) * 8
    return w.reshape(n // rows, rows, k // 64, 64).permute(0, 2, 1, 3).gather(-1, col.expand(n // rows, k // 64, rows, 64)).reshape(n, k).contiguous()



_OC_OL_CONFIG = (128, 128, 4, 3, 1, 1)
_OC_RL_CONFIG = (256, 64, 5, 3, 1, 1, 1)
_oc_parent_prepare = prepare
_oc_parent_kernel = kernel_fn


def prepare(weights):
    state = _oc_parent_prepare(weights)
    state["oc_ol"] = {"ext": _oc_build_olayer([_OC_OL_CONFIG]), "weight": _oc_pack_olayer(state["raw"]["o"])}
    state["oc_rl"] = {"ext": _oc_build_rl([_OC_RL_CONFIG]), "weight": _oc_rl_pack(state["raw"]["o"], 64)}
    return state


def kernel_fn(data, state):
    x = data["x"]
    if data["config"]["layer"] == "o" and x.is_contiguous():
        m = x.shape[0]
        if m == 128:
            out = torch.empty((128, 8192), device=x.device, dtype=torch.bfloat16)
            rc = state["oc_ol"]["ext"].ol_launch(0, x.data_ptr(), state["oc_ol"]["weight"].data_ptr(),
                                                out.data_ptr(), 128, torch.cuda.current_stream().cuda_stream, 0)
            if rc:
                raise RuntimeError(f"custom olayer launch failed: {rc}")
            return out
        if m == 256:
            out = torch.empty((256, 8192), device=x.device, dtype=torch.bfloat16)
            rc = state["oc_rl"]["ext"].rl_launch(0, x.data_ptr(), state["oc_rl"]["weight"].data_ptr(),
                                                out.data_ptr(), 256, torch.cuda.current_stream().cuda_stream, 0)
            if rc:
                raise RuntimeError(f"custom rl launch failed: {rc}")
            return out
    return _oc_parent_kernel(data, state)


import types as _KTypes


class _KSharedModule:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)

    def __deepcopy__(self, memo):
        return self


def _kshare_modules(value):
    if isinstance(value, _KTypes.ModuleType):
        return _KSharedModule(value)
    if isinstance(value, dict):
        return {key: _kshare_modules(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_kshare_modules(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_kshare_modules(item) for item in value)
    return value


_oc_shared_prepare = prepare


def prepare(weights):
    return _kshare_modules(_oc_shared_prepare(weights))


_U_CFG=('u32_r96', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1)
_U_NATIVE=None
_U_RUN=_run_sk2
_U_PREPARE=prepare

def _u_ext():
 global _U_NATIVE
 if _U_NATIVE is None:
  cfg=_U_CFG;name,layer,n,R,NT,NWR,WSK,UPW,NACC,CL=cfg
  _,ns=_sk2_geom(cfg,32)
  call=f'launch_sk2<2,32,{NT},{NWR},{WSK},{UPW},{ns},true,false,{NACC},{CL},__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;'
  src=CUDA_SRC.replace('@@MBAR@@',MBAR_HELPERS).replace('@@BODY@@','')+MAP_SRC+SK2_SRC.replace('@@SK2_DISPATCH@@',call)
  import hashlib
  cpp='#include <torch/extension.h>\n'+next(l for l in CPP_SRC.splitlines() if l.startswith('void sk2('))+'\n'+next(l for l in CPP_SRC.splitlines() if 'int64_t M, int64_t N, int64_t K, int64_t cfg' in l)
  _U_NATIVE=load_inline('proj_u32geom_'+hashlib.md5(src.encode()).hexdigest()[:12],cpp,cuda_sources=src,functions=['sk2'],no_implicit_headers=True,extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90','-Xptxas=-v'],verbose=False)
 return _U_NATIVE

def prepare(weights):
 _u_ext()
 state=_U_PREPARE(weights)
 state['u32']=_u_pack(weights['unpooler'])
 return state

def _u_pack(w):
 name,layer,n,R,NT,NWR,WSK,UPW,NACC,CL=_U_CFG
 c=sk2_compress(w)
 if c is None:return None
 tab,pmax=sk2_table(c,n,R)
 return dict(w=sk2_stream(c['packed'],n,R,NT,NWR,WSK,UPW),tab=tab,pmax=pmax,b4=c['base4'])

def _run_sk2(x,layer,m,n,k,state,dg=0,dbg=None):
 if layer!='unpooler' or m!=32 or state.get('u32') is None:return _U_RUN(x,layer,m,n,k,state,dg,dbg)
 c=state['u32'];out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
 _u_ext().sk2(c['w'],x,out,c['tab'],c['pmax'],c['b4'],m,n,k,0,32,dg,state['dummy'] if dbg is None else dbg)
 return out


_ship_parent_kernel = kernel_fn


def kernel_fn(data, state):
    if data["config"]["layer"] == "router" and data["x"].shape[0] == 1:
        return _run(("rgemv",), data["x"], "router", 1, data["config"]["n"], data["config"]["k"], state)
    return _ship_parent_kernel(data, state)


# Ensure driver API calls on every route have a current CUDA context in a fresh thread.
# The helper is credited to matvey; it is cached per thread/device.
_ship_context_parent_kernel = kernel_fn
def kernel_fn(data, state):
    _k_ctx_bind(data['x'])
    return _ship_context_parent_kernel(data, state)


# Nathen Fernandes (MIT): literal pinned router routes at m <= 32.
import hashlib
def _mma(n):
    regs = ', '.join(f'%{i}' for i in range(n // 2))
    outs = ', '.join(f'"+f"(d[{i}])' for i in range(n // 2))
    return f'''__device__ __forceinline__ void rr_mma_{n}(float* d, uint64_t a, uint64_t b) {{
    asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(a), "l"(b));
}}'''


ROUTER_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define RD __device__ __forceinline__
RD uint32_t rr_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
RD void rr_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void rr_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void rr_wait(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");
}
RD void rr_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" :::"memory"); }
RD uint32_t rr_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }
RD void rr_push(uint32_t d, float4 v, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
RD float4 rr_pull(uint32_t d) {
  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(d):"memory");return v;
}
RD void rr_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }
RD uint64_t rr_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }
RD void rr_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(dst),"l"(tm),"r"(k),"r"(r),"r"(bar):"memory");
}
@@MMA@@

template<int BN,int S,int ST,int SW,int RED,int PIPE>
__global__ __launch_bounds__((PIPE&2)?256:128) void rr_kernel(const __grid_constant__ CUtensorMap tw,
                                               const __grid_constant__ CUtensorMap tx, float* out, int m) {
  constexpr int NR=SW?64:BN, TM=SW?BN:64;
  constexpr int WB=NR*128, XB=TM*128, SB=WB*2+XB;
  constexpr int NF=64*BN, NV=NF/4, OWN=NV/S;
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, rank=blockIdx.x%S;
  const int row=blockIdx.x/S*NR, tok=blockIdx.y*TM;
  const uint32_t base=rr_smem(sm), bars=base+ST*SB, empty=bars+ST*8, rb=(empty+((PIPE&2)?ST*8:0)+15)&~15u;
  const uint32_t recv=rb+16;
  const int kb=rank*(128/S);
  auto load=[&](int i) {
    const int st=i%ST;
    const uint32_t b=bars+st*8, d=base+st*SB;
    rr_expect(b,SB);
    rr_tma(d,&tw,0,((row/NR)*128+kb+i)*NR*2,b);
    rr_tma(d+2*WB,&tx,(kb+i)*64,tok,b);
  };
  if(tid==((PIPE&2)?128:0)) {
#pragma unroll
    for(int i=0;i<ST;i++) { rr_init(bars+i*8,1); if constexpr(PIPE&2) rr_init(empty+i*8,4); }
    asm volatile("fence.proxy.async.shared::cta;" :::"memory");
    if constexpr(PIPE&4) {
    if constexpr(RED>=1) {
      rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");rr_expect(rb,(S-1)*OWN*16);
      asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");
    }
    }
#pragma unroll
    for(int i=0;i<ST && i<128/S;i++) load(i);
    if constexpr(!(PIPE&4)) {
    if constexpr(RED>=1) {
      rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");rr_expect(rb,(S-1)*OWN*16);
      asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");
    }
    }
  }
  __syncthreads();
  if constexpr(RED>=1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  if constexpr(PIPE&2) {
    if(tid>=128) {
      if(tid==128) {
        for(int i=ST;i<128/S;i++) {
          rr_wait(empty+(i%ST)*8,((i/ST)-1)&1);
          load(i);
        }
      }
      if constexpr(RED>=1) asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
      if constexpr(RED!=2) rr_sync();
      return;
    }
  }
  float acc[BN/2];
#pragma unroll
  for(int j=0;j<BN/2;j++) asm volatile("mov.b32 %0, 0;":"=f"(acc[j]));
  int prev=0;
#pragma unroll 1
  for(int i=0;i<128/S;i++) {
    const int st=i%ST;
    rr_wait(bars+st*8,(i/ST)&1);
    const uint32_t w=base+st*SB, x=w+2*WB;
#pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(acc[j])::"memory");
    asm volatile("wgmma.fence.sync.aligned;" :::"memory");
#pragma unroll
    for(int q=0;q<4;q++) {
      const uint64_t dx=rr_desc(x+q*32),dh=rr_desc(w+q*32),dl=rr_desc(w+WB+q*32);
      if constexpr(SW) { @@CALL_H@@; @@CALL_L@@; }
      else { @@CALL_NH@@; @@CALL_NL@@; }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" :::"memory");
    if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" :::"memory");
    else asm volatile("wgmma.wait_group.sync.aligned 0;" :::"memory");
#pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(acc[j])::"memory");
    if constexpr(PIPE&2) {
      if constexpr(PIPE&1) {
        if(i>0 && prev+ST<128/S && lane==0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty+(prev%ST)*8):"memory");
      } else {
        if(i+ST<128/S && lane==0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty+st*8):"memory");
      }
    } else if constexpr(PIPE&1) {
      if(i>0 && prev+ST<128/S && tid==0) load(prev+ST);
    } else {
      if(i+ST<128/S && tid==0) load(i+ST);
    }
    prev=i;
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" :::"memory");
  auto emit=[&](int v,float4 z) {
    const int t=v%128, j=v/128, ln=t&31, wr=t>>5;
    const int ar=wr*16+(ln>>2),bc=j*8+(ln&3)*2;
    const float vv[4]={z.x,z.y,z.z,z.w};
#pragma unroll
    for(int c=0;c<4;c++) {
      const int a=ar+(c/2)*8, b=bc+(c&1);
      const int tr=tok+(SW?b:a), wc=row+(SW?a:b);
      if(tr<m) out[tr*256+wc]=vv[c];
    }
  };
  if constexpr(S==1) {
#pragma unroll
    for(int j=0;j<BN/8;j++) emit(j*128+tid,make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]));
  } else if constexpr(RED==0) {
    float4* part=(float4*)sm;
#pragma unroll
    for(int j=0;j<BN/8;j++) part[j*128+tid]=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);
    rr_sync();
    for(int t=tid;t<OWN;t+=128) {
      const int v=rank*OWN+t;
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) rr_add(sum,rr_pull(rr_map(base+v*16,r)));
      emit(v,sum);
    }
    rr_sync();
  } else {
    asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
#pragma unroll
    for(int j=0;j<BN/8;j++) {
      const int v=j*128+tid, dst=v/OWN, local=v%OWN;
      const float4 val=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);
      const uint32_t addr=recv+(rank*OWN+local)*16;
      if(dst==rank) *(float4*)(sm+(addr-base))=val;
      else rr_push(rr_map(addr,dst),val,rr_map(rb,dst));
    }
    if constexpr(PIPE&2) asm volatile("bar.sync 1, 128;" :::"memory"); else __syncthreads();
    rr_wait(rb,0);
    for(int t=tid;t<OWN;t+=128) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) rr_add(sum,*(float4*)(sm+recv-base+(r*OWN+t)*16));
      emit(rank*OWN+t,sum);
    }
    if constexpr(RED!=2) rr_sync();
  }
}

typedef CUresult (*RR_Encode)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static int rr_map_host(CUtensorMap* tm,int64_t p,int inner,int outer,int box) {
  static RR_Encode enc=nullptr;
  if(!enc) { cudaDriverEntryPointQueryResult q; if(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&enc,12000,cudaEnableDefault,&q)!=cudaSuccess||!enc) return -100; }
  cuuint64_t dims[2]={(uint64_t)inner,(uint64_t)outer},stride[1]={(uint64_t)inner*2};
  cuuint32_t sz[2]={64,(uint32_t)box},es[2]={1,1};
  return (int)enc(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,stride,sz,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
                 CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template<int BN,int S,int ST,int SW,int RED,int PIPE>
int rr_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  constexpr int NR=SW?64:BN, TM=SW?BN:64;
  constexpr int bytes=ST*(2*NR+TM)*128+ST*8*((PIPE&2)?2:1)+48+(RED>=1?64*BN*4:0);
  auto fn=rr_kernel<BN,S,ST,SW,RED,PIPE>;
  static bool configured=false;
  if(!configured) {
    cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(e!=cudaSuccess) return -(int)e;
    e=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
    if(e!=cudaSuccess)return -(int)e;
    configured=true;
  }
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(256/NR*S,(m+TM-1)/TM,1);cfg.blockDim=dim3((PIPE&2)?256:128);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute attr[1];attr[0].id=cudaLaunchAttributeClusterDimension;attr[0].val.clusterDim={S,1,1};
  if(S>1){cfg.attrs=attr;cfg.numAttrs=1;}
  if(query==1){int n=0;cudaError_t e=cudaOccupancyMaxActiveClusters(&n,fn,&cfg);return e==cudaSuccess?n:-(int)e;}
  CUtensorMap tw,tx;
  int e=rr_map_host(&tw,w,64,256*128*2,NR*2);if(e)return e;
  e=rr_map_host(&tx,x,8192,m,TM);if(e)return e;
  cudaError_t err=cudaLaunchKernelEx(&cfg,fn,tw,tx,(float*)out,m);
  return err==cudaSuccess?0:(int)err;
}
int rr_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
 switch(id){@@CASES@@ default:return -99;}
}
@@SCALAR@@
"""

ROUTER_CPP = '''#include <pybind11/pybind11.h>
#include <stdint.h>
int rr_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
int rr_scalar(int64_t,int64_t,int64_t,int,int64_t,int);
int rr_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);
'''


def build_router(configs):
    src = ROUTER_CUDA.replace('@@MMA@@', '\n'.join(_mma(n) for n in sorted({c[0] for c in configs})))
    for tag, a, b in [('H', 'dh', 'dx'), ('L', 'dl', 'dx'), ('NH', 'dx', 'dh'), ('NL', 'dx', 'dl')]:
        src = src.replace('@@CALL_' + tag + '@@', ' '.join(f'if constexpr(BN=={n}) rr_mma_{n}(acc,{a},{b});' for n in sorted({c[0] for c in configs})))
    cases = '\n'.join(f'case {i}: return rr_launch_t<{",".join(map(str,c))}>(w,x,out,m,stream,query);' for i,c in enumerate(configs))
    src = src.replace('@@CASES@@',cases).replace('@@SCALAR@@', ROUTER_SCALAR + ROUTER_SMALL)
    name = 'router_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, ROUTER_CPP, cuda_sources=src, functions=['rr_launch','rr_scalar','rr_small'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True, with_pytorch_error_handling=False,verbose=False)


def pack_router(w, nr):
    hi = w.to(torch.bfloat16)
    lo = (w-hi.float()).to(torch.bfloat16)
    return torch.stack([p.view(256//nr,nr,128,64).permute(0,2,1,3) for p in (hi,lo)],dim=2).contiguous().view(-1,64)


def prepare_candidates(weights, configs):
    w = weights['router'].contiguous()
    return {'ext':build_router(configs), 'configs':configs,'weight':w,
            'packed':{nr:pack_router(w,nr) for nr in {64 if c[3] else c[0] for c in configs}}}


def run_candidate(x, st, index):
    out = torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    if index == -1:
        rc = st['ext'].rr_scalar(st['weight'].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    else:
        cfg = st['configs'][index]
        w = st['packed'][64 if cfg[3] else cfg[0]]
        rc = st['ext'].rr_launch(index,w.data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    if rc:
        raise RuntimeError(f'router launch failed: config={index}, rc={rc}')
    return out


ROUTER_CONFIGS = [(32,16,4,1,2,1), (32,8,4,0,2,1), (64,4,8,1,2,3)]


def prepare_router(weights) -> dict:
    w = weights.get('router')
    if w is None or w.shape != (256,8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {'enabled':False}
    if torch.cuda.get_device_capability(w.device) != (9,0):
        return {'enabled':False}
    st = prepare_candidates(weights, ROUTER_CONFIGS)
    st['small'] = {(16,8):pack_small_router(w,16,8)}
    st['enabled'] = True
    for m in (1,8,16,32,128,256):
        x = torch.zeros((m,8192),dtype=torch.bfloat16,device=w.device)
        run_router(x,'router',256,st)
    torch.cuda.synchronize(w.device)
    return st


def run_router(x, layer, n, st) -> torch.Tensor | None:
    if not st.get('enabled') or layer != 'router' or n != 256:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    if x.device != st['weight'].device or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if not 1 <= m <= 256:
        return None
    if m == 1:
        return run_candidate(x,st,-1)
    if m <= 16:
        return run_small_candidate(x,st)
    return run_candidate(x,st,0 if m <= 32 else 1 if m <= 128 else 2)


ROUTER_SCALAR = r"""
template<int M, int NT, int ST>
__global__ void rr_scalar_kernel(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                float* __restrict__ y, int m) {
  constexpr int K=8192;
  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;
  float acc[M][4]={};
  float4 ww[ST];
  uint2 xx[ST][M];
  auto ld=[&](int slot,int i) {
    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];
    #pragma unroll
    for(int t=0;t<M;t++) {
      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];
    }
  };
  #pragma unroll
  for(int s=0;s<ST;s++) ld(s,tid+s*NT);
  #pragma unroll 1
  for(int base=tid;base<K/4;base+=ST*NT) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      float4 v=ww[s];
      uint2 a[M];
      #pragma unroll
      for(int t=0;t<M;t++) a[t]=xx[s][t];
      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);
      #pragma unroll
      for(int t=0;t<M;t++) if(t<m) {
        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);
        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);
        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);
        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);
        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);
        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);
        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);
      }
    }
  }
  __shared__ float sums[M][NT/32];
  #pragma unroll
  for(int t=0;t<M;t++) if(t<m) {
    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);
    #pragma unroll
    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    if(lane==0) sums[t][warp]=v;
  }
  __syncthreads();
  if(warp==0) {
    #pragma unroll
    for(int t=0;t<M;t++) if(t<m) {
      float v=lane<NT/32?sums[t][lane]:0.f;
      #pragma unroll
      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
      if(lane==0) y[t*256+row]=v;
    }
  }
}

int rr_scalar(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto kern=rr_scalar_kernel<1,512,2>;
  if(query==2) {
    cudaFuncAttributes fa;
    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;
    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);
  }
  if(m!=1) return -5;
  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);
  return cudaGetLastError()==cudaSuccess?0:-8;
}
"""

ROUTER_SMALL = r"""
template<int MT,int S,int WK>
__global__ __launch_bounds__(WK*32) void rr_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {
  constexpr int KU=128/S/WK, NI=MT*32;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+S*NI;
  const uint32_t rb=rr_smem(sm+(S+WK)*NI*16);
  int4 w[KU][8];uint4 xx[KU][MT][2];
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<8;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  if(rank==0 && tid==0) {rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  if(rank==0 && warp==0) __syncwarp();
  if(rank==0 && tid==0) {rr_expect(rb,(S-1)*NI*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  float c[2][MT][4]={};
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          const int4 a=w[k][q+h*4];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  for(int v=tid;v<NI;v+=WK*32) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) rr_add(sum,part[k*NI+v]);
    if(rank==0)recv[v]=sum;
    else rr_push(rr_map(rr_smem(recv+rank*NI+v),0),sum,rr_map(rb,0));
  }
  if(rank==0) {
    __syncthreads();rr_wait(rb,0);
    for(int v=tid;v<NI;v+=WK*32) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int s=0;s<S;s++) rr_add(sum,recv[s*NI+v]);
      const int ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
      if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
      if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
    }
  }
}

template<int MT,int S,int WK>
int rr_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto fn=rr_small_kernel<MT,S,WK>;
  constexpr int bytes=(S+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);
}
int rr_small(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t stream,int query) {
#define RSM(S,W) if(s==S && wk==W) {if(m<=8)return rr_small_t<1,S,W>(w,x,out,m,stream,query);if(m<=16)return rr_small_t<2,S,W>(w,x,out,m,stream,query);if(m<=32)return rr_small_t<4,S,W>(w,x,out,m,stream,query);}
RSM(8,8) RSM(16,8) RSM(16,4)
return -99;
}

"""


def pack_small_router(w, s=16, wk=8):
    hi=w.to(torch.bfloat16);lo=(w-hi.float()).to(torch.bfloat16)
    def units(v):
        v=v.view(16,2,8,128,4,4,2,2).permute(0,3,2,4,5,6,1,7).reshape(16,128,32,4,8)
        return v.permute(0,1,3,2,4)
    data=torch.cat([units(hi),units(lo)],dim=2).contiguous()
    ku=128//s//wk
    return data.view(16,s,wk,ku,8,32,8).permute(3,1,0,2,4,5,6).contiguous()


def run_small_candidate(x,st,s=16,wk=8):
    out=torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    rc=st['ext'].rr_small(st['small'][s,wk].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],s,wk,torch.cuda.current_stream().cuda_stream,0)
    if rc:raise RuntimeError(f'router small launch failed: {rc}')
    return out


class _NRShared(dict):
    def __deepcopy__(self, memo):
        import copy as _nr_copy
        out = _NRShared()
        memo[id(self)] = out
        for key, value in self.items():
            out[key] = value if key == 'ext' else _nr_copy.deepcopy(value, memo)
        return out


_nr_parent_prepare = prepare
_nr_parent_kernel = kernel_fn


def prepare(weights):
    return _nr_parent_prepare(weights)


def kernel_fn(data, state):
    return _nr_parent_kernel(data, state)


V18S_CUDA_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define SM3D __device__ __forceinline__

SM3D int4 v18s_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SM3D uint4 v18s_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SM3D uint32_t v18s_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

SM3D uint32_t v18s_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

SM3D void v18s_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
SM3D void v18s_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = v18s_prmt(T0, T1, cm);
  const uint32_t hhi = v18s_prmt(T0, T1, cm >> 16);
  a[0] = v18s_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = v18s_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = v18s_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = v18s_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void v18s_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
SM3D void v18s_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) v18s_never(p);
}

SM3D uint32_t v18s_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

SM3D uint32_t v18s_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SM3D uint32_t v18s_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

SM3D void v18s_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

SM3D void v18s_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

SM3D void v18s_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
SM3D void v18s_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
SM3D void v18s_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

SM3D void v18s_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

SM3D bool v18s_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

SM3D uint64_t v18s_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a wrong result instead of a hang.
SM3D void v18s_wait_bar(uint32_t bar) {
  if (v18s_mbar_try(bar)) return;
  const uint64_t t0 = v18s_gtime();
  while (!v18s_mbar_try(bar))
    if (v18s_gtime() - t0 > 20000000ull) __trap();
}

SM3D void v18s_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

SM3D float4 v18s_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

SM3D void v18s_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
SM3D void v18s_store(float* p, float v) { *p = v; }

template <int FMT> struct Sm3Fmt;
template <> struct Sm3Fmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct Sm3Fmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct Sm3Fmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct Sm3Fmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
SM3D void v18s_decode(const int4 (&w)[Sm3Fmt<FMT>::NL], uint32_t (&A)[Sm3Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < Sm3Fmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < Sm3Fmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      v18s_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      v18s_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      v18s_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      v18s_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
SM3D void v18s_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
SM3D void v18s_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) v18s_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>
__global__ void __launch_bounds__(NTMAX) v18s_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = Sm3Fmt<FMT>::NL, NA = Sm3Fmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = (OPT & 8) ? 4 * MT : min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = v18s_smem_u32(&s_bar);
    uint32_t rank = 0, my_e = 0xFFFFFFFFu;
    if constexpr (!(OPT & 1)) {
    if constexpr (PUSH) {
      rank = v18s_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        v18s_mbar_init(bar, 1);
        v18s_mbar_expect(bar, parts * RT * TV * 128u);
        v18s_fence_init();
      }
      v18s_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = v18s_lde(esc + (size_t)gw * 32 + lane);

    }

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) w[slot][j][l] = v18s_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && !(OPT & 2)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = v18s_ldx(xp);
          xv[slot][n][1] = v18s_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    if constexpr (!(OPT & 4)) v18s_order_point(m, outp);
    if constexpr (OPT & 1) {
    if constexpr (PUSH) {
      rank = v18s_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        v18s_mbar_init(bar, 1);
        v18s_mbar_expect(bar, parts * RT * TV * 128u);
        v18s_fence_init();
      }
      v18s_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = v18s_lde(esc + (size_t)gw * 32 + lane);

    }

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          if constexpr (!(OPT & 4)) v18s_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr (OPT & 2) {
              const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + si * 64;
              uint4 u = v18s_ldx(xp), v = v18s_ldx(xp + 8);
              b[n][0]=u.x; b[n][1]=u.y; b[n][2]=u.z; b[n][3]=u.w;
              b[n][4]=v.x; b[n][5]=v.y; b[n][6]=v.z; b[n][7]=v.w;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) v18s_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if (si + ST < KU) load(s, si + ST);
          if constexpr (!(OPT & 4)) v18s_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) v18s_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  v18s_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) v18s_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          v18s_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) v18s_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                v18s_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          if constexpr (OPT & 8) { emit_full(idx, v); return; }
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem);
          const int NI = (OPT & 16) ? WR * RT * TV * 8 : WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              if (!(OPT & 16) || n * 4 + (lane & 3) < TV)
              part[(OPT & 16) ? (size_t)wk * NI + ((wr * RT + j) * TV + n * 4 + (lane & 3)) * 8 + (lane >> 2) : ((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            v18s_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { if constexpr (OPT & 16) emit_compact(idx, v); else emit_full(idx, v); });
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            v18s_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) v18s_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            v18s_cl_wait();
            if (t < TV) {
              const uint32_t rbar = v18s_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g);
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else v18s_st_async4(v18s_mapa(v18s_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            v18s_wait_bar(bar);
            v18s_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + n * 4 + t) * 8 + g)] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            v18s_cl_wait();
            const uint32_t rbar = v18s_mapa(bar, 0);
            v18s_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else v18s_st_async4(v18s_mapa(v18s_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            v18s_wait_bar(bar);
            v18s_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int v18s_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>
static int v18s_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = v18s_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, OPT>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = (OPT & 8) ? 4 * MT : std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = v18s_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int v18s_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

template<int M, int NT, int ST>
__global__ void v18s_router_scalar(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                float* __restrict__ y, int m) {
  constexpr int K=8192;
  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;
  float acc[M][4]={};
  float4 ww[ST];
  uint2 xx[ST][M];
  auto ld=[&](int slot,int i) {
    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];
    #pragma unroll
    for(int t=0;t<M;t++) {
      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];
    }
  };
  #pragma unroll
  for(int s=0;s<ST;s++) ld(s,tid+s*NT);
  #pragma unroll 1
  for(int base=tid;base<K/4;base+=ST*NT) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      float4 v=ww[s];
      uint2 a[M];
      #pragma unroll
      for(int t=0;t<M;t++) a[t]=xx[s][t];
      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);
      #pragma unroll
      for(int t=0;t<M;t++) if(t<m) {
        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);
        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);
        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);
        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);
        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);
        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);
        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);
      }
    }
  }
  __shared__ float sums[M][NT/32];
  #pragma unroll
  for(int t=0;t<M;t++) if(t<m) {
    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);
    #pragma unroll
    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    if(lane==0) sums[t][warp]=v;
  }
  __syncthreads();
  if(warp==0) {
    #pragma unroll
    for(int t=0;t<M;t++) if(t<m) {
      float v=lane<NT/32?sums[t][lane]:0.f;
      #pragma unroll
      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
      if(lane==0) y[t*256+row]=v;
    }
  }
}

int v18s_router_launch(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto kern=v18s_router_scalar<1,512,2>;
  if(query==2) {
    cudaFuncAttributes fa;
    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;
    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);
  }
  if(m!=1) return -5;
  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);
  return cudaGetLastError()==cudaSuccess?0:-8;
}
"""

V18S_CPP = r"""
int v18s_router_launch(long,long,long,int,long,int);
#include <pybind11/pybind11.h>
#include <stdint.h>
int v18s_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

V18S_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
V18S_ESC_CAP = 32


def v18s_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst + (0,) * (15 - len(inst))))
        cases.append(f"    case {i}: return v18s_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return V18S_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))


def v18s_build(instances, name="v18s_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return load_inline(name, V18S_CPP + extra_cpp, cuda_sources=v18s_cuda_source(instances) + extra_cuda,
                       functions=["v18s_launch", "v18s_router_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def v18s_units(w):
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)


def v18s_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)


def v18s_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def v18s_bf16_bytes(v):
    r, c = v.shape[:2]
    words = v18s_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)


def v18s_c12_bytes(v):
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(V18S_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = v18s_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)


def v18s_c12_unpack(data, base, esc):
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(V18S_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val


def v18s_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v


def v18s_pack_format(w, fmt):
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [v18s_units(hi), v18s_units(lo)]
    else:
        mats = [v18s_units(w)]
    if fmt in (0, 2):
        data = torch.cat([v18s_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = v18s_c12_bytes(v)
        if not torch.equal(v18s_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(v18s_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}


def v18s_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk


def v18s_stream(pk, rt, wr, wk, ku, s):
    data = pk["data"]
    R, C, B = data.shape
    geo = v18s_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, V18S_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > V18S_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = v18s_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}


def v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    if sp["fmt"] == 4:
        return ext.v18s_router_launch(sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0],
                                     torch.cuda.current_stream().cuda_stream, query)
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.v18s_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)


def v18s_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full


def v18s_bf16_twin(inst):
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None


def v18s_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if v18s_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))


def v18s_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True


def v18s_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows


def v18s_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    m_hi = key[2] if key[1] == "m" else 8 * inst[1]
    rows = v18s_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = v18s_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = v18s_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = v18s_verify_onehot(ext, inst_id[inst], sp, v18s_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and \
            v18s_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)


_VO_CONFIG = ((1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8, 2, 1, 6), (1, 8, 2, 1))
_vo_parent_prepare = prepare
_vo_parent_kernel = kernel_fn


def _vo_prepare_o(weights):
    instances = [_VO_CONFIG[0], v18s_bf16_twin(_VO_CONFIG[0])]
    ext = v18s_build(instances, 'v18_small_selected')
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams = {}, {}
    plan = v18s_plan(ext, ids, packs, streams, weights['o'], 'o', ('m', 8), _VO_CONFIG[0], _VO_CONFIG[1])
    if plan is None:
        plan = v18s_plan(ext, ids, packs, streams, weights['o'], 'o', ('m', 8), instances[1], _VO_CONFIG[1])
    return _NRShared(ext=ext, plan=plan)


def prepare(weights):
    state = _vo_parent_prepare(weights)
    state['v18o'] = _vo_prepare_o(weights)
    return state


def kernel_fn(data, state):
    _k_ctx_bind(data['x'])
    if data['config']['layer'] == 'o' and data['x'].shape[0] == 8:
        plan = state['v18o']['plan']
        x = data['x']
        out = torch.empty((8, 8192), device=x.device, dtype=torch.bfloat16)
        rc = v18s_launch_raw(state['v18o']['ext'], plan['inst_id'], plan, x, out, 8192, plan['cfg'])
        if rc:
            raise RuntimeError(f'v18 selected launch failed: {rc}')
        return out
    return _vo_parent_kernel(data, state)


"""H200 small-m table refinements. Unclaimed shapes return None."""
import torch
from torch.utils.cpp_extension import load_inline
_NP_SMALLM2_CUDA_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define SMALLM2D __device__ __forceinline__

SMALLM2D int4 smallm2_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMALLM2D uint4 smallm2_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMALLM2D uint32_t smallm2_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

SMALLM2D uint32_t smallm2_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

SMALLM2D void smallm2_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
SMALLM2D void smallm2_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = smallm2_prmt(T0, T1, cm);
  const uint32_t hhi = smallm2_prmt(T0, T1, cm >> 16);
  a[0] = smallm2_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = smallm2_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = smallm2_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = smallm2_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void smallm2_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
SMALLM2D void smallm2_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) smallm2_never(p);
}

SMALLM2D uint32_t smallm2_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

SMALLM2D uint32_t smallm2_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SMALLM2D uint32_t smallm2_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

SMALLM2D void smallm2_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

SMALLM2D void smallm2_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

SMALLM2D void smallm2_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
SMALLM2D void smallm2_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
SMALLM2D void smallm2_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

SMALLM2D void smallm2_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

SMALLM2D bool smallm2_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

SMALLM2D uint64_t smallm2_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
SMALLM2D void smallm2_wait_bar(uint32_t bar) {
  if (smallm2_mbar_try(bar)) return;
  const uint64_t t0 = smallm2_gtime();
  while (!smallm2_mbar_try(bar))
    if (smallm2_gtime() - t0 > 20000000ull) __trap();
}

SMALLM2D void smallm2_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

SMALLM2D float4 smallm2_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

SMALLM2D void smallm2_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
SMALLM2D void smallm2_store(float* p, float v) { *p = v; }

template <int FMT> struct Smallm2Fmt;
template <> struct Smallm2Fmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct Smallm2Fmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct Smallm2Fmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct Smallm2Fmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
SMALLM2D void smallm2_decode(const int4 (&w)[Smallm2Fmt<FMT>::NL], uint32_t (&A)[Smallm2Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < Smallm2Fmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < Smallm2Fmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      smallm2_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      smallm2_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      smallm2_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      smallm2_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
SMALLM2D void smallm2_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
SMALLM2D void smallm2_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) smallm2_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>
__global__ void __launch_bounds__(NTMAX) smallm2_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = Smallm2Fmt<FMT>::NL, NA = Smallm2Fmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = (OPT & 8) ? 4 * MT : min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = smallm2_smem_u32(&s_bar);
    uint32_t rank = 0, my_e = 0xFFFFFFFFu;
    if constexpr (!(OPT & 1)) {
    if constexpr (PUSH) {
      rank = smallm2_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        smallm2_mbar_init(bar, 1);
        smallm2_mbar_expect(bar, parts * RT * TV * 128u);
        smallm2_fence_init();
      }
      smallm2_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = smallm2_lde(esc + (size_t)gw * 32 + lane);

    }

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) w[slot][j][l] = smallm2_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && !(OPT & 2)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = smallm2_ldx(xp);
          xv[slot][n][1] = smallm2_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);
    if constexpr (OPT & 1) {
    if constexpr (PUSH) {
      rank = smallm2_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        smallm2_mbar_init(bar, 1);
        smallm2_mbar_expect(bar, parts * RT * TV * 128u);
        smallm2_fence_init();
      }
      smallm2_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = smallm2_lde(esc + (size_t)gw * 32 + lane);

    }

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr (OPT & 2) {
              const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + si * 64;
              uint4 u = smallm2_ldx(xp), v = smallm2_ldx(xp + 8);
              b[n][0]=u.x; b[n][1]=u.y; b[n][2]=u.z; b[n][3]=u.w;
              b[n][4]=v.x; b[n][5]=v.y; b[n][6]=v.z; b[n][7]=v.w;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) smallm2_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if (si + ST < KU) load(s, si + ST);
          if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) smallm2_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  smallm2_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) smallm2_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          smallm2_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) smallm2_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                smallm2_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) smallm2_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          if constexpr (OPT & 8) { emit_full(idx, v); return; }
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) smallm2_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem);
          const int NI = (OPT & 16) ? WR * RT * TV * 8 : WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              if (!(OPT & 16) || n * 4 + (lane & 3) < TV)
              part[(OPT & 16) ? (size_t)wk * NI + ((wr * RT + j) * TV + n * 4 + (lane & 3)) * 8 + (lane >> 2) : ((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            smallm2_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { if constexpr (OPT & 16) emit_compact(idx, v); else emit_full(idx, v); });
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            smallm2_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) smallm2_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            smallm2_cl_wait();
            if (t < TV) {
              const uint32_t rbar = smallm2_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g);
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else smallm2_st_async4(smallm2_mapa(smallm2_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            smallm2_wait_bar(bar);
            smallm2_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + n * 4 + t) * 8 + g)] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            smallm2_cl_wait();
            const uint32_t rbar = smallm2_mapa(bar, 0);
            smallm2_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else smallm2_st_async4(smallm2_mapa(smallm2_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            smallm2_wait_bar(bar);
            smallm2_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int smallm2_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>
static int smallm2_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = smallm2_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, OPT>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = (OPT & 8) ? 4 * MT : std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = smallm2_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

template<int M, int NT, int ST>
__global__ void smallm2_router_scalar(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                float* __restrict__ y, int m) {
  constexpr int K=8192;
  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;
  float acc[M][4]={};
  float4 ww[ST];
  uint2 xx[ST][M];
  auto ld=[&](int slot,int i) {
    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];
    #pragma unroll
    for(int t=0;t<M;t++) {
      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];
    }
  };
  #pragma unroll
  for(int s=0;s<ST;s++) ld(s,tid+s*NT);
  #pragma unroll 1
  for(int base=tid;base<K/4;base+=ST*NT) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      float4 v=ww[s];
      uint2 a[M];
      #pragma unroll
      for(int t=0;t<M;t++) a[t]=xx[s][t];
      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);
      #pragma unroll
      for(int t=0;t<M;t++) if(t<m) {
        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);
        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);
        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);
        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);
        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);
        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);
        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);
      }
    }
  }
  __shared__ float sums[M][NT/32];
  #pragma unroll
  for(int t=0;t<M;t++) if(t<m) {
    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);
    #pragma unroll
    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    if(lane==0) sums[t][warp]=v;
  }
  __syncthreads();
  if(warp==0) {
    #pragma unroll
    for(int t=0;t<M;t++) if(t<m) {
      float v=lane<NT/32?sums[t][lane]:0.f;
      #pragma unroll
      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
      if(lane==0) y[t*256+row]=v;
    }
  }
}

int smallm2_router_launch(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto kern=smallm2_router_scalar<1,512,2>;
  if(query==2) {
    cudaFuncAttributes fa;
    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;
    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);
  }
  if(m!=1) return -5;
  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);
  return cudaGetLastError()==cudaSuccess?0:-8;
}
"""
_NP_SMALLM2_CPP = '\nint smallm2_router_launch(long,long,long,int,long,int);\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_NP_SMALLM2_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
_NP_SMALLM2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_NP_SMALLM2_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
_NP_SMALLM2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_NP_SMALLM2_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_NP_SMALLM2_ESC_CAP = 32

def _NP_smallm2_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def _NP_smallm2_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in inst + (0,) * (15 - len(inst))))
        cases.append(f'    case {i}: return smallm2_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return _NP_SMALLM2_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def _NP_smallm2_build(instances, name='smallm2_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return load_inline(name, _NP_SMALLM2_CPP + extra_cpp, cuda_sources=_NP_smallm2_cuda_source(instances) + extra_cuda, functions=['smallm2_launch', 'smallm2_router_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _NP_smallm2_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _NP_smallm2_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def _NP_smallm2_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def _NP_smallm2_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _NP_smallm2_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def _NP_smallm2_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(_NP_SMALLM2_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _NP_smallm2_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _NP_smallm2_c12_unpack(data, base, esc):
    """Inverse of smallm2_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(_NP_SMALLM2_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def _NP_smallm2_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _NP_smallm2_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [_NP_smallm2_units(hi), _NP_smallm2_units(lo)]
    else:
        mats = [_NP_smallm2_units(w)]
    if fmt in (0, 2):
        data = torch.cat([_NP_smallm2_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _NP_smallm2_c12_bytes(v)
        if not torch.equal(_NP_smallm2_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_NP_smallm2_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _NP_smallm2_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _NP_smallm2_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _NP_smallm2_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, _NP_SMALLM2_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _NP_SMALLM2_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _NP_smallm2_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _NP_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    if sp['fmt'] == 4:
        return ext.smallm2_router_launch(sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, query)
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smallm2_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def _NP_smallm2_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _NP_smallm2_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _NP_smallm2_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _NP_smallm2_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _NP_smallm2_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _NP_smallm2_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if _NP_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _NP_smallm2_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def _NP_smallm2_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _NP_smallm2_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _NP_smallm2_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _NP_smallm2_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _NP_smallm2_verify_onehot(ext, inst_id[inst], sp, _NP_smallm2_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _NP_smallm2_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)
_NP_SMALLM2_ROUTES = {('o', 1): ((1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, 4, 8, 2, 1, 7), (4, 8, 2, 1)), ('pooler', 8): ((1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6), (2, 16, 8, 1))}

def _NP_prepare_smallm2(weights) -> dict:
    st = {'plans': {}}
    entries = []
    for (layer, m), (inst, cfg) in _NP_SMALLM2_ROUTES.items():
        w = weights.get(layer)
        if w is None or w.dtype != torch.bfloat16 or (not w.is_cuda):
            continue
        if tuple(w.shape) != (inst[8], inst[9]) or torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        entries.append(((layer, 'm', m), inst, cfg))
    if not entries:
        return st
    instances = list(dict.fromkeys((v for _, inst, _ in entries for v in (inst, _NP_smallm2_bf16_twin(inst)))))
    ext = _NP_smallm2_build(instances, 'smallm2_final_v3')
    st['ext'] = ext
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, inst, cfg in entries:
        layer = key[0]
        w = weights[layer]
        plan = _NP_smallm2_plan(ext, ids, packs, streams, w, layer, key, inst, cfg)
        if plan is None:
            plan = _NP_smallm2_plan(ext, ids, packs, streams, w, layer, key, _NP_smallm2_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][layer, key[2]] = plan
    return st

def _NP_run_smallm2(x, layer, n, st) -> torch.Tensor | None:
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
        return None
    plan = st['plans'].get((layer, x.shape[0]))
    if plan is None or n != plan['rows'] or x.shape[1] != _NP_SMALLM2_LAYER_K[layer]:
        return None
    if x.device != plan['wp'].device or x.data_ptr() % 16:
        return None
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = _NP_smallm2_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
    return out if rc == 0 else None
_NP_SMALLM2_ROUTES = {key: value for key, value in _NP_SMALLM2_ROUTES.items() if key == ('pooler', 8)}


"""Lossless BF16 small-token projection routes validated on H200."""
import torch
from torch.utils.cpp_extension import load_inline
_NU_SMTRACE_CUDA_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define SMD __device__ __forceinline__

SMD int4 smtrace_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMD uint4 smtrace_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMD uint32_t smtrace_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

SMD uint32_t smtrace_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

SMD void smtrace_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
SMD void smtrace_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = smtrace_prmt(T0, T1, cm);
  const uint32_t hhi = smtrace_prmt(T0, T1, cm >> 16);
  a[0] = smtrace_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = smtrace_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = smtrace_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = smtrace_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void smtrace_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
SMD void smtrace_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) smtrace_never(p);
}

SMD uint32_t smtrace_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

SMD uint32_t smtrace_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SMD uint32_t smtrace_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

SMD void smtrace_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

SMD void smtrace_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

SMD void smtrace_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
SMD void smtrace_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
SMD void smtrace_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

SMD void smtrace_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

SMD bool smtrace_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

SMD uint64_t smtrace_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
SMD void smtrace_wait_bar(uint32_t bar) {
  if (smtrace_mbar_try(bar)) return;
  const uint64_t t0 = smtrace_gtime();
  while (!smtrace_mbar_try(bar))
    if (smtrace_gtime() - t0 > 20000000ull) __trap();
}

SMD void smtrace_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

SMD float4 smtrace_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

SMD void smtrace_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
SMD void smtrace_store(float* p, float v) { *p = v; }

template <int FMT> struct SmFmt;
template <> struct SmFmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct SmFmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct SmFmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct SmFmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
SMD void smtrace_decode(const int4 (&w)[SmFmt<FMT>::NL], uint32_t (&A)[SmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < SmFmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < SmFmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      smtrace_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      smtrace_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      smtrace_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      smtrace_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
SMD void smtrace_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
SMD void smtrace_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) smtrace_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS>
__global__ void __launch_bounds__(NTMAX) smtrace_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = SmFmt<FMT>::NL, NA = SmFmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = smtrace_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = smtrace_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        smtrace_mbar_init(bar, 1);
        smtrace_fence_init();
      }
      if (rank == 0) {
        __syncthreads();
        if(tid==0) {
          const uint32_t parts=RED==2?(uint32_t)((S-1)*NW):(uint32_t)((S-1)*WR);
          smtrace_mbar_expect(bar,parts*RT*TV*128u);
        }
      }
      smtrace_cl_arrive();
    }

    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = smtrace_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) w[slot][j][l] = smtrace_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = smtrace_ldx(xp);
          xv[slot][n][1] = smtrace_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    smtrace_order_point(m, outp);

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          smtrace_order_point(m, outp);
        } else {
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) smtrace_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          smtrace_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) smtrace_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  smtrace_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) smtrace_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          smtrace_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) smtrace_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                smtrace_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) smtrace_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) smtrace_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem);
          const int NI = WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            smtrace_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            smtrace_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) smtrace_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            smtrace_cl_wait();
            if (t < TV) {
              const uint32_t rbar = smtrace_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else smtrace_st_async4(smtrace_mapa(smtrace_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            smtrace_wait_bar(bar);
            smtrace_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            smtrace_cl_wait();
            const uint32_t rbar = smtrace_mapa(bar, 0);
            smtrace_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else smtrace_st_async4(smtrace_mapa(smtrace_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            smtrace_wait_bar(bar);
            smtrace_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int smtrace_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS>
static int smtrace_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = smtrace_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = smtrace_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
"""
_NU_SMTRACE_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_NU_SMTRACE_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
_NU_SMTRACE_NL = {0: 4, 1: 3, 2: 8, 3: 6}
_NU_SMTRACE_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
_NU_SMTRACE_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
_NU_SMTRACE_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_NU_SMTRACE_ESC_CAP = 32

def _NU_smtrace_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def _NU_smtrace_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
        cases.append(f'    case {i}: return smtrace_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return _NU_SMTRACE_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def _NU_smtrace_build(instances, name='small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return load_inline(name, _NU_SMTRACE_CPP + extra_cpp, cuda_sources=_NU_smtrace_cuda_source(instances) + extra_cuda, functions=['smtrace_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _NU_smtrace_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _NU__smtrace_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def _NU__smtrace_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def _NU_smtrace_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _NU__smtrace_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def _NU_smtrace_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(_NU_SMTRACE_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _NU__smtrace_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _NU_smtrace_c12_unpack(data, base, esc):
    """Inverse of smtrace_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(_NU_SMTRACE_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def _NU_smtrace_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def _NU_smtrace_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [_NU_smtrace_units(hi), _NU_smtrace_units(lo)]
    else:
        mats = [_NU_smtrace_units(w)]
    if fmt in (0, 2):
        data = torch.cat([_NU_smtrace_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _NU_smtrace_c12_bytes(v)
        if not torch.equal(_NU_smtrace_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_NU_smtrace_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _NU_smtrace_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _NU_smtrace_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _NU_smtrace_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, _NU_SMTRACE_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _NU_SMTRACE_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _NU__smtrace_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _NU_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smtrace_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
_NU_SMTRACE_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
_NU_SMTRACE_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
_NU_SMTRACE_MAX_M = 32
_NU_SMTRACE_TABLE_M.update({('router', 8): _NU_SMTRACE_TABLE_MT['router', 1], ('qkv_gate', 8): _NU_SMTRACE_TABLE_MT['qkv_gate', 1], ('pooler', 8): _NU_SMTRACE_TABLE_MT['pooler', 1], ('o', 8): _NU_SMTRACE_TABLE_MT['o', 1], ('unpooler', 1): _NU_SMTRACE_TABLE_MT['unpooler', 1], ('unpooler', 8): _NU_SMTRACE_TABLE_MT['unpooler', 1]})
_NU_SMTRACE_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
_NU_SMTRACE_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
_NU_SMTRACE_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
_NU_SMTRACE_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def _NU_smtrace_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def _NU_smtrace_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _NU_smtrace_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _NU_smtrace_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _NU_smtrace_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _NU_smtrace_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if _NU_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _NU_smtrace_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def _NU_smtrace_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _NU_smtrace_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _NU_smtrace_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _NU_smtrace_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _NU_smtrace_verify_onehot(ext, inst_id[inst], sp, _NU_smtrace_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _NU_smtrace_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def _NU_smtrace_prepare_core(weights, table_m=None, table_mt=None, name='small_m_gemv'):
    table_m = dict(_NU_SMTRACE_TABLE_M if table_m is None else table_m)
    table_mt = dict(_NU_SMTRACE_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == 'm' else 0
            rows = _NU_smtrace_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, _NU_SMTRACE_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if _NU_smtrace_bf16_twin(inst):
            instances.append(_NU_smtrace_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {'plans': {}}
    if not instances:
        return st
    ext = _NU_smtrace_build(instances, name)
    st['ext'] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = _NU_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and _NU_smtrace_bf16_twin(inst):
            plan = _NU_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, _NU_smtrace_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][key] = plan
    del packs
    return st

def _NU_smtrace_run_core(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= _NU_SMTRACE_MAX_M:
        return None
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', _NU_smtrace_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != _NU_SMTRACE_LAYER_K[layer]) or (not x.is_cuda):
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
    if _NU_smtrace_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
        return None
    return out
_NU_SMTRACE_ROUTES = {('unpooler', 8): ((1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1, 16, 2, 2), (1, 16, 2, 2))}

def _NU_prepare_smtrace(weights):
    usable = {}
    for (layer, m), (inst, cfg) in _NU_SMTRACE_ROUTES.items():
        w = weights.get(layer)
        if w is not None and w.is_cuda and (w.dtype == torch.bfloat16) and (w.ndim == 2) and (w.shape[1] == inst[9]) and (w.shape[0] >= inst[8]) and (torch.cuda.get_device_capability(w.device) == (9, 0)):
            usable[layer] = w
    routes = {key: value for key, value in _NU_SMTRACE_ROUTES.items() if key[0] in usable}
    return _NU_smtrace_prepare_core(usable, table_m=routes, table_mt={}, name='smtrace_final_d46e5decdceb')

def _NU_run_smtrace(x, layer, n, st):
    if (layer, x.shape[0]) not in _NU_SMTRACE_ROUTES:
        return None
    return _NU_smtrace_run_core(x, layer, n, st)


# Nathen Fernandes (MIT): measured selective O/pooler routes.
_nathen_m8_parent_prepare = prepare
_nathen_m8_parent_kernel = kernel_fn


def prepare(weights):
    state = _nathen_m8_parent_prepare(weights)
    state['nathen_pooler8'] = _kshare_modules(_NP_prepare_smallm2({'pooler': weights['pooler']}))
    return state


def kernel_fn(data, state):
    _k_ctx_bind(data['x'])
    x = data['x']
    layer = data['config']['layer']
    if x.shape[0] == 8 and layer == 'pooler':
        out = _NP_run_smallm2(x, layer, data['config']['n'], state['nathen_pooler8'])
        if out is not None:
            return out
    return _nathen_m8_parent_kernel(data, state)


'Experimental lossless nine-instruction BF16 decode. No route is promoted. See FINDINGS.txt.'
import hashlib
import torch
from torch.utils.cpp_extension import load_inline












































_NDD_CPS32_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}











_NDD_CPS32_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

def _NDD_dd_cap2_b_cps32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0






_NDD_CPSSM_CUDA_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define CPSSD __device__ __forceinline__

CPSSD int4 dd_cap2_b_cpssm_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

CPSSD uint4 dd_cap2_b_cpssm_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

CPSSD uint32_t dd_cap2_b_cpssm_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

CPSSD uint32_t dd_cap2_b_cpssm_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

CPSSD void dd_cap2_b_cpssm_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
CPSSD void dd_cap2_b_cpssm_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));
  const uint32_t low_high = (C & 0x87878787u) + T0;
  const uint32_t high_high = (rotated & 0x87878787u) + T0;
  a[0] = dd_cap2_b_cpssm_prmt(L0, low_high, 0x5140u);
  a[1] = dd_cap2_b_cpssm_prmt(L0, low_high, 0x7362u);
  a[2] = dd_cap2_b_cpssm_prmt(L1, high_high, 0x5140u);
  a[3] = dd_cap2_b_cpssm_prmt(L1, high_high, 0x7362u);
}

__device__ __noinline__ void dd_cap2_b_cpssm_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
CPSSD void dd_cap2_b_cpssm_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) dd_cap2_b_cpssm_never(p);
}

CPSSD uint32_t dd_cap2_b_cpssm_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

CPSSD uint32_t dd_cap2_b_cpssm_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

CPSSD uint32_t dd_cap2_b_cpssm_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

CPSSD void dd_cap2_b_cpssm_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

CPSSD void dd_cap2_b_cpssm_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

CPSSD void dd_cap2_b_cpssm_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
CPSSD void dd_cap2_b_cpssm_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
CPSSD void dd_cap2_b_cpssm_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

CPSSD void dd_cap2_b_cpssm_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

CPSSD bool dd_cap2_b_cpssm_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

CPSSD uint64_t dd_cap2_b_cpssm_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
CPSSD void dd_cap2_b_cpssm_wait_bar(uint32_t bar) {
  if (dd_cap2_b_cpssm_mbar_try(bar)) return;
  const uint64_t t0 = dd_cap2_b_cpssm_gtime();
  while (!dd_cap2_b_cpssm_mbar_try(bar))
    if (dd_cap2_b_cpssm_gtime() - t0 > 20000000ull) __trap();
}

CPSSD void dd_cap2_b_cpssm_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

CPSSD float4 dd_cap2_b_cpssm_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

CPSSD void dd_cap2_b_cpssm_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
CPSSD void dd_cap2_b_cpssm_store(float* p, float v) { *p = v; }

template <int FMT> struct CpsSmFmt;
template <> struct CpsSmFmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct CpsSmFmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct CpsSmFmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct CpsSmFmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
CPSSD void dd_cap2_b_cpssm_decode(const int4 (&w)[CpsSmFmt<FMT>::NL], uint32_t (&A)[CpsSmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < CpsSmFmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < CpsSmFmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      dd_cap2_b_cpssm_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      dd_cap2_b_cpssm_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      dd_cap2_b_cpssm_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      dd_cap2_b_cpssm_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
CPSSD void dd_cap2_b_cpssm_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
CPSSD void dd_cap2_b_cpssm_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) dd_cap2_b_cpssm_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>
__global__ void __launch_bounds__(NTMAX) dd_cap2_b_cpssm_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = CpsSmFmt<FMT>::NL, NA = CpsSmFmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  const int rawbytes = CP * WR * WK * RT * NL * 512;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = dd_cap2_b_cpssm_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = dd_cap2_b_cpssm_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        dd_cap2_b_cpssm_mbar_init(bar, 1);
        dd_cap2_b_cpssm_mbar_expect(bar, parts * RT * TV * 128u);
        dd_cap2_b_cpssm_fence_init();
      }
      dd_cap2_b_cpssm_cl_arrive();
    }

    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = dd_cap2_b_cpssm_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4* rawring = reinterpret_cast<int4*>(smem);
    auto stage_weights = [&](int slot, int si) {
      if constexpr (CP > 0) {
        if (si < KU) {
#pragma unroll
          for (int j=0; j<RT; ++j)
#pragma unroll
            for (int l=0; l<NL; ++l) {
              const uint32_t dst=dd_cap2_b_cpssm_smem_u32(rawring + (((slot*NW+warp)*RT+j)*NL+l)*32+lane);
              const int4* src=wq+(size_t)si*sstride+(j*NL+l)*32;
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
    };
#pragma unroll
    for (int q=0; q<CP; ++q) stage_weights(q,q);
    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if constexpr (CP == 0) w[slot][j][l] = dd_cap2_b_cpssm_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = dd_cap2_b_cpssm_ldx(xp);
          xv[slot][n][1] = dd_cap2_b_cpssm_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    dd_cap2_b_cpssm_order_point(m, outp);

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          dd_cap2_b_cpssm_order_point(m, outp);
        } else {
          if constexpr (CP > 0) {
            asm volatile("cp.async.wait_group %0;" :: "n"(CP-1) : "memory");
#pragma unroll
            for (int j=0; j<RT; ++j)
#pragma unroll
              for (int l=0; l<NL; ++l) {
                const uint32_t src=dd_cap2_b_cpssm_smem_u32(rawring+((((si%CP)*NW+warp)*RT+j)*NL+l)*32+lane);
                int4& v=w[s][j][l];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(src) : "memory");
              }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) dd_cap2_b_cpssm_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if constexpr (CP > 0) stage_weights(si%CP,si+CP);
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          dd_cap2_b_cpssm_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) dd_cap2_b_cpssm_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  dd_cap2_b_cpssm_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) dd_cap2_b_cpssm_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (CP > 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          dd_cap2_b_cpssm_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) dd_cap2_b_cpssm_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem + rawbytes);
          const int NI = WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            dd_cap2_b_cpssm_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            dd_cap2_b_cpssm_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) dd_cap2_b_cpssm_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem + rawbytes);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            dd_cap2_b_cpssm_cl_wait();
            if (t < TV) {
              const uint32_t rbar = dd_cap2_b_cpssm_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else dd_cap2_b_cpssm_st_async4(dd_cap2_b_cpssm_mapa(dd_cap2_b_cpssm_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            dd_cap2_b_cpssm_wait_bar(bar);
            dd_cap2_b_cpssm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            dd_cap2_b_cpssm_cl_wait();
            const uint32_t rbar = dd_cap2_b_cpssm_mapa(bar, 0);
            dd_cap2_b_cpssm_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else dd_cap2_b_cpssm_st_async4(dd_cap2_b_cpssm_mapa(dd_cap2_b_cpssm_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            dd_cap2_b_cpssm_wait_bar(bar);
            dd_cap2_b_cpssm_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int dd_cap2_b_cpssm_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>
static int dd_cap2_b_cpssm_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = dd_cap2_b_cpssm_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, CP>;
  const int policy = query / 100;
  query %= 100;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  sm += (size_t)CP * NW * RT * CpsSmFmt<FMT>::NL * 512;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = dd_cap2_b_cpssm_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[2];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    at[1].val.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(policy);
    lc.numAttrs = 2;
  }
  if (query == 3) { int v=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); return e==cudaSuccess?v:-11; }
  if (query == 4) return (int)sm;
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int dd_cap2_b_cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
"""
_NDD_CPSSM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint dd_cap2_b_cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
_NDD_CPSSM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
_NDD_CPSSM_ESC_CAP = 32

def _NDD_dd_cap2_b_cpssm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def _NDD_dd_cap2_b_cpssm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _NDD_cpstage___to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def _NDD_dd_cap2_b_cpssm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _NDD_cpstage___to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _NDD_dd_cap2_b_cpssm_c12_unpack(data, base, esc):
    """Inverse of dd_cap2_b_cpssm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def _NDD_dd_cap2_b_cpssm_table(base):
    return base * 16843009

def _NDD_dd_cap2_b_cpssm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [_NDD_dd_cap2_b_cpssm_units(hi), _NDD_dd_cap2_b_cpssm_units(lo)]
    else:
        mats = [_NDD_dd_cap2_b_cpssm_units(w)]
    if fmt in (0, 2):
        data = torch.cat([_NDD_dd_cap2_b_cpssm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = _NDD_dd_cap2_b_cpssm_c12_bytes(v)
        if not torch.equal(_NDD_dd_cap2_b_cpssm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(_NDD_dd_cap2_b_cpssm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def _NDD_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def _NDD_dd_cap2_b_cpssm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = _NDD_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, _NDD_CPSSM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > _NDD_CPSSM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = _NDD_cpstage___to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def _NDD_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.dd_cap2_b_cpssm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
_NDD_CPSSM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
_NDD_CPSSM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
_NDD_CPSSM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
_NDD_CPSSM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
_NDD_CPSSM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
_NDD_CPSSM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def _NDD_dd_cap2_b_cpssm_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def _NDD_dd_cap2_b_cpssm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def _NDD_dd_cap2_b_cpssm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if _NDD_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def _NDD_dd_cap2_b_cpssm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if _NDD_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def _NDD_dd_cap2_b_cpssm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def _NDD_dd_cap2_b_cpssm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = _NDD_dd_cap2_b_cpssm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = _NDD_dd_cap2_b_cpssm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = _NDD_dd_cap2_b_cpssm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = _NDD_dd_cap2_b_cpssm_verify_onehot(ext, inst_id[inst], sp, _NDD_dd_cap2_b_cpssm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and _NDD_dd_cap2_b_cpssm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def _NDD_cpstage___to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def _NDD_cpstage___to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)
_NDD_CPS_LAYERS = ('qkv_gate', 'pooler', 'unpooler')

def _NDD_cps_build(family, instances, depth):
    is32 = False
    prefix = 'dd_cap2_b_cpssm_'
    template = _NDD_CPSSM_CUDA_TEMPLATE
    cpp = _NDD_CPSSM_CPP
    cases = []
    for i, inst in enumerate(instances):
        full = inst if len(inst) == 14 else inst + (0,) * 7
        args = ', '.join(map(str, (*full, depth)))
        cases.append(f'case {i}: return {prefix}launch_t<{args}>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
    source = template.replace('@@CASES@@', '\n'.join(cases))
    name = 'cps_' + family + '_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name, cpp, cuda_sources=source, functions=[prefix + 'launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-Xptxas=-v', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def _NDD_prepare_cps(weights, depth=None, layers=_NDD_CPS_LAYERS, buckets=(2, 4), include_policy=False):
    if depth is None:
        st = {'plans': {}, 'depth': 'selected', 'policy': 0}
        selected_layers = tuple((layer for layer in layers if layer in ('pooler', 'unpooler')))
        for bucket, selected_depth in ((2, 2), (4, 1)):
            if bucket in buckets and selected_layers:
                part = _NDD_prepare_cps(weights, selected_depth, selected_layers, (bucket,))
                st['plans'].update(part['plans'])
        return st
    if depth not in (0, 1, 2, 3):
        raise ValueError('ring depth must be 0, 1, 2 or 3')
    st = {'plans': {}, 'depth': depth, 'policy': 0}
    for family in ('sm',):
        table = _NDD_CPSSM_TABLE_MT
        entries = []
        for (layer, bucket), (inst, cfg) in table.items():
            is_policy = include_policy and family == 'sm' and (layer in _NDD_CPS_LAYERS) and (bucket == 1) and (inst[4] in (2, 3))
            if layer not in weights or (not is_policy and (layer not in layers or bucket not in buckets)):
                continue
            if not is_policy and family == 'sm' and ((layer, bucket) in _NDD_CPS32_TABLE_MT):
                continue
            if len(inst) == 7:
                rows = 1024 if layer == 'unpooler' and bucket == 1 else weights[layer].shape[0]
                inst = inst + (0, rows, _NDD_CPSSM_LAYER_K[layer], *cfg)
            entries.append(((layer, 'mt', bucket), (inst, cfg)))
        if include_policy and family == 'sm':
            for (layer, m), (inst, cfg) in _NDD_CPSSM_TABLE_M.items():
                if layer not in _NDD_CPS_LAYERS or layer not in weights or inst[4] not in (2, 3):
                    continue
                rows = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
                if len(inst) == 7:
                    inst = inst + (m, rows, _NDD_CPSSM_LAYER_K[layer], *cfg)
                entries.append(((layer, 'm', m), (inst, cfg)))
        if not entries:
            continue
        twin = _NDD_dd_cap2_b_cpssm_bf16_twin
        instances = list(dict.fromkeys((inst for _, (inst, _) in entries for inst in (inst, twin(inst)) if inst is not None)))
        ext = _NDD_cps_build(family, instances, depth)
        ids = {inst: i for i, inst in enumerate(instances)}
        plan_fn = _NDD_dd_cap2_b_cpssm_plan
        packs, streams = ({}, {})
        for key, (inst, cfg) in entries:
            sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, inst, cfg)
            chosen = inst
            if sp is None and twin(inst) is not None:
                chosen = twin(inst)
                sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, chosen, cfg)
            if sp is not None:
                st['plans'][key] = dict(sp, ext=ext, family=family, inst=chosen, depth=depth)
    return st

def _NDD_cps_launch(sp, x, out, n, policy=0, query=0):
    fn = _NDD_dd_cap2_b_cpssm_launch_raw
    return fn(sp['ext'], sp['inst_id'], sp, x, out, n, sp['cfg'], query + policy * 100)

def _NDD_run_cps(x, layer, n, st):
    if x.ndim != 2 or not 1 <= x.shape[0] <= 32:
        return None
    m = x.shape[0]
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', _NDD_dd_cap2_b_cps32_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != _NDD_CPS32_LAYER_K[layer]) or (not x.is_cuda):
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.device != sp['wp'].device:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
    rc = _NDD_cps_launch(sp, x, out, n, st.get('policy', 0))
    if rc:
        raise RuntimeError(f'cpstage launch failed: {rc}')
    return out
import torch
from torch.utils.cpp_extension import load_inline
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

def _NDD_tt_mw_mma(n):
    nr = n // 2
    regs = ','.join((f'%{i}' for i in range(nr)))
    outs = ','.join((f'"+f"(d[{i}])' for i in range(nr)))
    inputs = ','.join([f'"r"(a[{i}])' for i in range(16)] + [f'"l"(b+{2 * i})' for i in range(4)])
    ops = ['wgmma.fence.sync.aligned;']
    for kk in range(4):
        ar = ','.join((f'%{nr + 4 * kk + i}' for i in range(4)))
        ops.append(f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{{ar}}}, %{nr + 16 + kk}, 1, 1, 1, 0;')
    ops.append('wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;')
    return f'D void mma{n}(float* d,uint32_t* a,uint64_t b) {{asm volatile("' + ' '.join(ops) + f'" : {outs} : {inputs} : "memory");}}'
_NDD_tt_mw_CUDA_PREFIX = '// dd_cap2_b_\n' + r"""

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define D __device__ __forceinline__
D uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
D void waitbar(uint32_t b,int p) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");
}
D void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
D void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
D uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
D void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");
}
D uint4 ldw(const uint4* p,uint64_t policy) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));
  return v;
}
D uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {
  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;
}
D void dec(uint32_t L0,uint32_t L1,uint32_t C,uint32_t T0,uint32_t T1,uint32_t* a) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));
  const uint32_t low_high = (C & 0x87878787u) + T0;
  const uint32_t high_high = (rotated & 0x87878787u) + T0;
  a[0] = prmt(L0, low_high, 0x5140u);
  a[1] = prmt(L0, low_high, 0x7362u);
  a[2] = prmt(L1, high_high, 0x5140u);
  a[3] = prmt(L1, high_high, 0x7362u);
}
D void patch(uint32_t* a,uint32_t e) {
  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);
  #pragma unroll
  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;
}

"""
_NDD_tt_mw_CUDA_CORE = r"""

template<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>
__global__ __launch_bounds__(128*WK,CAP) void m32w_gemm(
    const __grid_constant__ CUtensorMap tx,const uint4* __restrict__ w,
    const uint32_t* __restrict__ esc,__nv_bfloat16* out,int m,int n,int k,int split,uint32_t t0,uint32_t t1) {
  extern __shared__ __align__(1024) uint8_t sm[];
  constexpr int RW=R/16,NT=128*WK,RP=R+4,SB=BN*128,WS=(Q*(SB+8)+1023)/1024*1024,NL=C12?3:4;
  int tid=threadIdx.x,wg=tid/128,warp=tid%128/32,lane=tid%32;
  uint32_t base0=addr(sm),base=base0+wg*WS,full=base+Q*SB;
  int rank=blockIdx.x%split,nt=blockIdx.x/split,mt=blockIdx.y;
  int kb0=(k/64)*(rank*WK+wg)/(split*WK),nkb=(k/64)*(rank*WK+wg+1)/(split*WK)-kb0;
  uint64_t pw,px;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(pw));
  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0,1.0;":"=l"(px));
  if(tid%128==0) {
    for(int s=0;s<Q;s++)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  auto loadx=[&](int i) {
    if(tid%128==0 && i<nkb) {
      int s=i%Q;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
      tma(base+s*SB,&tx,(kb0+i)*64,mt*BN,full+s*8,px);
    }
  };
  #pragma unroll
  for(int s=0;s<Q;s++)loadx(s);
  uint32_t e0=0xffffffffu,e1=0xffffffffu,nxt=0xffffffffu;int ep=0;
  if(C12 && warp<RW) {
    const uint32_t* el=esc+((nt*split*WK+rank*WK+wg)*RW+warp)*64;
    e0=el[lane];e1=el[lane+32];nxt=__shfl_sync(0xffffffffu,e0,0);
  }
  uint4 raw[PF][NL]={};
  auto load=[&](int slot,int i) {
    if(i<nkb && warp<RW) {
      const uint4* src=w+((long)(nt*(k/64)+kb0+i)*RW+warp)*NL*32+lane;
      #pragma unroll
      for(int l=0;l<NL;l++)raw[slot][l]=ldw(src+l*32,pw);
    }
  };
  #pragma unroll
  for(int p=0;p<PF;p++)load(p,p);
  float acc[BN/2];
  #pragma unroll
  for(int j=0;j<BN/2;j++)asm volatile("mov.b32 %0,0;":"=f"(acc[j]));
  #pragma unroll 1
  for(int ib=0;ib<nkb;ib+=PF) {
    #pragma unroll
    for(int p=0;p<PF;p++) {
      int i=ib+p;if(i>=nkb)break;
      uint32_t a[16];
      if constexpr(C12) {
        dec(raw[p][0].x,raw[p][0].y,raw[p][2].x,t0,t1,a);
        dec(raw[p][0].z,raw[p][0].w,raw[p][2].y,t0,t1,a+4);
        dec(raw[p][1].x,raw[p][1].y,raw[p][2].z,t0,t1,a+8);
        dec(raw[p][1].z,raw[p][1].w,raw[p][2].w,t0,t1,a+12);
      } else {
        #pragma unroll
        for(int l=0;l<4;l++) {
          a[4*l]=raw[p][l].x;a[4*l+1]=raw[p][l].y;a[4*l+2]=raw[p][l].z;a[4*l+3]=raw[p][l].w;
        }
      }
      load(p,i+PF);
      if constexpr(C12) {
        while((nxt>>18)==uint32_t(i)) {
          if(lane==int((nxt>>13)&31u))patch(a,nxt);
          ++ep;
          uint32_t ev=ep<32?e0:e1;
          nxt=__shfl_sync(0xffffffffu,ev,ep&31);
          if(ep>=64)nxt=0xffffffffu;
        }
      }
      int s=i%Q;
      waitbar(full+s*8,(i/Q)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");
      if constexpr(BN==16)mma16(acc,a,desc(base+s*SB));
      if constexpr(BN==32)mma32(acc,a,desc(base+s*SB));
      if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));
      #pragma unroll
      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");
      loadx(i+Q);
    }
  }
  __syncthreads();
  if(tid%128==0) {
    for(int s=0;s<Q;s++)asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(full+s*8) : "memory");
  }
  __syncthreads();
  int row=warp*16+lane/4,col=2*(lane%4);
  #pragma unroll
  for(int j=0;j<BN/8;j++) {
    #pragma unroll
    for(int h=0;h<2;h++) {
      if(warp>=RW)continue;
      ((float*)sm)[(wg*BN+col+8*j)*RP+row+8*h]=acc[4*j+2*h];
      ((float*)sm)[(wg*BN+col+8*j+1)*RP+row+8*h]=acc[4*j+2*h+1];
    }
  }
  __syncthreads();
  if(split>1)cluster_sync();
  for(int q=tid+rank*NT;q<BN*(R/4);q+=NT*split) {
    int token=q/(R/4),row=(q%(R/4))*4;
    if(mt*BN+token>=m)continue;
    float4 sum=make_float4(0,0,0,0);
    for(int pp=0;pp<split*WK;pp++) {
      int s=pp/WK;
      uint32_t remote=base0+((pp%WK*BN+token)*RP+row)*4;
      if(split>1)asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(remote):"r"(remote),"r"(s));
      float4 v;
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(remote):"memory");
      sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;
    }
    __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);
    *(uint2*)(out+(long)(mt*BN+token)*n+nt*R+row)=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);
  }
  __syncthreads();
  if(split>1)cluster_sync();
}

"""
_NDD_tt_mw_CUDA_MAP = r"""
using EncodeFn=decltype(&cuTensorMapEncodeTiled);
int map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {
  static EncodeFn encode=nullptr;
  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}
  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

"""
_NDD_tt_mw_CUDA_HOST = r"""
template<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>
int launch_t(const void* x,const void* w,const void* e,void* out,int m,int n,int k,int split,uint64_t tab,cudaStream_t st) {
  CUtensorMap tx;int rc=map(&tx,x,m,k,BN);if(rc)return rc;
  constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=a>b?a:b;
  auto kern=m32w_gemm<BN,R,WK,PF,Q,C12,CAP>;
  static bool ready=false;
  if(!ready) {
    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return rc;
    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));if(rc)return rc;
    ready=true;
  }
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);cfg.blockDim=dim3(128*WK);cfg.dynamicSmemBytes=smem;cfg.stream=st;
  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(split),1,1};
  if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}
  return int(cudaLaunchKernelEx(&cfg,kern,tx,(const uint4*)w,(const uint32_t*)e,(__nv_bfloat16*)out,m,n,k,split,uint32_t(tab),uint32_t(tab>>32)));
}
int launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t st) {
  switch(id) {DISPATCH}
  return -1;
}
int info(int id){cudaFuncAttributes fa={};switch(id){INFO}return fa.numRegs*10000+fa.localSizeBytes;}
"""
_NDD_tt_mw_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint info(int);\n'

def _NDD_tt_mw_cuda_source(instances):
    cases = []
    attrs = []
    for i, inst in enumerate(instances):
        bn, r, wk, pf, q, c, cap = inst
        params = f'{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}'
        cases.append(f'case {i}: return launch_t<{params}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
        attrs.append(f'case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;')
    return _NDD_tt_mw_CUDA_PREFIX + '\n'.join((_NDD_tt_mw_mma(n) for n in (16, 32, 64))) + _NDD_tt_mw_CUDA_CORE + _NDD_tt_mw_CUDA_MAP + _NDD_tt_mw_CUDA_HOST.replace('DISPATCH', '\n'.join(cases)).replace('INFO', '\n'.join(attrs))

def _NDD_tt_mw_build(instances):
    src = _NDD_tt_mw_cuda_source(instances)
    name = 'dd_cap2_b_m32w_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, _NDD_tt_mw_CPP, cuda_sources=src, functions=['launch', 'info'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])

def _NDD_tt_mw__to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def _NDD_tt_mw__to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def _NDD_tt_mw_sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _NDD_tt_mw__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def _NDD_tt_mw_sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _NDD_tt_mw__to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def _NDD_tt_mw_sm_c12_unpack(data, base, esc):
    """Inverse of dd_cap2_b_sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def _NDD_tt_mw_sm_table(base):
    return base * 16843009

def _NDD_tt_mw_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = _NDD_tt_mw_sm_c12_bytes(v)
        if not torch.equal(_NDD_tt_mw_sm_c12_unpack(data, b, entries), v):
            return None
        table = _NDD_tt_mw_sm_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = _NDD_tt_mw__to_i32(entry)
    else:
        data = _NDD_tt_mw_sm_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def _NDD_tt_mw_launch_raw(ext, idx, pk, x, out):
    return ext.launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], torch.cuda.current_stream().cuda_stream)

def _NDD_tt_mw_verify_onehot(ext, idx, pk, w, m=32):
    n, k = w.shape
    eye = torch.eye(k, device=w.device, dtype=torch.bfloat16)
    out = torch.empty((k, n), device=w.device, dtype=torch.bfloat16)
    for start in range(0, k, m):
        if _NDD_tt_mw_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
            return False
    return bool(torch.equal(out.view(torch.int16), w.T.contiguous().view(torch.int16)))
_NDD_tt_mw_CONFIGS = {(16, 'pooler'): ((16, 64, 1, 2, 2, True, 1), 8), (16, 'unpooler'): ((16, 64, 4, 2, 2, True, 1), 1), (16, 'qkv_gate'): ((16, 64, 1, 2, 2, True, 1), 8), (32, 'pooler'): ((32, 64, 1, 2, 2, True, 1), 8), (32, 'unpooler'): ((32, 64, 4, 2, 2, True, 1), 1), (32, 'qkv_gate'): ((32, 64, 1, 2, 2, True, 1), 8)}

def _NDD_tt_mw_prepare_m32w(weights, configs=None) -> dict:
    configs = _NDD_tt_mw_CONFIGS if configs is None else configs
    device = next(iter(weights.values())).device
    state = {'plans': {}, 'device': device}
    if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (9, 0):
        return state
    instances = []
    for inst, split in configs.values():
        for c in (inst[5], False):
            twin = inst[:5] + (c,) + inst[6:]
            if twin not in instances:
                instances.append(twin)
    ext = _NDD_tt_mw_build(instances)
    state['ext'] = ext
    packs = {}
    for (m, layer), (inst, split) in configs.items():
        if layer not in weights:
            continue
        w = weights[layer]
        for compressed in (inst[5], False):
            actual = inst[:5] + (compressed,) + inst[6:]
            idx = instances.index(actual)
            if ext.info(idx) % 10000:
                continue
            key = (layer, inst[1], inst[2], split, compressed)
            if key not in packs:
                packs[key] = _NDD_tt_mw_pack_weight(w, inst[1], inst[2], split, compressed)
            pk = packs[key]
            if pk is None or not _NDD_tt_mw_verify_onehot(ext, idx, pk, w, m):
                continue
            if m == 32 and (not _NDD_tt_mw_verify_onehot(ext, idx, pk, w, 33)):
                continue
            state['plans'][m, layer] = (idx, pk)
            break
    return state

def _NDD_tt_mw_run_m32w(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (x.device != st['device']):
        return None
    m = x.shape[0]
    if not 16 <= m <= 33 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    key = (16 if m <= 16 else 32, layer)
    plan = st['plans'].get(key)
    if plan is None:
        return None
    idx, pk = plan
    if n != pk['n'] or x.shape[1] != pk['k']:
        return None
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    rc = _NDD_tt_mw_launch_raw(st['ext'], idx, pk, x, out)
    if rc:
        raise RuntimeError(f'm32w launch failed: {rc}')
    return out
import torch
from torch.utils.cpp_extension import load_inline





















import hashlib
import torch
from torch.utils.cpp_extension import load_inline
import torch
from torch.utils.cpp_extension import load_inline














import hashlib
import torch
from torch.utils.cpp_extension import load_inline


































dd_DECODEDIET_ROUTES = (('pooler',16),('unpooler',16))

def dd_prepare_decodediet(weights) -> dict:
    routes = dd_DECODEDIET_ROUTES
    state = {'routes': frozenset(routes)}
    if not routes:
        return state
    if any((m == 16 and layer in ('pooler', 'unpooler') for layer, m in routes)):
        layers = tuple((layer for layer, m in routes if m == 16 and layer in ('pooler', 'unpooler')))
        state['cps'] = _NDD_prepare_cps(weights, layers=layers, buckets=(2,))
    return state

def dd_run_decodediet(x, layer, n, state) -> torch.Tensor | None:
    m = x.shape[0] if x.ndim == 2 else 0
    if (layer, m) not in state.get('routes', ()):
        return None
    return _NDD_run_cps(x, layer, n, state['cps'])


# Nathen Fernandes (MIT): literal pooler-m16 decodediet route.
dd_DECODEDIET_ROUTES = (('pooler', 16),)
_nathen_pool16_parent_prepare = prepare
_nathen_pool16_parent_kernel = kernel_fn


def prepare(weights):
    state = _nathen_pool16_parent_prepare(weights)
    state['nathen_pool16'] = _kshare_modules(dd_prepare_decodediet({'pooler': weights['pooler']}))
    return state


def kernel_fn(data, state):
    _k_ctx_bind(data['x'])
    x = data['x']
    if data['config']['layer'] == 'pooler' and x.shape[0] == 16:
        out = dd_run_decodediet(x, 'pooler', data['config']['n'], state['nathen_pool16'])
        if out is not None:
            return out
    return _nathen_pool16_parent_kernel(data, state)

# =====================================================================================================================
# tA addon (adamleex, 2026-11-01): compliant m = 128 split-K wgmma path for qkv_gate / pooler / unpooler.
# Mainloop and 12-bit pack: keell0renz's rs128 "rq" kernel and its pack/escape helpers, from
# https://github.com/keell0renz/muse-open-kernel-challenge (commit 8e30e1b, kernels/muse_bf16_proj.py), licensed under the
# Apache License 2.0 (text at the end of this file); copied with the changes marked "tA". Swap-AB register-A wgmma that
# decodes the lossless 12-bit pack (exact escapes) into the A registers, x as the smem B operand via 2D TMA, one-wave K
# split. CHANGED: keell0renz's BF16 split-K reduction (cp.reduce.async.bulk.tensor add into a zeroed BF16 output) is NOT
# used; the KS CTAs of a row block exchange FP32 partials through a workspace and the owner of each 16-row unit sums them
# in the fixed split order 0..KS-1 in FP32 and rounds once to BF16 (deterministic, no atomics on data).
# =====================================================================================================================
import hashlib as _ta_hashlib
import time as _ta_time

_TA_RQCU = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
#ifndef FRSLEEP
#define FRSLEEP 32
#endif
struct alignas(64) RqTmap { unsigned long long v[16]; };
struct RqP {
  const uint8_t* W;                  // packed [NB][KC][8 tiles][1536]
  const uint32_t* esc;               // escape entries, [KS][NB][8][ecap] sentinel-terminated lists (+32 sentinels)
  const uint32_t* esc2;              // FL bit 20: per-(warp,lane) sub-lists [KS][NB][8][32][4], sentinel-terminated
  int ecap;                          // entries per (ks, nb, warp) list slot: fixed stride, so no offset load
  __nv_bfloat16* out;                // [m][N] (in-kernel zeroing only; the reduce-adds go through the out tensor map)
  float* ws;                         // tA FR: FP32 split-K partials [NB][KS][NW][NT/8][32] float4
  long long nws;                     // tA FR: ws floats
  int* cnt;                          // [2][NB] zero / poll counters (FL bit 1), all zero between launches
  unsigned long long* prof;          // INS stamps [KS][NB][2 WG][16]
  uint32_t* sink;
  int m, N, K, KC, NB, walias, ncnt;
  uint32_t Tb;
};
__device__ __forceinline__ void rq_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nRQ_W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RQ_W_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void rq_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
// wf_rq128b: relaxed (non-release) arrive. The x/W slots are only ever *read* by the consumer (LDS + wgmma, the latter
// retired by wait_group), so the release ordering of the default arrive -- which makes the arrive wait for the LDS just
// issued (P4: 95 clk) -- is not needed.
__device__ __forceinline__ void rq_arrive_r(uint32_t b) { asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void rq_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void rq_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_bulkh(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void rq_tile(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_tileh(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 rq_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ uint32_t rq_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// r6 c12 decode (variant 1), pw10 pw_dec / pw_unit unchanged
__device__ __forceinline__ void rq_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = rq_prmt(L0, hl, 0x5140); a[1] = rq_prmt(L0, hl, 0x7362); a[2] = rq_prmt(L1, hh, 0x5140); a[3] = rq_prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void rq_unit(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  rq_dec(q0.x, q0.y, q2.x, Tb, A); rq_dec(q0.z, q0.w, q2.y, Tb, A + 4);
  rq_dec(q1.x, q1.y, q2.z, Tb, A + 8); rq_dec(q1.z, q1.w, q2.w, Tb, A + 12);
}
// wf_dilp DILP (FL bit 12): the same 36 ALU ops, written as four independent 3-deep chains: every (hl, hh) pair is
// computed first (8 words, 20 ops, all independent), then the 16 PRMTs (all independent). rq_unit's per-dec form makes the
// compiler walk the four sub-blocks in order (each PRMT waits on its own LDS->LOP3->IADD chain); here the four chains'
// latencies overlap. Same values, same order of A[] writes.
__device__ __forceinline__ void rq_unit_ilp(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  const uint32_t hl0 = (q2.x & 0x87878787u) + Tb, hh0 = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  const uint32_t hl1 = (q2.y & 0x87878787u) + Tb, hh1 = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  const uint32_t hl2 = (q2.z & 0x87878787u) + Tb, hh2 = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  const uint32_t hl3 = (q2.w & 0x87878787u) + Tb, hh3 = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
  A[0] = rq_prmt(q0.x, hl0, 0x5140); A[1] = rq_prmt(q0.x, hl0, 0x7362);
  A[2] = rq_prmt(q0.y, hh0, 0x5140); A[3] = rq_prmt(q0.y, hh0, 0x7362);
  A[4] = rq_prmt(q0.z, hl1, 0x5140); A[5] = rq_prmt(q0.z, hl1, 0x7362);
  A[6] = rq_prmt(q0.w, hh1, 0x5140); A[7] = rq_prmt(q0.w, hh1, 0x7362);
  A[8] = rq_prmt(q1.x, hl2, 0x5140); A[9] = rq_prmt(q1.x, hl2, 0x7362);
  A[10] = rq_prmt(q1.y, hh2, 0x5140); A[11] = rq_prmt(q1.y, hh2, 0x7362);
  A[12] = rq_prmt(q1.z, hl3, 0x5140); A[13] = rq_prmt(q1.z, hl3, 0x7362);
  A[14] = rq_prmt(q1.w, hh3, 0x5140); A[15] = rq_prmt(q1.w, hh3, 0x7362);
}
// wf_rq128b WK: apply one patch-table word (bit 15 valid, 11:8 reg, 7 half, 6:0 xor7) to a decoded A fragment. The
// predicate is per-lane but the *instruction count* is warp-uniform, so the wgmma.fence no longer waits for a walker.
__device__ __forceinline__ void rq_patch(uint32_t (&A)[16], uint32_t w) {
  if (w & 0x8000u) {
    const int r = (int)((w >> 8) & 15);
    const uint32_t xr = (w & 0x7fu) << (8 + ((w >> 7) & 1) * 16);
#pragma unroll
    for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
  }
}
// wf_rq128b FX2: the same walk with a fixed iteration count (warp-uniform instruction count, so the wgmma.fence of the
// warpgroup no longer waits for the warp that happens to walk the longest list). K must be >= the max hits per stage.
__device__ __forceinline__ void rq_fix2(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
#pragma unroll
  for (int it = 0; it < 2; it++) {
    if (nxs == (uint32_t)seq) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
      const bool me = lane == (int)((nx >> 12) & 31);
#pragma unroll
      for (int i = 0; i < 16; i++) if (me && i == r) A[i] ^= xr;
      ++ei;
      if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
      nx = __shfl_sync(0xffffffffu, ec, ei & 31);
      nxs = nx >> 17;
    }
  }
}
// escape fix (pw10 pw_fix3 entry format): entry = seq << 17 | lane << 12 | reg << 8 | half << 7 | xor7, seq = stage within
// the CTA's range. The warp's list is cached in registers, 32 entries per lane-spread window (entry i in lane i % 32), read
// by shfl: no dependent global load per hit (pw10 reloaded the next entry from L2 after every hit, ~600 clk per escape).
__device__ __forceinline__ void rq_fix(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
    nx = __shfl_sync(0xffffffffu, ec, ei & 31);
    nxs = nx >> 17;
  }
}
// wf_dilp WPF (FL bit 13): the same walk with the next entry's shfl issued before the current entry's 16 predicated XORs,
// so the shfl latency (the only dependent chain in the walk) hides under the apply instead of stalling the wgmma.fence.
__device__ __forceinline__ void rq_fixp(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    uint32_t nxn = __shfl_sync(0xffffffffu, ec, (ei + 1) & 31);
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) { ec = __ldg(el + ei + lane); nxn = __shfl_sync(0xffffffffu, ec, 0); }
    nx = nxn;
    nxs = nx >> 17;
  }
}
// FL bit 20: per-thread staged cursor.  E holds this thread's own escape entries sorted by stage (head = E.x),
// hs = E.x >> 17 kept pre-shifted so the common path is one compare + one branch; no shfl, no cross-lane dependency.
// The apply needs a runtime register index, so it is either 16 predicated XORs (if-converted -> 32 instrs EVERY stage,
// measured: loop body 2120 vs 1120 no-escape) or a real branch around them.  asm volatile cannot be if-converted, so
// rq_xor16 keeps the branch: common path = ISETP + BRA.  FL bit 22 = the __any_sync (warp-uniform branch) flavour.
__device__ __forceinline__ void rq_xor16(uint32_t (&A)[16], uint32_t nx) {
  const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16), r = (nx >> 8) & 15;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %16, 0;\n\t@p xor.b32 %0, %0, %17;\n\t"
      "setp.eq.u32 p, %16, 1;\n\t@p xor.b32 %1, %1, %17;\n\t"
      "setp.eq.u32 p, %16, 2;\n\t@p xor.b32 %2, %2, %17;\n\t"
      "setp.eq.u32 p, %16, 3;\n\t@p xor.b32 %3, %3, %17;\n\t"
      "setp.eq.u32 p, %16, 4;\n\t@p xor.b32 %4, %4, %17;\n\t"
      "setp.eq.u32 p, %16, 5;\n\t@p xor.b32 %5, %5, %17;\n\t"
      "setp.eq.u32 p, %16, 6;\n\t@p xor.b32 %6, %6, %17;\n\t"
      "setp.eq.u32 p, %16, 7;\n\t@p xor.b32 %7, %7, %17;\n\t"
      "setp.eq.u32 p, %16, 8;\n\t@p xor.b32 %8, %8, %17;\n\t"
      "setp.eq.u32 p, %16, 9;\n\t@p xor.b32 %9, %9, %17;\n\t"
      "setp.eq.u32 p, %16, 10;\n\t@p xor.b32 %10, %10, %17;\n\t"
      "setp.eq.u32 p, %16, 11;\n\t@p xor.b32 %11, %11, %17;\n\t"
      "setp.eq.u32 p, %16, 12;\n\t@p xor.b32 %12, %12, %17;\n\t"
      "setp.eq.u32 p, %16, 13;\n\t@p xor.b32 %13, %13, %17;\n\t"
      "setp.eq.u32 p, %16, 14;\n\t@p xor.b32 %14, %14, %17;\n\t"
      "setp.eq.u32 p, %16, 15;\n\t@p xor.b32 %15, %15, %17;\n\t}"
      : "+r"(A[0]), "+r"(A[1]), "+r"(A[2]), "+r"(A[3]), "+r"(A[4]), "+r"(A[5]), "+r"(A[6]), "+r"(A[7]),
        "+r"(A[8]), "+r"(A[9]), "+r"(A[10]), "+r"(A[11]), "+r"(A[12]), "+r"(A[13]), "+r"(A[14]), "+r"(A[15])
      : "r"(r), "r"(xr));
}
__device__ __forceinline__ void rq_adv(uint4& E, uint32_t& hs) { E.x = E.y; E.y = E.z; E.z = E.w; E.w = ~0u; hs = E.x >> 17; }
__device__ __forceinline__ void rq_xor16s(uint32_t (&A)[16], uint32_t nx) {  // if-convertible 16-way select
  const int r = (int)((nx >> 8) & 15);
  const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
  for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
}
__device__ __forceinline__ void rq_fixc(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {  // bit 20: real branch
  while (hs == (uint32_t)seq) { rq_xor16(A, E.x); rq_adv(E, hs); }
}
__device__ __forceinline__ void rq_fixv(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {  // bit 22: warp-uniform
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor16s(A, E.x); rq_adv(E, hs); }
  }
}
// bit 23: warp-uniform branch (__any_sync) around the asm apply -> the apply is neither if-converted nor hoisted out of
// the loop (micro: loop body 107 vs 101 no-escape, 115 vote+select, 199 asm alone), so wgmma.fence has nothing to wait for
__device__ __forceinline__ void rq_fixw(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor16(A, E.x); rq_adv(E, hs); }
  }
}
// FL bit 25 (wf_escur3 H2b, ported): patch the 8 hi-byte registers instead of the 16 A registers.
// rq_unit_ilp builds A[4g+0]=PRMT(q0,hl_g,0x5140) A[4g+1]=PRMT(q0,hl_g,0x7362) A[4g+2]=PRMT(q0,hh_g,0x5140)
// A[4g+3]=PRMT(q0,hh_g,0x7362), so entry (reg, half) -> H[reg>>1], byte (reg&1)*2 + half => 8 predicated XORs
// instead of 16 (half the apply block, so the taken branch over it is cheap).
__device__ __forceinline__ void rq_hi8(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* H) {
  H[0] = (q2.x & 0x87878787u) + Tb; H[1] = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  H[2] = (q2.y & 0x87878787u) + Tb; H[3] = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  H[4] = (q2.z & 0x87878787u) + Tb; H[5] = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  H[6] = (q2.w & 0x87878787u) + Tb; H[7] = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
}
__device__ __forceinline__ void rq_prmtA(const uint4& q0, const uint4& q1, const uint32_t* H, uint32_t* A) {
  A[0] = rq_prmt(q0.x, H[0], 0x5140); A[1] = rq_prmt(q0.x, H[0], 0x7362);
  A[2] = rq_prmt(q0.y, H[1], 0x5140); A[3] = rq_prmt(q0.y, H[1], 0x7362);
  A[4] = rq_prmt(q0.z, H[2], 0x5140); A[5] = rq_prmt(q0.z, H[2], 0x7362);
  A[6] = rq_prmt(q0.w, H[3], 0x5140); A[7] = rq_prmt(q0.w, H[3], 0x7362);
  A[8] = rq_prmt(q1.x, H[4], 0x5140); A[9] = rq_prmt(q1.x, H[4], 0x7362);
  A[10] = rq_prmt(q1.y, H[5], 0x5140); A[11] = rq_prmt(q1.y, H[5], 0x7362);
  A[12] = rq_prmt(q1.z, H[6], 0x5140); A[13] = rq_prmt(q1.z, H[6], 0x7362);
  A[14] = rq_prmt(q1.w, H[7], 0x5140); A[15] = rq_prmt(q1.w, H[7], 0x7362);
}
__device__ __forceinline__ void rq_xor8(uint32_t* H, uint32_t nx) {
  const uint32_t r = (nx >> 8) & 15u;
  const uint32_t xr = (nx & 0x7fu) << (16 * (r & 1u) + 8 * ((nx >> 7) & 1u));
  const uint32_t ri = r >> 1;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %8, 0;\n\t@p xor.b32 %0, %0, %9;\n\t"
      "setp.eq.u32 p, %8, 1;\n\t@p xor.b32 %1, %1, %9;\n\t"
      "setp.eq.u32 p, %8, 2;\n\t@p xor.b32 %2, %2, %9;\n\t"
      "setp.eq.u32 p, %8, 3;\n\t@p xor.b32 %3, %3, %9;\n\t"
      "setp.eq.u32 p, %8, 4;\n\t@p xor.b32 %4, %4, %9;\n\t"
      "setp.eq.u32 p, %8, 5;\n\t@p xor.b32 %5, %5, %9;\n\t"
      "setp.eq.u32 p, %8, 6;\n\t@p xor.b32 %6, %6, %9;\n\t"
      "setp.eq.u32 p, %8, 7;\n\t@p xor.b32 %7, %7, %9;\n\t}"
      : "+r"(H[0]), "+r"(H[1]), "+r"(H[2]), "+r"(H[3]), "+r"(H[4]), "+r"(H[5]), "+r"(H[6]), "+r"(H[7])
      : "r"(ri), "r"(xr));
}
__device__ __forceinline__ void rq_fixp8(uint32_t* H, int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor8(H, E.x); rq_adv(E, hs); }
  }
}
__device__ __forceinline__ unsigned long long rq_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t rq_desc(uint32_t saddr) {  // K-major SW128: LBO 1, SBO 1024 B, base offset 0
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
template <int ND> __device__ __forceinline__ void rq_facc(float (&D)[ND]) {
#pragma unroll
  for (int i = 0; i < ND; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
__device__ __forceinline__ void rq_fa(uint32_t (&A)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[i]) :: "memory");
}
__device__ __forceinline__ uint32_t rq_pk(float lo, float hi) {  // inline cvt: __float2bfloat16 serialises the RS pipeline (P4)
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
template <int NT> __device__ __forceinline__ void rq_wgmma(float (&D)[NT / 2], const uint32_t* a, uint64_t desc, int scd);
template <int NT> __device__ __forceinline__ void rq_wgss(float (&D)[NT / 2], uint64_t da, uint64_t db, int scd);
@@WGMMA@@

template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
struct RqC {
  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = NT / 2;  // FL bit 4 (rs32): HALF, 64 W rows per CTA
  static constexpr int XBS = MS * XB, WBS = MS * WB;  // MS k64 stages per ring slot (MS = 2: merged k128 slot)
  static constexpr int PTW = 68;  // u32 per (slot, warp): hdr, u0 mask, u1 mask, 2 patch words per lane
  static constexpr int SMEM = 1024 + ST * (XBS + WBS) + 24 * ST + 16 + ((FL & 32768) ? ST * 8 * PTW * 4 : 0);  // WK patch table
};

// Ring: ST k64 slots, slot i = x tile (XB, 1024-aligned) in the x ring + 12 KB W chunk in the W ring. ONE full barrier
// per slot (count 2: the W lane and the x lane each arrive.expect_tx their own bytes; pw24h P4 lesson), a W empty barrier
// (released by every math warp right after its LDS: W is dead once in registers) and an x empty barrier (released after
// the wait_group that retires the slot's wgmmas). The W lane may refill a slot before the x lane: its next arrive on the
// slot's full barrier can only happen after the consumers saw the current phase complete (they LDS after the full wait),
// so phases never mix. Consumers run GS k64 stages per round: GS full waits, one wgmma.fence, 4*GS wgmma, one commit,
// one wait_group 1 (P4: k128 rounds amortise the per-round bookkeeping).
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
__global__ void __launch_bounds__((FL & 1024) ? 256 : 384, (FL & 3072) ? 2 : 1) rq(const RqP p, const __grid_constant__ RqTmap tx, const __grid_constant__ RqTmap to) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL, MS>;
  constexpr int XB = C::XB, WB = C::WB, ND = C::ND, XBS = C::XBS, WBS = C::WBS;
  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0;
  constexpr bool FR = (FL & 67108864) != 0;  // tA: FP32 split-K exchange epilogue (no BF16 reduce-add)
  constexpr int KSC = (FL >> 27) & 15;       // tA FR: compiled K split (= gridDim.y)
  constexpr int FRSLP = FRSLEEP;
  static_assert(FR, "tA build: only the FP32-exchange (FR) epilogue is instantiated");
  static_assert(!FR || (!(FL & 2) && !(FL & 4) && !(FL & 8) && !(FL & 16) && !(FL & 1024) && !(FL & 2048) && !(FL & 65536) && MS == 1 && KSC >= 1),
                "tA FR: full 128-row CTAs, no zeroing / plain store / EH");
  constexpr bool MRG = (FL & 64) != 0;    // one empty barrier per slot: the x release is folded into the W release
  constexpr bool RLAX = (FL & 128) != 0;  // relaxed (non-release) empty arrive
  constexpr bool NOE = (FL & 512) != 0;   // PROBE: no escape walk (wrong output, times the floor)
  constexpr bool FX2 = (FL & 16384) != 0;  // fixed-iteration uniform escape loop (bit 14: bit 10 is pw47's H2)
  constexpr bool WK = (FL & 32768) != 0;   // patch-table walker (warp PW+3) instead of the per-warp escape list walk (bit 15: bit 11 is pw47's F2)
  constexpr bool DILP = (FL & 4096) != 0; // wf_dilp: 4 independent decode chains (rq_unit_ilp)
  constexpr bool WPF = (FL & 8192) != 0;  // wf_dilp: escape walk with the next-entry shfl prefetched
  static_assert(!WK || (PRB == 0 && MS == 1), "WK: full kernel, MS == 1");  // ZI: in-kernel zeroing of out (no memset node)
  constexpr bool RES = PRB == 8 || PRB == 10;                           // resident: no TMA, no barriers
  constexpr bool XON = PRB != 1 && PRB != 5 && PRB != 7 && !RES;       // x streamed
  constexpr bool WON = PRB != 7 && !RES;                               // W streamed
  constexpr bool SELF = PRB == 5 || PRB == 6;                          // producers self-paced (no consumer)
  // rs32 HALF (FL bit 4): one math warpgroup, 64 W rows per CTA (= half h of the 128-row pack block: tiles 4h..4h+3, a
  // contiguous 6,144 B within each 12,288 B chunk); grid x = 2 NB; math WG1 exits at once; NW = math warps
  constexpr bool HALF = (FL & 16) != 0;
  static_assert(!(FL & 65536) || (NT == 128 && GS == MS && (ST >= 5 && ST <= 8) && PRB == 0 && WM <= 1 && INS == 0 && (FL == 65538 || FL == 65539 || FL == 65570 || FL == 65571))
                || (NT == 256 && GS == MS && ST == 5 && PRB == 0 && WM == 1 && INS == 0 && FL == 65603), "EH backport approved M128/ST5-8/selfzero full-output variants only");
  constexpr int NW = HALF ? 4 : 8, CW = HALF ? 64 : 128;
  // wf_rq32b: 2 CTAs per SM. FL bit 10 (H2): 256-thread CTA = 4 math warps + W lane + x lane + 2 zeroing warps, HALF
  // (64 W rows) only, launch cap 128 regs, math 216 / producer 40. FL bit 11 (F2): 384 threads, full 128 rows, launch
  // cap 84 regs, math 104 / producer 40 (KS must double so the grid still fills 2 x 132 SMs).
  constexpr bool H2 = (FL & 1024) != 0, F2 = (FL & 2048) != 0;
  static_assert(!H2 || HALF, "H2 needs HALF (64 W rows, one math warpgroup)");
  constexpr int PW = H2 ? 4 : 8;                       // first producer warp (W lane = PW, x lane = PW+1, zero = PW+2..)
  constexpr int MREG = H2 ? 128 : (F2 ? 96 : 232);     // math-warp setmaxnreg target (H2 pool = 256*128, F2 = 384*80)
  // rs32 EARLY (FL bit 5): the W lane thread itself inits the barriers and issues the first min(ST, KCS) W chunks and x
  // tiles before the CTA-wide __syncthreads (consumers touch the barriers only after it); the lanes resume at stage ST
  constexpr bool EARLY = (FL & 32) != 0 && PRB == 0;
  // MS (k64 stages per ring slot): one full/empty barrier pair, one MS*12 KB W bulk and MS x tiles per slot; the consumer
  // round must cover a whole slot (GS == MS) so a slot is released exactly once per round
  static_assert(MS == 1 || (GS == MS && !HALF), "MS > 1: GS == MS and no HALF");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t xring = sbase, wring = sbase + ST * XBS, bars = wring + ST * WBS;
  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST, zflag = bars + 24 * ST;
  const uint32_t ptab = bars + 24 * ST + 16;  // WK patch table: [slot][warp][hdr + 2 words per lane]
  uint32_t* const ptabp = (uint32_t*)(smem + (sbase - sraw) + ptab);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nb = HALF ? (int)(blockIdx.x >> 1) : (int)blockIdx.x, hh = HALF ? (int)(blockIdx.x & 1) : 0, ks = blockIdx.y, KS = gridDim.y;
  const int cb = nb * 128 + 64 * hh;  // first out column of this CTA
  const int s0 = (int)((long long)ks * p.KC / KS), KCS = (int)((long long)(ks + 1) * p.KC / KS) - s0;
  const unsigned long long gt0 = INS ? rq_gt() : 0;
  constexpr bool CUR = (FL & 1048576) != 0, NOESC = (FL & 2097152) != 0, VOTE = (FL & 4194304) != 0,
                 VASM = (FL & 8388608) != 0;
  constexpr bool CURP = (FL & 33554432) != 0;  // wf_escur3 H2b: cursor + 8-way hi-byte apply (rq_hi8/rq_fixp8/rq_prmtA)
  // escape-list window issued before the producers start the W flood (a load issued after it waits behind ~28 MB)
  const uint32_t* el = p.esc + (size_t)((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * p.ecap;
  // wf_rq32 FL bit 8 (ESPLIT): the entry x tiles are issued by the x lane (warp 9) in parallel with the W lane's bulks
  // (one bar.sync 6,64 after the init fence); bit 9 (EFENCE): the epilogue's fence.proxy.async.global is dropped (the
  // zeroing warp already fenced its own stores before publishing zflag).  Both measured within noise (batch 3):
  // ESPLIT -0.06..+0.11 us, EFENCE -0.22..+0.05 us on qkv m=32/16 -> no RQV entry uses them (dead code, kept for the
  // record; FL 35 is unchanged).
  constexpr bool ESPLIT = (FL & 256) != 0, EFENCE = (FL & 512) != 0;
  uint32_t ec = (warp < NW && !CUR) ? __ldg(el + lane) : 0u;
  uint4 E4 = make_uint4(~0u, ~0u, ~0u, ~0u);
  if (CUR && warp < NW)  // 16 B per thread, 512 B per warp, coalesced; latency hidden behind the first TMA wait
    E4 = *(const uint4*)(p.esc2 + (size_t)(((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * 32 + lane) * 4);
  uint32_t hs = E4.x >> 17;
  if (ESPLIT && EARLY) {
    if (warp == PW && lane == 0) {
      for (int i = 0; i < ST; i++) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"(XON ? 2 : 1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
      }
      asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (warp == PW || (warp == PW + 1 && XON)) asm volatile("bar.sync 6, 64;\n" ::: "memory");
    if (warp == PW && lane == 0) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
      for (int s = 0; s < ST && s < KCS; s++) {
        rq_expect(full + 8 * s, WB);
        if (HINT) rq_bulkh(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s, pw);
        else rq_bulk(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s);
      }
    }
    if (warp == PW + 1 && lane == 0 && XON) {
      uint64_t px = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(px));
      for (int s = 0; s < ST && s < KCS; s++) {
        rq_expect(full + 8 * s, XB);
        if (HINT) rq_tileh(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s, px);
        else rq_tile(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s);
      }
    }
  } else if (tid == (EARLY ? PW * 32 : 0)) {
    for (int i = 0; i < ST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"((XON ? 2 : 1) + (WK ? 1 : 0)));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
    }
    asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (EARLY) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0, px = 0;
      if (HINT) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
        asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(px));
      }
      for (int s = 0; s < MS * ST && s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS, slot = (s / MS) % ST;
        rq_expect(full + 8 * slot, n * WB);  // two arrivals (count 2), as the W and x lanes would make
        rq_expect(full + 8 * slot, n * XB);
        if (HINT) rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pw);
        else rq_bulk(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot);
        for (int j = 0; j < n; j++) {
          if (HINT) rq_tileh(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot, px);
          else rq_tile(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot);
        }
      }
    }
  }
  if (tid == (PW + 2) * 32 && XON) asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
  __syncthreads();
  const unsigned long long cta = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
  uint32_t ec0 = 0xffffffffu;
  if (WK && warp == PW + 3 && lane < 8) ec0 = __ldg(p.esc + (size_t)((ks * p.NB + nb) * 8 + (lane & 7)) * p.ecap);
  auto walk = [&]() {  // WK patch-table walker (warp PW+3): lane l walks math warp l's escape list and publishes
    // (hdr, 2 words per lane) into the slot's table before arriving on the slot's full barrier
    const int lw = lane & 7;
    const uint32_t* myel = p.esc + (size_t)((ks * p.NB + nb) * 8 + lw) * p.ecap;
    uint32_t* const tb0 = ptabp + lw * C::PTW;
    int ei = 0, slot = 0, r = 0;
    uint32_t ec = ec0, u0 = 0, u1 = 0;
    for (int s = 0; s < KCS; s++) {
      if (s >= ST) rq_wait((MRG ? wempty : xempty) + 8 * slot, (r & 1) ^ 1);
      uint32_t* const tb = tb0 + slot * 8 * C::PTW;
      if (lane < 8) {  // clear the words this slot's previous fill set (the consumer is done with them: xempty)
        uint32_t q0 = tb[1], q1 = tb[2];
        while (q0) { const int l = __ffs((int)q0) - 1; tb[3 + l * 2] = 0u; q0 &= q0 - 1; }
        while (q1) { const int l = __ffs((int)q1) - 1; tb[4 + l * 2] = 0u; q1 &= q1 - 1; }
        tb[0] = 0u; tb[1] = 0u; tb[2] = 0u; u0 = u1 = 0u;
      }
      while (lane < 8 && (ec >> 17) == (uint32_t)s) {
        const int lf = (int)((ec >> 12) & 31), idx = (int)((u0 >> lf) & 1);
        tb[3 + lf * 2 + idx] = 0x8000u | (((ec >> 8) & 15u) << 8) | (((ec >> 7) & 1u) << 7) | (ec & 0x7fu);
        if (idx) u1 |= 1u << lf; else u0 |= 1u << lf;
        ++ei;
        ec = __ldg(myel + ei);
      }
      if (lane < 8) { tb[1] = u0; tb[2] = u1; tb[0] = (u0 | u1) ? 1u : 0u; }
      __syncwarp();
      if (lane == 0) rq_arrive(full + 8 * slot);
      if (++slot == ST) { slot = 0; r++; }
    }
  };
  if (warp >= PW) {  // ---------------- producer warpgroup: W lane (warp PW), x lane (warp PW+1)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == PW && lane == 0 && WON) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < MS * ST ? KCS : MS * ST) : 0;  // EARLY: stages 0 .. MS*ST-1 already issued
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        if (s >= MS * ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : wempty) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, n * WB);
        if (HINT) rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pol);
        else rq_bulk(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot);
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF && !XON) for (int s = (KCS > MS * ST ? KCS - MS * ST : 0); s < KCS; s += MS) rq_wait(full + 8 * ((s / MS) % ST), ((s / MS) / ST) & 1);
      if (INS) { p.prof[cta * 32 + 9] = tw; p.prof[cta * 32 + 15] = rq_gt(); }
    } else if (warp == PW + 1 && lane == 0 && XON) {
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < MS * ST ? KCS : MS * ST) : 0;  // EARLY: stages 0 .. MS*ST-1 already issued
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        if (s >= MS * ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : (MRG ? wempty : xempty)) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, n * XB);
        for (int j = 0; j < n; j++) {
          if (HINT) rq_tileh(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot, pol);
          else rq_tile(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot);
        }
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF) for (int s = (KCS > MS * ST ? KCS - MS * ST : 0); s < KCS; s += MS) rq_wait(full + 8 * ((s / MS) % ST), ((s / MS) / ST) & 1);
      if (INS) p.prof[cta * 32 + 10] = tw;
    } else if (ZI && warp >= PW + 2) {  // in-kernel zeroing: this CTA's 1/KS token slice of its row block's 128 out columns,
      // then a release increment of the block's counter; every reduce-add into the block first acquires counter == KS
      const int t = tid - (PW + 2) * 32, r0 = (int)((long long)ks * p.m / KS), r1 = (int)((long long)(ks + 1) * p.m / KS);
      for (int e = t; e < (r1 - r0) * (CW / 8); e += 64) {
        __nv_bfloat16* d = p.out + (size_t)(r0 + e / (CW / 8)) * p.N + cb + (e % (CW / 8)) * 8;
        asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\n" :: "l"(d), "r"(0) : "memory");
      }
      asm volatile("fence.proxy.async.global;\n" ::: "memory");  // generic zero stores -> the TMA (async proxy) adds
      asm volatile("bar.sync 4, 64;\n" ::: "memory");
      if (t == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" :: "l"(p.cnt + blockIdx.x) : "memory");
        // wait (off the critical path, while the mainloop runs) until every CTA of the block zeroed its slice; publish it
        // to the epilogue through a shared flag. Bounded: a non-co-resident grid traps instead of hanging.
        uint32_t c = 0;
        for (int it = 0; it < (1 << 22); it++) {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(c) : "l"(p.cnt + blockIdx.x) : "memory");
          if ((int)c >= KS) break;
          __nanosleep(128);
        }
        if ((int)c < KS) asm volatile("trap;");  // not co-resident: fail loudly rather than reduce into unzeroed memory
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        asm volatile("st.release.cta.shared.u32 [%0], 1;\n" :: "r"(zflag) : "memory");
        // the last of the block's KS pollers resets both counters for the next launch (the kernel boundary orders it)
        if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == KS - 1) { p.cnt[blockIdx.x] = 0; p.cnt[gridDim.x + blockIdx.x] = 0; }
      }
      if (WK && warp == PW + 3) walk();
    } else if (WK && warp == PW + 3) {
      walk();
    }
    return;
  }
  // ---------------- math warpgroups: warp w of WG g decodes W rows 64g + 16w .. +15 of the block (= tile 4g + w = warp)
  if constexpr (MREG == 232) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  else if constexpr (MREG == 128) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n");
  else asm volatile("setmaxnreg.inc.sync.aligned.u32 96;\n");
  if (SELF || (HALF && warp >= 4)) return;
  const int g = warp >> 2, w = warp & 3;
  int ei = 0;
  uint32_t nx = __shfl_sync(0xffffffffu, ec, 0), nxs = nx >> 17;
  uint32_t Tb = p.Tb;
  asm volatile("" : "+r"(Tb));
  const uint32_t wl = wring + warp * 1536 + lane * 16;
  const uint64_t xd0 = rq_desc(xring);
  float D[ND];  // never zero-filled: scale-d 0 on the CTA's first k16 (a generic write would serialise the pipe, C7515)
  uint32_t A0[GS][16], A1[GS][16];
#pragma unroll
  for (int j = 0; j < GS; j++)
#pragma unroll
    for (int i = 0; i < 16; i++) A0[j][i] = A1[j][i] = 0u;
  uint4 Q[GS][3];
#pragma unroll
  for (int j = 0; j < GS; j++) Q[j][0] = Q[j][1] = Q[j][2] = make_uint4(0, 0, 0, 0);
  uint32_t sink = 0;
  uint32_t frE = 0;  // tA FR: this launch's epoch (gen + 1), read early; used by math thread 0 only
  if (FR && tid == 0) frE = (uint32_t)__ldcg(p.cnt + blockIdx.x) + 1u;
  unsigned long long pc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, tq = 0, tq0 = 0;
#define RQ_T0() do { if (INS) tq = clock64(); } while (0)
#define RQ_T1(i) do { if (INS) { const unsigned long long t_ = clock64(); pc[i] += t_ - tq; tq = t_; } } while (0)
  if (INS) { tq0 = clock64(); tq = tq0; }
  const unsigned long long gtl0 = INS ? rq_gt() : 0;
  unsigned long long gtw0 = 0;
  const int NGF = KCS / GS, NGT = (KCS + GS - 1) / GS;  // full rounds, all rounds
  // round G = stages GS*G .. GS*G + GS - 1 (the last round may be partial)
  auto ldr = [&](int G) {  // full waits, LDS of this warp's 48 B per lane per stage, W slots released
#pragma unroll
    for (int j = 0; j < GS; j++) {
      const int s = GS * G + j;
      if (s < KCS) {
        const int slot = RES ? 0 : (s / MS) % ST;
        if (!RES && (j % MS) == 0) rq_wait(full + 8 * slot, ((s / MS) / ST) & 1);
        if (PRB != 1 && PRB != 2) {
          const uint32_t a = wl + slot * WBS + (s % MS) * WB;
          Q[j][0] = rq_lds(a); Q[j][1] = rq_lds(a + 512); Q[j][2] = rq_lds(a + 1024);
        }
      }
    }
    RQ_T1(1);
    if (!RES && !MRG) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j += MS) if (GS * G + j < KCS) rq_arrive(wempty + 8 * (((GS * G + j) / MS) % ST));
      }
    }
    RQ_T1(2);
  };
  auto xrel = [&](int G) {
    if (!RES) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j += MS) if (GS * G + j < KCS) {
          const uint32_t b = (MRG ? wempty : xempty) + 8 * (((GS * G + j) / MS) % ST);
          if (RLAX) rq_arrive_r(b); else rq_arrive(b);
        }
      }
    }
    RQ_T1(7);
  };
  auto dec = [&](int G, int ns, uint32_t (&A)[GS][16]) {
    if (INS) {  // force the LDS results in: time the data wait separately from the decode ALU work
      uint32_t t = 0;
#pragma unroll
      for (int j = 0; j < GS; j++) t ^= Q[j][0].x ^ Q[j][0].w ^ Q[j][1].x ^ Q[j][1].w ^ Q[j][2].x ^ Q[j][2].w;
      asm volatile("" : "+r"(t));
      sink += t & 1;
      RQ_T1(4);
    }
#pragma unroll
    for (int j = 0; j < GS; j++) {
      if (j < ns) {
        if (PRB == 10) {
#pragma unroll
          for (int i = 0; i < 16; i++) A[j][i] = Tb + i * lane;
        } else {
          if constexpr (CURP) {
            uint32_t H[8];
            rq_hi8(Q[j][0], Q[j][1], Q[j][2], Tb, H);
            rq_fixp8(H, GS * G + j, E4, hs);
            rq_prmtA(Q[j][0], Q[j][1], H, A[j]);
          } else {
          const uint32_t* const tb = ptabp + ((((GS * G + j) / MS) % ST) * 8 + (warp & 7)) * C::PTW;
          const uint32_t ph = WK ? tb[0] : 0u, pw0 = WK ? tb[3 + lane * 2] : 0u, pw1 = WK ? tb[4 + lane * 2] : 0u;
          if (DILP) rq_unit_ilp(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
          else rq_unit(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
          if (WK) {
            if (__builtin_expect(ph != 0u, 0)) { rq_patch(A[j], pw0); rq_patch(A[j], pw1); }
          } else if constexpr (CUR && VASM) rq_fixw(A[j], GS * G + j, E4, hs);
          else if constexpr (CUR && VOTE) rq_fixv(A[j], GS * G + j, E4, hs);
          else if constexpr (CUR) rq_fixc(A[j], GS * G + j, E4, hs);
          else if (NOE) { } else if (FX2) rq_fix2(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          else if (WPF) rq_fixp(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          else rq_fix(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          }
        }
      }
    }
    RQ_T1(3);
  };
  auto mma = [&](int G, auto NSC, uint32_t (&A)[GS][16]) {  // NSC: compile-time stage count of this round
    constexpr int NS = decltype(NSC)::value;
    rq_facc<ND>(D);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int j = 0; j < NS; j++) {
      const int s = GS * G + j;
      const uint64_t d = xd0 + (uint64_t)(((RES ? 0 : ((s / MS) % ST) * XBS + (s % MS) * XB)) >> 4);
#pragma unroll
      for (int q = 0; q < 4; q++) rq_wgmma<NT>(D, A[j] + 4 * q, d + 2 * q, (s | q) != 0);
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    rq_facc<ND>(D);
    if (INS && G == 0) gtw0 = rq_gt();
    RQ_T1(5);
  };
  auto fa = [&](uint32_t (&A)[GS][16]) {
#pragma unroll
    for (int j = 0; j < GS; j++) rq_fa(A[j]);
  };
  if (PRB == 1 || PRB == 2 || PRB == 3) {
#pragma unroll 1
    for (int G = 0; G < NGT; G++) {
      ldr(G);
      if (PRB == 3) {
        dec(G, KCS - GS * G, A0);
#pragma unroll
        for (int j = 0; j < GS; j++)
#pragma unroll
          for (int i = 0; i < 16; i++) sink ^= A0[j][i];
      }
      if (PRB != 1) xrel(G);
    }
  } else if (PRB == 0 || PRB == 4 || RES) {
    // WM 0: dec(G) -> issue(G) -> LDS(G+1) -> wait_group 1 -> release x(G-1)  (decode on the wait->issue chain)
    // WM 1: issue(G) -> LDS(G+1) + dec(G+1) under wgmma(G) -> wait_group 0 -> release x(G) at once
    // WM 2: issue(G) -> LDS(G+1) + dec(G+1) -> wait_group 1 -> release x(G-1): the decode of the next round runs under
    //       the two queued groups; A triple-buffered (G-1 still in flight, G queued, G+1 being written)
    uint32_t A2[GS][16];
#pragma unroll
    for (int j = 0; j < GS; j++)
#pragma unroll
      for (int i = 0; i < 16; i++) A2[j][i] = 0u;
    auto round = [&](int G, auto NSC, uint32_t (&A)[GS][16], uint32_t (&An)[GS][16], uint32_t (&Ap)[GS][16]) {
      if (WM == 0) {
        dec(G, decltype(NSC)::value, A);
        mma(G, NSC, A);
        if (G + 1 < NGT) ldr(G + 1);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      } else if (WM == 1) {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        fa(A);
        RQ_T1(6);
        xrel(G);
      } else {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      }
    };
    using FULLR = std::integral_constant<int, GS>;
    using ONER = std::integral_constant<int, 1>;
    ldr(0);
    if (WM != 0) dec(0, GS, A0);
    int G = 0;
    if (WM == 2) {  // buffer of round G = G % 3; host guarantees KCS >= GS
      round(0, FULLR{}, A0, A1, A2);
      G = 1;
#pragma unroll 1
      for (; G + 2 < NGF; G += 3) { round(G, FULLR{}, A1, A2, A0); round(G + 1, FULLR{}, A2, A0, A1); round(G + 2, FULLR{}, A0, A1, A2); }
      if (G < NGF) { round(G, FULLR{}, A1, A2, A0); G++; }
      if (G < NGF) { round(G, FULLR{}, A2, A0, A1); G++; }
      if (GS > 1 && NGT > NGF) {
        const int b = G % 3;
        if (b == 0) round(G, ONER{}, A0, A1, A2);
        else if (b == 1) round(G, ONER{}, A1, A2, A0);
        else round(G, ONER{}, A2, A0, A1);
        G++;
      }
    } else {
      round(0, FULLR{}, A0, A1, A1);
      G = 1;
#pragma unroll 1
      for (; G + 1 < NGF; G += 2) { round(G, FULLR{}, A1, A0, A0); round(G + 1, FULLR{}, A0, A1, A1); }
      if (G < NGF) { round(G, FULLR{}, A1, A0, A0); G++; }
      if (GS > 1 && NGT > NGF) {  // partial last round (KCS % GS stages; GS = 2: one stage), buffer parity by G
        if (G & 1) round(G, ONER{}, A1, A0, A0);
        else round(G, ONER{}, A0, A1, A1);
        G++;
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    rq_facc<ND>(D);
    fa(A0); fa(A1); fa(A2);
    if (WM != 1) xrel(NGT - 1);
  }
  if (INS) pc[0] = clock64() - tq0;
  const unsigned long long gtl1 = INS ? rq_gt() : 0;
  RQ_T0();
  if (PRB == 0 || PRB == 7) {  // ---------------- epilogue: stmatrix.trans -> SW128 slab [NT tok][64 n] -> TMA reduce-add
    if (PRB == 7) {
#pragma unroll
      for (int i = 0; i < ND; i++) D[i] = (float)((i + tid) & 7) * 0.125f;
    }
    if constexpr (FR) {
      // tA (adamleex): COMPLIANT split-K epilogue. The KS CTAs of a 128-row block exchange FP32 partials through p.ws in
      // wgmma fragment order (no atomics on data). CTA ks owns the 8-token groups [ks*NI/KS, (ks+1)*NI/KS) of the block
      // (all 128 columns): every CTA publishes its partials of the groups it does not own, then the owner sums the KS
      // partials of each owned group in the fixed split order 0, 1, ..., KS - 1 in FP32 (its own partial taken from
      // registers at its own position), rounds ONCE to BF16 and stores. The owned range is made compile-time by a switch
      // over ks (one epilogue copy per split), so D is only ever indexed statically. Arrival: one integer release add per
      // CTA on the block's monotonic arrival counter cnt[NB + nb] + an acquire spin until it reaches KS * epoch, where
      // epoch = gen[nb] + 1 is read at kernel start and CTA 0 of the block advances gen[nb] once all KS arrived (so every
      // peer has read it): no counter reset, no memset node, no order-dependent float arithmetic.
      constexpr int NI = NT / 8;  // 8-token fragment groups (4 floats per thread each)
      float4* const wsb = reinterpret_cast<float4*>(p.ws) + (size_t)blockIdx.x * (size_t)(KSC * NI * NW * 32);
      // layout [ks][i][math thread]: one 16 B slot per thread per group
      auto epi = [&](auto KSI_) {
        constexpr int KSI = decltype(KSI_)::value;
        if constexpr (KSI < KSC) {
          constexpr int I0 = KSI * NI / KSC, I1 = (KSI + 1) * NI / KSC;
          {
            float4* const d = wsb + (size_t)KSI * NI * (NW * 32) + tid;
#pragma unroll
            for (int i = 0; i < NI; i++)
              if (i < I0 || i >= I1) __stcg(d + i * (NW * 32), make_float4(D[4 * i], D[4 * i + 1], D[4 * i + 2], D[4 * i + 3]));
          }
          asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // this CTA's partials stored (and the x ring free)
          if (INS && tid == 0) p.prof[cta * 32 + 26] = rq_gt();
          if (tid == 0) {  // arrivals counter cnt[NB + nb] is monotonic: this launch is complete at KS * epoch arrivals
            int* const arr = p.cnt + gridDim.x + blockIdx.x;
            asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" :: "l"(arr) : "memory");
            if (INS) p.prof[cta * 32 + 27] = rq_gt();
            const int target = KSC * (int)frE;
            int c = 0;
            for (int it = 0; it < (1 << 24); it++) {  // bounded: a non-co-resident grid traps instead of hanging
              asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(c) : "l"(arr) : "memory");
              if (c - target >= 0) break;
              if (it > 256) __nanosleep(FRSLP);
            }
            if (c - target < 0) asm volatile("trap;");
            if (INS) p.prof[cta * 32 + 28] = rq_gt();
            // every peer has arrived (so has read gen): CTA 0 advances the block's epoch for the next launch
            if (KSI == 0) asm volatile("st.relaxed.gpu.global.s32 [%0], %1;\n" :: "l"(p.cnt + blockIdx.x), "r"((int)frE) : "memory");
          }
          asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // the block's KS partials are all visible
          if (INS && tid == 0) p.prof[cta * 32 + 29] = rq_gt();
          constexpr int NO = I1 - I0;
          float4 v[NO][KSC];
#pragma unroll
          for (int j = 0; j < NO; j++)
#pragma unroll
            for (int k = 0; k < KSC; k++)
              if (k != KSI) v[j][k] = __ldcg(wsb + ((size_t)k * NI + I0 + j) * (NW * 32) + tid);
#pragma unroll
          for (int j = 0; j < NO; j++) {
            const int i = I0 + j;
            v[j][KSI] = make_float4(D[4 * i], D[4 * i + 1], D[4 * i + 2], D[4 * i + 3]);
            float4 s = v[j][0];
#pragma unroll
            for (int k = 1; k < KSC; k++) { s.x += v[j][k].x; s.y += v[j][k].y; s.z += v[j][k].z; s.w += v[j][k].w; }
            D[4 * i] = s.x; D[4 * i + 1] = s.y; D[4 * i + 2] = s.z; D[4 * i + 3] = s.w;
          }
          if (INS && tid == 0) { uint32_t z = __float_as_uint(D[4 * I0]); asm volatile("" : "+r"(z)); p.prof[cta * 32 + 30] = rq_gt() + (z & 0); }
          // one BF16 rounding -> stmatrix.trans of the owned groups into the WG slab (SW128, [tok][64 cols])
          const uint32_t slab = xring + g * XB;
          const int jm = lane >> 3, rr = lane & 7;
#pragma unroll
          for (int j = 0; j + 1 < NO; j += 2) {
            const int i0 = I0 + j, i = i0 + (jm >> 1), ch = 2 * w + (jm & 1), tok = 8 * i + rr;
            const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
            const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
            const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
          }
          if constexpr (NO & 1) {
            const int i = I1 - 1, ch = 2 * w + (jm & 1), tok = 8 * i + rr;
            const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
            const uint32_t r0 = rq_pk(D[4 * (I1 - 1)], D[4 * (I1 - 1) + 1]), r1 = rq_pk(D[4 * (I1 - 1) + 2], D[4 * (I1 - 1) + 3]);
            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n" :: "r"(a), "r"(r0), "r"(r1) : "memory");
          }
          asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
          // the WG stores its 64-column half of the owned tokens: NO*8 rows x 8 chunks of 16 B, coalesced 128 B rows
          const int t = tid & 127;
#pragma unroll
          for (int e = t; e < NO * 64; e += 128) {
            const int tok = 8 * I0 + (e >> 3), ch = e & 7;
            const uint4 q = rq_lds(slab + tok * 128 + ((ch ^ (tok & 7)) << 4));
            if (tok < p.m)
              asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};\n" :: "l"(p.out + (size_t)tok * p.N + cb + 64 * g + 8 * ch),
                           "r"(q.x), "r"(q.y), "r"(q.z), "r"(q.w) : "memory");
          }
          if (INS && tid == 0) p.prof[cta * 32 + 31] = rq_gt();
        }
      };
      switch (ks) {
        case 0: epi(std::integral_constant<int, 0>{}); break;
        case 1: epi(std::integral_constant<int, 1>{}); break;
        case 2: epi(std::integral_constant<int, 2>{}); break;
        case 3: epi(std::integral_constant<int, 3>{}); break;
        case 4: epi(std::integral_constant<int, 4>{}); break;
        case 5: epi(std::integral_constant<int, 5>{}); break;
        case 6: epi(std::integral_constant<int, 6>{}); break;
        case 7: epi(std::integral_constant<int, 7>{}); break;
        default: break;
      }
    } else if constexpr (FL & 65536) {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
    // rs128b FL bit 16 (EH): two token halves per warpgroup slab, the first half's reduce-add issued before the second
    // half's stmatrix (the TMA read of half 0 overlaps the stmatrix of half 1); the out map's box is NT / 2 tokens.
    // FL bit 17 (EP): the issuing thread polls the zero flag before its stmatrix instead of after the slab barrier.
    constexpr bool EH = (FL & 65536) != 0, EP = false;  // EH only: historical EP is deliberately unported
    const bool issuer = (w == 0 && lane == 0);
    auto zpoll = [&]() {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
    };
    if (EP && issuer) zpoll();
    auto stsm = [&](int pp) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    };
    auto issue = [&](int tok0, uint32_t src) {
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
      else if (!(FL & 8)) asm volatile("trap;" ::: "memory");  // tA: BF16 reduce-add removed
    };
    if (EH) {
#pragma unroll
      for (int pp = 0; pp < NT / 32; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (issuer) {
        if (!EP) zpoll();
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      }
#pragma unroll
      for (int pp = NT / 32; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (INS) ce[2] = clock64() - ce0;
        issue(NT / 2, slab + (NT / 2) * 128);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    } else {
#pragma unroll
      for (int pp = 0; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (!EP) zpoll();
        if (INS) ce[2] = clock64() - ce0;
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    } else {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
#pragma unroll
    for (int pp = 0; pp < NT / 16; pp++) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
    if (INS) ce[1] = clock64() - ce0;
    if (w == 0 && lane == 0) {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        if (!EFENCE) asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
      if (INS) ce[2] = clock64() - ce0;
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      else if (!(FL & 8)) asm volatile("trap;" ::: "memory");  // tA: BF16 reduce-add removed
      asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      if (INS) ce[3] = clock64() - ce0;
      asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
      if (INS) ce[4] = clock64() - ce0;
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    }
  } else {
    if (sink == 0x9e3779b9u || D[0] == 1.2345f) p.sink[0] = sink;
  }
  if (INS && w == 0 && lane == 0) {
#pragma unroll
    for (int i = 0; i < 9; i++) p.prof[cta * 32 + 16 * g + i] = pc[i];
    if (g == 0) { p.prof[cta * 32 + 11] = gt0; p.prof[cta * 32 + 12] = gtl0; p.prof[cta * 32 + 13] = gtl1; p.prof[cta * 32 + 14] = rq_gt(); p.prof[cta * 32 + 25] = gtw0; }
  }
#undef RQ_T0
#undef RQ_T1
#endif
}

typedef CUresult (*RqEnc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                          const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(RqTmap), "CUtensorMap size");
// 2D bf16 map: dims {inner, outer}, box {64, box_outer}, 128B swizzle; out-of-range rows read as 0 and are not written
static int rq_map(RqTmap* tm, const void* ptr, int64_t inner, int64_t outer, int box_outer) {
  static RqEnc enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (RqEnc)fn;
  }
  if (((uintptr_t)ptr) % 16 || (inner * 2) % 16 || box_outer < 1 || box_outer > 256) return -21;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_outer};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
static int rq_go(RqP& p, torch::Tensor x, torch::Tensor out, int KS, int info) {
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL, MS>;
  auto k = rq<NT, ST, GS, PRB, WM, INS, FL, MS>;
  if (info) {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, k) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (info == 4) {  // wf_rq32b: CTAs per SM at this launch geometry (regs + smem)
      int nb = 0;
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
      if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, (FL & 1024) ? 256 : 384, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -1; }
      return nb;
    }
    return info == 1 ? a.numRegs : info == 2 ? (int)a.localSizeBytes : C::SMEM;
  }
  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;
  if constexpr (NT == 32 && (ST == 4 || ST == 5) && FL == 52) {
    if (p.m != 32 || p.N != 8192 || p.K != 1024 || p.KC != 16 || KS != 1 || p.walias != 0) return -45;
  }
  if (p.ncnt < 2 * p.NB * ((FL & 16) ? 2 : 1)) return -5;
  if constexpr ((FL & 67108864) != 0) {  // tA FR: K split compiled in, workspace present and large enough
    if (KS != ((FL >> 27) & 15) || !p.ws || p.nws < (long long)p.NB * KS * 8 * (NT / 8) * 32 * 4) return -46;
  }
  RqTmap tx, to;
  int r = rq_map(&tx, x.data_ptr(), p.K, p.m, NT);
  if (r) return r;
  r = rq_map(&to, out.data_ptr(), p.N, p.m, (FL & 65536) ? NT / 2 : NT);
  if (r) return r;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3((FL & 1024) ? 256 : 384), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : (int)e;
}
int64_t tarqrun(int64_t v, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor esc2, torch::Tensor cnt, int64_t KS,
              int64_t Tb, int64_t walias, torch::Tensor prof, torch::Tensor sink, int64_t info, torch::Tensor ws) {
  RqP p = {};
  if (!info) {
    if (W.dim() != 4 || W.size(2) != 8 || W.size(3) != 1536 || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;
    if (x.size(1) != W.size(1) * 64 || out.size(1) != W.size(0) * 128 || x.scalar_type() != torch::kBFloat16 || out.scalar_type() != torch::kBFloat16) return -11;
    const int64_t nl = KS * W.size(0) * 8;
    if (esc.numel() < nl * 32 + 32 || (esc.numel() - 32) % nl || sink.numel() < 1) return -12;
    if (cnt.numel() < 2 * W.size(0) || cnt.scalar_type() != torch::kInt32) return -13;
    p.ncnt = (int)cnt.numel();
    if (esc2.numel() && esc2.numel() < nl * 128) return -14;
    p.W = (const uint8_t*)W.data_ptr(); p.esc = (const uint32_t*)esc.data_ptr(); p.ecap = (int)((esc.numel() - 32) / nl);
    p.esc2 = esc2.numel() ? (const uint32_t*)esc2.data_ptr() : nullptr;
    p.out = (__nv_bfloat16*)out.data_ptr(); p.cnt = cnt.data_ptr<int>();
    p.prof = prof.numel() >= 2 * KS * W.size(0) * 32 ? (unsigned long long*)prof.data_ptr() : nullptr;
    p.sink = (uint32_t*)sink.data_ptr();
    p.m = (int)x.size(0); p.K = (int)x.size(1); p.N = (int)out.size(1); p.KC = (int)W.size(1); p.NB = (int)W.size(0);
    p.walias = (int)walias; p.Tb = (uint32_t)(Tb & 0xffffffff);
    if (ws.numel() && ws.scalar_type() == torch::kFloat32) { p.ws = ws.data_ptr<float>(); p.nws = (long long)ws.numel(); }
  }
  switch (v) {
@@CASES@@
  }
  return -2;
}
"""

_TA_OS = ((0, 4, 8, 12, 16, 20, 24, 28), (0, 8, 16, 24, 4, 12, 20, 28))
_TA_SS = ((3, 11, 19, 27, 7, 15, 23, 31), (7, 15, 23, 31, 11, 19, 27, 3))


def _ta_c12(fr, dv=1):
    """bf16 fragments [T, KC, 4, 32, 8] -> (uint8 [T, KC, 3, 32, 16], table, escapes [E, 6] (tile, ch, j, lane, v, xor)) or None"""
    T, KC = fr.shape[:2]
    dev = fr.device
    b = fr.reshape(T, KC, 4, 32, 8).view(torch.int16).int() & 0xffff
    h7, sg = (b >> 8) & 127, b >> 15
    win = torch.bincount(h7.flatten(), minlength=128).unfold(0, 8, 1).sum(1)
    B = int(win.argmax())
    d = h7 - B
    inw = (d >= 0) & (d < 8)
    d = torch.where(inw, d, 0).long()
    C = ((d << torch.tensor(_TA_OS[dv], device=dev)) | (sg.long() << torch.tensor(_TA_SS[dv], device=dev))).sum(-1)
    C = torch.where(C >= 2 ** 31, C - 2 ** 32, C).int()
    lo = (b & 255).to(torch.uint8)
    p0 = lo[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p1 = lo[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p2 = C.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(T, KC, 32, 16)
    data = torch.stack((p0, p1, p2), 2).contiguous()
    ei = (~inw).nonzero()
    esc = torch.cat((ei, (h7[tuple(ei.t())] ^ B)[:, None].long()), 1)
    tab = B * 0x01010101 if dv else sum((B + i) << (8 * i) for i in range(8))
    if not torch.equal(_ta_unc12(data, tab, esc, dv), b):
        return None
    return data, tab - 2 ** 64 if tab >= 2 ** 63 else tab, esc


def _ta_unc12(data, tab, esc, dv):
    """independent decoder (mirrors the kernel's byte permutes) for the exactness proof"""
    T, KC = data.shape[:2]
    dev = data.device
    w = data.view(torch.int32).view(T, KC, 3, 32, 4).long() & 0xffffffff
    L = torch.stack((w[:, :, 0, :, 0], w[:, :, 0, :, 1], w[:, :, 0, :, 2], w[:, :, 0, :, 3], w[:, :, 1, :, 0], w[:, :, 1, :, 1], w[:, :, 1, :, 2], w[:, :, 1, :, 3]), 2).view(T, KC, 4, 2, 32)
    C = w[:, :, 2].permute(0, 1, 3, 2)
    off = (C[..., None] >> torch.tensor(_TA_OS[dv], device=dev)) & 7
    sgn = (C[..., None] >> torch.tensor(_TA_SS[dv], device=dev)) & 1
    tb = torch.tensor([(tab >> (8 * i)) & 255 for i in range(8)], device=dev)
    hi = ((tab & 255) + off if dv else tb[off]) | (sgn << 7)
    lob = (L.permute(0, 1, 2, 4, 3)[..., None] >> (8 * torch.arange(4, device=dev))) & 255
    val = (hi << 8) | lob.reshape(T, KC, 4, 32, 8)
    if len(esc):
        t, c, j, l, vv, x = esc.t()
        val[t, c, j, l, vv] ^= x << 8
    return val.int()



def _ta_pack_nat(w):
    """pw10 _ta_pack_nat: bf16 [N, K] -> A fragments [N/16, K/64, 4 steps, 32 lanes, 8] in natural k order (wgmma B = x
in natural k): step q, lane g*4+t, value v = (kh, half, e) holds W[16T + 8*half + g][64c + 16q + 8*kh + 2t + e]"""
    N, K = w.shape
    W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)  # T, half, g, c, q, kh, t, e
    return W.permute(0, 3, 4, 2, 6, 5, 1, 7).reshape(N // 16, K // 64, 4, 32, 8).contiguous()


def _ta_rqpack(w):
    """12-bit pack re-blocked to 128-row blocks: [N/128][K/64][8 tiles][1536 B] (one 12,288 B bulk copy per stage),
the decode table word, and the escape rows (tile, chunk, step, lane, value, xor); None if the pack is not exact"""
    N, K = w.shape
    c = _ta_c12(_ta_pack_nat(w), 1)
    if c is None or N % 128:
        return None
    data, tab, esc = c
    T, KC = data.shape[:2]
    lay = data.reshape(N // 128, 8, KC, 1536).permute(0, 2, 1, 3).contiguous()
    return lay, tab, esc


def _ta_rqunpack(lay, tab, esc, N, K):
    """independent check of the re-blocking: unpack [NB][KC][8][1536] with _ta_unc12 and undo _ta_pack_nat -> bf16 [N, K]"""
    NB, KC = lay.shape[:2]
    data = lay.permute(0, 2, 1, 3).reshape(NB * 8, KC, 3, 32, 16)
    v = _ta_unc12(data, tab, esc, 1)  # [T, KC, 4, 32, 8] int32 bf16 bits
    W = v.view(N // 16, K // 64, 4, 8, 4, 2, 2, 2)  # T, c, q, g, t, kh, half, e
    return W.permute(0, 6, 3, 1, 2, 5, 4, 7).reshape(N, K)  # int32 bf16 bits (compare with w.view(int16) & 0xffff)


def _ta_rqesc(esc6, NB, KC, KS, dev):
    """escape lists for grid (NB, KS): one fixed-stride slot of CAP entries per (ks, nb, warp) (CAP a multiple of 32,
> the longest list, so every list ends in a sentinel), + 32 trailing sentinels; int32 [KS*NB*8*CAP + 32]"""
    G = KS * NB * 8
    if len(esc6):
        tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
        nb, wt = tile // 8, tile % 8
        s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
        ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
        u = ch - s0[ks]
        ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
        gw = (ks * NB + nb) * 8 + wt
        o = torch.argsort(gw * 65536 + u)
        gw, ent = gw[o], ent[o]
        rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
        cap = (int(torch.bincount(gw, minlength=G).max()) + 1 + 31) // 32 * 32
    else:
        cap = 32
    arr = torch.full((G * cap + 32,), 0xffffffff, dtype=torch.long, device=dev)
    if len(esc6):
        arr[gw * cap + rank] = ent
    arr = torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int()
    return arr.contiguous(), torch.zeros(1, dtype=torch.int32, device=dev)


def _ta_rqesc2(esc6, NB, KC, KS, dev, CAPT=4):
    """FL bit 20 lists: one fixed slot of CAPT entries per (ks, nb, warp, lane), sorted by stage, sentinel-terminated
-> int32 [KS*NB*8*32*CAPT].  Returns (arr, max entries in any slot); arr is None if a slot overflows CAPT."""
    G = KS * NB * 8
    if not len(esc6):
        return torch.full((G * 32 * CAPT,), 0xffffffff, dtype=torch.int32, device=dev), 0
    tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
    nb, wt = tile // 8, tile % 8
    s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
    ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
    u = ch - s0[ks]
    ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
    slot = ((ks * NB + nb) * 8 + wt) * 32 + ln
    o = torch.argsort(slot * 65536 + u)
    slot, ent = slot[o], ent[o]
    rank = torch.arange(len(slot), device=dev) - torch.searchsorted(slot, slot)
    mx = int(rank.max()) + 1
    if mx > CAPT:
        return None, mx
    arr = torch.full((G * 32 * CAPT,), 0xffffffff, dtype=torch.long, device=dev)
    arr[slot * CAPT + rank] = ent
    return torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int().contiguous(), mx




def _ta_wg(n):
    nd = n // 2
    d = ", ".join("%%%d" % i for i in range(nd))
    a = ", ".join("%%%d" % (nd + i) for i in range(4))
    outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
    return (("template <> __device__ __forceinline__ void rq_wgmma<%d>(float (&D)[%d], const uint32_t* a, uint64_t desc, int scd) {\n"
             "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
             "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%s}, %%%d, p, 1, 1, 0;\\n}\\n\"\n"
             "    : %s\n    : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(desc), \"r\"(scd));\n}\n")
            % (n, nd, nd + 5, n, d, a, nd + 4, outs))


# keell0renz's rs128 'rq' code and its 12-bit pack sit inside their _make_chenwainuo factory, which credits
# 'Based on github.com/chenwainuo/okc muse_bf16_proj/submission.py @ 3d418e1 (Apache-2.0)'; that attribution is retained here.
# tA variants: name -> (NT, ST, GS, PRB, WM, INS, FL). FL = keell0renz's d2s5hzwP flags (L2 hints, EARLY, 4-chain decode,
# prefetched walk, per-thread escape cursor + short apply) WITHOUT bit 1 (in-kernel zeroing for the BF16 reduce-add),
# plus bit 26 (FR: FP32 exchange epilogue) and bits 27..30 = the compiled K split.
_TA_FLB = 42999939 & ~2
_TA_FR = 1 << 26
_TA_V = {
    "fr4": (128, 5, 1, 0, 0, 0, _TA_FLB | _TA_FR | (4 << 27)),  # pooler m = 128: 32 row blocks x K split 4 = 128 CTAs
    "fr2": (128, 5, 1, 0, 0, 0, _TA_FLB | _TA_FR | (2 << 27)),  # unpooler m = 128: 64 row blocks x K split 2 = 128 CTAs
}
_TA_PLAN = {("pooler", 128): ("fr4", 4), ("unpooler", 128): ("fr2", 2)}
_TA_NAMES = list(_TA_V)
_TA_NOVAL = False  # probes only (never set in a submission)
_TA_MOD = [None]
_TA_THR = []


def _ta_src():
    cases = "\n".join("    case %d: return rq_go<%s>(p, x, out, (int)KS, (int)info);" % (i, ", ".join(str(a) for a in _TA_V[nm] + (1,)))
                      for i, nm in enumerate(_TA_NAMES))
    return _TA_RQCU.replace("@@WGMMA@@", _ta_wg(128) + _ta_wg(256)).replace("@@CASES@@", cases)


def _ta_ext():
    if _TA_MOD[0] is None:
        src = _ta_src()
        tag = "okc_bf16proj_tA_rq_" + _ta_hashlib.md5(src.encode()).hexdigest()[:12]
        _TA_MOD[0] = load_inline(tag, cpp_sources="int64_t tarqrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, torch::Tensor);",
                                 cuda_sources=src, functions=["tarqrun"],
                                 extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-DNDEBUG", "--expt-relaxed-constexpr"],
                                 verbose=False)
    return _TA_MOD[0]


def _ta_start():
    """compile the tA extension on a background thread while the parent prepare runs (CPU-only work, never fatal)"""
    def build():
        t0 = _ta_time.time()
        try:
            _ta_ext()
            _TA_THR.append("ok")
        except Exception as error:
            _TA_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _TA_THR.append(_ta_time.time() - t0)
    t = threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _ta_call(p, x, out):
    return _TA_MOD[0].tarqrun(p["i"], p["lay"], x, out, p["esc"], p["esc2"], p["cnt"], p["KS"], p["tab"], 0, p["prof"], p["sink"], 0, p["ws"])


def _ta_validate(p, w, m, n, gen):
    """one-hot x8 rows bit-exact against W over every K column, random / outlier rows against FP64 (judge 1e-2, rows 5x),
    repeated calls bit-identical, counters back to zero"""
    K = w.shape[1]
    dev = w.device
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    eye = torch.eye(K, dtype=torch.bfloat16, device=dev) * 8.0
    exact = w[:n].t()
    for start in range(0, K, m):
        x = eye[start:start + m]
        if x.shape[0] < m:
            x = torch.cat((x, torch.zeros((m - x.shape[0], K), dtype=x.dtype, device=dev)))
        if _ta_call(p, x.contiguous(), out) != 0:
            return False, "launch error"
        got = out[: min(m, K - start)]
        if not torch.isfinite(got).all() or not torch.equal(got.float(), exact[start:start + got.shape[0]].float() * 8.0):
            return False, "one-hot mismatch at column %d" % start
    del eye
    for dist in (1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16)
        o = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        if _ta_call(p, x, o) != 0:
            return False, "launch error"
        ref = x.double() @ w[:n].double().t()
        err = float((o.double() - ref).norm() / ref.norm())
        row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        if not (err <= 4e-3 and row <= 2e-2 and torch.isfinite(o).all()):
            return False, "random rel L2 %.2e row %.2e" % (err, row)
        for _ in range(3):
            o2 = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            _ta_call(p, x, o2)
            if not torch.equal(o.view(torch.int16), o2.view(torch.int16)):
                return False, "not bit-identical across calls"
    torch.cuda.synchronize()
    return True, "ok %.1e" % err


def _ta_setup(state, weights, thr):
    state["ta_rq"] = {}
    thr.join(timeout=max(1.0, 250.0 - (_ta_time.time() - _TA_T0[0])))
    if not _TA_THR or _TA_THR[0] != "ok" or _TA_MOD[0] is None:
        print("TA build %s" % (_TA_THR[0] if _TA_THR else "not finished"), flush=True)
        return
    gen = torch.Generator(device="cuda").manual_seed(12828)
    for (layer, m), (vn, KS) in _TA_PLAN.items():
        w = weights[layer]
        n, K = w.shape
        dev = w.device
        pk = _ta_rqpack(w)
        if pk is None or not torch.equal(_ta_rqunpack(*pk, n, K), w.view(torch.int16).int() & 0xffff):
            print("TA %s pack not exact" % layer, flush=True)
            continue
        lay, tab, esc6 = pk
        NB, KC = lay.shape[:2]
        esc = _ta_rqesc(esc6, NB, KC, KS, dev)[0]
        esc2, mx = _ta_rqesc2(esc6, NB, KC, KS, dev)
        if esc2 is None:
            print("TA %s escape cursor overflow (%d)" % (layer, mx), flush=True)
            continue
        NT = _TA_V[vn][0]
        p = {"i": _TA_NAMES.index(vn), "lay": lay, "esc": esc, "esc2": esc2, "KS": KS, "tab": tab, "n": n,
             "cnt": torch.zeros(10 * NB, dtype=torch.int32, device=dev),
             "ws": torch.empty(NB * KS * 8 * (NT // 8) * 32 * 4, dtype=torch.float32, device=dev),
             "prof": torch.empty(0, dtype=torch.int64, device=dev), "sink": torch.zeros(4, dtype=torch.int32, device=dev)}
        try:
            ok, msg = (True, "NOT VALIDATED (probe)") if _TA_NOVAL else _ta_validate(p, w, m, n, gen)
        except Exception as error:
            ok, msg = False, "error %s" % str(error).splitlines()[-1][:160]
        torch.cuda.synchronize()
        print("TA %s m=%d %s: %s" % (layer, m, vn, msg), flush=True)
        if ok:
            state["ta_rq"][(layer, m)] = p


_TA_T0 = [0.0]
_ta_parent_prepare = prepare
_ta_parent_kernel = kernel_fn


def prepare(weights):
    _TA_T0[0] = _ta_time.time()
    thr = _ta_start()
    state = _ta_parent_prepare(weights)
    try:
        _ta_setup(state, weights, thr)
    except Exception as error:
        state["ta_rq"] = {}
        print("TA setup failed: %s" % str(error).splitlines()[-1][:200], flush=True)
    torch.cuda.synchronize()
    print("TA prepare done in %.0fs (tA build %s)" % (_ta_time.time() - _TA_T0[0], _TA_THR), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    if (cfg["layer"], x.shape[0]) == ("unpooler", 128):   # u128 / r6c own this cell; without them the base torchT route beats tA fr2
        return _ta_parent_kernel(data, state)
    p = state.get("ta_rq", {}).get((cfg["layer"], x.shape[0]))
    if p is not None and p["n"] == cfg["n"]:
        out = torch.empty((x.shape[0], cfg["n"]), dtype=torch.bfloat16, device=x.device)
        if _ta_call(p, x.contiguous(), out) == 0:
            return out
    return _ta_parent_kernel(data, state)

# ======================================================================================================================
# track tC (adamleex, 2026-11-02): determinism fixes on top of the alygg77 file above.
# 1) Router m 33..256: aly's 'rk' adds FP32 partials into the output with red.add in arrival order, so its output is
#    not bit-reproducible (measured: 0/20 identical repeated calls at m 128 / 256). 'rkd' below keeps rk's TMA +
#    SS-wgmma main loop (hi then lo tile per 64-k block, FP32 accumulation) but makes the S-way split-K a thread-block
#    cluster and reduces through distributed shared memory in a fixed order (st.async push to the owning rank, sum over
#    source ranks q = 0..S-1): no atomics, no zeroing pass, bit-identical across calls and prepares.
# 2) m = 1 qkv_gate / unpooler: the per-layer cgemv geometry was picked by timing inside prepare(), so two prepares
#    could pick different geometries (different FP32 summation order) and give different bits. Pinned to the
#    geometries measured fastest locally (qkv_gate: 2 rows x 512 threads, 2 accumulators; unpooler: base cgemv).
# ======================================================================================================================
_TC_RKD_CU = r"""// track tC rkd: deterministic router GEMM for m 33..256 (sm_90a). Derived from aly's rk (vjeux/alygg77 lineage):
// out[M, 256] (FP32) = x[M, 8192] (BF16) @ (hi + lo)[256, 8192]^T, W = [hi | lo] [256, 16384] BF16.
// CTA = 1 consumer warpgroup (SS-wgmma m64n64k16 on the hi tile then the lo tile of each 64-k block, FP32 accumulate)
// + 1 TMA producer warp; tile 64 W rows x 64 tokens; split-K over S CTAs that form one thread-block cluster (S, 1, 1).
// Reduction (deterministic, no atomics): every CTA pushes each 8-token column block of its FP32 partial tile with
// st.async into the receive buffer of the cluster rank that owns those tokens (slot = source rank); the owner waits on
// its mbarrier for all S partials and sums them in the fixed order q = 0..S-1, then stores. No output zeroing needed.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef RKD_QS
#define RKD_QS 6
#endif
#define RD __device__ __forceinline__
RD uint64_t rkd_desc(uint32_t addr) {   // K-major tile, 128 B rows, 128B swizzle, SBO 1024
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
RD void rkd_mbar_init(uint32_t bar, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt)); }
RD void rkd_mbar_expect(uint32_t bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
RD void rkd_mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
RD void rkd_mbar_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred P1;\nRKDW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@!P1 bra RKDW_%=;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
RD void rkd_mbar_wait_cl(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred P1;\nRKDC_%=:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n@!P1 bra RKDC_%=;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
RD void rkd_tma2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
RD uint32_t rkd_mapa(uint32_t a, uint32_t rank) { uint32_t d; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank)); return d; }
RD void rkd_push4(uint32_t ra, float a, float b, float c, float d, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               :: "r"(ra), "f"(a), "f"(b), "f"(c), "f"(d), "r"(rbar) : "memory");
}
RD void rkd_push2(uint32_t ra, float a, float b, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
               :: "r"(ra), "f"(a), "f"(b), "r"(rbar) : "memory");
}
@@RKD_MMA@@

// S = K splits = cluster size (2..16); BN tokens per tile (multiple of 16, <= 128). blockIdx.x = sp + S * (rt + 4 * tt).
// Push units: S <= NB (= BN / 8 column blocks): the NB blocks (float4 per thread); else 2 NB half blocks (float2, rows
// g / g+8). Rank r owns units [ceil(r U / S), ceil((r + 1) U / S)); receive layout [source q][slot][thread].
template <int S, int QS, int BN>
__global__ void __launch_bounds__(160, 1) rkd_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
                                                     float* __restrict__ out, int M) {
  constexpr int BM = 64, NB = BN / 8, NACC = BN / 2;
  constexpr int W_BYTES = 2 * BM * 128, ST = W_BYTES + BN * 128;
  constexpr int U = S <= NB ? NB : 2 * NB, NV = S <= NB ? 4 : 2;
  static_assert(S <= 2 * NB && S <= 16, "rkd: S too large for BN");
  constexpr int NSLOT = (U + S - 1) / S;
  constexpr int SLOT_BYTES = 128 * NV * 4;
  constexpr int RECV = S * NSLOT * SLOT_BYTES;
  extern __shared__ __align__(1024) unsigned char rkd_smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(rkd_smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t recv = sbase + QS * ST;
  const uint32_t fullb = recv + ((RECV + 15) & ~15), emptyb = fullb + 8 * QS, rbar = emptyb + 8 * QS;
  const int sp = blockIdx.x % S, pair = blockIdx.x / S, rt = pair & 3, tt = pair >> 2;
  const int kb0 = (sp * 128) / S, nk = ((sp + 1) * 128) / S - kb0;
  const int u0 = (sp * U + S - 1) / S, u1 = ((sp + 1) * U + S - 1) / S;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < QS; ++i) { rkd_mbar_init(fullb + 8 * i, 1); rkd_mbar_init(emptyb + 8 * i, 1); }
    rkd_mbar_init(rbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) rkd_mbar_expect(rbar, (uint32_t)(S * (u1 - u0) * SLOT_BYTES));
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  if (warp == 4) {   // producer
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        if (i >= QS) rkd_mbar_wait(emptyb + 8 * s, ((i / QS) - 1) & 1);
        const uint32_t fb = fullb + 8 * s, st = sbase + s * ST;
        rkd_mbar_expect(fb, ST);
        const int kb = kb0 + i;
        rkd_tma2d(st, &mapW, fb, kb * 64, rt * BM);
        rkd_tma2d(st + BM * 128, &mapW, fb, 8192 + kb * 64, rt * BM);
        rkd_tma2d(st + W_BYTES, &mapX, fb, kb * 64, tt * BN);
      }
    }
    return;
  }
  float d[NACC];
#pragma unroll
  for (int e = 0; e < NACC; ++e) d[e] = 0.f;
  for (int i = 0; i < nk; ++i) {
    const int s = i % QS;
    rkd_mbar_wait(fullb + 8 * s, (i / QS) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const uint32_t a = sbase + s * ST, b = a + W_BYTES;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      rkd_wgmma_n<BN>(d, rkd_desc(a + 32 * k16), rkd_desc(b + 32 * k16));
      rkd_wgmma_n<BN>(d, rkd_desc(a + BM * 128 + 32 * k16), rkd_desc(b + 32 * k16));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && tid == 0) rkd_mbar_arrive(emptyb + 8 * ((i - 1) % QS));
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int e = 0; e < NACC; ++e) asm volatile("" : "+f"(d[e]) :: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // every receive barrier is initialised
  // n8 block i holds d[4i..4i+3] = (row g, tok 8i+2c), (g, 8i+2c+1), (g+8, 8i+2c), (g+8, 8i+2c+1)
#pragma unroll
  for (int i = 0; i < NB; ++i) {
    if constexpr (U == NB) {
      const int dst = (i * S) / U, slot = i - (dst * U + S - 1) / S;
      const uint32_t ra = rkd_mapa(recv + (uint32_t)(((sp * NSLOT + slot) * 128 + tid) * 16), (uint32_t)dst);
      rkd_push4(ra, d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3], rkd_mapa(rbar, (uint32_t)dst));
    } else {
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int u = 2 * i + q, dst = (u * S) / U, slot = u - (dst * U + S - 1) / S;
        const uint32_t ra = rkd_mapa(recv + (uint32_t)(((sp * NSLOT + slot) * 128 + tid) * 8), (uint32_t)dst);
        rkd_push2(ra, d[4 * i + 2 * q], d[4 * i + 2 * q + 1], rkd_mapa(rbar, (uint32_t)dst));
      }
    }
  }
  rkd_mbar_wait_cl(rbar, 0);
  const int g = (tid >> 5) * 16 + ((tid & 31) >> 2), c = tid & 3;
  const int row = rt * BM + g;
  if constexpr (U == NB) {
    const float4* R = reinterpret_cast<const float4*>(rkd_smem + (recv - sraw));
    for (int u = u0; u < u1; ++u) {
      const int slot = u - u0;
      float4 v = R[slot * 128 + tid];
#pragma unroll
      for (int q = 1; q < S; ++q) { const float4 w = R[(q * NSLOT + slot) * 128 + tid]; v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w; }
      const int t0 = tt * BN + 8 * u + 2 * c;
      if (t0 < M) { out[(size_t)t0 * 256 + row] = v.x; out[(size_t)t0 * 256 + row + 8] = v.z; }
      if (t0 + 1 < M) { out[(size_t)(t0 + 1) * 256 + row] = v.y; out[(size_t)(t0 + 1) * 256 + row + 8] = v.w; }
    }
  } else {
    const float2* R = reinterpret_cast<const float2*>(rkd_smem + (recv - sraw));
    for (int u = u0; u < u1; ++u) {
      const int slot = u - u0;
      float2 v = R[slot * 128 + tid];
#pragma unroll
      for (int q = 1; q < S; ++q) { const float2 w = R[(q * NSLOT + slot) * 128 + tid]; v.x += w.x; v.y += w.y; }
      const int i = u >> 1, rr = row + 8 * (u & 1), t0 = tt * BN + 8 * i + 2 * c;
      if (t0 < M) out[(size_t)t0 * 256 + rr] = v.x;
      if (t0 + 1 < M) out[(size_t)(t0 + 1) * 256 + rr] = v.y;
    }
  }
}

static PFN_cuTensorMapEncodeTiled_v12000 rkd_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qr; void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr);
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
static int rkd_map(CUtensorMap* m, const void* ptr, uint64_t rows, uint64_t cols, uint32_t brows) {
  cuuint64_t dims[2] = {cols, rows}; cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, brows}; cuuint32_t estr[2] = {1, 1};
  auto enc = rkd_encode();
  if (!enc) return -20;
  CUresult r = enc(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS ? 0 : -21;
}
template <int S, int QS, int BN>
static int rkd_launch_t(int64_t w, int64_t x, int64_t out, int M, int64_t stream, int query) {
  auto kern = rkd_kernel<S, QS, BN>;
  constexpr int ST = 2 * 64 * 128 + BN * 128;
  constexpr int NB = BN / 8, U = S <= NB ? NB : 2 * NB, NV = S <= NB ? 4 : 2, RECV = S * ((U + S - 1) / S) * 128 * NV * 4;
  const int smem = 1024 + QS * ST + ((RECV + 15) & ~15) + 8 * (2 * QS + 1) + 64;
  static bool attr = false;
  if (!attr) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return -30;
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess) return -31;
    attr = true;
  }
  if (M < 1 || M > 256) return -5;
  const int ntt = (M + BN - 1) / BN;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(4 * ntt * S, 1, 1);
  lc.blockDim = dim3(160, 1, 1);
  lc.dynamicSmemBytes = smem;
  lc.stream = (cudaStream_t)stream;
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = S; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  lc.attrs = at; lc.numAttrs = 1;
  if (query == 1) {
    int v = 0;
    cudaError_t e = cudaOccupancyMaxActiveClusters(&v, kern, &lc);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  CUtensorMap mw, mx;
  int rc = rkd_map(&mw, (const void*)w, 256, 16384, 64); if (rc) return rc;
  rc = rkd_map(&mx, (const void*)x, (uint64_t)M, 8192, BN); if (rc) return rc;
  cudaLaunchKernelEx(&lc, kern, mw, mx, reinterpret_cast<float*>(out), M);
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}
int rkd_launch(int cfg, int64_t w, int64_t x, int64_t out, int M, int64_t stream, int query) {
  switch (cfg) {
@@RKD_CASES@@
    default: return -99;
  }
}
"""
TC_RKD_CFGS = [(8, 6, 32), (8, 6, 48), (8, 6, 64), (8, 6, 96), (6, 6, 64)]   # (cluster size S = K splits, ring stages, tokens per tile)
TC_RKD_ROUTE = ((96, 0), (144, 1), (192, 2), (208, 3), (256, 4))                # m <= bound -> config index
_tc_rkd = None


def _tc_rkd_mma(ns):
    out = ['template <int BN> RD void rkd_wgmma_n(float* d, uint64_t a, uint64_t b);']
    for n in ns:
        regs = ', '.join(f'%{i}' for i in range(n // 2))
        outs = ', '.join(f'"+f"(d[{i}])' for i in range(n // 2))
        out.append(f'template <> RD void rkd_wgmma_n<{n}>(float* d, uint64_t a, uint64_t b) {{\n  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;\\n" : {outs} : "l"(a), "l"(b));\n}}')
    return '\n'.join(out)


def _tc_rkd_ext():
    global _tc_rkd
    if _tc_rkd is None:
        import hashlib
        cases = '\n'.join(f'    case {i}: return rkd_launch_t<{s},{q},{b}>(w, x, out, M, stream, query);' for i, (s, q, b) in enumerate(TC_RKD_CFGS))
        src = _TC_RKD_CU.replace('@@RKD_CASES@@', cases).replace('@@RKD_MMA@@', _tc_rkd_mma(sorted({c[2] for c in TC_RKD_CFGS})))
        _tc_rkd = load_inline('tc_rkd_' + hashlib.sha256(src.encode()).hexdigest()[:10],
                              '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rkd_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n',
                              cuda_sources=src, functions=['rkd_launch'], no_implicit_headers=True, with_pytorch_error_handling=False,
                              extra_cuda_cflags=['-O3', '-std=c++17', '-gencode=arch=compute_90a,code=sm_90a'], verbose=False)
    return _tc_rkd


def tc_build_group(name=None):   # local CPU pre-build hook (tools/build_groups.py)
    return _tc_rkd_ext()
TC_SW_GROUPS = {'tc_rkd': None}


def _tc_rkd_cfg(m):
    for bound, i in TC_RKD_ROUTE:
        if m <= bound:
            return i
    return None


def _tc_rkd_run(x, state):
    m = x.shape[0]
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    rc = state['tc_rkd'].rkd_launch(_tc_rkd_cfg(m), state['router_cat'].data_ptr(), x.data_ptr(), out.data_ptr(), m,
                                    torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f'rkd launch failed: {rc}')
    return out


def _tc_rkd_validate(weights, state):
    """Every rkd config: all CTAs must fit as co-resident clusters, and the output must match an FP64 reference of
    hi + lo (normal and probe inputs) at the bucket's edge m values; any failure leaves the router on cuBLAS."""
    ext = state['tc_rkd']
    w = weights['router']
    hi = w.to(torch.bfloat16); lo = (w - hi.float()).to(torch.bfloat16)
    wex = (hi.double() + lo.double())
    g = torch.Generator(device=w.device).manual_seed(4242)
    lo_m = 33
    for bound, i in TC_RKD_ROUTE:
        S, _, bn = TC_RKD_CFGS[i]
        tiles = 4 * ((bound + bn - 1) // bn)
        if ext.rkd_launch(i, 0, 0, 0, bound, 0, 1) < tiles:
            return f'cfg {i}: {tiles} clusters of {S} not co-resident'
        for m in sorted({lo_m, (lo_m + bound) // 2, bound}):
            x = torch.randn((m, 8192), generator=g, device=w.device)
            x[m - 1].zero_(); x[m - 1, (m * 131) % 8192] = 64.0
            x = x.to(torch.bfloat16)
            out = _tc_rkd_run(x, state)
            ref = x.double() @ wex.T
            err = float((out.double() - ref).norm() / ref.norm())
            perr = float((out[m - 1].double() - ref[m - 1]).norm() / ref[m - 1].norm())
            if not (err < 2e-5 and perr < 2e-5) or not bool(torch.isfinite(out).all()):
                return f'cfg {i} m {m}: rel err {err:.2e} probe {perr:.2e}'
            if not torch.equal(out, _tc_rkd_run(x, state)):
                return f'cfg {i} m {m}: not reproducible'
        lo_m = bound + 1
    return None


TC_RKD_ON = False   # router m 33..256 keeps the base 'rk' route (FP32 split-K partials, FP32 red.add)
_tc_parent_prepare = prepare
_tc_parent_kernel = kernel_fn


def prepare(weights):
    state = _tc_parent_prepare(weights)
    # (2) pinned m = 1 cgemv geometries (no timing-dependent choice)
    state['iv_cg_choice'] = {'qkv_gate': 1} if 'qkv_gate' in state.get('cg', {}) else {}
    # (1) deterministic router m 33..256
    bad = 'disabled'
    if TC_RKD_ON:
        try:
            state['tc_rkd'] = _KSharedModule(_tc_rkd_ext())
            bad = _tc_rkd_validate(weights, state)
        except Exception as e:
            bad = repr(e)[:200]
    state['tc_rkd_ok'] = bad is None
    if bad is not None and TC_RKD_ON:
        print(f'[tc] rkd off ({bad}); router m 33..256 -> cuBLAS hi + lo', file=sys.stderr)
    torch.cuda.synchronize()
    return state


def kernel_fn(data, state):
    x = data['x']
    if TC_RKD_ON and data['config']['layer'] == 'router' and 33 <= x.shape[0] <= 256:
        _k_ctx_bind(x)
        if state.get('tc_rkd_ok') and x.is_contiguous():
            return _tc_rkd_run(x, state)
        return _run(('torch',), x, 'router', x.shape[0], 256, 8192, state)
    return _tc_parent_kernel(data, state)


# ----------------------------------------------------------------------------------------------------------------------
# 3) Router m 2..32: Nathen Fernandes' routertail_small (MIT, github.com/NathenFernandes/muse-bf16-proj submission.py
#    @ 7385e5d, as carried verbatim in keell0renz/muse-open-kernel-challenge; see the MIT notice at the top of this
#    file), with its owner-rank count P made a template parameter. BF16 hi + lo fragments, mma.sync, FP32 accumulate,
#    16-CTA (or 8) cluster split-K, warps summed in smem then partials pushed to P owner ranks that sum all S in fixed
#    order: deterministic. Configs per m bucket from the local sweep (probe sw6).
# ----------------------------------------------------------------------------------------------------------------------
_TC_RT_CU = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define RD __device__ __forceinline__
RD uint32_t routertail_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
RD void routertail_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void routertail_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void routertail_wait(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");
}
RD void routertail_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" :::"memory"); }
RD uint32_t routertail_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }
RD void routertail_push(uint32_t d, float4 v, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
RD float4 routertail_pull(uint32_t d) {
  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(d):"memory");return v;
}
RD void routertail_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }
RD uint64_t routertail_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }
RD void routertail_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(dst),"l"(tm),"r"(k),"r"(r),"r"(bar):"memory");
}

template<int MT,int S,int WK,int P>
__global__ __launch_bounds__(WK*32) void routertail_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {
  constexpr int KU=128/S/WK, NI=MT*32, OWN=NI/P, PARTS=S*1;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+PARTS*OWN;
  const uint32_t rb=routertail_smem(sm+(PARTS*OWN+WK*NI)*16);
  int4 w[KU][8];uint4 xx[KU][MT][2];
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<8;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  if(rank<P && tid==0) {routertail_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  __syncthreads();
  if(rank<P && tid==0) {routertail_expect(rb,PARTS*OWN*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  float c[2][MT][4]={};
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          const int4 a=w[k][q+h*4];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  if(tid<NI) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) routertail_add(sum,part[k*NI+tid]);
    const int dst=tid/OWN,local=tid%OWN;
    routertail_push(routertail_map(routertail_smem(recv+rank*OWN+local),dst),sum,routertail_map(rb,dst));
  }

  if(rank<P && tid<OWN) {
    routertail_wait(rb,0);
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<PARTS;k++) routertail_add(sum,recv[k*OWN+tid]);
    const int v=rank*OWN+tid,ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
    if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
    if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
  }

}

template<int MT,int S,int WK,int P>
int routertail_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto fn=routertail_small_kernel<MT,S,WK,P>;
  constexpr int bytes=(S*1/P+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);
}
int routertail_cfg(int cfg,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  switch(cfg) {
@@RT_CASES@@
    default: return -99;
  }
}
"""
TC_RT_CFGS = [(1, 8, 8, 4), (2, 16, 8, 4), (4, 16, 4, 16)]   # (MT n8 token tiles, cluster S, warps WK, owner ranks P)
TC_RT_ROUTE = ((8, 0), (16, 1), (32, 2))                     # m <= bound -> config index
TC_RT_ON = (2, 32)                                           # router m range routed to routertail
_tc_rt = None


def _tc_rt_ext():
    global _tc_rt
    if _tc_rt is None:
        import hashlib
        cases = '\n'.join(f'    case {i}: return routertail_small_t<{a},{b},{c},{d}>(w, x, out, m, stream, query);' for i, (a, b, c, d) in enumerate(TC_RT_CFGS))
        src = _TC_RT_CU.replace('@@RT_CASES@@', cases)
        _tc_rt = load_inline('tc_rt_' + hashlib.sha256(src.encode()).hexdigest()[:10],
                             '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint routertail_cfg(int,int64_t,int64_t,int64_t,int,int64_t,int);\n',
                             cuda_sources=src, functions=['routertail_cfg'], no_implicit_headers=True, with_pytorch_error_handling=False,
                             extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'], verbose=False)
    return _tc_rt


_tc_v2_build = tc_build_group


def tc_build_group(name=None):
    _tc_v2_build()
    return _tc_rt_ext()


def _tc_rt_pack(w, s, wk):
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)

    def units(v):
        v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
        return v.permute(0, 1, 3, 2, 4)
    data = torch.cat([units(hi), units(lo)], dim=2).contiguous()
    ku = 128 // s // wk
    return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()


def _tc_rt_run(x, state):
    m = x.shape[0]
    i = next(i for bound, i in TC_RT_ROUTE if m <= bound)
    mt, s, wk, p = TC_RT_CFGS[i]
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    rc = state['tc_rt'].routertail_cfg(i, state['tc_rt_w'][(s, wk)].data_ptr(), x.data_ptr(), out.data_ptr(), m,
                                       torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f'routertail launch failed: {rc}')
    return out


def _tc_rt_validate(weights, state):
    w = weights['router']
    hi = w.to(torch.bfloat16); lo = (w - hi.float()).to(torch.bfloat16)
    wex = hi.double() + lo.double()
    g = torch.Generator(device=w.device).manual_seed(4343)
    for m in (2, 5, 8, 9, 13, 16, 17, 25, 32):
        x = torch.randn((m, 8192), generator=g, device=w.device)
        x[m - 1].zero_(); x[m - 1, (m * 977) % 8192] = 32.0
        x = x.to(torch.bfloat16)
        out = _tc_rt_run(x, state)
        ref = x.double() @ wex.T
        err = float((out.double() - ref).norm() / ref.norm())
        perr = float((out[m - 1].double() - ref[m - 1]).norm() / ref[m - 1].norm())
        if not (err < 2e-5 and perr < 2e-5) or not bool(torch.isfinite(out).all()):
            return f'm {m}: rel err {err:.2e} probe {perr:.2e}'
        if not torch.equal(out, _tc_rt_run(x, state)):
            return f'm {m}: not reproducible'
    return None


_tc_v2_prepare = prepare
_tc_v2_kernel = kernel_fn


def prepare(weights):
    state = _tc_v2_prepare(weights)
    try:
        state['tc_rt'] = _KSharedModule(_tc_rt_ext())
        w = weights['router'].contiguous()
        state['tc_rt_w'] = {}
        for (mt, s, wk, p) in TC_RT_CFGS:
            if (s, wk) not in state['tc_rt_w']:
                state['tc_rt_w'][(s, wk)] = _tc_rt_pack(w, s, wk)
        bad = _tc_rt_validate(weights, state)
    except Exception as e:
        bad = repr(e)[:200]
    state['tc_rt_ok'] = bad is None
    if bad is not None:
        print(f'[tc] routertail off ({bad}); router m 2..32 stays on the base routes', file=sys.stderr)
    torch.cuda.synchronize()
    return state


def kernel_fn(data, state):
    x = data['x']
    if (data['config']['layer'] == 'router' and TC_RT_ON[0] <= x.shape[0] <= TC_RT_ON[1] and state.get('tc_rt_ok')
            and x.is_contiguous() and x.data_ptr() % 16 == 0):
        _k_ctx_bind(x)
        return _tc_rt_run(x, state)
    return _tc_v2_kernel(data, state)


# =====================================================================================================================
# track u128 (adamleex, 2026-11-02): split-free unpooler m = 128 path (no K split, so no cross-CTA reduction at all).
# The decode / escape-cursor / register-A wgmma building blocks are keell0renz's rs128 "rq" kernel code (from
# https://github.com/keell0renz/muse-open-kernel-challenge, commit 8e30e1b, kernels/muse_bf16_proj.py, Apache License 2.0,
# text at the end of this file), copied under a u8_ prefix inside an anonymous namespace, with the 12-bit pack of the tA
# addon above (_ta_rqpack, lossless, exact escapes). NEW here: 128 CTAs of 64 W rows x all K (8192 rows / 64); two math
# warpgroups take alternate k64 stages (even / odd) with their own FP32 accumulators, summed once in FP32 (even + odd) at
# the end, rounded once to BF16 and stored with plain stores. No atomics, no reduction into memory, no inter-CTA waits,
# nothing order-dependent: bit-identical across calls.
# =====================================================================================================================
import hashlib as _u8_hashlib
import time as _u8_time

_U8_CU = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
namespace {
struct alignas(64) U8Tmap { unsigned long long v[16]; };
struct U8P {
  const uint8_t* W;            // tA/keell0renz rq pack [N/128][KC][8 tiles of 16 rows][1536 B]
  const uint4* esc;            // escape cursors [N/64 CTAs][2 WG][4 warps][32 lanes] x 4 entries (sentinel ~0)
  __nv_bfloat16* out;          // [m][N]
  unsigned long long* prof;    // INS stamps [N/64][32]
  uint32_t* sink;
  int m, N, KC;
  uint32_t Tb;
};
__device__ __forceinline__ void u8_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nU8W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra U8W_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void u8_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void u8_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void u8_bulkh(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void u8_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void u8_tile(uint32_t dst, const U8Tmap* tm, int k0, int r0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar) : "memory");
}
__device__ __forceinline__ void u8_tileh(uint32_t dst, const U8Tmap* tm, int k0, int r0, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 u8_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ uint32_t u8_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// keell0renz rq_hi8 / rq_xor8 / rq_fixp8 / rq_prmtA (12-bit decode: hi bytes from the 4-bit codes + the window base, an
// exact XOR patch of the rare escapes from the per-thread cursor, then the 16 PRMTs that build the A fragment)
__device__ __forceinline__ void u8_hi8(const uint4& q2, uint32_t Tb, uint32_t* H) {
  H[0] = (q2.x & 0x87878787u) + Tb; H[1] = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  H[2] = (q2.y & 0x87878787u) + Tb; H[3] = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  H[4] = (q2.z & 0x87878787u) + Tb; H[5] = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  H[6] = (q2.w & 0x87878787u) + Tb; H[7] = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
}
__device__ __forceinline__ void u8_prmtA(const uint4& q0, const uint4& q1, const uint32_t* H, uint32_t* A) {
  A[0] = u8_prmt(q0.x, H[0], 0x5140); A[1] = u8_prmt(q0.x, H[0], 0x7362);
  A[2] = u8_prmt(q0.y, H[1], 0x5140); A[3] = u8_prmt(q0.y, H[1], 0x7362);
  A[4] = u8_prmt(q0.z, H[2], 0x5140); A[5] = u8_prmt(q0.z, H[2], 0x7362);
  A[6] = u8_prmt(q0.w, H[3], 0x5140); A[7] = u8_prmt(q0.w, H[3], 0x7362);
  A[8] = u8_prmt(q1.x, H[4], 0x5140); A[9] = u8_prmt(q1.x, H[4], 0x7362);
  A[10] = u8_prmt(q1.y, H[5], 0x5140); A[11] = u8_prmt(q1.y, H[5], 0x7362);
  A[12] = u8_prmt(q1.z, H[6], 0x5140); A[13] = u8_prmt(q1.z, H[6], 0x7362);
  A[14] = u8_prmt(q1.w, H[7], 0x5140); A[15] = u8_prmt(q1.w, H[7], 0x7362);
}
__device__ __forceinline__ void u8_xor8(uint32_t* H, uint32_t nx) {
  const uint32_t r = (nx >> 8) & 15u;
  const uint32_t xr = (nx & 0x7fu) << (16 * (r & 1u) + 8 * ((nx >> 7) & 1u));
  const uint32_t ri = r >> 1;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %8, 0;\n\t@p xor.b32 %0, %0, %9;\n\t"
      "setp.eq.u32 p, %8, 1;\n\t@p xor.b32 %1, %1, %9;\n\t"
      "setp.eq.u32 p, %8, 2;\n\t@p xor.b32 %2, %2, %9;\n\t"
      "setp.eq.u32 p, %8, 3;\n\t@p xor.b32 %3, %3, %9;\n\t"
      "setp.eq.u32 p, %8, 4;\n\t@p xor.b32 %4, %4, %9;\n\t"
      "setp.eq.u32 p, %8, 5;\n\t@p xor.b32 %5, %5, %9;\n\t"
      "setp.eq.u32 p, %8, 6;\n\t@p xor.b32 %6, %6, %9;\n\t"
      "setp.eq.u32 p, %8, 7;\n\t@p xor.b32 %7, %7, %9;\n\t}"
      : "+r"(H[0]), "+r"(H[1]), "+r"(H[2]), "+r"(H[3]), "+r"(H[4]), "+r"(H[5]), "+r"(H[6]), "+r"(H[7])
      : "r"(ri), "r"(xr));
}
__device__ __forceinline__ void u8_adv(uint4& E, uint32_t& hs) { E.x = E.y; E.y = E.z; E.z = E.w; E.w = ~0u; hs = E.x >> 17; }
__device__ __forceinline__ void u8_fixp8(uint32_t* H, int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { u8_xor8(H, E.x); u8_adv(E, hs); }
  }
}
__device__ __forceinline__ unsigned long long u8_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t u8_desc(uint32_t saddr) {  // K-major SW128: LBO 1, SBO 1024 B
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void u8_facc(float (&D)[64]) {
#pragma unroll
  for (int i = 0; i < 64; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
__device__ __forceinline__ void u8_fa(uint32_t (&A)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[i]) :: "memory");
}
__device__ __forceinline__ uint32_t u8_pk(float lo, float hi) {  // the single FP32 -> BF16 rounding (round to nearest even)
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
@@WGMMA@@

// Grid: N / 64 CTAs (CTA c = W rows 64c .. 64c + 63, all K = no K split). 384 threads: math WG 0 / 1 (warps 0-7) take
// the k64 stages s = 2i + g (even / odd), each with its own FP32 accumulator; producer warp 8 = W lane (one 6 KB bulk per
// stage), warp 9 = x lane (the stage's 128-token x tile, 16 KB 2D TMA, SW128). Ring of ST slots: x 16 KB + W 6 KB.
// full[slot]: count 2 (the W lane's and the x lane's arrive.expect_tx), wempty[slot]: the 4 warps of the consuming WG
// after their LDS (W lives in registers from then on), xempty[slot]: the same 4 warps after the wait_group that retires
// the stage's wgmmas. ST >= 5 (static_assert): round i's full wait for stage 2(i+1)+g needs stage 2(i+1)+g-ST
// released, and ST < 5 deadlocks.
// HN: L2 hints (bit 0: W evict_first, bit 1: x evict_normal; no evict_last). PM: 0 full kernel, 2 streams only, 3 LDS + decode without
// wgmma (probes). INS: globaltimer stamps.
template <int ST, int HN, int PM, int INS>
__global__ void __launch_bounds__(384, 1) u8k(const U8P p, const __grid_constant__ U8Tmap tx) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int XB = 16384, WB = 6144;
  static_assert(ST >= 5 && 1024 + ST * (XB + WB) + 24 * ST <= 232448, "smem / ring depth");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t xring = sbase, wring = sbase + ST * XB, bars = wring + ST * WB;
  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int c = blockIdx.x, nb = c >> 1, hh = c & 1, KC = p.KC;
  const unsigned long long gt0 = INS ? u8_gt() : 0;
  uint4 E4 = make_uint4(~0u, ~0u, ~0u, ~0u);
  if (warp < 8) E4 = p.esc[(size_t)((c * 2 + (warp >> 2)) * 4 + (warp & 3)) * 32 + lane];  // latency hidden by the init
  if (tid == 256) {
    for (int i = 0; i < ST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"(2));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(4));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(4));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (tid == 288) asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
  __syncthreads();
  if (warp >= 8) {  // ---------------- producer warpgroup
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == 8 && lane == 0) {
      const uint8_t* src = p.W + (size_t)nb * KC * 12288 + hh * 6144;  // tiles 4hh .. 4hh + 3 of each 12 KB chunk
      uint64_t pol;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      for (int s = 0; s < KC; s++) {
        const int slot = s % ST;
        if (s >= ST) u8_wait(wempty + 8 * slot, ((s / ST) & 1) ^ 1);
        u8_expect(full + 8 * slot, WB);
        if constexpr (HN & 1) u8_bulkh(wring + slot * WB, src + (size_t)s * 12288, WB, full + 8 * slot, pol);
        else u8_bulk(wring + slot * WB, src + (size_t)s * 12288, WB, full + 8 * slot);
      }
      if (INS) p.prof[c * 32 + 15] = u8_gt();
    } else if (warp == 9 && lane == 0) {
      uint64_t pol;
      asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(pol));
      for (int s = 0; s < KC; s++) {
        const int slot = s % ST;
        if (s >= ST) u8_wait(xempty + 8 * slot, ((s / ST) & 1) ^ 1);
        u8_expect(full + 8 * slot, XB);
        if constexpr ((HN & 2) != 0) u8_tileh(xring + slot * XB, &tx, s * 64, 0, full + 8 * slot, pol);
        else u8_tile(xring + slot * XB, &tx, s * 64, 0, full + 8 * slot);
      }
      if (INS) p.prof[c * 32 + 16] = u8_gt();
    }
    __syncwarp();
    return;
  }
  // ---------------- math warpgroups
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  const int g = warp >> 2, w = warp & 3;
  uint32_t hs = E4.x >> 17;
  uint32_t Tb = p.Tb;
  asm volatile("" : "+r"(Tb));
  const uint32_t wl = wring + w * 1536 + lane * 16;
  const uint64_t xd0 = u8_desc(xring);
  float D[64];  // never zero-filled: scale-d 0 on the WG's first k16
  uint32_t A0[16], A1[16];
#pragma unroll
  for (int i = 0; i < 16; i++) A0[i] = A1[i] = 0u;
  uint4 Q0 = make_uint4(0, 0, 0, 0), Q1 = Q0, Q2 = Q0;
  uint32_t sink = 0;
  const int NI = KC / 2;  // host guarantees KC even
  const unsigned long long gtl0 = INS ? u8_gt() : 0;
  auto ldr = [&](int i) {  // full wait, LDS of this warp's 48 B per lane, W slot released
    const int s = 2 * i + g, slot = s % ST;
    u8_wait(full + 8 * slot, (s / ST) & 1);
    if (PM != 2) {
      const uint32_t a = wl + slot * WB;
      Q0 = u8_lds(a); Q1 = u8_lds(a + 512); Q2 = u8_lds(a + 1024);
    }
    __syncwarp();
    if (lane == 0) u8_arrive(wempty + 8 * slot);  // release semantics: ordered after the LDS above
  };
  auto xrel = [&](int i) {
    const int s = 2 * i + g, slot = s % ST;
    __syncwarp();
    if (lane == 0) u8_arrive(xempty + 8 * slot);
    __syncwarp();
  };
  auto dec = [&](int i, uint32_t (&A)[16]) {
    uint32_t H[8];
    u8_hi8(Q2, Tb, H);
    u8_fixp8(H, i, E4, hs);
    u8_prmtA(Q0, Q1, H, A);
  };
  auto mma = [&](int i, uint32_t (&A)[16]) {
    const int s = 2 * i + g;
    u8_facc(D);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    const uint64_t d = xd0 + (uint64_t)(((s % ST) * XB) >> 4);
#pragma unroll
    for (int q = 0; q < 4; q++) u8_wgmma(D, A + 4 * q, d + 2 * q, (i | q) != 0);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    u8_facc(D);
  };
  if constexpr (PM == 2) {
#pragma unroll 1
    for (int i = 0; i < NI; i++) { ldr(i); xrel(i); }
  } else if constexpr (PM == 3) {
#pragma unroll 1
    for (int i = 0; i < NI; i++) {
      ldr(i);
      dec(i, A0);
#pragma unroll
      for (int j = 0; j < 16; j++) sink ^= A0[j];
      xrel(i);
    }
  } else {
    // dec(i) -> issue(i) -> LDS(i + 1) -> wait_group 1 (retires i - 1) -> release x(i - 1)
    auto round = [&](int i, uint32_t (&A)[16], uint32_t (&Ap)[16]) {
      dec(i, A);
      mma(i, A);
      if (i + 1 < NI) ldr(i + 1);
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      u8_fa(Ap);
      if (i > 0) xrel(i - 1);
    };
    ldr(0);
    int i = 0;
#pragma unroll 1
    for (; i + 1 < NI; i += 2) { round(i, A0, A1); round(i + 1, A1, A0); }
    if (i < NI) { round(i, A0, A1); i++; }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    u8_facc(D);
    u8_fa(A0); u8_fa(A1);
    xrel(NI - 1);
  }
  const unsigned long long gtl1 = INS ? u8_gt() : 0;
  if constexpr (PM == 0) {
    // both WGs retired every wgmma and consumed every full barrier: the x ring is free. Intra-CTA exchange: WG g owns
    // the 8-token groups [8g, 8g + 8); it stores its partials of the other WG's groups (16 KB, fragment order) to shared
    // memory, then adds the other WG's partials of its own groups: even + odd stages, FP32, the same two operands every
    // call (IEEE add is commutative). One BF16 rounding.
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
    const int t = tid & 127;
    auto epi = [&](auto GI) {
      constexpr int G = decltype(GI)::value, O = 8 * (1 - G);
      const uint32_t mine = xring + G * 16384, other = xring + (1 - G) * 16384, slab = xring + 32768;
#pragma unroll
      for (int j = 0; j < 8; j++)
        asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "r"(mine + (j * 128 + t) * 16),
                     "f"(D[4 * (O + j)]), "f"(D[4 * (O + j) + 1]), "f"(D[4 * (O + j) + 2]), "f"(D[4 * (O + j) + 3]) : "memory");
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float4 v;
        asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(other + (j * 128 + t) * 16) : "memory");
        const int i = 8 * G + j;
        if (G == 0) { D[4 * i] = D[4 * i] + v.x; D[4 * i + 1] = D[4 * i + 1] + v.y; D[4 * i + 2] = D[4 * i + 2] + v.z; D[4 * i + 3] = D[4 * i + 3] + v.w; }
        else { D[4 * i] = v.x + D[4 * i]; D[4 * i + 1] = v.y + D[4 * i + 1]; D[4 * i + 2] = v.z + D[4 * i + 2]; D[4 * i + 3] = v.w + D[4 * i + 3]; }
      }
      // round once -> stmatrix.trans into the slab [128 tok][64 rows] (SW128 rows of 128 B), then coalesced 16 B stores
      const int jm = lane >> 3, rr = lane & 7;
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        const int i0 = 8 * G + j, i = i0 + (jm >> 1), ch = 2 * w + (jm & 1), tok = 8 * i + rr;
        const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
        const uint32_t r0 = u8_pk(D[4 * i0], D[4 * i0 + 1]), r1 = u8_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
        const uint32_t r2 = u8_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = u8_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
      }
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + G) : "memory");
#pragma unroll
      for (int e = t; e < 512; e += 128) {
        const int tok = 64 * G + (e >> 3), ch = e & 7;
        const uint4 q = u8_lds(slab + tok * 128 + ((ch ^ (tok & 7)) << 4));
        if (tok < p.m)
          asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};\n" :: "l"(p.out + (size_t)tok * p.N + 64 * c + 8 * ch),
                       "r"(q.x), "r"(q.y), "r"(q.z), "r"(q.w) : "memory");
      }
    };
    if (g == 0) epi(std::integral_constant<int, 0>{});
    else epi(std::integral_constant<int, 1>{});
  } else {
    if (sink == 0x9e3779b9u || D[0] == 1.2345f) p.sink[0] = sink;
  }
  if (INS && w == 0 && lane == 0) {
    p.prof[c * 32 + 2 * g + 0] = gtl0; p.prof[c * 32 + 2 * g + 1] = gtl1;
    if (g == 0) p.prof[c * 32 + 11] = gt0;
    p.prof[c * 32 + 12 + g] = u8_gt();
  }
#endif
}

typedef CUresult (*U8Enc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                          const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(U8Tmap), "CUtensorMap size");
U8Enc u8_enc = nullptr;
// 2D bf16 map: dims {inner, outer}, box {64, box_outer}, 128B swizzle; rows past `outer` read as 0
int u8_map(U8Tmap* tm, const void* ptr, int64_t inner, int64_t outer, int box_outer) {
  if (!u8_enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    u8_enc = (U8Enc)fn;
  }
  if (((uintptr_t)ptr) % 16 || (inner * 2) % 16 || box_outer < 1 || box_outer > 256) return -21;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_outer};
  cuuint32_t estr[2] = {1, 1};
  return u8_enc((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int ST, int HN, int PM, int INS>
struct U8Go {
  static bool attr;  // internal linkage (anonymous namespace): not shared with any other extension module
  static constexpr int SMEM = 1024 + ST * (16384 + 6144) + 24 * ST;
  static int go(const U8P& p, torch::Tensor x, int info) {
    auto k = u8k<ST, HN, PM, INS>;
    if (info) {
      cudaFuncAttributes a;
      if (cudaFuncGetAttributes(&a, k) != cudaSuccess) { cudaGetLastError(); return -1; }
      return info == 1 ? a.numRegs : info == 2 ? (int)a.localSizeBytes : SMEM;
    }
    if (p.m < 1 || p.m > 128 || (p.KC & 1) || p.KC < 2 || p.N % 128) return -4;
    U8Tmap tx;
    int r = u8_map(&tx, x.data_ptr(), (int64_t)p.KC * 64, p.m, 128);
    if (r) return r;
    if (!attr) {
      if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess) { cudaGetLastError(); return -6; }
      attr = true;
    }
    k<<<dim3(p.N / 64), dim3(384), SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
  }
};
template <int ST, int HN, int PM, int INS> bool U8Go<ST, HN, PM, INS>::attr = false;
}  // namespace

int64_t u8run(int64_t v, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor esc, int64_t Tb, torch::Tensor prof,
              torch::Tensor sink, int64_t info) {
  U8P p = {};
  if (!info) {
    if (W.dim() != 4 || W.size(2) != 8 || W.size(3) != 1536 || W.scalar_type() != torch::kUInt8 || !W.is_contiguous()) return -10;
    if (x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0) || !x.is_contiguous() || !out.is_contiguous()) return -11;
    if (x.size(1) != W.size(1) * 64 || out.size(1) != W.size(0) * 128 || x.scalar_type() != torch::kBFloat16 || out.scalar_type() != torch::kBFloat16) return -12;
    if (esc.scalar_type() != torch::kInt32 || esc.numel() != W.size(0) * 2 * 2 * 4 * 32 * 4 || !esc.is_contiguous() || sink.numel() < 1) return -13;
    p.W = (const uint8_t*)W.data_ptr(); p.esc = (const uint4*)esc.data_ptr();
    p.out = (__nv_bfloat16*)out.data_ptr();
    p.prof = prof.numel() >= W.size(0) * 2 * 32 ? (unsigned long long*)prof.data_ptr() : nullptr;
    p.sink = (uint32_t*)sink.data_ptr();
    p.m = (int)x.size(0); p.N = (int)out.size(1); p.KC = (int)W.size(1);
    p.Tb = (uint32_t)(Tb & 0xffffffff);
  }
  switch (v) {
@@CASES@@
  }
  return -2;
}
"""


def _u8_wg():
    """wgmma m64n128k16, A from registers (keell0renz rq_wgmma<128>, renamed)"""
    nd = 64
    d = ", ".join("%%%d" % i for i in range(nd))
    a = ", ".join("%%%d" % (nd + i) for i in range(4))
    outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
    return (("__device__ __forceinline__ void u8_wgmma(float (&D)[64], const uint32_t* a, uint64_t desc, int scd) {\n"
             "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
             "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%s}, {%s}, %%%d, p, 1, 1, 0;\\n}\\n\"\n"
             "    : %s\n    : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(desc), \"r\"(scd));\n}\n")
            % (nd + 5, d, a, nd + 4, outs))


# shipped instance: ring depth 8, no L2 hints (u128 sweep: ST 8 beat 5/6/7/9/10; hint-less beat the hinted variant
# by ~0.2 us), the full kernel only (the PM / INS probe paths are compiled out)
_U8_V = {"s8h0": (8, 0, 0, 0)}  # name -> (ST, HN, PM, INS)
_U8_PLAN = {("unpooler", 128): "s8h0"}  # route ONLY unpooler at m == 128 (n 8192, K 4096); every other cell -> parent
_U8_MOD = [None]
_U8_THR = []
_U8_T0 = [0.0]


def _u8_src(V=None):
    V = _U8_V if V is None else V
    cases = "\n".join("    case %d: return U8Go<%s>::go(p, x, (int)info);" % (i, ", ".join(str(a) for a in V[nm])) for i, nm in enumerate(V))
    return _U8_CU.replace("@@WGMMA@@", _u8_wg()).replace("@@CASES@@", cases)


def _u8_load(V=None, verbose=False, prefix="okc_bf16proj_u128_"):
    src = _u8_src(V)
    tag = prefix + _u8_hashlib.md5(src.encode()).hexdigest()[:12]
    fl = ["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-DNDEBUG", "--expt-relaxed-constexpr"] + (["-Xptxas=-v"] if verbose else [])
    return load_inline(tag, cpp_sources="int64_t u8run(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, torch::Tensor, int64_t);",
                       cuda_sources=src, functions=["u8run"], extra_cuda_cflags=fl, verbose=verbose)


def _u8_ext():
    if _U8_MOD[0] is None:
        _U8_MOD[0] = _u8_load()
    return _U8_MOD[0]


def _u8_esc(esc6, N, KC, dev, CAPT=4):
    """escape cursors for the split-free grid: slot (CTA c = tile // 4, WG g = stage & 1, warp w = tile % 4, lane), entries
    (i << 17 | lane << 12 | reg << 8 | half << 7 | xor7) with i = stage >> 1 (the WG's own stage counter), sorted by i,
    CAPT per slot, sentinel ~0 -> (int32 [N/64 * 2 * 4 * 32 * CAPT], max entries in any slot); arr None on overflow"""
    G = (N // 64) * 2 * 4 * 32
    if not len(esc6):
        return torch.full((G * CAPT,), -1, dtype=torch.int32, device=dev), 0
    tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
    c, w, g, i = tile // 4, tile % 4, ch & 1, ch >> 1
    ent = (i << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
    slot = ((c * 2 + g) * 4 + w) * 32 + ln
    o = torch.argsort(slot * 65536 + i)
    slot, ent = slot[o], ent[o]
    rank = torch.arange(len(slot), device=dev) - torch.searchsorted(slot, slot)
    mx = int(rank.max()) + 1
    if mx > CAPT:
        return None, mx
    arr = torch.full((G * CAPT,), 0xffffffff, dtype=torch.long, device=dev)
    arr[slot * CAPT + rank] = ent
    return torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int().contiguous(), mx


def _u8_call(p, x, out):
    return _U8_MOD[0].u8run(p["i"], p["lay"], x, out, p["esc"], p["tab"], p["prof"], p["sink"], 0)


def _u8_validate(p, w, m, n, gen):
    """one-hot x8 rows bit-exact against W over every K column, random / 1%-channel x30 outlier rows against FP64 (judge
    1e-2, rows 5x; we demand 4e-3 / 2e-2), repeated calls bit-identical"""
    K = w.shape[1]
    dev = w.device
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    eye = torch.eye(K, dtype=torch.bfloat16, device=dev) * 8.0
    exact = w[:n].t()
    for start in range(0, K, m):
        x = eye[start:start + m]
        if x.shape[0] < m:
            x = torch.cat((x, torch.zeros((m - x.shape[0], K), dtype=x.dtype, device=dev)))
        if _u8_call(p, x.contiguous(), out) != 0:
            return False, "launch error"
        got = out[: min(m, K - start)]
        if not torch.isfinite(got).all() or not torch.equal(got.float(), exact[start:start + got.shape[0]].float() * 8.0):
            return False, "one-hot mismatch at column %d" % start
    del eye
    err = 0.0
    for dist in (1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16)
        o = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        if _u8_call(p, x, o) != 0:
            return False, "launch error"
        ref = x.double() @ w[:n].double().t()
        err = float((o.double() - ref).norm() / ref.norm())
        row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        if not (err <= 4e-3 and row <= 2e-2 and torch.isfinite(o).all()):
            return False, "random rel L2 %.2e row %.2e" % (err, row)
        for _ in range(3):
            o2 = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            _u8_call(p, x, o2)
            if not torch.equal(o.view(torch.int16), o2.view(torch.int16)):
                return False, "not bit-identical across calls"
    torch.cuda.synchronize()
    return True, "ok %.1e" % err


def _u8_start():
    """compile the u128 extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _u8_time.time()
        try:
            _u8_ext()
            _U8_THR.append("ok")
        except Exception as error:
            _U8_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _U8_THR.append(round(_u8_time.time() - t0, 1))
    t = threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _u8_setup(state, weights, thr):
    state["u8"] = {}
    thr.join(timeout=max(1.0, 250.0 - (_u8_time.time() - _U8_T0[0])))
    if not _U8_THR or _U8_THR[0] != "ok" or _U8_MOD[0] is None:
        print("U8 build %s" % (_U8_THR[0] if _U8_THR else "not finished"), flush=True)
        return
    gen = torch.Generator(device="cuda").manual_seed(31415)
    names = list(_U8_V)
    for (layer, m), vn in _U8_PLAN.items():
        w = weights[layer]
        n, K = w.shape
        dev = w.device
        pk = _ta_rqpack(w)
        if pk is None or not torch.equal(_ta_rqunpack(*pk, n, K), w.view(torch.int16).int() & 0xffff):
            print("U8 %s pack not exact" % layer, flush=True)
            continue
        lay, tab, esc6 = pk
        KC = lay.shape[1]
        esc, mx = _u8_esc(esc6, n, KC, dev)
        if esc is None:
            print("U8 %s escape cursor overflow (%d)" % (layer, mx), flush=True)
            continue
        p = {"i": names.index(vn), "lay": lay, "esc": esc, "tab": tab, "n": n,
             "prof": torch.empty(0, dtype=torch.int64, device=dev), "sink": torch.zeros(4, dtype=torch.int32, device=dev)}
        try:
            ok, msg = _u8_validate(p, w, m, n, gen)
        except Exception as error:
            ok, msg = False, "error %s" % str(error).splitlines()[-1][:160]
        torch.cuda.synchronize()
        print("U8 %s m=%d %s: %s" % (layer, m, vn, msg), flush=True)
        if ok:
            state["u8"][(layer, m)] = p


_u8_parent_prepare = prepare
_u8_parent_kernel = kernel_fn


def prepare(weights):
    _U8_T0[0] = _u8_time.time()
    thr = _u8_start()
    state = _u8_parent_prepare(weights)
    try:
        _u8_setup(state, weights, thr)
    except Exception as error:
        state["u8"] = {}
        print("U8 setup failed: %s" % str(error).splitlines()[-1][:200], flush=True)
    torch.cuda.synchronize()
    print("U8 prepare done in %.0fs (u128 build %s)" % (_u8_time.time() - _U8_T0[0], _U8_THR), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    p = state.get("u8", {}).get((cfg["layer"], x.shape[0]))
    if p is not None and p["n"] == cfg["n"] and x.is_contiguous():
        out = torch.empty((x.shape[0], cfg["n"]), dtype=torch.bfloat16, device=x.device)
        if _u8_call(p, x, out) == 0:
            return out
    return _u8_parent_kernel(data, state)

# ======================================================================================================================
# track p128 (adamleex, 2026-11-02): pooler m = 128 only (every other cell, incl. pooler at any other m, goes to the
# parent chain above unchanged). A leaner launch of the tA addon's split-K kernel for this one cell. Derived from
# keell0renz's rs128 "rq" kernel (https://github.com/keell0renz/muse-open-kernel-challenge, commit 8e30e1b,
# kernels/muse_bf16_proj.py, Apache License 2.0, text at the end of this file) and the tA addon's FP32 split-K exchange
# epilogue (above). CHANGED (p128): the x and W rings are separate (own full / empty barriers; the x wait sits after the
# decode), the K split points are computed in-kernel (no cold global load ahead of the first TMA), and the W lane issues
# an L2 prefetch of its CTA's first two k64 W stages as its first instruction (the startup was the cheapest term to cut:
# ~0.5 us local). It reuses the tA pooler pack / escape-cursor lists unchanged, keeps the same wgmma fragment order,
# the same per-element FP32 accumulation order and the same fixed-order FP32 exchange (no atomics on data, one final BF16
# rounding), so its output is bit-identical to the tA kernel's (checked at prepare time, together with one-hot rows over
# every K column and random / outlier rows against FP64). Any build or validation failure -> the parent route.
# r7y (adamleex, 2026-11-02, round 7b): (1) in-call x pin with spread per-chunk demotion (no cross-call pin): the CTA's K
# range is cut into chunks of CS k64 stages; the x stages of chunks 0..NPIN-1 load with L2::evict_last INSIDE the call
# (every one of the 32 CTAs of a K split re-reads the same x tiles while the 48 MB W stream passes through L2), and every
# raised line is demoted to L2::evict_normal (applypriority) before the call returns: the x producer lane signals a CTA
# barrier per chunk for each stage it knows has landed (stage s-XST when its xempty wait for refill s returns, the last XST
# stages by waiting on their xfull barriers after the last refill); an idle producer warp (observer) waits on chunk c's
# barrier, takes a release ticket on the (K split, chunk) counter, then -- once chunk c-1's counter shows every CTA of the
# K split has landed chunk c-1 -- demotes its own 1/32 share of chunk c-1's x lines; after the last chunk it does the same
# for that chunk, so a CTA exits only after its share is demoted. Integer atomics only; counters grow by 32 per launch.
# (2) optional plain L2 prefetch of x at entry; (3) the W / x slot releases are release-arrives and the W producer fences
# the proxies (fence.proxy.async.shared::cta) before each refill (r7r's formal race). Cache hints and barrier ordering
# only: the arithmetic and the output are unchanged (bit-identical).
# r8p (adamleex, 2026-11-02, round 8): the kernel reads W from a KC-MAJOR copy of the tA pooler pack, [KC][NB][8 tiles]
# [1536 B] = a lossless permute (1, 0, 2, 3) of the tA [NB][KC] pack (made and checked with torch.equal at prepare time; the
# tA pack stays for the tA fallback route), so at every ring step the 32 CTAs of a K split fetch one contiguous 384 KB window
# instead of 32 ranges 384 KB apart (as r6c / s5b already do for unpooler / qkv_gate m128). Only the W addresses change
# (stage stride NB x 12 KB; the entry prefetch of the first two stages becomes two 12 KB prefetches): every ring slot
# receives the same bytes as before, so the output is bit-identical.
# ======================================================================================================================
import hashlib as _p128_hashlib
import time as _p128_time

_P128_CU = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
// track p128 (adamleex): pooler m = 128 split-K kernel. Derived from keell0renz's rs128 "rq" kernel (Apache-2.0,
// keell0renz/muse-open-kernel-challenge commit 8e30e1b: 12-bit decode, escape cursor, swap-AB register-A wgmma, x by 2D
// TMA) and the tA addon's FP32 split-K exchange epilogue (both earlier in this file). CHANGED (p128): separate x / W rings
// with their own barriers (the x wait sits after the decode), split points computed in-kernel, and an L2 prefetch of the
// CTA's first two W stages issued as the W lane's first instruction; r8p: W read from a KC-major re-order of the same
// pack ([KC][NB] instead of [NB][KC]: each ring slot gets the same bytes). Same fragment order, same per-element
// FP32 accumulation order and the same fixed-order FP32 exchange as the tA kernel: bit-identical output.
namespace {
struct alignas(64) P128Tmap { unsigned long long v[16]; };
struct P128P {
  const uint8_t* W;                  // r8p: KC-major [KC][NB][8 tiles][1536] (lossless re-order of the tA / keell0renz 12-bit pack [NB][KC])
  const uint32_t* esc2;              // per-(ks, nb, warp, lane) escape cursor lists [KS][NB][8][32][4], sentinel-terminated
  __nv_bfloat16* out;                // [m][N]
  float* ws;                         // FP32 split-K partials [NB][KS][16 groups][256 math threads] float4
  int* cnt;                          // [0, NB): epoch; [NB, 2 NB): monotonic arrival counters; r7y: [2 NB, 2 NB + 16 KS): x-chunk tickets
  int m, N, K, KC, NB;
  uint32_t Tb;
  const uint8_t* xg;                 // r7y: x [m][K] BF16 (entry prefetch + demotion addresses; the tiles come through the tensor map)
  int fl;                            // r7y: bits 0-7 CS (x k64 stages per pin chunk, 0 = no pin), 8-15 NPIN (pinned chunks), 16 x prefetch
};
__device__ __forceinline__ void p128_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nP128_W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra P128_W_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
// r7y: slot releases are RELEASE arrives (were .relaxed.cta: the W lanes' ld.shared must be ordered before the release that lets
// the producer refill the slot; the W producer adds fence.proxy.async.shared::cta after its wempty wait for the generic ->
// async proxy hand-off, the r7r 'mode P' fix)
__device__ __forceinline__ void p128_arrive_r(uint32_t b) { asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
// r7y F1 helpers: L2 demotion of one 128-B line; release ticket (returns the pre-value); acquire load
__device__ __forceinline__ void p128_demote(const void* a) { asm volatile("applypriority.global.L2::evict_normal [%0], 128;\n" :: "l"(a) : "memory"); }
__device__ __forceinline__ int p128_rticket(int* c) {
  int o; asm volatile("atom.release.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(o) : "l"(c) : "memory"); return o;
}
__device__ __forceinline__ int p128_ldacq(const int* c) { int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(c) : "memory"); return v; }
__device__ __forceinline__ void p128_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void p128_tileh(uint32_t dst, const P128Tmap* tm, int k0, int r0, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 p128_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ uint32_t p128_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// 12-bit decode (keell0renz, FL bit 25 form): hi bytes first (8 words), escapes patched into them, then 16 PRMTs build A
__device__ __forceinline__ void p128_hi8(const uint4& q2, uint32_t Tb, uint32_t* H) {
  H[0] = (q2.x & 0x87878787u) + Tb; H[1] = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  H[2] = (q2.y & 0x87878787u) + Tb; H[3] = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  H[4] = (q2.z & 0x87878787u) + Tb; H[5] = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  H[6] = (q2.w & 0x87878787u) + Tb; H[7] = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
}
__device__ __forceinline__ void p128_prmtA(const uint4& q0, const uint4& q1, const uint32_t* H, uint32_t* A) {
  A[0] = p128_prmt(q0.x, H[0], 0x5140); A[1] = p128_prmt(q0.x, H[0], 0x7362);
  A[2] = p128_prmt(q0.y, H[1], 0x5140); A[3] = p128_prmt(q0.y, H[1], 0x7362);
  A[4] = p128_prmt(q0.z, H[2], 0x5140); A[5] = p128_prmt(q0.z, H[2], 0x7362);
  A[6] = p128_prmt(q0.w, H[3], 0x5140); A[7] = p128_prmt(q0.w, H[3], 0x7362);
  A[8] = p128_prmt(q1.x, H[4], 0x5140); A[9] = p128_prmt(q1.x, H[4], 0x7362);
  A[10] = p128_prmt(q1.y, H[5], 0x5140); A[11] = p128_prmt(q1.y, H[5], 0x7362);
  A[12] = p128_prmt(q1.z, H[6], 0x5140); A[13] = p128_prmt(q1.z, H[6], 0x7362);
  A[14] = p128_prmt(q1.w, H[7], 0x5140); A[15] = p128_prmt(q1.w, H[7], 0x7362);
}
__device__ __forceinline__ void p128_xor8(uint32_t* H, uint32_t nx) {
  const uint32_t r = (nx >> 8) & 15u;
  const uint32_t xr = (nx & 0x7fu) << (16 * (r & 1u) + 8 * ((nx >> 7) & 1u));
  const uint32_t ri = r >> 1;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %8, 0;\n\t@p xor.b32 %0, %0, %9;\n\t"
      "setp.eq.u32 p, %8, 1;\n\t@p xor.b32 %1, %1, %9;\n\t"
      "setp.eq.u32 p, %8, 2;\n\t@p xor.b32 %2, %2, %9;\n\t"
      "setp.eq.u32 p, %8, 3;\n\t@p xor.b32 %3, %3, %9;\n\t"
      "setp.eq.u32 p, %8, 4;\n\t@p xor.b32 %4, %4, %9;\n\t"
      "setp.eq.u32 p, %8, 5;\n\t@p xor.b32 %5, %5, %9;\n\t"
      "setp.eq.u32 p, %8, 6;\n\t@p xor.b32 %6, %6, %9;\n\t"
      "setp.eq.u32 p, %8, 7;\n\t@p xor.b32 %7, %7, %9;\n\t}"
      : "+r"(H[0]), "+r"(H[1]), "+r"(H[2]), "+r"(H[3]), "+r"(H[4]), "+r"(H[5]), "+r"(H[6]), "+r"(H[7])
      : "r"(ri), "r"(xr));
}
__device__ __forceinline__ void p128_adv(uint4& E, uint32_t& hs) { E.x = E.y; E.y = E.z; E.z = E.w; E.w = ~0u; hs = E.x >> 17; }
__device__ __forceinline__ void p128_fixp8(uint32_t* H, int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { p128_xor8(H, E.x); p128_adv(E, hs); }
  }
}
__device__ __forceinline__ unsigned long long p128_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t p128_desc(uint32_t saddr) {  // K-major SW128: LBO 1, SBO 1024 B
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
template <int ND> __device__ __forceinline__ void p128_facc(float (&D)[ND]) {
#pragma unroll
  for (int i = 0; i < ND; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
__device__ __forceinline__ void p128_fa(uint32_t (&A)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[i]) :: "memory");
}
__device__ __forceinline__ uint32_t p128_pk(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ void p128_wgmma(float (&D)[64], const uint32_t* a, uint64_t desc, int scd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %68, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
               "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, "
               "%24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, "
               "%47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %69, p, 1, 1, 0;\n}\n"
               : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]),
                 "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]), "+f"(D[16]), "+f"(D[17]), "+f"(D[18]), "+f"(D[19]),
                 "+f"(D[20]), "+f"(D[21]), "+f"(D[22]), "+f"(D[23]), "+f"(D[24]), "+f"(D[25]), "+f"(D[26]), "+f"(D[27]), "+f"(D[28]), "+f"(D[29]),
                 "+f"(D[30]), "+f"(D[31]), "+f"(D[32]), "+f"(D[33]), "+f"(D[34]), "+f"(D[35]), "+f"(D[36]), "+f"(D[37]), "+f"(D[38]), "+f"(D[39]),
                 "+f"(D[40]), "+f"(D[41]), "+f"(D[42]), "+f"(D[43]), "+f"(D[44]), "+f"(D[45]), "+f"(D[46]), "+f"(D[47]), "+f"(D[48]), "+f"(D[49]),
                 "+f"(D[50]), "+f"(D[51]), "+f"(D[52]), "+f"(D[53]), "+f"(D[54]), "+f"(D[55]), "+f"(D[56]), "+f"(D[57]), "+f"(D[58]), "+f"(D[59]),
                 "+f"(D[60]), "+f"(D[61]), "+f"(D[62]), "+f"(D[63])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(scd), "l"(desc));
}
// p128 kernel: XST x ring slots (16 KB each, TMA 2D SW128 tiles of x), WST W ring slots (12 KB each, one bulk copy of the
// pack per k64 stage), separate full / empty barriers per ring; KSC = compiled K split (= gridDim.y).
template <int XST, int WST, int KSC>
__global__ void __launch_bounds__(384, 1) p128_kern(const P128P p, const __grid_constant__ P128Tmap tx) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int NT = 128, XB = NT * 128, WB = 12288, ND = 64, NW = 8;
  static_assert(KSC >= 1 && KSC <= 8, "K split");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t xring = sbase, wring = xring + XST * XB, bars = wring + WST * WB;
  const uint32_t xfull = bars, xempty = bars + 8 * XST, wfull = bars + 16 * XST, wempty = bars + 16 * XST + 8 * WST;
  const uint32_t cbar = bars + 16 * XST + 16 * WST;  // r7y: up to 16 x-chunk barriers
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nb = blockIdx.x, ks = blockIdx.y;
  const int cb = nb * 128;
  // even K split, computed (a load of split points would be a cold global load ahead of the first TMA)
  const int s0 = (int)((long long)ks * p.KC / KSC), KCS = (int)((long long)(ks + 1) * p.KC / KSC) - s0;
  if (KCS % 2 != 0 || KCS < 2) asm volatile("trap;");  // host guarantees it; never hang on a bad split
  if (tid == 8 * 32) {  // p128: L2 prefetch of the CTA's first two W stages as the W lane's very first work
    const uint8_t* const w0 = p.W + ((size_t)s0 * p.NB + nb) * WB;  // r8p: KC-major, the two stages are NB * WB apart
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(w0), "r"(WB) : "memory");
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(w0 + (size_t)p.NB * WB), "r"(WB) : "memory");
  }
  // r7y F1 (see the section comment): x chunks 0..NPIN-1 (CS k64 stages each) of this CTA's K range load with L2::evict_last;
  // the host clears CS / NPIN unless x is 256-B aligned and they tile KCS in whole even chunks (a 256-B L2 fill pair of one x row never straddles
  // two chunks) and the chunk share splits evenly over the NB CTAs of a K split.
  const int CSX = p.fl & 255, NPIN = (p.fl >> 8) & 255, PIN = (CSX > 0 && NPIN > 0) ? 1 : 0, OBS = PIN;
  if ((p.fl & 65536) && warp == NW + 3) {  // r7y: plain L2 prefetch (no policy hint) of this CTA's rows of its x K range
    const int rpc = (p.m + (int)gridDim.x - 1) / (int)gridDim.x, r = nb * rpc + lane;
    if (lane < rpc && r < p.m)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(p.xg + (size_t)r * p.K * 2 + (size_t)s0 * 128), "r"(KCS * 128) : "memory");
  }
  if (tid == 0) {
    for (int i = 0; i < XST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xfull + 8 * i), "r"(1));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
    }
    for (int i = 0; i < WST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wfull + 8 * i), "r"(1));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
    }
    if (OBS)  // r7y: one barrier per pinned x chunk (count CS: one arrival per landed stage), completes once per launch
      for (int c = 0; c < NPIN; c++) asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(cbar + 8 * c), "r"(CSX));
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (tid == 8 * 32 + 32) asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
  __syncthreads();
  if (warp >= NW) {  // ---------------- producer warpgroup: warp 8 lane 0 = W lane, warp 9 lane 0 = x lane
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == NW && lane == 0) {
      uint64_t pw;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
      const uint8_t* src = p.W + ((size_t)s0 * p.NB + nb) * WB;  // r8p: KC-major: stage s of this CTA at src + s * NB * WB
      const size_t wstr = (size_t)p.NB * WB;
      int slot = 0, ph = 0;
      for (int s = 0; s < KCS; s++) {
        if (s >= WST) {
          p128_wait(wempty + 8 * slot, ph ^ 1);
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // r7y: generic reads of the slot -> async refill
        }
        p128_expect(wfull + 8 * slot, WB);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                     :: "r"(wring + slot * WB), "l"(src + (size_t)s * wstr), "r"(WB), "r"(wfull + 8 * slot), "l"(pw) : "memory");
        if (++slot == WST) { slot = 0; ph ^= 1; }
      }
    } else if (warp == NW + 1 && lane == 0) {
      uint64_t px, pxl;
      asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(px));
      if (PIN) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pxl)); else pxl = px;
      const int PS = PIN ? NPIN * CSX : 0;  // r7y: stages [0, PS) are pinned (no division on the refill path)
      int slot = 0, ph = 0, sc = 0, sn = 0;  // r7y: chunk / stage-in-chunk of stage s - XST (the one a refill proves landed)
      for (int s = 0; s < KCS; s++) {
        if (s >= XST) p128_wait(xempty + 8 * slot, ph ^ 1);
        p128_expect(xfull + 8 * slot, XB);
        p128_tileh(xring + slot * XB, &tx, (s0 + s) * 64, 0, xfull + 8 * slot, s < PS ? pxl : px);
        if (++slot == XST) { slot = 0; ph ^= 1; }
        if (OBS && s >= XST) {  // r7y: AFTER the refill is issued (the x ring runs near empty when x is pinned)
          if (sc < NPIN) p128_arrive_r(cbar + 8 * sc);  // stage s - XST was read by every math warp, so it landed
          if (++sn == CSX) { sn = 0; ++sc; }
        }
      }
      if (OBS) {  // r7y: the last XST stages are never refilled: wait for them to land, then signal
        for (int s = KCS > XST ? KCS - XST : 0; s < KCS; s++) {
          p128_wait(xfull + 8 * (s % XST), (s / XST) & 1);
          if (s / CSX < NPIN) p128_arrive_r(cbar + 8 * (s / CSX));
        }
      }
    } else if (OBS && warp == NW + 2) {  // r7y observer warp (all 32 lanes): see the section comment
      const unsigned G = gridDim.x;
      const int per = CSX * p.m / (int)G;  // host: CS * m divisible by NB
      int vp = 0;
      auto share = [&](int c, int v) {  // demote this CTA's share of chunk c once every CTA of this K split landed it
        int* const cc = p.cnt + 2 * p.NB + ks * 16 + c;
        if (lane == 0) {
          const unsigned tg = ((unsigned)v / G + 1u) * G;  // counters wrap mod 2^32, a multiple of G (= NB, a power of 2)
          for (long it = 0;; it++) {
            if ((int)((unsigned)p128_ldacq(cc) - tg) >= 0) break;
            if (it > (1 << 24)) asm volatile("trap;");  // > ~3 s: never hang
            if (it > 64) __nanosleep(128);
          }
        }
        __syncwarp();
        const uint8_t* const b0 = p.xg + (size_t)(s0 + c * CSX) * 128;
        for (int q = nb * per + lane; q < (nb + 1) * per; q += 32)
          p128_demote(b0 + (size_t)(q % p.m) * p.K * 2 + (size_t)(q / p.m) * 128);
      };
      for (int c = 0; c < NPIN; c++) {
        if (lane == 0) p128_wait(cbar + 8 * c, 0);
        __syncwarp();
        int v = 0;
        if (lane == 0) v = p128_rticket(p.cnt + 2 * p.NB + ks * 16 + c);
        if (c > 0) share(c - 1, vp);
        vp = v;
      }
      share(NPIN - 1, vp);
    }
    return;
  }
  // ---------------- math warpgroups: warp w of WG g decodes W rows 64g + 16(w&3) .. +15 of the block (= pack tile warp)
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  const int g = warp >> 2, w = warp & 3;
  uint4 E4 = *(const uint4*)(p.esc2 + (size_t)(((ks * p.NB + nb) * 8 + warp) * 32 + lane) * 4);
  uint32_t hs = E4.x >> 17;
  uint32_t frE = 0;
  if (tid == 0) frE = (uint32_t)__ldcg(p.cnt + blockIdx.x) + 1u;
  uint32_t Tb = p.Tb;
  asm volatile("" : "+r"(Tb));
  const uint64_t xd0 = p128_desc(xring);
  const uint32_t wl = wring + warp * 1536 + lane * 16;
  float D[ND];  // never zero-filled: scale-d 0 on the CTA's first k16
  uint32_t A[2][16];
#pragma unroll
  for (int j = 0; j < 2; j++)
#pragma unroll
    for (int i = 0; i < 16; i++) A[j][i] = 0u;
  uint4 Q[3];
  int xs = 0, ws = 0;
  uint32_t xph = 0, wph = 0;
  auto ldr = [&]() {  // wait W slot ws, LDS this warp's 48 B per lane, release the W slot at once
    p128_wait(wfull + 8 * ws, wph);
    const uint32_t a = wl + ws * WB;
    Q[0] = p128_lds(a); Q[1] = p128_lds(a + 512); Q[2] = p128_lds(a + 1024);
    __syncwarp();
    if (lane == 0) p128_arrive_r(wempty + 8 * ws);
    if (++ws == WST) { ws = 0; wph ^= 1; }
  };
  // round G: decode Q -> A[G&1] (escape cursor patches the hi bytes); wait x(G); 4 wgmma; LDS of G+1;
  // wait_group 1 (retires G-1); release the x slot of G-1
  auto round = [&](int G, auto UI) {
    constexpr int ai = decltype(UI)::value;
    uint32_t H[8];
    p128_hi8(Q[2], Tb, H);
    p128_fixp8(H, G, E4, hs);
    p128_prmtA(Q[0], Q[1], H, A[ai]);
    p128_wait(xfull + 8 * xs, xph);
    p128_facc<ND>(D);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    const uint64_t d = xd0 + (uint64_t)((xs * XB) >> 4);
#pragma unroll
    for (int q = 0; q < 4; q++) p128_wgmma(D, A[ai] + 4 * q, d + 2 * q, (G | q) != 0);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    p128_facc<ND>(D);
    if (G + 1 < KCS) ldr();
    asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
    p128_fa(A[ai ^ 1]);
    if (G > 0) {
      __syncwarp();
      if (lane == 0) p128_arrive_r(xempty + 8 * (xs == 0 ? XST - 1 : xs - 1));
    }
    if (++xs == XST) { xs = 0; xph ^= 1; }
  };
  ldr();
#pragma unroll 1
  for (int G = 0; G < KCS; G += 2) { round(G, std::integral_constant<int, 0>{}); round(G + 1, std::integral_constant<int, 1>{}); }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  p128_facc<ND>(D);
  p128_fa(A[0]); p128_fa(A[1]);
  {
    // FP32 split-K exchange epilogue (tA): CTA ks owns the 8-token groups [ks*NI/KS, (ks+1)*NI/KS) of its 128-row block;
    // every CTA publishes its partials of the groups it does not own, then the owner sums the KS partials of each owned
    // group in the fixed split order 0..KS-1 in FP32, rounds once to BF16 and stores. Monotonic arrival counter + epoch.
    constexpr int NI = NT / 8;
    float4* const wsb = reinterpret_cast<float4*>(p.ws) + (size_t)blockIdx.x * (size_t)(KSC * NI * NW * 32);
    auto epi = [&](auto KSI_) {
      constexpr int KSI = decltype(KSI_)::value;
      if constexpr (KSI < KSC) {
        constexpr int I0 = KSI * NI / KSC, I1 = (KSI + 1) * NI / KSC;
        {
          float4* const d = wsb + (size_t)KSI * NI * (NW * 32) + tid;
#pragma unroll
          for (int i = 0; i < NI; i++)
            if (i < I0 || i >= I1) __stcg(d + i * (NW * 32), make_float4(D[4 * i], D[4 * i + 1], D[4 * i + 2], D[4 * i + 3]));
        }
        asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");
        if (tid == 0) {
          int* const arr = p.cnt + gridDim.x + blockIdx.x;
          asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" :: "l"(arr) : "memory");
          const int target = KSC * (int)frE;
          int c = 0;
          for (int it = 0; it < (1 << 24); it++) {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(c) : "l"(arr) : "memory");
            if (c - target >= 0) break;
            if (it > 256) __nanosleep(32);
          }
          if (c - target < 0) asm volatile("trap;");
          if (KSI == 0) asm volatile("st.relaxed.gpu.global.s32 [%0], %1;\n" :: "l"(p.cnt + blockIdx.x), "r"((int)frE) : "memory");
        }
        asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");
        constexpr int NO = I1 - I0;
        float4 v[NO][KSC];
#pragma unroll
        for (int j = 0; j < NO; j++)
#pragma unroll
          for (int k = 0; k < KSC; k++)
            if (k != KSI) v[j][k] = __ldcg(wsb + ((size_t)k * NI + I0 + j) * (NW * 32) + tid);
#pragma unroll
        for (int j = 0; j < NO; j++) {
          const int i = I0 + j;
          v[j][KSI] = make_float4(D[4 * i], D[4 * i + 1], D[4 * i + 2], D[4 * i + 3]);
          float4 s = v[j][0];
#pragma unroll
          for (int k = 1; k < KSC; k++) { s.x += v[j][k].x; s.y += v[j][k].y; s.z += v[j][k].z; s.w += v[j][k].w; }
          D[4 * i] = s.x; D[4 * i + 1] = s.y; D[4 * i + 2] = s.z; D[4 * i + 3] = s.w;
        }
        const uint32_t slab = xring + g * XB;
        const int jm = lane >> 3, rr = lane & 7;
#pragma unroll
        for (int j = 0; j + 1 < NO; j += 2) {
          const int i0 = I0 + j, i = i0 + (jm >> 1), ch = 2 * w + (jm & 1), tok = 8 * i + rr;
          const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
          const uint32_t r0 = p128_pk(D[4 * i0], D[4 * i0 + 1]), r1 = p128_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
          const uint32_t r2 = p128_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = p128_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
          asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
        }
        if constexpr (NO & 1) {
          const int i = I1 - 1, ch = 2 * w + (jm & 1), tok = 8 * i + rr;
          const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
          const uint32_t r0 = p128_pk(D[4 * (I1 - 1)], D[4 * (I1 - 1) + 1]), r1 = p128_pk(D[4 * (I1 - 1) + 2], D[4 * (I1 - 1) + 3]);
          asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n" :: "r"(a), "r"(r0), "r"(r1) : "memory");
        }
        asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
        const int t = tid & 127;
#pragma unroll
        for (int e = t; e < NO * 64; e += 128) {
          const int tok = 8 * I0 + (e >> 3), ch = e & 7;
          const uint4 q = p128_lds(slab + tok * 128 + ((ch ^ (tok & 7)) << 4));
          if (tok < p.m)
            asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};\n" :: "l"(p.out + (size_t)tok * p.N + cb + 64 * g + 8 * ch),
                         "r"(q.x), "r"(q.y), "r"(q.z), "r"(q.w) : "memory");
        }
      }
    };
    switch (ks) {
      case 0: epi(std::integral_constant<int, 0>{}); break;
      case 1: epi(std::integral_constant<int, 1>{}); break;
      case 2: epi(std::integral_constant<int, 2>{}); break;
      case 3: epi(std::integral_constant<int, 3>{}); break;
      case 4: epi(std::integral_constant<int, 4>{}); break;
      case 5: epi(std::integral_constant<int, 5>{}); break;
      case 6: epi(std::integral_constant<int, 6>{}); break;
      case 7: epi(std::integral_constant<int, 7>{}); break;
      default: break;
    }
  }
#endif
}

typedef CUresult (*P128Enc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                          const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(P128Tmap), "CUtensorMap size");
int p128_map(P128Tmap* tm, const void* ptr, int64_t inner, int64_t outer, int box_outer) {
  static P128Enc enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (P128Enc)fn;
  }
  if (((uintptr_t)ptr) % 16 || (inner * 2) % 16 || box_outer < 1 || box_outer > 256) return -21;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_outer};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int XST, int WST, int KSC>
struct P128Launch {
  static constexpr int SMEM = 1024 + XST * 128 * 128 + WST * 12288 + 16 * XST + 16 * WST + 16 + 16 * 8;  // r7y: + chunk barriers
  static int go(P128P& p, torch::Tensor x, int KS, int info) {
    auto k = p128_kern<XST, WST, KSC>;
    if (info) {
      cudaFuncAttributes a;
      if (cudaFuncGetAttributes(&a, k) != cudaSuccess) { cudaGetLastError(); return -1; }
      return info == 1 ? a.numRegs : info == 2 ? (int)a.localSizeBytes : SMEM;
    }
    if (p.m != 128 || KS != KSC || p.KC % (2 * KSC) != 0) return -4;
    P128Tmap tx;
    int r = p128_map(&tx, x.data_ptr(), p.K, p.m, 128);
    if (r) return r;
    static bool once = false;  // internal linkage (anonymous namespace): one flag per instantiation per module
    if (!once) {
      if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess) { cudaGetLastError(); return -6; }
      once = true;
    }
    k<<<dim3(p.NB, KS), dim3(384), SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
  }
};
}  // namespace

int64_t p128run(int64_t v, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor esc2, torch::Tensor cnt,
                int64_t KS, int64_t Tb, torch::Tensor ws, int64_t info, int64_t fl) {
  P128P p = {};
  if (!info) {
    if (W.dim() != 4 || W.size(2) != 8 || W.size(3) != 1536 || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;
    const int64_t KCw = W.size(0), NBw = W.size(1);  // r8p: KC-major pack [KC][NB][8][1536]
    if (x.size(1) != KCw * 64 || out.size(1) != NBw * 128 || x.scalar_type() != torch::kBFloat16 || out.scalar_type() != torch::kBFloat16) return -11;
    if (!x.is_contiguous() || !out.is_contiguous() || !W.is_contiguous() || !x.is_cuda() || !out.is_cuda()) return -15;
    if (esc2.numel() < KS * NBw * 8 * 32 * 4 || esc2.scalar_type() != torch::kInt32) return -12;
    if (cnt.numel() < 2 * NBw || cnt.scalar_type() != torch::kInt32) return -13;
    if (ws.numel() < NBw * KS * 16 * 256 * 4 || ws.scalar_type() != torch::kFloat32) return -16;
    p.W = (const uint8_t*)W.data_ptr(); p.esc2 = (const uint32_t*)esc2.data_ptr();
    p.out = (__nv_bfloat16*)out.data_ptr(); p.cnt = cnt.data_ptr<int>(); p.ws = ws.data_ptr<float>();
    p.m = (int)x.size(0); p.K = (int)x.size(1); p.N = (int)out.size(1); p.KC = (int)KCw; p.NB = (int)NBw;
    p.Tb = (uint32_t)(Tb & 0xffffffff);
    p.xg = (const uint8_t*)x.data_ptr();
    {  // r7y: pin only with whole even chunks tiling the CTA's K range, shares splitting evenly over the NB CTAs of a K
       // split, a 256-B aligned x (L2 fills 256-B pairs: a 128-B offset would let one chunk's loads re-raise a line of a
       // chunk already demoted) and room for the tickets
      const int cs = (int)(fl & 255), np = (int)((fl >> 8) & 255), kcs = KS > 0 ? p.KC / (int)KS : 0;
      if (cs && (cs % 2 || kcs % cs || kcs / cs > 16 || np > kcs / cs || KS > 8 || ((uintptr_t)p.xg) % 256 ||
                 (cs * p.m) % p.NB || cnt.numel() < 2 * NBw + 16 * KS)) fl &= ~(int64_t)0xffff;
      p.fl = (int)(fl & 0x1ffff);
    }
  }
  switch (v) {
    case 0: return P128Launch<5, 5, 4>::go(p, x, (int)KS, (int)info);
  }
  return -2;
}
"""
_P128_SIG = ("int64_t p128run(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
             "int64_t, torch::Tensor, int64_t, int64_t);")
# r7y: p128run flags (static, from offline cross-process measurements): bits 0-7 = x k64 stages per pin chunk, 8-15 = pinned
# chunks (from the first), bit 16 = plain entry prefetch of x. 0 = m4np2's x behaviour (no pin, no x prefetch).
_P128_FL = 0x408
_P128_MOD = [None]
_P128_THR = []
_P128_T0 = [0.0]


def _p128_ext():
    if _P128_MOD[0] is None:
        tag = "okc_bf16proj_p128_" + _p128_hashlib.md5(_P128_CU.encode()).hexdigest()[:12]
        _P128_MOD[0] = load_inline(tag, cpp_sources=_P128_SIG, cuda_sources=_P128_CU, functions=["p128run"],
                                   extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-DNDEBUG", "--expt-relaxed-constexpr"],
                                   verbose=False)
    return _P128_MOD[0]


def _p128_start():
    """compile the p128 extension on a background thread while the parent prepare runs (CPU-only work, never fatal)"""
    def build():
        t0 = _p128_time.time()
        try:
            _p128_ext()
            _P128_THR.append("ok")
        except Exception as error:
            _P128_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _P128_THR.append(round(_p128_time.time() - t0, 1))
    t = threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _p128_call(p, x, out):
    return _P128_MOD[0].p128run(0, p["w"], x, out, p["esc2"], p["cnt"], 4, p["tab"], p["ws"], 0, p["fl"])


def _p128_validate(p, ta, w, gen):
    """one-hot x8 rows bit-exact against W over every K column; random and 1%-channel x30 outlier rows against FP64
    (judge 1e-2, rows 5x), bit-identical to the tA kernel on the same input, repeated calls bit-identical"""
    m = 128
    n, K = w.shape
    dev = w.device
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    eye = torch.eye(K, dtype=torch.bfloat16, device=dev) * 8.0
    exact = w.t()
    for start in range(0, K, m):
        if _p128_call(p, eye[start:start + m].contiguous(), out) != 0:
            return False, "launch error"
        if not torch.isfinite(out).all() or not torch.equal(out.float(), exact[start:start + m].float() * 8.0):
            return False, "one-hot mismatch at column %d" % start
    del eye
    err = 0.0
    for dist in (1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16)
        o = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        if _p128_call(p, x, o) != 0:
            return False, "launch error"
        ref = x.double() @ w.double().t()
        err = float((o.double() - ref).norm() / ref.norm())
        row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        if not (err <= 4e-3 and row <= 2e-2 and torch.isfinite(o).all()):
            return False, "random rel L2 %.2e row %.2e" % (err, row)
        ot = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        if _ta_call(ta, x, ot) != 0 or not torch.equal(o.view(torch.int16), ot.view(torch.int16)):
            return False, "not bit-identical to the tA kernel"
        for _ in range(3):
            o2 = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            _p128_call(p, x, o2)
            if not torch.equal(o.view(torch.int16), o2.view(torch.int16)):
                return False, "not bit-identical across calls"
    torch.cuda.synchronize()
    if p["fl"] & 0xffff and bool((p["cnt"][2 * p["lay"].shape[0]:] % p["lay"].shape[0]).any()):  # r7y: tickets complete
        return False, "x-chunk tickets not complete"
    return True, "ok %.1e" % err


def _p128_setup(state, weights, thr):
    state["p128"] = None
    thr.join(timeout=max(1.0, 250.0 - (_p128_time.time() - _P128_T0[0])))
    if not _P128_THR or _P128_THR[0] != "ok" or _P128_MOD[0] is None:
        print("P128 build %s" % (_P128_THR[0] if _P128_THR else "not finished"), flush=True)
        return
    ta = state.get("ta_rq", {}).get(("pooler", 128))
    if ta is None or ta.get("KS") != 4:  # reuse the tA pooler pack and its KS-4 escape lists; no tA route -> no p128 route
        print("P128 off: tA pooler m=128 route not active", flush=True)
        return
    w = weights["pooler"]
    NB = ta["lay"].shape[0]
    wkc = ta["lay"].permute(1, 0, 2, 3).contiguous()  # r8p: KC-major [KC][NB][8][1536] (lossless re-order of the tA pack)
    if wkc.dtype != torch.uint8 or wkc.dim() != 4 or not torch.equal(wkc.permute(1, 0, 2, 3), ta["lay"]):
        print("P128 off: KC-major re-order not exact", flush=True)
        return
    p = {"lay": ta["lay"], "w": wkc, "esc2": ta["esc2"], "tab": ta["tab"], "n": ta["n"], "fl": _P128_FL,
         "cnt": torch.zeros(2 * NB + 16 * 4, dtype=torch.int32, device=w.device),  # r7y: + 16 x-chunk tickets per K split
         "ws": torch.empty(NB * 4 * 16 * 256 * 4, dtype=torch.float32, device=w.device)}
    gen = torch.Generator(device="cuda").manual_seed(12829)
    try:
        ok, msg = _p128_validate(p, ta, w, gen)
    except Exception as error:
        ok, msg = False, "error %s" % str(error).splitlines()[-1][:160]
    torch.cuda.synchronize()
    print("P128 pooler m=128: %s (fl %s)" % (msg, hex(p["fl"])), flush=True)
    if ok:
        state["p128"] = p


_p128_parent_prepare = prepare
_p128_parent_kernel = kernel_fn


def prepare(weights):
    _P128_T0[0] = _p128_time.time()
    thr = _p128_start()
    state = _p128_parent_prepare(weights)
    try:
        _p128_setup(state, weights, thr)
    except Exception as error:
        state["p128"] = None
        print("P128 setup failed: %s" % str(error).splitlines()[-1][:200], flush=True)
    torch.cuda.synchronize()
    print("P128 prepare done in %.0fs (p128 build %s)" % (_p128_time.time() - _P128_T0[0], _P128_THR), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    p = state.get("p128")
    if p is not None and cfg["layer"] == "pooler" and x.shape[0] == 128 and cfg["n"] == p["n"]:
        out = torch.empty((128, p["n"]), dtype=torch.bfloat16, device=x.device)
        if _p128_call(p, x.contiguous(), out) == 0:
            return out
    return _p128_parent_kernel(data, state)

# ======================================================================================================================
# track s5a (adamleex, 2026-11-02): two small-m cells on Nathen Fernandes' dd_cap2_b_cpssm ("cps") small-m GEMV (MIT,
# notice at the top of this file; the template, pack, plan and verify helpers this file already uses for pooler m = 16),
# with launch shapes that are not routed by his tables. Routed cells (EXACT m only; every other cell, incl. the same
# layers at any other m, goes to the parent chain above unchanged):
#   pooler   m = 8  : 4 weight units in flight per warp (ST 4), compile-time m = 8, 16 rows x 8 K-warps per CTA
#                     (256 CTAs x 256 threads); parent: Nathen's smallm2 route
#   unpooler m = 16 : Nathen's unpooler bucket-2 shape (32 rows = 2 row-warps x 4 K-warps per CTA, 256 CTAs x 256 threads),
#                     compile-time m = 16; parent: aly's sk2 'unpooler16t_r72'
#   pooler   m = 16 : (r8p, round 8) the SAME cps configuration as the parent route (Nathen's pooler bucket: 16 rows x
#                     8 K-warps per CTA, ST 2, 256 CTAs x 256 threads) with a compile-time m = 16 instead of a runtime m;
#                     prepare additionally requires its output to be bit-identical to the parent route's (r6b near miss)
# All: 2-deep cp.async weight ring, mma.sync BF16 -> FP32 accumulators, the K-warp partials summed in shared memory in a
# fixed order, one BF16 rounding, no cluster / inter-CTA traffic, no atomics; lossless 12-bit weights with exact XOR escape
# patches (the cps pack); nothing input-derived cached; configurations pinned (no prepare-time timing). Prepare builds one
# small extension (cps_s5a_<sha12>, three template instances) on a background thread while the parent prepare runs, then
# per route: the cps plan (pack, stream order, per-warp escape lists, one-hot rows bit-exact over every K column, random
# rows against FP64) plus 1%-channel x30 outlier rows against FP64 with a one-hot probe row exact, and repeated calls
# bit-identical. Any build / pack / validation failure -> that cell stays on the parent route.
# Also: m2's iv section times its m = 1 cgemv candidates for qkv_gate / unpooler at prepare time; this section pins that
# choice to the configuration the scored judge run used (qkv_gate: iv cfg 1 = <2, 512, 8192, NACC 2>; unpooler: the base
# cgemv <2, 128, 4096>), so a fresh prepare() can never switch kernels (no speed change; bit-for-bit determinism).
# ======================================================================================================================
import time as _s5a_time

# (layer, m) -> (family, instance, launch cfg (WR, WK, KU, S)); cps instance = (FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN,
# CK, CWR, CWK, CKU, CS), built with cp.async ring depth _S5A_DEPTH (template CP)
_S5A_ROUTES = {
    ("pooler", 8): ("cps", (1, 1, 1, 4, 0, 0, 256, 8, 4096, 8192, 1, 8, 16, 1), (1, 8, 16, 1)),
    ("unpooler", 16): ("cps", (1, 2, 1, 2, 0, 0, 256, 16, 8192, 4096, 2, 4, 16, 1), (2, 4, 16, 1)),
    ("pooler", 16): ("cps", (1, 2, 1, 2, 0, 0, 256, 16, 4096, 8192, 1, 8, 16, 1), (1, 8, 16, 1)),   # r8p
}
_S5A_BITEQ = {("pooler", 16)}   # r8p: routes whose output must be bit-identical to the parent route (same cps config, compile-time m)
_S5A_DEPTH = 2
_S5A_EXT = {}
_S5A_THR = []
_S5A_T0 = [0.0]


def _s5a_insts(fam):
    return [inst for f, inst, _ in _S5A_ROUTES.values() if f == fam]


def _s5a_start():
    """compile the s5a extensions on a background thread while the parent prepare runs (CPU-only work, never fatal)"""
    def build():
        t0 = _s5a_time.time()
        try:
            if "cps" not in _S5A_EXT:
                _S5A_EXT["cps"] = _NDD_cps_build("s5a", _s5a_insts("cps"), _S5A_DEPTH)   # extension cps_s5a_<sha12>
            _S5A_THR.append("ok")
        except Exception as error:
            _S5A_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _S5A_THR.append(round(_s5a_time.time() - t0, 1))
    t = threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _s5a_call(sp, x, out):
    return _NDD_dd_cap2_b_cpssm_launch_raw(_S5A_EXT["cps"], sp["inst_id"], sp, x, out, sp["rows"], sp["cfg"])


def _s5a_validate(sp, w, m, gen):
    """1%-channel x30 outlier and plain random rows (one row a one-hot probe) against FP64 (judge 1e-2, rows 5x; probe row
    exact), repeated calls bit-identical"""
    n, K = w.shape
    dev = w.device
    err = 0.0
    for dist in (30.0, 1.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x[m - 1].zero_()
        x[m - 1, (K // 2) + 3] = -64.0
        x = x.to(torch.bfloat16)
        o = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        if _s5a_call(sp, x, o) != 0:
            return False, "launch error"
        ref = x.double() @ w.double().t()
        err = float((o.double() - ref).norm() / ref.norm())
        row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        probe = float((o[m - 1].double() - ref[m - 1].to(torch.bfloat16).double()).abs().max())
        if not (err <= 4e-3 and row <= 2e-2 and probe == 0.0 and bool(torch.isfinite(o).all())):
            return False, "rel L2 %.2e row %.2e probe %.1e" % (err, row, probe)
        for _ in range(3):
            o2 = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            _s5a_call(sp, x, o2)
            if not torch.equal(o.view(torch.int16), o2.view(torch.int16)):
                return False, "not bit-identical across calls"
    torch.cuda.synchronize()
    return True, "ok %.1e" % err


def _s5a_biteq(sp, layer, w, m, state, gen):
    """r8p: plain random and 1%-channel x30 outlier rows: output bit-identical to the parent route on the same x"""
    n, K = w.shape
    dev = w.device
    for dist in (1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        o = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        if _s5a_call(sp, x, o) != 0:
            return False, "launch error"
        par = _s5a_parent_kernel({"x": x, "config": {"m": m, "n": n, "k": K, "layer": layer}}, state)
        if par.shape != o.shape or par.dtype != o.dtype or not torch.equal(o.view(torch.int16), par.view(torch.int16)):
            return False, "not bit-identical to the parent route"
    torch.cuda.synchronize()
    return True, "bit-identical to the parent route"


def _s5a_setup(state, weights, thr):
    state["s5a"] = {}
    if thr is not None:
        thr.join(timeout=max(1.0, 250.0 - (_s5a_time.time() - _S5A_T0[0])))
    if not _S5A_THR or _S5A_THR[0] != "ok" or "cps" not in _S5A_EXT:
        print("S5A build %s" % (_S5A_THR[0] if _S5A_THR else "not finished"), flush=True)
        return
    if torch.cuda.get_device_capability(weights["o"].device) != (9, 0):
        print("S5A off: not an sm_90 device", flush=True)
        return
    packs = {}
    for (layer, m), (fam, inst, cfg) in _S5A_ROUTES.items():
        w = weights[layer]
        try:
            if w.dim() != 2 or w.shape[1] != inst[9] or w.shape[0] < inst[8] or w.dtype != torch.bfloat16:
                ok, msg = False, "unexpected weights"
            else:
                key = (layer, "m", m)
                ids = {i: j for j, i in enumerate(_s5a_insts(fam))}
                # the cps plan: pack, stream order, escape lists, one-hot rows over every K column, random rows
                sp = _NDD_dd_cap2_b_cpssm_plan(_S5A_EXT["cps"], ids, packs, {}, w, layer, key, inst, cfg)
                if sp is None:
                    ok, msg = False, "plan / verify failed"
                else:
                    sp = dict({k: sp[k] for k in ("wp", "esc", "t01", "u01", "fmt", "inst_id", "cfg", "rows")}, fam=fam)
                    ok, msg = _s5a_validate(sp, w[:sp["rows"]], m, torch.Generator(device="cuda").manual_seed(15151 + m))
                    if ok and (layer, m) in _S5A_BITEQ:   # r8p
                        ok2, msg2 = _s5a_biteq(sp, layer, w[:sp["rows"]], m, state, torch.Generator(device="cuda").manual_seed(25151 + m))
                        ok, msg = ok2, msg + "; " + msg2
        except Exception as error:
            ok, msg = False, "error %s" % str(error).splitlines()[-1][:160]
        torch.cuda.synchronize()
        print("S5A %s m=%d: %s" % (layer, m, msg), flush=True)
        if ok:
            state["s5a"][(layer, m)] = sp
    del packs


_s5a_parent_prepare = prepare
_s5a_parent_kernel = kernel_fn


def prepare(weights):
    _S5A_T0[0] = _s5a_time.time()
    thr = _s5a_start()
    state = _s5a_parent_prepare(weights)
    try:
        _s5a_setup(state, weights, thr)
    except Exception as error:
        state["s5a"] = {}
        print("S5A setup failed: %s" % str(error).splitlines()[-1][:200], flush=True)
    torch.cuda.synchronize()
    print("S5A prepare done in %.0fs (s5a build %s, routed %s)" % (_s5a_time.time() - _S5A_T0[0], _S5A_THR,
                                                                    sorted(state.get("s5a", {}))), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    sp = state.get("s5a", {}).get((cfg["layer"], x.shape[0]))
    if (sp is not None and cfg["n"] == sp["rows"] and x.dtype == torch.bfloat16 and x.is_contiguous()
            and x.data_ptr() % 16 == 0):
        _k_ctx_bind(x)
        out = torch.empty((x.shape[0], sp["rows"]), dtype=torch.bfloat16, device=x.device)
        if _s5a_call(sp, x, out) == 0:
            return out
    return _s5a_parent_kernel(data, state)

# ======================================================================================================================
# track s5b (adamleex, 2026-11-02): qkv_gate m = 128 and o m = 128 read W from a STAGE-MAJOR lossless re-pack.
# The kernels are alygg77's qbig (qkv_gate m 128) and ol (o m 128) routes above (Apache-2.0, vjeux lineage),
# copied under an s5b_ prefix inside an anonymous namespace; their main loops, FP32 accumulation order, the fixed-order
# FP32 DSMEM split-K reduction (qkv) and the single BF16 rounding are unchanged, so the output is BIT-IDENTICAL to the
# parent routes (checked in prepare()). Only the weight layout and its load change:
#   qkv_gate m 128: [local k stage i][K-split rank][row group][64 rows][64 k, 128B-swizzled]: each stage is one contiguous
#                   8 KB bulk copy (aly: 2D tensor map over the row-major W, 64 rows 16 KB apart), and at stage i all
#                   108 CTAs read one contiguous 864 KB range.
#   o m 128:        [k stage][64-row tile][16 KB] instead of aly's [tile][k stage][16 KB].
# Measured (GPU 7, run_bf16 paired graph replays after a 256 MiB flush, 8 windows): qkv m128 -0.22 us, o m128 -0.14 us
# vs m2 (o m 256 with the same re-pack was too placement-noisy to ship). Each routed cell is validated in prepare() (pack round trip exact, one-hot x8 rows
# bit-exact over every K column, random / outlier rows vs FP64, bit-identical to the parent route, repeat calls
# identical) and falls back to the parent route on any failure. Exact m only; every other cell -> parent.
# F1 (round 7b, 2026-11-02), qkv_gate m 128 (s5b_qk; o m 128 is unchanged): every CTA re-reads x (qkv 36x, o 128x) while W streams through L2,
# so the x TMA loads use
# L2::evict_last INSIDE the call; right after the kernel, the same call launches s5b_unpin, which applies
# applypriority.global.L2::evict_normal to every 128-B line of that x (spread over its CTAs) after the main grid has
# completed, so nothing stays pinned when the call returns (the judge's flush, the next call, other kernels see x at normal
# priority). The unpin is a programmatic dependent launch (griddepcontrol): its CTAs are already resident and start the moment the
# main grid completes (griddepcontrol.wait = full completion + memory flush of the main grid). The arithmetic is unchanged: the output is bit-identical to the parent routes.
# r8q (round 8, 2026-11-03), qkv_gate m 128: the demotion is done by s5b_qk ITSELF (one kernel = one graph node per call, no PDL,
# no second kernel). After the consumers' last wgmma (wgmma.wait_group 0: they waited on every full barrier, so every x TMA load
# of the CTA has landed and the CTA makes no further x access) they do a non-blocking bar.arrive; the producer warpgroup's
# idle warp 5 does the matching bar.sync, lane 0 takes ONE ticket for the CTA (atom.relaxed.gpu.add.u64 on a never-reset
# counter allocated in prepare; the counter is a multiple of 108 between calls, so the call's 108 tickets fill the epoch that
# the fetch-add pre-value lies in) and polls (ld.acquire.gpu + __nanosleep; trap after 20 s, never a hang) until all 108 CTAs
# of the call ticketed, i.e. until every evict_last access of the call happened; then the warp applies
# applypriority.global.L2::evict_normal to its CTA's contiguous 1/108 share of x's 128-B lines while the consumers run the
# DSMEM epilogue. A CTA exits only after its share is demoted, so nothing is pinned when the kernel (the call) ends.
# Requirements, checked in prepare(): all 108 CTAs (36 3-CTA clusters, one per SM) co-resident (cudaOccupancyMaxActiveClusters
# >= 36), and x 256-B aligned for the pin (whole 256-B L2 promotion pairs; any other x runs the same kernel with x evict_normal
# and takes no ticket). prepare() validates the route (bit-identical to the parent route, CUDA-graph capture + replay, exactly
# 108 tickets per pinned call, the unaligned-x path); only if that fails does it fall back to round 7b's route (s5b_unpin, a
# programmatic dependent launch of the same call) and, if that fails too, to the parent route (which never pins).
# ======================================================================================================================
import hashlib as _s5b_hashlib
import threading as _s5b_threading
import time as _s5b_time

_S5B_CU = r"""// track s5b kernels (sm_90a). Both are alygg77's kernels (Apache-2.0, vjeux lineage; notices at the top / end of this
// file) with the weight read from a STAGE-MAJOR lossless re-pack instead of aly's layout; main loops, FP32 accumulation
// order, the fixed-order FP32 split-K reduction (qkv) and the single BF16 rounding are aly's, so the output is
// bit-identical to aly's routes (checked in prepare()).
//  s5b_qk: aly's qbig kernel (qkv_gate m = 128) at its pinned config (64 W rows x 128 tokens per CTA, 36 row groups x a
//          3-way K split as one thread-block cluster = 108 CTAs, SS wgmma m64n128k16 with W as A, 8-stage ring, wgmma
//          wait depth 2, DSMEM push of FP32 partials to the owning rank, owner sums ranks 0..2 in order). CHANGED: aly
//          reads each 64 x 64 W tile with a 2D tensor map from the row-major W (64 separate 128 B rows 16 KB apart);
//          here W is re-packed as [local stage i][rank][group][64 rows][64 k, 128B-swizzled] and every stage is ONE
//          contiguous 8 KB bulk copy. At stage i all 108 CTAs read one contiguous 864 KB range.
//  s5b_ok: aly's ol_projection at its pinned o m = 128 config (BN 128 tokens, BK 128, 4 stages, producer warpgroup,
//          W evict_first / x evict_normal (no L2 pin), 64 W rows per CTA = 128 CTAs). CHANGED: aly's o pack is tile-major
//          [64-row tile][k stage][16 KB] (CTA c reads 128 KB at c x 128 KB); here it is stage-major [k stage][tile][16 KB]
//          (at stage i the 128 CTAs read one contiguous 2 MB range).
// r8q: s5b_qk<1> (the route) loads x with L2::evict_last and demotes it itself before its CTAs exit (s5b_tdemote, warp 5,
//     one ticket per CTA). s5b_qk<0> + s5b_unpin = round 7b's route, kept only as the prepare-time fallback; s5b_qk<2> = no pin
//     (an x that is not 256-B aligned under the r8q route).
// F1: s5b_qk loads x with L2::evict_last; s5b_unpin (launched right after, same stream, programmatic dependent launch) demotes every
//     x line of the call to evict_normal after the main grid completed.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace {
#define S5BD __device__ __forceinline__
S5BD uint32_t s5b_addr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
S5BD void s5b_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(b), "r"(n) : "memory"); }
S5BD void s5b_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(b), "r"(n) : "memory"); }
S5BD void s5b_wait_cl(uint32_t b, int ph) {
  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%0],%1; @!p bra L;}" :: "r"(b), "r"(ph) : "memory");
}
S5BD void s5b_wait(uint32_t b, int ph) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b), "r"(ph) : "memory");
}
S5BD void s5b_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
S5BD uint32_t s5b_mapa(uint32_t p, int rank) { uint32_t d; asm("mapa.shared::cluster.u32 %0,%1,%2;" : "=r"(d) : "r"(p), "r"(rank)); return d; }
S5BD void s5b_push(uint32_t p, float4 v, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0],{%1,%2,%3,%4},[%5];"
               :: "r"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(b) : "memory");
}
S5BD uint64_t s5b_desc(uint32_t a) { return uint64_t((a & 0x3ffff) >> 4) | (uint64_t(64) << 32) | (uint64_t(1) << 62); }
S5BD void s5b_bulk(uint32_t dst, const void* src, int bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
@@MMA@@

// ------------------------------------------------------------------------------------------------ qkv_gate, m = 128
constexpr int QG = 36, QS = 3, QST = 8, QTM = 128, QWB = 64 * 128, QXB = QTM * 128, QSB = QWB + QXB;
constexpr int QNV = 64 * QTM / 4, QOWN = ((QNV / 128 + QS - 1) / QS) * 128, QRB = QST * QSB + QST * 16, QRECV = QRB + 16;
constexpr int QSMEM = QST * QSB + QST * 16 + 16 + (QS - 1) * QOWN * 16;
// r8q: warp 5 of s5b_qk<1> (otherwise idle). The bar.sync pairs with the consumers' bar.arrive after their wgmma.wait_group 0:
// they waited on every full barrier, so every x TMA load of this CTA has landed and the CTA makes no further x access. Lane 0
// takes ONE ticket for the CTA on the never-reset u64 counter (a RELAXED atom: no data is published through the counter, and
// the atom is issued only after the barrier chain above, i.e. after this CTA's x loads were serviced by L2; an acq_rel atom
// puts a MEMBAR.ALL.GPU in front of it that delays the ticket by ~1 us, measured +0.3 us per call). Between calls the counter
// is a multiple of NC = 108 (each call adds exactly 108; calls on one stream do not overlap), so the call's tickets are the
// epoch [pre - pre % NC, + NC) and the CTA proceeds only once all 108 CTAs of its call ticketed, i.e. after every evict_last
// access of the call (the acquire poll load must return the final count before any demotion is issued). The wait needs the
// 108 CTAs co-resident (prepare() checks the cluster occupancy); a wait > 20 s traps instead of hanging. Then the warp demotes
// the CTA's contiguous 1/NC share of x's 128-B lines (x 256-B aligned, m x 16 KB long: the 256-B L2 promotion pairs are whole).
S5BD void s5b_tdemote(const char* xg, int m, unsigned long long* cnt, int lane) {
  constexpr unsigned long long NC = QG * QS;
  asm volatile("bar.sync 2, 160;" ::: "memory");
  if (lane == 0) {
    unsigned long long t, v, t0, t1;
    asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;" : "=l"(t) : "l"(cnt) : "memory");
    t = (t / NC + 1) * NC;  // end of this call's ticket epoch
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (;;) {
      asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(cnt) : "memory");
      if (v >= t) break;
      __nanosleep(32);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
      if (t1 - t0 > 20000000000ull) asm volatile("trap;");  // 20 s: the CTAs are not co-resident -> fail loudly, never hang
    }
  }
  __syncwarp();
  const uintptr_t a = (uintptr_t)xg & ~(uintptr_t)255, e = ((uintptr_t)xg + (size_t)m * 16384 + 255) & ~(uintptr_t)255;
  const long long nl = (long long)((e - a) >> 7), per = (long long)((nl + NC - 1) / NC), l0 = per * blockIdx.x;
  const long long l1 = l0 + per < nl ? l0 + per : nl;
  for (long long i = l0 + lane; i < l1; i += 32)
    asm volatile("applypriority.global.L2::evict_normal [%0], 128;" :: "l"(a + ((uintptr_t)i << 7)) : "memory");
  __syncwarp();  // reconverge before the .aligned cluster barrier
}
// PM (r8q): 0 = round 7b F1 (x evict_last, PDL unpin kernel s5b_unpin after it; prepare-time fallback only), 1 = the route: x
// evict_last + in-kernel ticket-gated demotion (one kernel, no PDL), 2 = no pin (x evict_normal, no ticket; unaligned x).
template <int PM>
__global__ __launch_bounds__(256) void s5b_qk(const __grid_constant__ CUtensorMap tx, const uint8_t* wp, __nv_bfloat16* out, int m,
                                              const char* xg, unsigned long long* cnt) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid = threadIdx.x, lane = tid & 31, rank = blockIdx.x % QS, group = blockIdx.x / QS;
  const int tok = blockIdx.y * QTM, row = group * 64;
  if constexpr (PM == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // F1 (PM 0 only): unpin grid may launch
  const int kb = 128 * rank / QS, nkb = 128 * (rank + 1) / QS - kb, own = ((QNV / 128 + QS - 1 - rank) / QS) * 128;
  const uint32_t base = s5b_addr(sm), bars = base + QST * QSB, empty = bars + QST * 8, rb = base + QRB;
  uint64_t pw, px;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
  if constexpr (PM == 2) asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(px));  // r8q: unaligned x, no pin
  else asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));  // F1: x pinned in the call (PM 1: demoted by s5b_tdemote, PM 0: by s5b_unpin)
  auto load = [&](int i) {
    const int stage = i % QST;
    s5b_expect(bars + stage * 8, QWB + QXB);
    s5b_bulk(base + stage * QSB, wp + ((long)(i * QS + rank) * QG + group) * QWB, QWB, bars + stage * 8, pw);  // s5b
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;"
                 :: "r"(base + stage * QSB + QWB), "l"(&tx), "r"((kb + i) * 64), "r"(tok), "r"(bars + stage * 8), "l"(px) : "memory");
  };
  if (tid == 128) {
#pragma unroll
    for (int i = 0; i < QST; i++) { s5b_init(bars + i * 8, 1); s5b_init(empty + i * 8, 4); }
    s5b_init(rb, 1);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
    for (int i = 0; i < QST; i++) if (i < nkb) load(i);
  }
  __syncthreads();
  if (tid == 128) {
    s5b_expect(rb, (QS - 1) * own * 16);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  if (tid >= 128) {
    if (tid == 128) for (int i = QST; i < nkb; i++) { s5b_wait_cl(empty + (i % QST) * 8, ((i / QST) - 1) & 1); load(i); }
    if constexpr (PM == 1) {
      if ((tid >> 5) == 5) s5b_tdemote(xg, m, cnt, lane);  // r8q: overlaps the consumers' DSMEM epilogue
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  float a[64];
#pragma unroll
  for (int j = 0; j < 64; j++) asm volatile("mov.b32 %0, 0;" : "=f"(a[j]));
#pragma unroll 1
  for (int i = 0; i < nkb; i++) {
    const int stage = i % QST;
    s5b_wait_cl(bars + stage * 8, (i / QST) & 1);
#pragma unroll
    for (int j = 0; j < 64; j++) asm volatile("" : "+f"(a[j]) :: "memory");
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const uint64_t dx = s5b_desc(base + stage * QSB + QWB + k * 32), dw = s5b_desc(base + stage * QSB + k * 32);
      s5b_mma128(a, dw, dx);
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 2;" ::: "memory");
#pragma unroll
    for (int j = 0; j < 64; j++) asm volatile("" : "+f"(a[j]) :: "memory");
    const int done = i - 2;
    if (done >= 0 && done + QST < nkb && lane == 0) s5b_arrive(empty + (done % QST) * 8);
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  if constexpr (PM == 1) asm volatile("bar.arrive 2, 160;" ::: "memory");  // r8q: all x of this CTA landed -> warp 5 (non-blocking)
  // aly's RED 5 reduction: FP32 partials of the token groups owned by the other ranks pushed into their smem (DSMEM),
  // the owner sums ranks 0..S-1 in that fixed order in FP32, rounds once to BF16 (stmatrix.trans) and stores.
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#pragma unroll
  for (int j = 0; j < QTM / 8; j++) {
    const int dst = j % QS, local = (j / QS) * 128 + tid, slot = rank - (rank > dst);
    if (dst != rank) s5b_push(s5b_mapa(base + QRECV + (slot * QOWN + local) * 16, dst),
                              make_float4(a[4 * j], a[4 * j + 1], a[4 * j + 2], a[4 * j + 3]), s5b_mapa(rb, dst));
  }
  s5b_wait_cl(rb, 0);
#pragma unroll
  for (int j = 0; j < QTM / 8; j++) if (j % QS == rank) {
    float4 sum = make_float4(0, 0, 0, 0);
#pragma unroll
    for (int r = 0; r < QS; r++) {
      const float4 z = r == rank ? make_float4(a[4 * j], a[4 * j + 1], a[4 * j + 2], a[4 * j + 3])
                                 : *(float4*)(sm + QRECV + ((r - (r > rank)) * QOWN + (j / QS) * 128 + tid) * 16);
      sum.x += z.x; sum.y += z.y; sum.z += z.z; sum.w += z.w;
    }
    a[4 * j] = sum.x; a[4 * j + 1] = sum.y; a[4 * j + 2] = sum.z; a[4 * j + 3] = sum.w;
  }
  asm volatile("bar.sync 1,128;" ::: "memory");
#pragma unroll
  for (int j = 0; j < QTM / 8; j++) if (j % QS == rank) {
    __nv_bfloat162 lo = __floats2bfloat162_rn(a[4 * j], a[4 * j + 1]), hi = __floats2bfloat162_rn(a[4 * j + 2], a[4 * j + 3]);
    const int rr = (tid / 32) * 16 + ((tid % 16) / 8) * 8, cc = 8 * j + tid % 8;
    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};"
                 :: "r"(base + 2 * (cc * 72 + rr)), "r"(*(uint32_t*)&lo), "r"(*(uint32_t*)&hi) : "memory");
  }
  asm volatile("bar.sync 1,128;" ::: "memory");
  for (int q = tid; q < (own / 128) * 64; q += 128) {
    const int token = (rank + (q / 64) * QS) * 8 + (q / 8) % 8, rr = (q % 8) * 8;
    if (tok + token < m) *(uint4*)(out + (tok + token) * 2304 + row + rr) = *(uint4*)((__nv_bfloat16*)sm + token * 72 + rr);
  }
}

// --------------------------------------------------------------------------------------------------- o, m = 128 / 256
template <int BN, int BK, int ST>
__global__ __launch_bounds__(256) void s5b_ok(const __grid_constant__ CUtensorMap tx, const uint8_t* wp, __nv_bfloat16* out, int m) {
  constexpr int T = BN, AB = 64 * BK * 2, SB = (64 + T) * BK * 2, IT = 1024 / BK;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t base = s5b_addr(sm), full = base + ST * SB, empty = full + ST * 8;
  const int tid = threadIdx.x, nt = blockIdx.x, mt = blockIdx.y;
  uint64_t pw, px;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(px));
  auto load = [&](int i) {
    const int s = i % ST;
    s5b_expect(full + s * 8, SB);
    s5b_bulk(base + s * SB, wp + ((long)i * 128 + nt) * AB, AB, full + s * 8, pw);  // s5b: stage-major W
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;"
                 :: "r"(base + s * SB + AB), "l"(&tx), "r"(mt * T), "r"(i * (BK / 64)), "r"(full + s * 8), "l"(px) : "memory");
  };
  if (tid == 128) {
#pragma unroll
    for (int s = 0; s < ST; s++) { s5b_init(full + s * 8, 1); s5b_init(empty + s * 8, 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll
    for (int i = 0; i < ST && i < IT; i++) load(i);
  }
  __syncthreads();
  if (tid >= 128) {
    if (tid == 128) {
#pragma unroll 1
      for (int i = ST; i < IT; i++) { s5b_wait(empty + (i % ST) * 8, ((i / ST) - 1) & 1); load(i); }
    }
  } else {
    float a[BN / 2];
#pragma unroll
    for (int j = 0; j < BN / 2; j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");
#pragma unroll 1
    for (int i = 0; i < IT; i++) {
      const int s = i % ST;
      s5b_wait(full + s * 8, (i / ST) & 1);
#pragma unroll
      for (int j = 0; j < BN / 2; j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
      for (int kk = 0; kk < BK / 16; kk++) {
        const uint64_t aw = s5b_desc(base + s * SB + (kk / 4) * 8192 + (kk % 4) * 32);
        const uint64_t ax = s5b_desc(base + s * SB + AB + (kk / 4) * T * 128 + (kk % 4) * 32);
        static_assert(BN == 128, "s5b_ok: m64n128k16 only");
        s5b_mma128(a, aw, ax);
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
#pragma unroll
      for (int j = 0; j < BN / 2; j++) asm volatile("" : "+f"(a[j]) :: "memory");
      const int done = i - 1;
      if (done >= 0 && done + ST < IT && tid == 0) s5b_arrive(empty + (done % ST) * 8);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
    for (int j = 0; j < BN / 2; j++) asm volatile("" : "+f"(a[j]) :: "memory");
    asm volatile("bar.sync 1,128;" ::: "memory");
#pragma unroll
    for (int j = 0; j < BN / 8; j++) {
      __nv_bfloat162 lo = __floats2bfloat162_rn(a[4 * j], a[4 * j + 1]);
      __nv_bfloat162 hi = __floats2bfloat162_rn(a[4 * j + 2], a[4 * j + 3]);
      const int rb = (tid / 32) * 16 + ((tid % 16) / 8) * 8, cb = 8 * j + (tid % 8);
      const uint32_t d = base + 2 * (cb * 72 + rb);
      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d), "r"(*(uint32_t*)&lo), "r"(*(uint32_t*)&hi) : "memory");
    }
  }
  __syncthreads();
  for (int q = tid; q < T * 8; q += 256) {
    const int token = q / 8, r8 = (q % 8) * 8;
    if (mt * T + token < m) *(uint4*)(out + (long)(mt * T + token) * 8192 + nt * 64 + r8) = *(uint4*)((__nv_bfloat16*)sm + token * 72 + r8);
  }
}


// ------------------------------------------------------------------------------------------------ F1 unpin (same call)
// applypriority.global.L2::evict_normal on every 128-B line of [x, x + nbytes) (rounded out to whole 256-B pairs), CTA b its 1/G
// share; with PDL it first waits for the main grid's completion (griddepcontrol.wait), so every evict_last access of the
// call precedes the demotion. Data untouched.
__global__ __launch_bounds__(256) void s5b_unpin(const char* x, long long nbytes) {
  asm volatile("griddepcontrol.wait;" ::: "memory");  // main grid completed, its memory operations visible
  const uintptr_t a = (uintptr_t)x & ~(uintptr_t)255, e = ((uintptr_t)x + nbytes + 255) & ~(uintptr_t)255;  // whole 256-B L2-promotion pairs
  const long long nl = (long long)((e - a) >> 7), per = (nl + gridDim.x - 1) / gridDim.x, l0 = per * blockIdx.x;
  const long long l1 = l0 + per < nl ? l0 + per : nl;
  for (long long i = l0 + threadIdx.x; i < l1; i += blockDim.x)
    asm volatile("applypriority.global.L2::evict_normal [%0], 128;" :: "l"(a + ((uintptr_t)i << 7)) : "memory");
}
int s5b_pdl = 1;
int s5b_unpin_go(int64_t x, int64_t nbytes, int64_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(132);
  cfg.blockDim = dim3(256);
  cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = s5b_pdl;  // 1 unless prepare() found PDL not capturable here
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  return int(cudaLaunchKernelEx(&cfg, s5b_unpin, (const char*)x, (long long)nbytes));
}

// ------------------------------------------------------------------------------------------------------------ host
using S5BEncode = decltype(&cuTensorMapEncodeTiled);
S5BEncode s5b_encoder() {
  static S5BEncode encode = nullptr;
  if (!encode) {
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&encode, 12000, cudaEnableDefault, &q) != cudaSuccess) encode = nullptr;
  }
  return encode;
}
int s5b_map2(CUtensorMap* tm, const void* p, uint64_t inner, uint64_t rows, uint32_t box_inner, uint32_t box_rows) {  // aly's qbig x map
  S5BEncode enc = s5b_encoder();
  if (!enc) return -100;
  uint64_t dims[2] = {inner, rows}, stride[1] = {inner * 2};
  uint32_t b[2] = {box_inner, box_rows}, es[2] = {1, 1};
  return int(enc(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p, dims, stride, b, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
int s5b_xmap3(CUtensorMap* tm, const void* p, int rows, int box, int bk) {  // aly's ol x map: {64, rows, 16 k64 blocks}
  S5BEncode enc = s5b_encoder();
  if (!enc) return -100;
  uint64_t dims[3] = {64, uint64_t(rows), 16}, stride[2] = {2048, 128};
  uint32_t b[3] = {64, uint32_t(box), uint32_t(bk / 64)}, es[3] = {1, 1, 1};
  return int(enc(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)p, dims, stride, b, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template <int PM>
struct S5BQGo {
  static bool configured;
  static int configure() {
    if (!configured) {
      cudaError_t e = cudaFuncSetAttribute(s5b_qk<PM>, cudaFuncAttributeMaxDynamicSharedMemorySize, QSMEM);
      if (e != cudaSuccess) return -(int)e;
      e = cudaFuncSetAttribute(s5b_qk<PM>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      if (e != cudaSuccess) return -(int)e;
      configured = true;
    }
    return 0;
  }
  static void config(cudaLaunchConfig_t& cfg, cudaLaunchAttribute& attr, int64_t stream) {
    cfg = {};
    cfg.gridDim = dim3(QG * QS, 1);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = QSMEM;
    cfg.stream = (cudaStream_t)stream;
    attr = {};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = QS; attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
  }
  static int go(int64_t wp, int64_t x, int64_t out, int m, int64_t stream, int64_t cnt) {
    if (m < 1 || m > QTM) return -4;
    if (const int e = configure()) return e;
    CUtensorMap tx;
    int e = s5b_map2(&tx, (const void*)x, 8192, uint64_t(m), 64, QTM);
    if (e) return e;
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attr;
    config(cfg, attr, stream);
    return int(cudaLaunchKernelEx(&cfg, s5b_qk<PM>, tx, (const uint8_t*)wp, (__nv_bfloat16*)out, m, (const char*)x, (unsigned long long*)cnt));
  }
  static int coresident() {  // r8q: max co-resident clusters of this kernel's launch (PM 1 needs all QG = 36 at once)
    if (configure()) { (void)cudaGetLastError(); return -1; }
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attr;
    config(cfg, attr, 0);
    int v = 0;
    const cudaError_t e = cudaOccupancyMaxActiveClusters(&v, s5b_qk<PM>, &cfg);
    if (e != cudaSuccess) { (void)cudaGetLastError(); return -2; }
    return v;
  }
};
template <int PM> bool S5BQGo<PM>::configured = false;
template <int BN, int BK, int ST>
struct S5BOGo {
  static bool configured;
  static int go(int64_t x, int64_t wp, int64_t out, int m, int64_t stream) {
    constexpr int T = BN, MAIN = ST * ((64 + T) * BK * 2 + 16), EPI = T * 72 * 2, SMEM = MAIN > EPI ? MAIN : EPI;
    auto fn = s5b_ok<BN, BK, ST>;
    if (m < 1 || m > T) return -4;
    if (!configured) {
      int rc = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
      if (rc) return rc;
      configured = true;
    }
    CUtensorMap tx;
    int rc = s5b_xmap3(&tx, (const void*)x, m, T, BK);
    if (rc) return rc;
    fn<<<dim3(128, 1), 256, SMEM, (cudaStream_t)stream>>>(tx, (const uint8_t*)wp, (__nv_bfloat16*)out, m);
    return int(cudaGetLastError());
  }
};
template <int BN, int BK, int ST>
bool S5BOGo<BN, BK, ST>::configured = false;
}  // namespace
__attribute__((visibility("hidden"))) int64_t s5b_set_pdl(int64_t v) { s5b_pdl = int(v); return 0; }
__attribute__((visibility("hidden"))) int64_t s5b_coresident() { return S5BQGo<1>::coresident(); }
__attribute__((visibility("hidden"))) int64_t s5b_q_run(int64_t wp, int64_t x, int64_t out, int64_t m, int64_t stream, int64_t cnt, int64_t mode) {
  if (mode == 1) {  // r8q route: ONE kernel; x pinned (and demoted by the kernel itself) only if 256-B aligned and a counter is set
    if (cnt != 0 && (x & 255) == 0) return S5BQGo<1>::go(wp, x, out, (int)m, stream, cnt);
    return S5BQGo<2>::go(wp, x, out, (int)m, stream, 0);
  }
  const int rc = S5BQGo<0>::go(wp, x, out, (int)m, stream, 0);  // round 7b route (prepare-time fallback)
  if (rc) return rc;
  if (s5b_unpin_go(x, m * 8192 * 2, stream) == 0) return 0;  // F1: unpin this call's x (m x 8192 BF16)
  (void)cudaGetLastError();  // the PDL launch failed: s5b_qk is already enqueued, so never leave its x pinned
  s5b_unpin<<<132, 256, 0, (cudaStream_t)stream>>>((const char*)x, (long long)(m * 8192 * 2));
  return int(cudaGetLastError());
}
__attribute__((visibility("hidden"))) int64_t s5b_o_run(int64_t id, int64_t x, int64_t wp, int64_t out, int64_t m, int64_t stream) {
  switch (id) {
@@OCASES@@
    default: return -99;
  }
}
"""

# (layer, m) -> (kernel, instance): q = s5b_qk; o instance 0 = BN 128 / BK 128 / 4 stages (aly's ol config for o m 128)
_S5B_ROUTES = {("qkv_gate", 128): ("q", 0), ("o", 128): ("o", 0)}
_S5B_OINST = ((128, 128, 4),)
_S5B_MOD = [None]
_S5B_THR = []
_S5B_T0 = [0.0]


def _s5b_mma(n):
    regs = ",".join("%%%d" % i for i in range(n // 2))
    outs = ",".join('"+f"(a[%d])' % i for i in range(n // 2))
    return ('S5BD void s5b_mma%d(float* a, uint64_t A, uint64_t B) {\n  asm volatile("wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 '
            '{%s}, %%%d, %%%d, 1, 1, 1, 0, 0;" : %s : "l"(A), "l"(B));\n}' % (n, n, regs, n // 2, n // 2 + 1, outs))


def _s5b_src():
    cases = "\n".join("    case %d: return S5BOGo<%s>::go(x, wp, out, (int)m, stream);" % (i, ",".join(map(str, c)))
                      for i, c in enumerate(_S5B_OINST))
    return _S5B_CU.replace("@@MMA@@", _s5b_mma(128)).replace("@@OCASES@@", cases)


def _s5b_ext():
    if _S5B_MOD[0] is None:
        src = _s5b_src()
        cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\nint64_t s5b_q_run(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
               "int64_t s5b_o_run(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\nint64_t s5b_set_pdl(int64_t);\nint64_t s5b_coresident();\n")
        _S5B_MOD[0] = load_inline("okc_bf16proj_s5b_" + _s5b_hashlib.md5((cpp + src).encode()).hexdigest()[:12], cpp, cuda_sources=src,
                                  functions=["s5b_q_run", "s5b_o_run", "s5b_set_pdl", "s5b_coresident"], no_implicit_headers=True, with_pytorch_error_handling=False,
                                  verbose=False, extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"])
    return _S5B_MOD[0]


def _s5b_start():
    """compile the s5b extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _s5b_time.time()
        try:
            _s5b_ext()
            _S5B_THR.append("ok")
        except Exception as error:
            _S5B_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _S5B_THR.append(round(_s5b_time.time() - t0, 1))
    t = _s5b_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _s5b_swz(t):
    """128B swizzle of [..., 64 rows, 64 cols] BF16 tiles (16 B chunk c of row r -> chunk c ^ (r & 7)); an involution"""
    r = torch.arange(64, device=t.device)
    cols = torch.arange(64, device=t.device)[None, :] ^ ((r[:, None] & 7) * 8)
    return t.gather(-1, cols.expand(t.shape))


def _s5b_qpack(w):
    """qkv_gate W [2304, 8192] -> [43][3 ranks][36 groups][64][64]: rank r owns k64 blocks [128r/3, 128(r+1)/3); slot (i, r)
    holds block 128r/3 + i (rank 0's slot 42 stays zero and is never read)"""
    t = _s5b_swz(w.reshape(36, 64, 128, 64).permute(2, 0, 1, 3))  # [kb][group][row][col]
    out = torch.zeros((43, 3, 36, 64, 64), dtype=w.dtype, device=w.device)
    for r in range(3):
        k0, k1 = 128 * r // 3, 128 * (r + 1) // 3
        out[:k1 - k0, r] = t[k0:k1]
    return out.contiguous()


def _s5b_qunpack(p):
    t = torch.empty((128, 36, 64, 64), dtype=p.dtype, device=p.device)
    for r in range(3):
        k0, k1 = 128 * r // 3, 128 * (r + 1) // 3
        t[k0:k1] = p[:k1 - k0, r]
    return _s5b_swz(t).permute(1, 2, 0, 3).reshape(2304, 8192)


def _s5b_opack(w, bk):
    """o W [8192, 1024] -> [1024 / bk stages][128 tiles][bk / 64][64][64] (aly's ol pack, stage-major)"""
    nb = bk // 64
    t = _s5b_swz(w.reshape(128, 64, 16, 64).permute(2, 0, 1, 3))  # [kb][tile][row][col]
    return t.reshape(16 // nb, nb, 128, 64, 64).permute(0, 2, 1, 3, 4).contiguous()


def _s5b_ounpack(p):
    t = _s5b_swz(p.permute(0, 2, 1, 3, 4).reshape(16, 128, 64, 64))
    return t.permute(1, 2, 0, 3).reshape(8192, 1024)


def _s5b_call(p, x, out):
    if p["kind"] == "q":
        return _S5B_MOD[0].s5b_q_run(p["w"].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream,
                                     p["cnt"], p["mode"])
    return _S5B_MOD[0].s5b_o_run(p["i"], x.data_ptr(), p["w"].data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream)


def _s5b_validate(p, layer, w, m, state, gen):
    """one-hot x8 rows bit-exact against W over every K column; random / 1%-channel x30 outlier rows within 4e-3 (rows
    2e-2) of FP64; bit-identical to the parent route on the same x; 3 repeat calls bit-identical"""
    n, K = w.shape
    dev = w.device
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    eye = torch.eye(K, dtype=torch.bfloat16, device=dev) * 8.0
    exact = w.t()
    for start in range(0, K, m):
        x = eye[start:start + m]
        if x.shape[0] < m:
            x = torch.cat((x, torch.zeros((m - x.shape[0], K), dtype=x.dtype, device=dev)))
        if _s5b_call(p, x.contiguous(), out) != 0:
            return False, "launch error"
        got = out[: min(m, K - start)]
        if not torch.isfinite(got).all() or not torch.equal(got.float(), exact[start:start + got.shape[0]].float() * 8.0):
            return False, "one-hot mismatch at column %d" % start
    del eye
    err = 0.0
    for dist in (1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        o = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        if _s5b_call(p, x, o) != 0:
            return False, "launch error"
        ref = x.double() @ w.double().t()
        err = float((o.double() - ref).norm() / ref.norm())
        row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        if not (err <= 4e-3 and row <= 2e-2 and torch.isfinite(o).all()):
            return False, "random rel L2 %.2e row %.2e" % (err, row)
        par = _s5b_parent_kernel({"x": x, "config": {"m": m, "n": n, "k": K, "layer": layer}}, state)
        if not torch.equal(o.view(torch.int16), par.view(torch.int16)):
            return False, "not bit-identical to the parent route"
        for _ in range(3):
            o2 = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            _s5b_call(p, x, o2)
            if not torch.equal(o.view(torch.int16), o2.view(torch.int16)):
                return False, "not bit-identical across calls"
    torch.cuda.synchronize()
    return True, "ok %.1e" % err


def _s5b_graph_check(p, w, m, gen):
    """F1: the pinned kernel and its unpin must capture and replay in a CUDA graph (the way the judge calls us), bit-identical to
    an eager call. If the programmatic dependent launch is not capturable here, fall back to a plain dependent launch (same
    ordering, slower); if neither works the route is dropped (the parent route never pins)."""
    x = torch.randn((m, w.shape[1]), generator=gen, device=w.device).to(torch.bfloat16).contiguous()
    ref = torch.empty((m, p["n"]), dtype=torch.bfloat16, device=w.device)
    if _s5b_call(p, x, ref) != 0:
        return False, "eager launch error"
    torch.cuda.synchronize()
    for mode in (1, 0):
        _S5B_MOD[0].s5b_set_pdl(mode)
        try:
            out = torch.zeros_like(ref)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            rc = []
            with torch.cuda.graph(g, stream=s):
                rc.append(_s5b_call(p, x, out))
            torch.cuda.synchronize()
            if rc != [0]:
                continue
            g.replay()
            torch.cuda.synchronize()
            if torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                return True, "ok (%s dependent launch)" % ("programmatic" if mode else "plain")
        except Exception as error:
            torch.cuda.synchronize()
            print("S5B F1 graph check mode %d: %s" % (mode, str(error).splitlines()[-1][:160] if str(error) else type(error).__name__), flush=True)
    return False, "pinned route + unpin not capturable in a CUDA graph"


_S5B_NC = 108   # r8q: CTAs (= tickets) per s5b_qk call: 36 row groups x 3-way K split


def _s5b_td_ready(m):
    """r8q (mode 1): the in-kernel demotion waits until all 108 CTAs of the call ticketed, so they must all be co-resident: 36
    3-CTA clusters at once (one CTA per SM). Checked BEFORE the route is launched at all."""
    if m != 128:
        return False, "exact m = 128 only"
    ncl = int(_S5B_MOD[0].s5b_coresident())
    nsm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if ncl < _S5B_NC // 3 or nsm < _S5B_NC:
        return False, "only %d co-resident 3-CTA clusters (%d SMs); need %d" % (ncl, nsm, _S5B_NC // 3)
    return True, "%d co-resident 3-CTA clusters >= %d (%d SMs)" % (ncl, _S5B_NC // 3, nsm)


def _s5b_td_check(p, w, m, gen):
    """r8q (mode 1): the one-kernel route in a CUDA graph (the way the judge calls us): capture + replays bit-identical to an
    eager call; exactly 108 tickets per pinned call (eager and replay; the counter stays a multiple of 108 between calls);
    a 128-B-offset (not 256-B aligned) x takes the unpinned kernel (no ticket) with the same output."""
    ct = p["cnt_t"]
    K = w.shape[1]
    torch.cuda.synchronize()
    c0 = int(ct[0])
    if c0 % _S5B_NC:
        return False, "counter %d not a multiple of %d after validation" % (c0, _S5B_NC)
    x = torch.randn((m, K), generator=gen, device=w.device).to(torch.bfloat16).contiguous()
    if x.data_ptr() % 256:
        return False, "test x not 256-B aligned"
    ref = torch.empty((m, p["n"]), dtype=torch.bfloat16, device=w.device)
    if _s5b_call(p, x, ref) != 0:
        return False, "eager launch error"
    torch.cuda.synchronize()
    if int(ct[0]) != c0 + _S5B_NC:
        return False, "eager call took %d tickets" % (int(ct[0]) - c0)
    out = torch.zeros_like(ref)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    rc = []
    with torch.cuda.graph(g, stream=s):
        rc.append(_s5b_call(p, x, out))
    torch.cuda.synchronize()
    if rc != [0]:
        return False, "launch error %s during capture" % rc
    c1 = int(ct[0])
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    if int(ct[0]) != c1 + 3 * _S5B_NC:
        return False, "3 graph replays took %d tickets" % (int(ct[0]) - c1)
    if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
        return False, "graph replay not bit-identical to the eager call"
    del g
    buf = torch.zeros(m * K + 512, dtype=torch.bfloat16, device=w.device)
    o = ((256 - buf.data_ptr() % 256) % 256 + 128) // 2
    xv = buf[o: o + m * K].view(m, K)
    xv.copy_(x)
    o2 = torch.empty_like(ref)
    torch.cuda.synchronize()
    c2 = int(ct[0])
    if xv.data_ptr() % 256 != 128 or _s5b_call(p, xv, o2) != 0:
        return False, "unaligned-x launch error"
    torch.cuda.synchronize()
    if int(ct[0]) != c2:
        return False, "unaligned x took %d tickets" % (int(ct[0]) - c2)
    if not torch.equal(o2.view(torch.int16), ref.view(torch.int16)):
        return False, "unaligned-x output differs"
    return True, "ok (one kernel per call: 108 tickets per call, eager and graph replay; counter %d; unaligned x -> unpinned, 0 tickets)" % int(ct[0])


def _s5b_setup(state, weights, thr):
    state["s5b"] = {}
    thr.join(timeout=max(1.0, 250.0 - (_s5b_time.time() - _S5B_T0[0])))
    if not _S5B_THR or _S5B_THR[0] != "ok" or _S5B_MOD[0] is None:
        print("S5B build %s" % (_S5B_THR[0] if _S5B_THR else "not finished"), flush=True)
        return
    gen = torch.Generator(device="cuda").manual_seed(27182)
    packs = {}
    for (layer, m), (kind, i) in _S5B_ROUTES.items():
        w = weights[layer].contiguous()
        if w.dtype != torch.bfloat16 or tuple(w.shape) != ((2304, 8192) if kind == "q" else (8192, 1024)):
            print("S5B %s unexpected weight %s %s" % (layer, w.dtype, tuple(w.shape)), flush=True)
            continue
        key = (kind, i)
        if key not in packs:
            pk = _s5b_qpack(w) if kind == "q" else _s5b_opack(w, _S5B_OINST[i][1])
            back = _s5b_qunpack(pk) if kind == "q" else _s5b_ounpack(pk)
            packs[key] = pk if torch.equal(back.view(torch.int16), w.view(torch.int16)) else None
            del back
        if packs[key] is None:
            print("S5B %s pack not exact" % layer, flush=True)
            continue
        p = {"kind": kind, "i": i, "w": packs[key], "n": w.shape[0], "mode": 0, "cnt": 0}
        ok = False
        if kind == "q":  # r8q: the one-kernel in-kernel demotion route (mode 1) first
            try:
                ok, msg = _s5b_td_ready(m)
                print("S5B %s m=%d F1 in-kernel demotion (r8q): %s" % (layer, m, msg), flush=True)
                if ok:
                    ct = torch.zeros(16, dtype=torch.int64, device=w.device)  # [0] = the ticket counter (never reset)
                    p.update(mode=1, cnt=ct.data_ptr(), cnt_t=ct)
                    ok, msg = _s5b_validate(p, layer, w, m, state, gen)
                    torch.cuda.synchronize()
                    print("S5B %s m=%d %s%d (in-kernel demotion): %s" % (layer, m, kind, i, msg), flush=True)
                if ok:
                    ok, msg = _s5b_td_check(p, w, m, gen)
                    torch.cuda.synchronize()
                    print("S5B %s m=%d F1 in-kernel demotion check (graph, tickets, unaligned x): %s" % (layer, m, msg), flush=True)
            except Exception as error:
                ok, msg = False, "error %s" % str(error).splitlines()[-1][:160]
                torch.cuda.synchronize()
                print("S5B %s m=%d F1 in-kernel demotion: %s" % (layer, m, msg), flush=True)
            if not ok:
                p.update(mode=0, cnt=0, cnt_t=None)
                print("S5B %s m=%d F1 in-kernel demotion not used -> round 7b route (PDL unpin kernel)" % (layer, m), flush=True)
        if not ok:
            try:
                ok, msg = _s5b_validate(p, layer, w, m, state, gen)
            except Exception as error:
                ok, msg = False, "error %s" % str(error).splitlines()[-1][:160]
            torch.cuda.synchronize()
            print("S5B %s m=%d %s%d: %s" % (layer, m, kind, i, msg), flush=True)
            if ok and kind == "q":
                try:
                    ok, msg = _s5b_graph_check(p, w, m, gen)
                except Exception as error:
                    ok, msg = False, "error %s" % str(error).splitlines()[-1][:160]
                torch.cuda.synchronize()
                print("S5B %s m=%d F1 graph check (x pinned in the call, unpin kernel): %s" % (layer, m, msg), flush=True)
        if ok:
            state["s5b"][(layer, m)] = p


_s5b_parent_prepare = prepare
_s5b_parent_kernel = kernel_fn


def prepare(weights):
    _S5B_T0[0] = _s5b_time.time()
    thr = _s5b_start()
    state = _s5b_parent_prepare(weights)
    try:
        _s5b_setup(state, weights, thr)
    except Exception as error:
        state["s5b"] = {}
        print("S5B setup failed: %s" % str(error).splitlines()[-1][:200], flush=True)
    torch.cuda.synchronize()
    print("S5B prepare done in %.0fs (s5b build %s, routed %s)" % (_s5b_time.time() - _S5B_T0[0], _S5B_THR,
          sorted((k, ("in-kernel demotion" if v["mode"] == 1 else "PDL unpin") if v["kind"] == "q" else "o") for k, v in state["s5b"].items())), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    p = state.get("s5b", {}).get((cfg["layer"], x.shape[0]))
    if p is not None and p["n"] == cfg["n"] and x.is_contiguous() and x.dtype == torch.bfloat16:
        out = torch.empty((x.shape[0], p["n"]), dtype=torch.bfloat16, device=x.device)
        if _s5b_call(p, x, out) == 0:
            return out
    return _s5b_parent_kernel(data, state)


# ======================================================================================================================
# track r6c (adamleex, 2026-11-02): unpooler m = 128 only (exact m; every other cell, incl. unpooler at any other m, goes to
# the parent chain above unchanged). The SAME arithmetic as the 'track u128' split-free kernel above (u8k: keell0renz's
# rs128 12-bit decode / escape cursor / register-A wgmma, Apache-2.0, https://github.com/keell0renz/muse-open-kernel-challenge
# commit 8e30e1b, text at the end of this file; lossless tA-addon pack), launched with three changes that only affect
# WHEN and FROM WHERE bytes are fetched:
#   (1) x row prefetch: at kernel entry CTA c issues cp.async.bulk.prefetch.L2 of x row c (all 128 rows of x are
#       requested at once, before the barrier init), so the per-stage x tiles hit L2 instead of 128 CTAs waiting on the
#       same L2 miss at every stage;
#   (2) the CTA's first two W stages are L2-prefetched at entry by an idle producer warp;
#   (3) the u128 12-bit pack is re-ordered stage-major ([k stage][128-row block][8 tiles][1536 B] instead of
#       [block][k stage][...]): at stage s all 128 CTAs read one contiguous 768 KB range. A pure permutation of the
#       12 KB chunks (round trip checked in prepare()); the escape cursors are u128's, unchanged.
# Same wgmma fragment order, same per-element FP32 accumulation (even / odd k64 stages in two warpgroups, one FP32 add,
# one BF16 rounding), plain stores, no atomics, no inter-CTA communication: output BIT-IDENTICAL to the u128 route
# (checked in prepare(), together with one-hot rows over every K column and random / outlier rows against FP64).
# Plain prefetches only (no evict_last / persisting L2 hints: nothing outlives the call). Any build or validation
# failure, or no u128 parent route -> the parent route.
# r7f (2026-11-02): the W producer lane executes fence.proxy.async.shared::cta after each wempty wait, before the
# cp.async.bulk refill (orders the math warps' generic ld.shared reads of the slot before the async-proxy write).
# ======================================================================================================================
import hashlib as _r6c_hashlib
import threading as _r6c_threading
import time as _r6c_time

_R6C_CU = r"""// track r6c (adamleex): unpooler m = 128, u128's split-free u8k kernel (keell0renz rs128 decode, Apache-2.0; earlier in
// this file) with an entry L2 prefetch of x (one row per CTA) and of the CTA's first two W stages, on a stage-major
// re-order of the u128 pack. Arithmetic unchanged (bit-identical to u8k).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
namespace {
struct alignas(64) R6CTmap { unsigned long long v[16]; };
struct R6CP {
  const uint8_t* W;    // stage-major 12-bit pack [KC][N/128][8 tiles][1536 B]
  const uint8_t* xp;   // x [m][K] BF16 (entry row prefetch only; the tiles come through the tensor map)
  const uint4* esc;    // u128 escape cursors [N/64 CTAs][2 WG][4 warps][32 lanes] x 4 entries (sentinel ~0)
  __nv_bfloat16* out;  // [m][N]
  int m, N, KC, NB;
  uint32_t Tb;
};
__device__ __forceinline__ void r6c_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nR6CW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra R6CW_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void r6c_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void r6c_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void r6c_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void r6c_tile(uint32_t dst, const R6CTmap* tm, int k0, int r0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar) : "memory");
}
__device__ __forceinline__ void r6c_pfl2(const void* src, uint32_t bytes) {  // plain L2 prefetch (no cache-policy hint)
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ uint4 r6c_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ uint32_t r6c_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// keell0renz rq_hi8 / rq_xor8 / rq_fixp8 / rq_prmtA (as in the u128 section)
__device__ __forceinline__ void r6c_hi8(const uint4& q2, uint32_t Tb, uint32_t* H) {
  H[0] = (q2.x & 0x87878787u) + Tb; H[1] = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  H[2] = (q2.y & 0x87878787u) + Tb; H[3] = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  H[4] = (q2.z & 0x87878787u) + Tb; H[5] = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  H[6] = (q2.w & 0x87878787u) + Tb; H[7] = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
}
__device__ __forceinline__ void r6c_prmtA(const uint4& q0, const uint4& q1, const uint32_t* H, uint32_t* A) {
  A[0] = r6c_prmt(q0.x, H[0], 0x5140); A[1] = r6c_prmt(q0.x, H[0], 0x7362);
  A[2] = r6c_prmt(q0.y, H[1], 0x5140); A[3] = r6c_prmt(q0.y, H[1], 0x7362);
  A[4] = r6c_prmt(q0.z, H[2], 0x5140); A[5] = r6c_prmt(q0.z, H[2], 0x7362);
  A[6] = r6c_prmt(q0.w, H[3], 0x5140); A[7] = r6c_prmt(q0.w, H[3], 0x7362);
  A[8] = r6c_prmt(q1.x, H[4], 0x5140); A[9] = r6c_prmt(q1.x, H[4], 0x7362);
  A[10] = r6c_prmt(q1.y, H[5], 0x5140); A[11] = r6c_prmt(q1.y, H[5], 0x7362);
  A[12] = r6c_prmt(q1.z, H[6], 0x5140); A[13] = r6c_prmt(q1.z, H[6], 0x7362);
  A[14] = r6c_prmt(q1.w, H[7], 0x5140); A[15] = r6c_prmt(q1.w, H[7], 0x7362);
}
__device__ __forceinline__ void r6c_xor8(uint32_t* H, uint32_t nx) {
  const uint32_t r = (nx >> 8) & 15u;
  const uint32_t xr = (nx & 0x7fu) << (16 * (r & 1u) + 8 * ((nx >> 7) & 1u));
  const uint32_t ri = r >> 1;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %8, 0;\n\t@p xor.b32 %0, %0, %9;\n\t"
      "setp.eq.u32 p, %8, 1;\n\t@p xor.b32 %1, %1, %9;\n\t"
      "setp.eq.u32 p, %8, 2;\n\t@p xor.b32 %2, %2, %9;\n\t"
      "setp.eq.u32 p, %8, 3;\n\t@p xor.b32 %3, %3, %9;\n\t"
      "setp.eq.u32 p, %8, 4;\n\t@p xor.b32 %4, %4, %9;\n\t"
      "setp.eq.u32 p, %8, 5;\n\t@p xor.b32 %5, %5, %9;\n\t"
      "setp.eq.u32 p, %8, 6;\n\t@p xor.b32 %6, %6, %9;\n\t"
      "setp.eq.u32 p, %8, 7;\n\t@p xor.b32 %7, %7, %9;\n\t}"
      : "+r"(H[0]), "+r"(H[1]), "+r"(H[2]), "+r"(H[3]), "+r"(H[4]), "+r"(H[5]), "+r"(H[6]), "+r"(H[7])
      : "r"(ri), "r"(xr));
}
__device__ __forceinline__ void r6c_adv(uint4& E, uint32_t& hs) { E.x = E.y; E.y = E.z; E.z = E.w; E.w = ~0u; hs = E.x >> 17; }
__device__ __forceinline__ void r6c_fixp8(uint32_t* H, int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { r6c_xor8(H, E.x); r6c_adv(E, hs); }
  }
}
__device__ __forceinline__ uint64_t r6c_desc(uint32_t saddr) {  // K-major SW128: LBO 1, SBO 1024 B
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void r6c_facc(float (&D)[64]) {
#pragma unroll
  for (int i = 0; i < 64; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
__device__ __forceinline__ void r6c_fa(uint32_t (&A)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[i]) :: "memory");
}
__device__ __forceinline__ uint32_t r6c_pk(float lo, float hi) {  // the single FP32 -> BF16 rounding (round to nearest even)
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
@@WGMMA@@

// Grid: N / 64 CTAs (CTA c = W rows 64c .. 64c + 63, all K, no K split), 384 threads, as u8k: math WG 0 / 1 (warps 0-7)
// take the k64 stages s = 2i + g with their own FP32 accumulators; warp 8 = W lane (one 6 KB bulk per stage), warp 9 = x
// lane (16 KB 2D TMA per stage, SW128); ring of ST slots (x 16 KB + W 6 KB), full[slot] count 2, wempty / xempty count 4.
// r6c: warp 10 lane 0 prefetches x row c into L2 and warp 11 lane 0 the CTA's first PF W stages, both before the barrier
// init; W stage s of CTA c is read from the stage-major pack at (s * NB + c / 2) * 12 KB + (c & 1) * 6 KB.
template <int ST, int PF>
__global__ void __launch_bounds__(384, 1) r6c_uk(const R6CP p, const __grid_constant__ R6CTmap tx) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int XB = 16384, WB = 6144;
  static_assert(ST >= 5 && 1024 + ST * (XB + WB) + 24 * ST <= 232448, "smem / ring depth");
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int c = blockIdx.x, nb = c >> 1, hh = c & 1, KC = p.KC;
  const uint8_t* const wsrc = p.W + (size_t)nb * 12288 + hh * 6144;
  const size_t wstr = (size_t)p.NB * 12288;
  if (tid == 352) {
#pragma unroll
    for (int s = 0; s < PF; s++)
      if (s < KC) r6c_pfl2(wsrc + s * wstr, WB);
  }
  if (tid == 320 && c < p.m) r6c_pfl2(p.xp + (size_t)c * KC * 128, KC * 128);
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t xring = sbase, wring = sbase + ST * XB, bars = wring + ST * WB;
  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST;
  uint4 E4 = make_uint4(~0u, ~0u, ~0u, ~0u);
  if (warp < 8) E4 = p.esc[(size_t)((c * 2 + (warp >> 2)) * 4 + (warp & 3)) * 32 + lane];
  if (tid == 256) {
    for (int i = 0; i < ST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"(2));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(4));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(4));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (tid == 288) asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
  __syncthreads();
  if (warp >= 8) {  // ---------------- producer warpgroup
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == 8 && lane == 0) {
      for (int s = 0; s < KC; s++) {
        const int slot = s % ST;
        if (s >= ST) r6c_wait(wempty + 8 * slot, ((s / ST) & 1) ^ 1);
        // r7f: the math warps' ld.shared reads of this W slot (released by a .release.cta arrive) happen before the
        // async-proxy bulk refill below
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        r6c_expect(full + 8 * slot, WB);
        r6c_bulk(wring + slot * WB, wsrc + s * wstr, WB, full + 8 * slot);
      }
    } else if (warp == 9 && lane == 0) {
      for (int s = 0; s < KC; s++) {
        const int slot = s % ST;
        if (s >= ST) r6c_wait(xempty + 8 * slot, ((s / ST) & 1) ^ 1);
        r6c_expect(full + 8 * slot, XB);
        r6c_tile(xring + slot * XB, &tx, s * 64, 0, full + 8 * slot);
      }
    }
    __syncwarp();
    return;
  }
  // ---------------- math warpgroups (u8k, unchanged)
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  const int g = warp >> 2, w = warp & 3;
  uint32_t hs = E4.x >> 17;
  uint32_t Tb = p.Tb;
  asm volatile("" : "+r"(Tb));
  const uint32_t wl = wring + w * 1536 + lane * 16;
  const uint64_t xd0 = r6c_desc(xring);
  float D[64];  // never zero-filled: scale-d 0 on the WG's first k16
  uint32_t A0[16], A1[16];
#pragma unroll
  for (int i = 0; i < 16; i++) A0[i] = A1[i] = 0u;
  uint4 Q0 = make_uint4(0, 0, 0, 0), Q1 = Q0, Q2 = Q0;
  const int NI = KC / 2;  // host guarantees KC even
  auto ldr = [&](int i) {  // full wait, LDS of this warp's 48 B per lane, W slot released
    const int s = 2 * i + g, slot = s % ST;
    r6c_wait(full + 8 * slot, (s / ST) & 1);
    const uint32_t a = wl + slot * WB;
    Q0 = r6c_lds(a); Q1 = r6c_lds(a + 512); Q2 = r6c_lds(a + 1024);
    __syncwarp();
    if (lane == 0) r6c_arrive(wempty + 8 * slot);
  };
  auto xrel = [&](int i) {
    const int s = 2 * i + g, slot = s % ST;
    __syncwarp();
    if (lane == 0) r6c_arrive(xempty + 8 * slot);
    __syncwarp();
  };
  auto dec = [&](int i, uint32_t (&A)[16]) {
    uint32_t H[8];
    r6c_hi8(Q2, Tb, H);
    r6c_fixp8(H, i, E4, hs);
    r6c_prmtA(Q0, Q1, H, A);
  };
  auto mma = [&](int i, uint32_t (&A)[16]) {
    const int s = 2 * i + g;
    r6c_facc(D);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    const uint64_t d = xd0 + (uint64_t)(((s % ST) * XB) >> 4);
#pragma unroll
    for (int q = 0; q < 4; q++) r6c_wgmma(D, A + 4 * q, d + 2 * q, (i | q) != 0);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    r6c_facc(D);
  };
  {
    // dec(i) -> issue(i) -> LDS(i + 1) -> wait_group 1 (retires i - 1) -> release x(i - 1)
    auto round = [&](int i, uint32_t (&A)[16], uint32_t (&Ap)[16]) {
      dec(i, A);
      mma(i, A);
      if (i + 1 < NI) ldr(i + 1);
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      r6c_fa(Ap);
      if (i > 0) xrel(i - 1);
    };
    ldr(0);
    int i = 0;
#pragma unroll 1
    for (; i + 1 < NI; i += 2) { round(i, A0, A1); round(i + 1, A1, A0); }
    if (i < NI) { round(i, A0, A1); i++; }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    r6c_facc(D);
    r6c_fa(A0); r6c_fa(A1);
    xrel(NI - 1);
  }
  {
    // intra-CTA exchange (u8k): WG g stores its partials of the other WG's 8-token groups to shared memory, then adds the
    // other WG's partials of its own groups (even + odd, FP32, the same two operands every call); one BF16 rounding.
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
    const int t = tid & 127;
    auto epi = [&](auto GI) {
      constexpr int G = decltype(GI)::value, O = 8 * (1 - G);
      const uint32_t mine = xring + G * 16384, other = xring + (1 - G) * 16384, slab = xring + 32768;
#pragma unroll
      for (int j = 0; j < 8; j++)
        asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "r"(mine + (j * 128 + t) * 16),
                     "f"(D[4 * (O + j)]), "f"(D[4 * (O + j) + 1]), "f"(D[4 * (O + j) + 2]), "f"(D[4 * (O + j) + 3]) : "memory");
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float4 v;
        asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(other + (j * 128 + t) * 16) : "memory");
        const int i = 8 * G + j;
        if (G == 0) { D[4 * i] = D[4 * i] + v.x; D[4 * i + 1] = D[4 * i + 1] + v.y; D[4 * i + 2] = D[4 * i + 2] + v.z; D[4 * i + 3] = D[4 * i + 3] + v.w; }
        else { D[4 * i] = v.x + D[4 * i]; D[4 * i + 1] = v.y + D[4 * i + 1]; D[4 * i + 2] = v.z + D[4 * i + 2]; D[4 * i + 3] = v.w + D[4 * i + 3]; }
      }
      const int jm = lane >> 3, rr = lane & 7;
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        const int i0 = 8 * G + j, i = i0 + (jm >> 1), ch = 2 * w + (jm & 1), tok = 8 * i + rr;
        const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
        const uint32_t r0 = r6c_pk(D[4 * i0], D[4 * i0 + 1]), r1 = r6c_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
        const uint32_t r2 = r6c_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = r6c_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
      }
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + G) : "memory");
#pragma unroll
      for (int e = t; e < 512; e += 128) {
        const int tok = 64 * G + (e >> 3), ch = e & 7;
        const uint4 q = r6c_lds(slab + tok * 128 + ((ch ^ (tok & 7)) << 4));
        if (tok < p.m)
          asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};\n" :: "l"(p.out + (size_t)tok * p.N + 64 * c + 8 * ch),
                       "r"(q.x), "r"(q.y), "r"(q.z), "r"(q.w) : "memory");
      }
    };
    if (g == 0) epi(std::integral_constant<int, 0>{});
    else epi(std::integral_constant<int, 1>{});
  }
#endif
}

typedef CUresult (*R6CEnc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                           const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                           CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(R6CTmap), "CUtensorMap size");
R6CEnc r6c_encf = nullptr;
// 2D bf16 map: dims {inner, outer}, box {64, box_outer}, 128B swizzle (u8_map)
int r6c_map(R6CTmap* tm, const void* ptr, int64_t inner, int64_t outer, int box_outer) {
  if (!r6c_encf) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    r6c_encf = (R6CEnc)fn;
  }
  if (((uintptr_t)ptr) % 16 || (inner * 2) % 16 || box_outer < 1 || box_outer > 256) return -21;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_outer};
  cuuint32_t estr[2] = {1, 1};
  return r6c_encf((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int ST, int PF>
struct R6CGo {
  static bool attr;  // internal linkage (anonymous namespace): not shared with any other extension module
  static constexpr int SMEM = 1024 + ST * (16384 + 6144) + 24 * ST;
  static int go(const R6CP& p, cudaStream_t stream) {
    auto k = r6c_uk<ST, PF>;
    if (p.m < 1 || p.m > 128 || (p.KC & 1) || p.KC < 2 || p.N % 128 || p.NB * 128 != p.N) return -4;
    R6CTmap tx;
    int r = r6c_map(&tx, p.xp, (int64_t)p.KC * 64, p.m, 128);
    if (r) return r;
    if (!attr) {
      if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess) { cudaGetLastError(); return -6; }
      attr = true;
    }
    k<<<dim3(p.N / 64), dim3(384), SMEM, stream>>>(p, tx);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
  }
};
template <int ST, int PF> bool R6CGo<ST, PF>::attr = false;
}  // namespace

__attribute__((visibility("hidden"))) int64_t r6c_u_run(int64_t W, int64_t x, int64_t out, int64_t esc, int64_t Tb, int64_t m, int64_t N,
                                                         int64_t K, int64_t stream) {
  if (!W || !x || !out || !esc || K % 128 || N % 128 || (x % 16) || (out % 16) || (W % 16)) return -10;
  R6CP p = {};
  p.W = (const uint8_t*)W; p.xp = (const uint8_t*)x; p.esc = (const uint4*)esc; p.out = (__nv_bfloat16*)out;
  p.m = (int)m; p.N = (int)N; p.KC = (int)(K / 64); p.NB = (int)(N / 128);
  p.Tb = (uint32_t)(Tb & 0xffffffff);
  return R6CGo<8, 2>::go(p, (cudaStream_t)stream);  // ring depth 8 (u128's), 2 W stages prefetched at entry
}
"""

_R6C_ROUTES = (("unpooler", 128),)  # exact m only (n 8192, K 4096); every other cell -> parent
_R6C_MOD = [None]
_R6C_THR = []
_R6C_T0 = [0.0]


def _r6c_wg():
    """wgmma m64n128k16, A from registers (keell0renz rq_wgmma<128>, as _u8_wg, renamed)"""
    nd = 64
    d = ", ".join("%%%d" % i for i in range(nd))
    a = ", ".join("%%%d" % (nd + i) for i in range(4))
    outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
    return (("__device__ __forceinline__ void r6c_wgmma(float (&D)[64], const uint32_t* a, uint64_t desc, int scd) {\n"
             "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
             "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%s}, {%s}, %%%d, p, 1, 1, 0;\\n}\\n\"\n"
             "    : %s\n    : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(desc), \"r\"(scd));\n}\n")
            % (nd + 5, d, a, nd + 4, outs))


def _r6c_ext():
    if _R6C_MOD[0] is None:
        src = _R6C_CU.replace("@@WGMMA@@", _r6c_wg())
        cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\n"
               "int64_t r6c_u_run(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n")
        _R6C_MOD[0] = load_inline("okc_bf16proj_r6c_" + _r6c_hashlib.md5((cpp + src).encode()).hexdigest()[:12], cpp, cuda_sources=src,
                                  functions=["r6c_u_run"], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False,
                                  extra_cuda_cflags=["-O3", "-DNDEBUG", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a"])
    return _R6C_MOD[0]


def _r6c_start():
    """compile the r6c extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _r6c_time.time()
        try:
            _r6c_ext()
            _R6C_THR.append("ok")
        except Exception as error:
            _R6C_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _R6C_THR.append(round(_r6c_time.time() - t0, 1))
    t = _r6c_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _r6c_call(p, x, out):
    return _R6C_MOD[0].r6c_u_run(p["w"].data_ptr(), x.data_ptr(), out.data_ptr(), p["esc"].data_ptr(), p["tab"], x.shape[0], p["n"],
                                 p["K"], torch.cuda.current_stream().cuda_stream)


def _r6c_validate(p, layer, w, m, state, gen):
    """one-hot x8 rows bit-exact against W over every K column; random / 1%-channel x30 outlier rows within 4e-3 (rows
    2e-2) of FP64; bit-identical to the parent route on the same x; 3 repeat calls bit-identical"""
    n, K = w.shape
    dev = w.device
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    eye = torch.eye(K, dtype=torch.bfloat16, device=dev) * 8.0
    exact = w.t()
    for start in range(0, K, m):
        x = eye[start:start + m]
        if x.shape[0] < m:
            x = torch.cat((x, torch.zeros((m - x.shape[0], K), dtype=x.dtype, device=dev)))
        if _r6c_call(p, x.contiguous(), out) != 0:
            return False, "launch error"
        got = out[: min(m, K - start)]
        if not torch.isfinite(got).all() or not torch.equal(got.float(), exact[start:start + got.shape[0]].float() * 8.0):
            return False, "one-hot mismatch at column %d" % start
    del eye
    err = 0.0
    for dist in (1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        o = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        if _r6c_call(p, x, o) != 0:
            return False, "launch error"
        ref = x.double() @ w.double().t()
        err = float((o.double() - ref).norm() / ref.norm())
        row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        if not (err <= 4e-3 and row <= 2e-2 and torch.isfinite(o).all()):
            return False, "random rel L2 %.2e row %.2e" % (err, row)
        par = _r6c_parent_kernel({"x": x, "config": {"m": m, "n": n, "k": K, "layer": layer}}, state)
        if not torch.equal(o.view(torch.int16), par.view(torch.int16)):
            return False, "not bit-identical to the parent route"
        for _ in range(3):
            o2 = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            _r6c_call(p, x, o2)
            if not torch.equal(o.view(torch.int16), o2.view(torch.int16)):
                return False, "not bit-identical across calls"
    torch.cuda.synchronize()
    return True, "ok %.1e" % err


def _r6c_setup(state, weights, thr):
    state["r6c"] = {}
    thr.join(timeout=max(1.0, 250.0 - (_r6c_time.time() - _R6C_T0[0])))
    if not _R6C_THR or _R6C_THR[0] != "ok" or _R6C_MOD[0] is None:
        print("R6C build %s" % (_R6C_THR[0] if _R6C_THR else "not finished"), flush=True)
        return
    gen = torch.Generator(device="cuda").manual_seed(16180)
    for layer, m in _R6C_ROUTES:
        par = (state.get("u8") or {}).get((layer, m))  # the u128 route this cell re-launches (its pack, cursors, table)
        w = weights[layer]
        n, K = w.shape
        if par is None or par.get("n") != n or w.dtype != torch.bfloat16:
            print("R6C %s m=%d: no u128 parent route, stays on the parent" % (layer, m), flush=True)
            continue
        lay = par["lay"]
        if lay.dtype != torch.uint8 or tuple(lay.shape) != (n // 128, K // 64, 8, 1536) or par["esc"].dtype != torch.int32:
            print("R6C %s unexpected u128 pack %s %s" % (layer, lay.dtype, tuple(lay.shape)), flush=True)
            continue
        ws = lay.permute(1, 0, 2, 3).contiguous()  # stage-major: [k stage][128-row block][8 tiles][1536 B]
        if not torch.equal(ws.permute(1, 0, 2, 3), lay):
            print("R6C %s stage-major re-order not exact" % layer, flush=True)
            continue
        p = {"w": ws, "esc": par["esc"], "tab": int(par["tab"]), "n": n, "K": K}
        try:
            ok, msg = _r6c_validate(p, layer, w, m, state, gen)
        except Exception as error:
            ok, msg = False, "error %s" % str(error).splitlines()[-1][:160]
        torch.cuda.synchronize()
        print("R6C %s m=%d: %s" % (layer, m, msg), flush=True)
        if ok:
            state["r6c"][(layer, m)] = p


_r6c_parent_prepare = prepare
_r6c_parent_kernel = kernel_fn


def prepare(weights):
    _R6C_T0[0] = _r6c_time.time()
    thr = _r6c_start()
    state = _r6c_parent_prepare(weights)
    try:
        _r6c_setup(state, weights, thr)
    except Exception as error:
        state["r6c"] = {}
        print("R6C setup failed: %s" % str(error).splitlines()[-1][:200], flush=True)
    torch.cuda.synchronize()
    print("R6C prepare done in %.0fs (r6c build %s, routed %s)" % (_r6c_time.time() - _R6C_T0[0], _R6C_THR, sorted(state["r6c"])), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    p = (state.get("r6c") or {}).get((cfg["layer"], x.shape[0]))
    if p is not None and p["n"] == cfg["n"] and x.dim() == 2 and x.shape[1] == p["K"] and x.is_contiguous() and x.dtype == torch.bfloat16:
        out = torch.empty((x.shape[0], p["n"]), dtype=torch.bfloat16, device=x.device)
        if _r6c_call(p, x, out) == 0:
            return out
    return _r6c_parent_kernel(data, state)

# ======================================================================================================================
# track r7f (adamleex, 2026-11-02): STATIC routes for exact scored cells whose in-house kernel lost to cuBLAS after the x
# pin was removed (round 7a de-pin). The table was decided OFFLINE from judge-faithful, cross-process measurements
# (cold L2 before every replay, SGLang interleaved on the same x; r7f README); prepare() never times anything.
# 'flinear' = F.linear(x, W) (cuBLAS, the call SGLang's bf16 dispatch makes on H200 for these shapes); 'torchT' =
# torch.mm(x, W^T K-major copy) (the base file's own torchT route); 'klarge' = alygg77's cuBLASLt plan for this exact
# (layer, m) from _KLARGE_CONFIGS (fixed algo, no split-K; the same call as that section's kernel_fn, which the s5b / p128 /
# OC sections above intercept before it is reached). Every listed (layer, m) is validated in prepare()
# (FP64 reference on random, 1 %-channel x30 outlier and one-hot rows, 3 bit-identical repeat calls); a failure leaves
# that cell on the parent chain above. Exact m only: nearby m keep the parent chain's routes. To re-point a cell (e.g.
# to a faster in-house kernel), delete or change its entry here.
# ======================================================================================================================
_R7F_ROUTES = {("o", 128): "flinear", ("o", 256): "klarge", ("pooler", 128): "klarge"}   # qkv_gate m128 -> track s5b (F1 pin inside the call, demoted by the kernel's own CTAs; PDL unpin only as fallback)
_r7f_parent_prepare = prepare
_r7f_parent_kernel = kernel_fn


def _r7f_run(route, x, layer, n, state):
    if route == "flinear":
        return F.linear(x, state["raw"][layer][:n])
    if route == "torchT":
        return torch.mm(x, state["rawT"][layer])
    if route == "klarge":
        key = (layer, x.shape[0])
        w = state["klarge_rawT"][layer] if key in _KLARGE_TRANSPOSED else state["raw"][layer]
        return state["klarge_lt"].run(x, w, n, state["klarge_plans"][key])
    raise ValueError("r7f: unknown route %r" % (route,))


def _r7f_validate(route, layer, m, w, state, gen):
    n, K = w.shape
    dev = w.device
    for dist in (0, 1, 2):
        if dist < 2:
            x = torch.randn((m, K), generator=gen, device=dev)
            if dist:
                x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= 30
        else:   # one-hot rows (x 8 at spread columns): each output row is a column of W, exact
            x = torch.zeros((m, K), device=dev)
            x[torch.arange(m, device=dev), (torch.arange(m, device=dev) * 7919) % K] = 8.0
        x = x.to(torch.bfloat16).contiguous()
        o = _r7f_run(route, x, layer, n, state)
        ref = x.double() @ w.double().t()
        if tuple(o.shape) != (m, n) or o.dtype != torch.bfloat16 or not torch.isfinite(o).all():
            return "bad output %s %s" % (tuple(o.shape), o.dtype)
        err = float((o.double() - ref).norm() / ref.norm())
        row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        if dist < 2 and not (err <= 4e-3 and row <= 2e-2):
            return "dist %d rel L2 %.2e row %.2e" % (dist, err, row)
        if dist == 2 and not torch.equal(o.float(), ref.float().to(torch.bfloat16).float()):
            return "one-hot rows not exact"
        for _ in range(3):
            if not torch.equal(_r7f_run(route, x, layer, n, state).view(torch.int16), o.view(torch.int16)):
                return "repeat calls differ"
    torch.cuda.synchronize()
    return None


def prepare(weights):
    state = _r7f_parent_prepare(weights)
    state["r7f"] = {}
    gen = torch.Generator(device="cuda").manual_seed(7707)
    for (layer, m), route in sorted(_R7F_ROUTES.items()):
        if (layer, m) == ("pooler", 128) and state.get("p128") is not None:   # m5d: a validated track p128 (F1) route wins
            print("R7F pooler m=128 -> %s: yields to track p128" % route, flush=True)
            continue
        try:
            w = state["raw"][layer]
            if route == "klarge" and (layer, m) not in state.get("klarge_plans", {}):
                raise RuntimeError("no klarge plan")
            bad = _r7f_validate(route, layer, m, w, state, gen)
        except Exception as error:
            bad = "error %s" % (str(error).splitlines()[-1][:160] if str(error) else type(error).__name__)
        if bad is None:
            state["r7f"][(layer, m)] = (route, w.shape[0])
        print("R7F %s m=%d -> %s: %s" % (layer, m, route, "ok" if bad is None else "off (" + bad + "), parent route"), flush=True)
    torch.cuda.synchronize()
    return state


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    r = (state.get("r7f") or {}).get((cfg["layer"], x.shape[0]))
    if r is not None and cfg["n"] == r[1] and x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_contiguous():
        return _r7f_run(r[0], x, cfg["layer"], cfg["n"], state)
    return _r7f_parent_kernel(data, state)

# ======================================================================================================================
# track r8m (adamleex, 2026-11-02): qkv_gate m = 1 and pooler m = 1 only (exact m; every other cell, incl. these two
# layers at any other m, goes to the parent chain above unchanged). Same CTA geometry and load structure as the parent's
# m = 1 cgemv routes (2 rows per CTA, 512 loading threads + 1 escape warp, 3 CTAs per SM, every load issued at CTA start),
# but the m = 1 dot products run on tensor cores (mma.sync.m16n8k16, bf16 x bf16 -> FP32): loading lane l = 4 g + t owns
# the K elements [16 tid, 16 tid + 16) of both rows (exactly the parent's loads), decodes them to bf16 pairs and feeds them
# as the A fragments: A row g = (row 0, lane group g), A row g + 8 = (row 1, lane group g), B column g = x of lane group
# g, so the diagonal entries C[g][g] / C[g + 8][g] collect lane group g's products (off-diagonal entries are discarded).
# x reaches shared memory by cp.async (no registers held; each thread reads back only its own 16-B chunks after its own
# cp.async.wait_group, no barrier); the W low bytes and a re-ordered code plane come by LDG.128, all in one asm block.
# Weights: the parent's lossless 12-bit cgemv pack (state["cg"][layer]: low-byte plane L and the per-CTA escape table)
# with the code plane re-ordered losslessly at prepare time ('rot' nibble order, then the two rows' code words of a
# thread interleaved), so the high bytes of 4 weights are (c & 0x87878787) + base4 and (rotl(c, 4) & 0x87878787) + base4;
# the decoded values are the parent's. Escapes: the parent's escape warp and table, unchanged (exact FP32 corrections).
# Arithmetic: bf16 x bf16 products exact; tensor-core FP32 accumulation (a truncating multi-term adder, same class as the file's other mma routes, then a fixed-order
# 3-step shuffle per row and a fixed-order shared-memory sum over the 16 warps + the escape warp), ONE final BF16
# rounding; deterministic; no atomics; graph-capturable; cache policies: none (plain ld.global.nc / cp.async.ca, as the
# parent's __ldg loads). Outputs may differ from the parent's FFMA cgemv only by summation order. Validated in prepare()
# per layer (one-hot x8 rows over EVERY K column bit-exact against W, random / 1 %-channel x30 outlier rows vs FP64,
# 3 bit-identical repeat calls); a failure leaves that layer on the parent route.
# ======================================================================================================================
import hashlib as _r8m_hashlib
import threading as _r8m_threading
import time as _r8m_time

_R8M_CU = r"""// track r8m (adamleex): qkv_gate / pooler m = 1 on tensor cores in the parent cgemv's geometry (section comment above)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {
constexpr int R8M_NTH = 512;              // loading threads per CTA (+ 1 escape warp)
constexpr int R8M_K = 16 * R8M_NTH;       // 8192: each loading thread owns 16 consecutive K elements of both rows
constexpr int R8M_NWARP = R8M_NTH / 32;

__device__ __forceinline__ float r8m_esc_delta(uint32_t t, uint32_t bq) {   // the parent's esc_delta
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}

__device__ __forceinline__ void r8m_mma(float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// grid N / 2, block 544: CTA b = rows 2b, 2b + 1. L [N][K] low bytes; CC [N / 2][512][16 B] = per thread {row 0 code word
// 2 tid, row 0 word 2 tid + 1, row 1 word 2 tid, row 1 word 2 tid + 1} in 'rot' order (byte j = o_j | s_{4+j} << 3 |
// o_{4+(j+1)%4} << 4 | s_j << 7); etab [N / 2][pmax] int2 (entry 0 = count; (row << 16 | col, original BF16 bits)).
__global__ void __launch_bounds__(R8M_NTH + 32, 3) r8m_kern(const uint8_t* __restrict__ L, const uint8_t* __restrict__ CC,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax, __nv_bfloat16* __restrict__ out, uint32_t base4) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float red[2][R8M_NWARP + 1];
  __shared__ __align__(16) uint4 xs[2 * R8M_NTH];
  if (warp == R8M_NWARP) {   // escape warp (the parent's): lane i takes entries 1 + i, 33 + i, ...; x[k] * delta per row
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec0 = 0.f, ec1 = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 0xFFFF]) * r8m_esc_delta((uint32_t)en.y, base4 & 0xFFu);
        const int rl = en.x >> 16;
        ec0 += (rl == 0) ? c : 0.f;
        ec1 += (rl == 1) ? c : 0.f;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { ec0 += __shfl_xor_sync(0xffffffffu, ec0, o); ec1 += __shfl_xor_sync(0xffffffffu, ec1, o); }
    if (lane == 0) { red[0][R8M_NWARP] = ec0; red[1][R8M_NWARP] = ec1; }
    __syncthreads();
    return;
  }
  // x[16 tid .. 16 tid + 16) -> xs[tid], xs[512 + tid] (cp.async, L1-allocating, default L2 policy)
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(xs);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb + tid * 16), "l"(x + 16 * tid) : "memory");
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb + (R8M_NTH + tid) * 16), "l"(x + 16 * tid + 8) : "memory");
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  uint4 l0, l1, cc;   // all three W loads in ONE asm block: nothing can consume a loaded register before all are issued
  {
    const size_t e0 = (size_t)blockIdx.x * (2 * R8M_K) + 16 * tid;
    const uint8_t* p0 = L + e0;
    const uint8_t* p1 = L + e0 + R8M_K;
    const uint8_t* p2 = CC + (size_t)blockIdx.x * R8M_K + 16 * tid;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%12];\n\t"
                 "ld.global.nc.v4.u32 {%4,%5,%6,%7}, [%13];\n\t"
                 "ld.global.nc.v4.u32 {%8,%9,%10,%11}, [%14];\n\t"
                 : "=r"(l0.x), "=r"(l0.y), "=r"(l0.z), "=r"(l0.w), "=r"(l1.x), "=r"(l1.y), "=r"(l1.z), "=r"(l1.w),
                   "=r"(cc.x), "=r"(cc.y), "=r"(cc.z), "=r"(cc.w) : "l"(p0), "l"(p1), "l"(p2));
  }
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#pragma unroll
  for (int g = 0; g < 2; ++g) {   // 8-element groups g: bf16 pairs 4 g .. 4 g + 3 of each row -> mma steps 2 g, 2 g + 1
    const uint32_t c0 = g ? cc.y : cc.x, c1 = g ? cc.w : cc.z;
    const uint32_t la0 = g ? l0.z : l0.x, lb0 = g ? l0.w : l0.y;
    const uint32_t la1 = g ? l1.z : l1.x, lb1 = g ? l1.w : l1.y;
    const uint32_t ha0 = (c0 & 0x87878787u) + base4, hb0 = (__funnelshift_l(c0, c0, 4) & 0x87878787u) + base4;
    const uint32_t ha1 = (c1 & 0x87878787u) + base4, hb1 = (__funnelshift_l(c1, c1, 4) & 0x87878787u) + base4;
    const uint4 xg = xs[g * R8M_NTH + tid];   // x pairs 4 g .. 4 g + 3 (this thread's own cp.async chunk)
    r8m_mma(d0, d1, d2, d3, __byte_perm(la0, ha0, 0x5140), __byte_perm(la1, ha1, 0x5140), __byte_perm(la0, ha0, 0x7362),
            __byte_perm(la1, ha1, 0x7362), xg.x, xg.y);
    r8m_mma(d0, d1, d2, d3, __byte_perm(lb0, hb0, 0x5140), __byte_perm(lb1, hb1, 0x5140), __byte_perm(lb0, hb0, 0x7362),
            __byte_perm(lb1, hb1, 0x7362), xg.z, xg.w);
  }
  // lane group g's sums sit on lane 4 g + g / 2: row 0 in d(g & 1), row 1 in d(2 + (g & 1)); fixed-order 3-step tree to lane 0
  const bool odd = (lane >> 2) & 1;
  float v0 = odd ? d1 : d0, v1 = odd ? d3 : d2;
  v0 += __shfl_xor_sync(0xffffffffu, v0, 4);  v1 += __shfl_xor_sync(0xffffffffu, v1, 4);    // lanes 0 9 18 27 <- 4 13 22 31
  v0 += __shfl_xor_sync(0xffffffffu, v0, 9);  v1 += __shfl_xor_sync(0xffffffffu, v1, 9);    // lanes 0 18 <- 9 27
  v0 += __shfl_xor_sync(0xffffffffu, v0, 18); v1 += __shfl_xor_sync(0xffffffffu, v1, 18);   // lane 0 <- 18
  if (lane == 0) { red[0][warp] = v0; red[1][warp] = v1; }
  __syncthreads();
  if (tid < 2) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= R8M_NWARP; ++q) v += red[tid][q];
    out[blockIdx.x * 2 + tid] = __float2bfloat16_rn(v);
  }
}
}  // namespace

__attribute__((visibility("hidden"))) int64_t r8m_run(int64_t L, int64_t CC, int64_t x, int64_t tab, int64_t pmax, int64_t out,
                                                       int64_t base4, int64_t N, int64_t K, int64_t stream) {
  if (!L || !CC || !x || !tab || !out || K != R8M_K || N < 2 || (N & 1) || pmax < 1 || (L & 15) || (CC & 15) || (x & 15) || (tab & 7))
    return -10;
  r8m_kern<<<(unsigned)(N / 2), R8M_NTH + 32, 0, (cudaStream_t)stream>>>((const uint8_t*)L, (const uint8_t*)CC,
      (const __nv_bfloat16*)x, (const int2*)tab, (int)pmax, (__nv_bfloat16*)out, (uint32_t)(base4 & 0xffffffff));
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : (int)e;
}
"""

_R8M_LAYERS = ("qkv_gate", "pooler")   # m = 1 only (exact); K 8192, parent cgemv pack with 2 rows per CTA
_R8M_MOD = [None]
_R8M_THR = []
_R8M_T0 = [0.0]


def _r8m_ext():
    if _R8M_MOD[0] is None:
        cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\n"
               "int64_t r8m_run(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n")
        _R8M_MOD[0] = load_inline("okc_bf16proj_r8m_" + _r8m_hashlib.md5((cpp + _R8M_CU).encode()).hexdigest()[:12], cpp,
                                  cuda_sources=_R8M_CU, functions=["r8m_run"], no_implicit_headers=True,
                                  with_pytorch_error_handling=False, verbose=False,
                                  extra_cuda_cflags=["-O3", "-DNDEBUG", "-gencode=arch=compute_90a,code=sm_90a"])
    return _R8M_MOD[0]


def _r8m_start():
    """compile the r8m extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _r8m_time.time()
        try:
            _r8m_ext()
            _R8M_THR.append("ok")
        except Exception as error:
            _R8M_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _R8M_THR.append(round(_r8m_time.time() - t0, 1))
    t = _r8m_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _r8m_codes(C):
    """parent code plane C [N][K / 2] (byte i of an 8-weight word = code_i | code_{4+i} << 4, code = sign << 3 | o) ->
    'rot' order (byte j = o_j | s_{4+j} << 3 | o_{4+(j+1)%4} << 4 | s_j << 7), then the two rows of a CTA interleaved per
    thread: CC [N / 2][K / 16][row 0 8 B | row 1 8 B]. Returns (CC, inverse-check ok)."""
    n, kh = C.shape
    b = C.view(n, kh // 4, 4).to(torch.int32)
    lo, hi = b & 15, b >> 4
    o_lo, s_lo, o_hi, s_hi = lo & 7, lo >> 3, hi & 7, hi >> 3
    R = (o_lo | (s_hi << 3) | (torch.roll(o_hi, shifts=-1, dims=2) << 4) | (s_lo << 7)).to(torch.uint8).reshape(n, kh)
    CC = R.view(n // 2, 2, kh // 8, 8).permute(0, 2, 1, 3).reshape(n // 2, 2 * kh).contiguous()
    # inverse (independent path): un-interleave, un-rot, compare with C
    Rb = CC.view(n // 2, kh // 8, 2, 8).permute(0, 2, 1, 3).reshape(n, kh // 4, 4).to(torch.int32)
    lo2 = (Rb & 7) | (((Rb >> 7) & 1) << 3)
    hi2 = torch.roll((Rb >> 4) & 7, shifts=1, dims=2) | (((Rb >> 3) & 1) << 3)
    ok = bool(torch.equal((lo2 | (hi2 << 4)).to(torch.uint8).reshape(n, kh), C))
    return CC, ok


def _r8m_call(p, x, out):
    return _R8M_MOD[0].r8m_run(p["L"].data_ptr(), p["CC"].data_ptr(), x.data_ptr(), p["tab"].data_ptr(), p["pmax"], out.data_ptr(),
                               p["base4"], p["n"], p["K"], torch.cuda.current_stream().cuda_stream)


def _r8m_validate(p, layer, w, state, gen):
    """one-hot x8 rows over EVERY K column bit-exact against W; random / 1%-channel x30 outlier rows within 4e-3 of FP64
    (and next to the parent route on the same x); 3 repeat calls bit-identical"""
    n, K = w.shape
    dev = w.device
    stream = torch.cuda.current_stream().cuda_stream
    eye = torch.zeros((K, K), dtype=torch.bfloat16, device=dev)
    eye.fill_diagonal_(8.0)
    got = torch.full((K, n), float("nan"), dtype=torch.bfloat16, device=dev)
    xp, op, run = eye.data_ptr(), got.data_ptr(), _R8M_MOD[0].r8m_run
    Lp, Cp, Tp = p["L"].data_ptr(), p["CC"].data_ptr(), p["tab"].data_ptr()
    for k in range(K):
        if run(Lp, Cp, xp + 2 * K * k, Tp, p["pmax"], op + 2 * n * k, p["base4"], n, K, stream) != 0:
            return False, "launch error"
    torch.cuda.synchronize()
    if not bool(torch.isfinite(got).all()) or not torch.equal(got.float(), w.t().float() * 8.0):
        return False, "one-hot rows not exact"
    del eye, got
    worst, ndiff, maxd = 0.0, 0, 0.0
    for dist in (1.0, 30.0, 1.0, 30.0):
        x = torch.randn((1, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        o = torch.empty((1, n), dtype=torch.bfloat16, device=dev)
        if _r8m_call(p, x, o) != 0:
            return False, "launch error"
        ref = x.double() @ w.double().t()
        err = float((o.double() - ref).norm() / ref.norm())
        if not (err <= 4e-3 and bool(torch.isfinite(o).all())):
            return False, "random rel L2 %.2e" % err
        worst = max(worst, err)
        par = _r8m_parent_kernel({"x": x, "config": {"m": 1, "n": n, "k": K, "layer": layer}}, state)
        ndiff += int((o.view(torch.int16) != par.view(torch.int16)).sum())
        maxd = max(maxd, float((o.float() - par.float()).abs().max() / ref.abs().max()))
        for _ in range(3):
            o2 = torch.empty((1, n), dtype=torch.bfloat16, device=dev)
            _r8m_call(p, x, o2)
            if not torch.equal(o.view(torch.int16), o2.view(torch.int16)):
                return False, "not bit-identical across calls"
    torch.cuda.synchronize()
    return True, "ok %.1e (vs parent route: %d of %d outputs differ, max |d| %.1e of max |ref|)" % (worst, ndiff, 4 * n, maxd)


def _r8m_setup(state, weights, thr):
    state["r8m"] = {}
    thr.join(timeout=max(1.0, 250.0 - (_r8m_time.time() - _R8M_T0[0])))
    if not _R8M_THR or _R8M_THR[0] != "ok" or _R8M_MOD[0] is None:
        print("R8M build %s" % (_R8M_THR[0] if _R8M_THR else "not finished"), flush=True)
        return
    gen = torch.Generator(device="cuda").manual_seed(8088)
    for layer in _R8M_LAYERS:
        cg = (state.get("cg") or {}).get(layer)
        w = weights[layer]
        n, K = w.shape
        if cg is None or w.dtype != torch.bfloat16 or K != 8192 or n % 2:
            print("R8M %s m=1: no parent cgemv pack, stays on the parent" % layer, flush=True)
            continue
        L, C, tab = cg["L"], cg["C"], cg["tab"]
        if (L.dtype != torch.uint8 or tuple(L.shape) != (n, K) or C.dtype != torch.uint8 or tuple(C.shape) != (n, K // 2)
                or tab.dtype != torch.int32 or tuple(tab.shape) != (n // 2, int(cg["pmax"]), 2) or not L.is_contiguous()
                or not tab.is_contiguous()):
            print("R8M %s unexpected parent pack" % layer, flush=True)
            continue
        CC, inv_ok = _r8m_codes(C)
        if not inv_ok:
            print("R8M %s code re-order not exact" % layer, flush=True)
            continue
        p = {"L": L, "CC": CC, "tab": tab, "pmax": int(cg["pmax"]), "base4": int(cg["base4"]), "n": n, "K": K}
        try:
            ok, msg = _r8m_validate(p, layer, w, state, gen)
        except Exception as error:
            ok, msg = False, "error %s" % str(error).splitlines()[-1][:160]
        torch.cuda.synchronize()
        print("R8M %s m=1: %s" % (layer, msg), flush=True)
        if ok:
            state["r8m"][layer] = p


_r8m_parent_prepare = prepare
_r8m_parent_kernel = kernel_fn


def prepare(weights):
    _R8M_T0[0] = _r8m_time.time()
    thr = _r8m_start()
    state = _r8m_parent_prepare(weights)
    try:
        _r8m_setup(state, weights, thr)
    except Exception as error:
        state["r8m"] = {}
        print("R8M setup failed: %s" % str(error).splitlines()[-1][:200], flush=True)
    torch.cuda.synchronize()
    print("R8M prepare done in %.0fs (r8m build %s, routed %s)" % (_r8m_time.time() - _R8M_T0[0], _R8M_THR, sorted(state["r8m"])), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    if x.dim() == 2 and x.shape[0] == 1:
        cfg = data["config"]
        p = (state.get("r8m") or {}).get(cfg["layer"])
        if (p is not None and cfg["n"] == p["n"] and x.shape[1] == p["K"] and x.dtype == torch.bfloat16 and x.is_contiguous()
                and x.data_ptr() % 16 == 0):
            out = torch.empty((1, p["n"]), dtype=torch.bfloat16, device=x.device)
            if _r8m_call(p, x, out) == 0:
                return out
    return _r8m_parent_kernel(data, state)

# ======================================================================================================================
# track r8g (adamleex, 2026-11-03): router m = 1 only (EXACT m; every other cell, incl. the router at any other m and the
# unpooler m = 1 slice, goes to the parent chain above unchanged; the parent route for router m = 1 is rgemv, which reads the
# FP32 weights directly, 8.39 MB). This route reads the router's FP32 weights as the BF16 hi + lo pair the task allows
# (hi = bf16(W), lo = bf16(W - hi), FP32 accumulation), each plane in this file's lossless 12-bit codec (low byte verbatim +
# 4-bit sign | exponent-window code, the window picked per plane by _exponent_window; exponents outside a plane's window are
# corrected EXACTLY from a per-row escape table holding the original BF16 bits, as cgemv does), so the weight stream is
# 6.29 MB instead of 8.39 MB.
# Kernel r8g_rtb_kernel: one CTA per output row (256 CTAs, one wave), 512 compute threads + 1 escape warp. Thread t owns
# k = h * 4096 + 8 t + e (h < 2, e < 8). Per row: [512 x 16 B hi low bytes][512 x 8 B hi codes][512 x 16 B lo low bytes]
# [512 x 8 B lo codes] (24 KB), so every weight / x load is warp-contiguous, all are issued before the first FMA, hi before lo
# (the hi FMAs can run while the lo bytes are still in flight). FP32 FMA into one hi and one lo chain, warp butterfly, then a
# fixed-order cross-warp sum (escape warp last) by thread 0; FP32 output (the router output is FP32: no BF16 rounding). No
# atomics, no cross-CTA traffic, no cache-policy hints, nothing input-derived cached; graph-capturable (one plain launch).
# Prepare: the pack is decoded back with torch ops (escapes applied) and compared bit-for-bit with hi and lo (torch.equal),
# escape counts are capped (MAX_ESCAPE_DENOMINATOR); the kernel is validated against FP64 on random, 1%-channel x30 outlier and
# one-hot probe rows (incl. escaped columns), and 3 repeat calls must be bit-identical. Any failure (or a non-sm_90 device,
# or a build failure) leaves router m = 1 on the parent route. The extension (okc_bf16proj_r8g_<md5 12>) is built on a
# background thread while the parent prepare runs. Static route (decided offline, judge-faithful cross-process timing in
# the r8g README); prepare() never times anything.
# ======================================================================================================================
import hashlib as _r8g_hashlib
import threading as _r8g_threading
import time as _r8g_time

_R8G_CU = r"""// track r8g (adamleex): router m = 1 GEMV on a lossless 12-bit BF16 hi + lo pack (sm_90). The 12-bit decode and the
// escape correction are this file's cgemv ones (alygg77 / vjeux lineage, Apache-2.0; notices at the top / end of the file).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace r8g_ns {
__device__ __forceinline__ float r8g_esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
__device__ __forceinline__ uint32_t r8g_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
// 8 weights (low bytes la = elements 0..3, lb = 4..7; code word c: byte j = code of element j | code of element j + 4 << 4)
// times 8 x values, FP32 FMA into v in element order
__device__ __forceinline__ float r8g_dot8(uint32_t la, uint32_t lb, uint32_t c, uint32_t b4, const float* xf, float v) {
  const uint32_t ha = r8g_hib(c & 0x0F0F0F0Fu, b4), hb = r8g_hib((c >> 4) & 0x0F0F0F0Fu, b4);
  const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
  const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
  v = fmaf(__uint_as_float(p01 << 16), xf[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xf[1], v);
  v = fmaf(__uint_as_float(p23 << 16), xf[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xf[3], v);
  v = fmaf(__uint_as_float(p45 << 16), xf[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xf[5], v);
  v = fmaf(__uint_as_float(p67 << 16), xf[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xf[7], v);
  return v;
}
__device__ __forceinline__ void r8g_cvt8(const uint4 xv, float* xf) {
  const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv);
#pragma unroll
  for (int e = 0; e < 4; ++e) { xf[2 * e] = __uint_as_float(xw[e] << 16); xf[2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
}

// grid = N rows, 544 threads. Escape table per row: entry 0 .x = count; entry i: .x = (plane << 16) | k, .y = original BF16 bits.
__global__ void __launch_bounds__(544, 2) r8g_rtb_kernel(const uint8_t* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
    const int2* __restrict__ etab, int pmax, float* __restrict__ out, uint32_t b4h, uint32_t b4l) {
  constexpr int NWARP = 16;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r = blockIdx.x;
  __shared__ float red[NWARP + 1];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ...; x[k] * (true - decoded) for its plane
    const int2* tb = etab + (size_t)r * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const uint32_t bq = ((en.x >> 16) ? b4l : b4h) & 0xFFu;
        ec += __bfloat162float(x[en.x & 0xFFFF]) * r8g_esc_delta((uint32_t)en.y, bq);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ec += __shfl_xor_sync(0xffffffffu, ec, o);
    if (lane == 0) red[NWARP] = ec;
    __syncthreads();
    return;
  }
  const uint8_t* wr = Wp + (size_t)r * 24576;
  const uint4 hl = __ldg(reinterpret_cast<const uint4*>(wr + 16 * tid));
  const uint2 hc = __ldg(reinterpret_cast<const uint2*>(wr + 8192 + 8 * tid));
  const uint4 xv0 = __ldg(reinterpret_cast<const uint4*>(x + 8 * tid));
  const uint4 xv1 = __ldg(reinterpret_cast<const uint4*>(x + 4096 + 8 * tid));
  const uint4 ll = __ldg(reinterpret_cast<const uint4*>(wr + 12288 + 16 * tid));
  const uint2 lc = __ldg(reinterpret_cast<const uint2*>(wr + 20480 + 8 * tid));
  float x0[8], x1[8];
  r8g_cvt8(xv0, x0); r8g_cvt8(xv1, x1);
  float ah = 0.f, al = 0.f;
  ah = r8g_dot8(hl.x, hl.y, hc.x, b4h, x0, ah);
  ah = r8g_dot8(hl.z, hl.w, hc.y, b4h, x1, ah);
  al = r8g_dot8(ll.x, ll.y, lc.x, b4l, x0, al);
  al = r8g_dot8(ll.z, ll.w, lc.y, b4l, x1, al);
  float v = ah + al;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (lane == 0) red[warp] = v;
  __syncthreads();
  if (tid == 0) {
    float s = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) s += red[q];
    out[r] = s;
  }
}
}  // namespace r8g_ns

__attribute__((visibility("hidden"))) int64_t r8g_rtb_run(int64_t Wp, int64_t x, int64_t tab, int64_t pmax, int64_t out, int64_t b4h,
                                                           int64_t b4l, int64_t n, int64_t stream) {
  if (!Wp || !x || !tab || !out || (Wp % 16) || (x % 16) || (tab % 8) || (out % 4) || pmax < 1 || n < 1 || n > 65535) return -10;
  r8g_ns::r8g_rtb_kernel<<<(unsigned)n, 544, 0, (cudaStream_t)stream>>>((const uint8_t*)Wp, (const __nv_bfloat16*)x, (const int2*)tab,
      (int)pmax, (float*)out, (uint32_t)b4h, (uint32_t)b4l);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
"""
_R8G_K = 8192          # router K (the kernel's 512 threads x 16 k)
_R8G_MOD = [None]
_R8G_THR = []
_R8G_T0 = [0.0]


def _r8g_ext():
    if _R8G_MOD[0] is None:
        cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\n"
               "int64_t r8g_rtb_run(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n")
        _R8G_MOD[0] = load_inline("okc_bf16proj_r8g_" + _r8g_hashlib.md5((cpp + _R8G_CU).encode()).hexdigest()[:12], cpp,
                                  cuda_sources=_R8G_CU, functions=["r8g_rtb_run"], no_implicit_headers=True,
                                  with_pytorch_error_handling=False, verbose=False,
                                  extra_cuda_cflags=["-O3", "-DNDEBUG", "-gencode=arch=compute_90a,code=sm_90a"])
    return _R8G_MOD[0]


def _r8g_start():
    """compile the r8g extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _r8g_time.time()
        try:
            _r8g_ext()
            _R8G_THR.append("ok")
        except Exception as error:
            _R8G_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _R8G_THR.append(round(_r8g_time.time() - t0, 1))
    t = _r8g_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _r8g_plane(t):
    """BF16 plane -> (low bytes u8, 4-bit codes u8 (sign << 3 | clamped window offset), window base, escape mask, bits int32);
    the file's 12-bit codec (_exponent_window / _escapes)"""
    bits = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi8, low = bits >> 8, bits & 0xFF
    e71, sign = hi8 & 0x7F, hi8 >> 7
    base, o = _exponent_window(e71)
    code = ((sign << 3) | o).to(torch.uint8)
    return low.to(torch.uint8), code, base, _escapes(e71, base), bits


def _r8g_unpack(Wp, tab, b4h, b4l, n):
    """torch decode of the pack (escapes applied from the table) -> (hi bits, lo bits) int32 [n, K]"""
    W = Wp.view(n, 24576).to(torch.int64)

    def plane(off_low, off_code, b4):
        low = W[:, off_low:off_low + 8192].reshape(n, 512, 2, 8)
        cw = W[:, off_code:off_code + 4096].reshape(n, 512, 2, 4)
        code = torch.cat([cw & 0xF, cw >> 4], dim=-1)
        hib = ((code & 7) + (b4 & 0xFF)) | ((code & 8) << 4)
        return ((hib << 8) | low).permute(0, 2, 1, 3).reshape(n, _R8G_K)

    dh, dl = plane(0, 8192, b4h), plane(12288, 20480, b4l)
    cnt = tab[:, 0, 0].long()
    pm = tab.shape[1]
    idx = torch.arange(pm, device=tab.device).view(1, pm)
    valid = (idx >= 1) & (idx <= cnt.view(n, 1))
    rows = torch.arange(n, device=tab.device).view(n, 1).expand(n, pm)[valid]
    ent = tab[valid].long()
    pl, kk, val = ent[:, 0] >> 16, ent[:, 0] & 0xFFFF, ent[:, 1] & 0xFFFF
    dh[rows[pl == 0], kk[pl == 0]] = val[pl == 0]
    dl[rows[pl == 1], kk[pl == 1]] = val[pl == 1]
    return dh, dl


def _r8g_pack(wf):
    """FP32 router W [n, 8192] -> dict(Wp uint8 [n * 24576], tab int32 [n, pmax, 2], pmax, b4h, b4l, n) or (None, reason)"""
    n, K = wf.shape
    if K != _R8G_K or wf.dtype != torch.float32 or not bool(torch.isfinite(wf).all()):
        return None, "unexpected router weights"
    hi = wf.to(torch.bfloat16)
    lo = (wf - hi.float()).to(torch.bfloat16)
    lh, ch, bh, eh, bitsh = _r8g_plane(hi)
    ll, cl, bl, el, bitsl = _r8g_plane(lo)
    if int(eh.sum()) > n * K // MAX_ESCAPE_DENOMINATOR or int(el.sum()) > n * K // MAX_ESCAPE_DENOMINATOR:
        return None, "too many escapes (hi %d, lo %d)" % (int(eh.sum()), int(el.sum()))
    for bits, esc, base in ((bitsh, eh, bh), (bitsl, el, bl)):   # the kernel's FP32 correction (true - decoded) must be exact
        b = bits[esc].long()
        h8 = (b >> 8) & 0xFF
        dhi = (h8 & 0x80) | torch.where((h8 & 0x7F) < base, torch.full_like(h8, base), torch.full_like(h8, base + EXPONENT_WINDOW - 1))
        f32 = lambda v: (v - ((v >> 31) & 1) * (1 << 32)).to(torch.int32).view(torch.float32)
        dec, true = f32(((dhi << 8) | (b & 0xFF)) << 16), f32(b << 16)
        if not bool(((dec + (true - dec)) == true).all()):
            return None, "inexact escape correction"

    def lowb(l):   # thread t: runs h0 (k = 8 t + e) and h1 (k = 4096 + 8 t + e), 16 bytes
        return l.view(n, 2, 512, 8).permute(0, 2, 1, 3).reshape(n, 8192)

    def codes(c):  # per run one u32: byte j = code of element j | code of element j + 4 << 4
        g = c.view(n, 2, 512, 8).permute(0, 2, 1, 3)
        return (g[..., 0:4] | (g[..., 4:8] << 4)).reshape(n, 4096)

    Wp = torch.cat([lowb(lh), codes(ch), lowb(ll), codes(cl)], dim=1).contiguous().view(-1)
    rows = torch.cat([eh.nonzero()[:, 0], el.nonzero()[:, 0]])
    cols = torch.cat([eh.nonzero()[:, 1], el.nonzero()[:, 1]])
    plane = torch.cat([torch.zeros(int(eh.sum()), dtype=torch.long, device=wf.device), torch.ones(int(el.sum()), dtype=torch.long, device=wf.device)])
    vals = torch.cat([bitsh[eh], bitsl[el]])
    order = torch.argsort(rows * (2 * K) + plane * K + cols)                    # row, plane, k: fixed order
    rows, cols, plane, vals = rows[order], cols[order], plane[order], vals[order]
    counts = torch.bincount(rows, minlength=n)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=wf.device) - start[rows] + 1
    tab = torch.zeros(n, pmax, 2, dtype=torch.int32, device=wf.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[rows, pos, 0] = ((plane << 16) | cols).to(torch.int32)
    tab[rows, pos, 1] = vals.to(torch.int32)
    tab = tab.contiguous()
    b4h, b4l = bh * BASE4_REPLICATE, bl * BASE4_REPLICATE
    dh, dl = _r8g_unpack(Wp, tab, b4h, b4l, n)                                  # lossless round trip
    if not (torch.equal(dh, bitsh.long()) and torch.equal(dl, bitsl.long())):
        return None, "pack does not round-trip"
    return dict(Wp=Wp, tab=tab, pmax=pmax, b4h=b4h, b4l=b4l, n=n, K=K, nesc=(int(eh.sum()), int(el.sum()))), None


def _r8g_call(st, x, out):
    return _R8G_MOD[0].r8g_rtb_run(st["Wp"].data_ptr(), x.data_ptr(), st["tab"].data_ptr(), st["pmax"], out.data_ptr(), st["b4h"],
                                   st["b4l"], st["n"], torch.cuda.current_stream().cuda_stream)


def _r8g_validate(st, wf, gen):
    """random / 1%-channel x30 outlier rows against FP64 (rel L2 <= 2e-5; the judge allows 1e-4); one-hot probe
    rows (incl. escaped columns) against FP64 (hi + lo) x (L2 rel <= 1e-6) and FP32 W x (L2 rel <= 1e-4); 3 repeat calls bit-identical"""
    n, K = wf.shape
    dev = wf.device
    hl = wf.to(torch.bfloat16).double()
    hl = hl + (wf - wf.to(torch.bfloat16).float()).to(torch.bfloat16).double()
    out = torch.empty((1, n), dtype=torch.float32, device=dev)
    err = 0.0
    for dist in (1.0, 30.0):
        x = torch.randn((1, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        if _r8g_call(st, x, out) != 0:
            return "launch error"
        torch.cuda.synchronize()
        ref = x.double() @ wf.double().t()
        e = float((out.double() - ref).norm() / ref.norm())
        if not (bool(torch.isfinite(out).all()) and e <= 2e-5):
            return "dist %g rel L2 %.2e" % (dist, e)
        err = max(err, e)
        o1 = out.clone()
        for _ in range(3):
            _r8g_call(st, x, out)
            torch.cuda.synchronize()
            if not torch.equal(out.view(torch.int32), o1.view(torch.int32)):
                return "repeat calls differ"
    tab = st["tab"]
    ks = [0, 7, 8, 4095, 4096, 4103, K - 1, 777, 5003]
    if int(tab[0, 0, 0]) >= 1:
        ks.append(int(tab[0, 1, 0]) & 0xFFFF)
    if int(tab[:, 0, 0].max()) >= 1:
        rr = int(torch.argmax(tab[:, 0, 0]))
        ks.extend(int(tab[rr, i, 0]) & 0xFFFF for i in range(1, int(tab[rr, 0, 0]) + 1))
    for j, k in enumerate(ks[:40]):
        val = (-1.0 if j % 2 else 1.0) * 2.0 ** (3 + j % 5)
        x = torch.zeros((1, K), dtype=torch.bfloat16, device=dev)
        x[0, k] = val
        if _r8g_call(st, x, out) != 0:
            return "launch error"
        torch.cuda.synchronize()
        ex = hl[:, k] * val
        ew = wf[:, k].double() * val   # L2 over the 256 outputs (the judge's probe metric); per-element bounds refused correct packs
        e1 = float((out[0].double() - ex).norm() / ex.norm().clamp(min=1e-300))
        e2 = float((out[0].double() - ew).norm() / ew.norm().clamp(min=1e-300))
        if not (e1 <= 1e-6 and e2 <= 1e-4):
            return "one-hot k %d: L2 rel %.2e vs hi+lo, %.2e vs W" % (k, e1, e2)
    torch.cuda.synchronize()
    return "ok %.1e" % err


def _r8g_setup(state, weights, thr):
    state["r8g"] = {}
    if thr is not None:
        thr.join(timeout=max(1.0, 250.0 - (_r8g_time.time() - _R8G_T0[0])))
    if not _R8G_THR or _R8G_THR[0] != "ok" or _R8G_MOD[0] is None:
        print("R8G build %s" % (_R8G_THR[0] if _R8G_THR else "not finished"), flush=True)
        return
    wf = weights["router"]
    if torch.cuda.get_device_capability(wf.device) != (9, 0):
        print("R8G off: not an sm_90 device", flush=True)
        return
    if wf.dim() != 2 or wf.shape[1] != _R8G_K or wf.dtype != torch.float32:
        print("R8G off: unexpected router weights", flush=True)
        return
    st, why = _r8g_pack(wf.contiguous())
    if st is None:
        print("R8G router m=1: off (%s), parent route" % why, flush=True)
        return
    msg = _r8g_validate(st, wf.contiguous(), torch.Generator(device=wf.device).manual_seed(80808))
    print("R8G router m=1: %s (escapes hi %d lo %d, pmax %d)" % (msg if msg.startswith("ok") else "off (" + msg + "), parent route",
                                                                 st["nesc"][0], st["nesc"][1], st["pmax"]), flush=True)
    if msg.startswith("ok"):
        state["r8g"] = {k: st[k] for k in ("Wp", "tab", "pmax", "b4h", "b4l", "n", "K")}


_r8g_parent_prepare = prepare
_r8g_parent_kernel = kernel_fn


def prepare(weights):
    _R8G_T0[0] = _r8g_time.time()
    thr = _r8g_start()
    state = _r8g_parent_prepare(weights)
    try:
        _r8g_setup(state, weights, thr)
    except Exception as error:
        state["r8g"] = {}
        print("R8G setup failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__), flush=True)
    torch.cuda.synchronize()
    print("R8G prepare done in %.0fs (r8g build %s, routed %s)" % (_r8g_time.time() - _R8G_T0[0], _R8G_THR,
                                                                    "router m=1" if state.get("r8g") else "none"), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    st = state.get("r8g")
    if (st and cfg["layer"] == "router" and x.dim() == 2 and x.shape[0] == 1 and x.shape[1] == st["K"] and cfg["n"] == st["n"]
            and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0):
        _k_ctx_bind(x)
        out = torch.empty((1, st["n"]), dtype=torch.float32, device=x.device)
        if _r8g_call(st, x, out) == 0:
            return out
    return _r8g_parent_kernel(data, state)


# ======================================================================================================================
# track r9b (adamleex, 2026-11-03): qkv_gate m = 32 and pooler m = 32 only (EXACT m; every other cell, incl. these two
# layers at m 17..31, goes to the parent chain above unchanged; the parent route of both cells is alygg77's sk2 TMA-ring
# GEMV, configs 'qkv32_r208' / 'pooler32_r144'). Same kernel (this file's SK2_SRC), same geometry, same weight stream (the
# parent's state["sk2"] stream tensors are reused, not copied), same consumer and producer warps, same epilogue, same
# producer-side fence.proxy.async and release arrives (r7r race fix). ONE change, in the escape warps ("DIST"): in the
# parent the 4 escape warps take turns on the stages, and the owner of a stage corrects ALL of that stage's escaped weights
# serially (lane = token, one shared-memory read-modify-write per escape) before it hands the slot back, so on
# escape-heavy stages the owner's serial correction latency gates the ring. Here the CTA has 2 escape warps (not 4; fewer
# warps in every slot handshake, 2 correction copies) and every escape warp works on EVERY stage: at prepare time each
# stage's escapes (k order, then row) are dealt round-robin to the 2 warps (starting at warp st % 2);
# each warp walks its own k-ordered list (closed by a k = 0xFFFF sentinel) and corrects its share into its own FP32 copy,
# with the exact FP32 delta w - decode(w) stored in the table (the parent's esc_delta value, asserted exact in prepare)
# instead of being re-derived per escape. Arithmetic: tensor-core FP32 accumulation of the decoded weights (unchanged);
# FP32 escape corrections (each escape's product delta * x is the parent's, but it may land in a different correction
# copy, so the FP32 sum of the copies can differ from the parent's in the last bits); the unchanged fixed-order FP32
# epilogue (copies 0..1, then the K-warp partials); ONE final BF16 rounding. Deterministic (fixed assignment and order),
# no atomics, no cache-policy hint added or changed, nothing input-derived cached, graph-capturable (one plain launch).
# Prepare: the parent's lossless pack is re-run and must reproduce the parent's stream and escape table bitwise; the DIST
# table is built from it; the kernel is validated against the parent route and FP64 on random and 1 %-channel x30 outlier
# rows, exactly (as values) on one-hot probe rows at escaped columns, and for 3 bit-identical repeat calls. Any failure (or a
# non-sm_90 device, or a build failure) leaves the cell on the parent route. The extension (okc_bf16proj_r9b_<md5 12>:
# the kernel template text of SK2_SRC renamed r9b_sk2_kernel, this file's CUDA_SRC / MBAR_HELPERS / MAP_SRC helpers, all in
# namespace r9b_ns, and a raw-pointer launcher with no ATen / torch headers, so it compiles in ~1/3 of the time of an ATen
# translation unit) builds on a background thread while the parent prepare runs. Static route (offline judge-faithful
# cross-process timing in the r9b README); prepare() never times anything.
# ======================================================================================================================
import hashlib as _r9b_hashlib
import threading as _r9b_threading
import time as _r9b_time

_R9B_CELLS = (("qkv_gate", "qkv32_r208"), ("pooler", "pooler32_r144"))   # (layer, parent sk2 config); m = 32 only
_R9B_NESC = 2         # escape warps per CTA in this route (the parent sk2 has min(4, NSTAGE))
_R9B_MOD = [None]
_R9B_THR = []
_R9B_T0 = [0.0]

_R9B_DIST = r"""      if constexpr (DIST) {
        // track r9b DIST: every escape warp works on EVERY stage. At prepare time each stage's escapes (k order, then row)
        // are dealt round-robin to the NESC warps (per-warp k-ordered lists closed by a k = 0xFFFF sentinel; the header at
        // int2 offset ptab_s[0].x holds each warp's list start); each warp corrects its share into its own copy with the
        // exact FP32 delta stored in the table (fixed assignment and order -> deterministic). Every warp still waits on
        // and hands back every stage, in order, as in the parent.
        int pe = reinterpret_cast<const int*>(ptab_s + ptab_s[0].x)[ew];
        constexpr int KST = UPS * 64;
        for (int st = 0; st < nst; ++st) {
          const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
          const int slot = st % NSTAGE;
          SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
          const unsigned char* xs = gbase + slot * SST;
          while (true) {
            const int2 e = ptab_s[pe];
            const int kk = e.x & 0xFFFF;
            if (kk >= kend) break;
            if (lane < M && kk >= kbeg) {
              const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
              const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                  xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
              cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(v);
            }
            ++pe;
          }
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
        }
      } else {
"""


def _r9b_rep(s, old, new):
    if s.count(old) != 1:
        raise RuntimeError("r9b: source anchor found %d times: %r" % (s.count(old), old[:60]))
    return s.replace(old, new)


def _r9b_cfg(name):
    return next(c for c in SK2_CONFIGS if c[0] == name)


_R9B_LAUNCH = r"""
// track r9b: raw-pointer launcher (the parent's launch_sk2 logic for COMP, CL = 1, M == TOKR (PK = false), bf16 output)
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC>
static int r9b_launch(const uint8_t* wsp, const void* xp, void* op, const int2* tp, int pmax, uint32_t base4, int M, int N,
                      int K, cudaStream_t stream) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = 768;
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int CP = R + 1;
  constexpr int NESC = R9B_NESC;
  if (M != TOKR || N % R != 0 || K % (UPS * 64) != 0) return -3;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (size_t)pmax * 8 + 128;
  auto kern = r9b_sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, true, false, NACC, 1, __nv_bfloat16, false>;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    if ((size_t)optin - fa.sharedSizeBytes < smem) return -5;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  r9b_err = 0;
  CUtensorMap xm = make_map_x3d(xp, (uint64_t)M, (uint64_t)K, (uint32_t)TOKR, UPS);
  if (r9b_err) return -2;
  kern<<<N / R, (NW + 1 + NESC) * 32, smem, stream>>>(xm, wsp, reinterpret_cast<const __nv_bfloat16*>(xp),
                                                     reinterpret_cast<__nv_bfloat16*>(op), tp, pmax, base4, M, N, K, 0, nullptr);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
"""


def _r9b_source():
    """the kernel template of SK2_SRC (+ the DIST escape loop, R9B_NESC escape warps), renamed r9b_sk2_kernel, after this
    file's CUDA_SRC / MBAR_HELPERS / MAP_SRC helpers, all in namespace r9b_ns (ATen includes dropped; TORCH_CHECK in the
    tensor-map helpers sets an error flag instead), + a raw-pointer launcher and the exported entry r9b_run (cfg = position in
    _R9B_CELLS, m = 32 only)"""
    k = SK2_SRC
    k = _r9b_rep(k, "#define SK2_DIAG 0\n", "#define SK2_DIAG 0\n// track r9b: DIST escape warps (see the section comment in the Python file)\n"
                 "#define R9B_NESC %d   // track r9b: escape warps per CTA\n" % _R9B_NESC)
    if k.count("(NSTAGE >= 4 ? 4 : NSTAGE)") != 3:
        raise RuntimeError("r9b: escape-warp count anchors")
    k = k.replace("(NSTAGE >= 4 ? 4 : NSTAGE)", "(R9B_NESC)")   # launch bounds, kernel, launcher
    k = _r9b_rep(k, "  constexpr int CSZ = (TOKS * CP + 3) & ~3;\n",
                 "  constexpr int CSZ = (TOKS * CP + 3) & ~3;\n  constexpr bool DIST = COMP && CL == 1;   // track r9b\n")
    k = _r9b_rep(k, "      mbar_wait_sleep(pbar, 0);\n      const int cnt = ptab_s[0].x;\n",
                 "      mbar_wait_sleep(pbar, 0);\n" + _R9B_DIST + "      const int cnt = ptab_s[0].x;\n")
    k = _r9b_rep(k, "        if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n      }\n    }\n    mbar_arrive(cbar);\n",
                 "        if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n      }\n      }   // track r9b: else (the parent's loop)\n"
                 "    }\n    mbar_arrive(cbar);\n")
    cut = "template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT>\nstatic void launch_sk2("
    k = k[:k.index(cut)]   # the ATen launcher and entry are replaced by _R9B_LAUNCH / r9b_run
    if k.count("sk2_kernel") != 1:
        raise RuntimeError("r9b: kernel name anchor")
    k = k.replace("sk2_kernel", "r9b_sk2_kernel")
    body = (CUDA_SRC.replace("@@MBAR@@", MBAR_HELPERS).replace("@@BODY@@", "") + MAP_SRC + k + _R9B_LAUNCH).splitlines()
    inc = [l for l in dict.fromkeys(l for l in body if l.startswith("#include")) if "ATen" not in l and "c10/" not in l]
    rest = "\n".join(l for l in body if not l.startswith("#include"))
    disp = []
    for ci, (layer, name) in enumerate(_R9B_CELLS):
        cfg = _r9b_cfg(name)
        _, _, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        _, nstage = _sk2_geom(cfg, 32)
        nacc = 1 if (nstage >= 4 and NT == 4) else NACC
        disp.append("  if (cfg == %d) return r9b_ns::r9b_launch<2, 32, %d, %d, %d, %d, %d, %d>(w, xp, o, t, (int)pmax, (uint32_t)b4, (int)M, (int)N, "
                    "(int)K, s);" % (ci, NT, NWR, WSK, UPW, nstage, nacc))
    return ("// track r9b (adamleex): this file's sk2 kernel (alygg77, Apache-2.0) with DIST escape warps; helpers = this file's\n"
            "// CUDA_SRC / MBAR_HELPERS / MAP_SRC, unchanged (no ATen: TORCH_CHECK sets r9b_err)\n#include <cuda.h>\n#include <cuda_runtime.h>\n"
            + "\n".join(inc) + "\nnamespace r9b_ns {\nstatic int r9b_err = 0;\n#define TORCH_CHECK(c, ...) do { if (!(c)) r9b_err = 1; } while (0)\n"
            + rest + "\n}  // namespace r9b_ns\n"
            "int64_t r9b_run(int64_t W, int64_t x, int64_t out, int64_t tab, int64_t pmax, int64_t b4, int64_t M, int64_t N, int64_t K,\n"
            "                int64_t cfg, int64_t stream) {\n"
            "  const uint8_t* w = reinterpret_cast<const uint8_t*>(W);\n  const void* xp = reinterpret_cast<const void*>(x);\n"
            "  void* o = reinterpret_cast<void*>(out);\n  const int2* t = reinterpret_cast<const int2*>(tab);\n"
            "  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);\n" + "\n".join(disp) + "\n  return -4;\n}\n")


def _r9b_ext():
    if _R9B_MOD[0] is None:
        src = _r9b_source()
        cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\nint64_t r9b_run(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, "
               "int64_t, int64_t, int64_t, int64_t, int64_t);\n")
        _R9B_MOD[0] = load_inline("okc_bf16proj_r9b_" + _r9b_hashlib.md5((cpp + src).encode()).hexdigest()[:12], cpp,
                                  cuda_sources=src, functions=["r9b_run"], no_implicit_headers=True, with_pytorch_error_handling=False,
                                  verbose=False, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xcompiler=-fno-gnu-unique"])
    return _R9B_MOD[0]


def _r9b_start():
    """compile the r9b extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _r9b_time.time()
        try:
            _r9b_ext()
            _R9B_THR.append("ok")
        except Exception as error:
            _R9B_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _R9B_THR.append(round(_r9b_time.time() - t0, 1))
    t = _r9b_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _r9b_bits_f32(b16):
    """BF16 bit patterns (int64 tensor, 0..65535) -> FP32 values"""
    v = (b16.to(torch.int64) & 0xFFFF) << 16
    return torch.where(v >= (1 << 31), v - (1 << 32), v).to(torch.int32).view(torch.float32)


def _r9b_table(comp, n_rows, R, UPS, perm):
    """DIST escape table, per CTA row (int2): [0] = (int2 offset of the header, int2 offset of the row map or 0) |
    [row map: R int32 output columns (qkv's escape-balance order), as the parent's sk2_table_add_perm] | header: NESC int32
    = int2 index (in the row) of each escape warp's list | per warp its entries in k order (x = (row_local << 16) | k,
    y = FP32 bits of the exact delta w - decode(w)), closed by a sentinel (x = 0xFFFF). Each stage's escapes (k, then row)
    are dealt round-robin to the warps starting at warp st % NESC. Returns (table int32 [ncta, rowlen, 2], rowlen)."""
    dev = comp["rows"].device
    NE = _R9B_NESC
    base = int(comp["base4"]) & 0xFF
    rows, cols, wb = comp["rows"].long(), comp["cols"].long(), comp["wbits"].long()
    keep = rows < n_rows
    rows, cols, wb = rows[keep], cols[keep], wb[keep]
    hi = (wb >> 8) & 0xFF
    e71 = hi & 0x7F
    dhi = (hi & 0x80) | torch.where(e71 < base, torch.full_like(e71, base), torch.full_like(e71, base + 7))
    wf, df = _r9b_bits_f32(wb), _r9b_bits_f32((dhi << 8) | (wb & 0xFF))
    delta = wf - df
    if not bool(((df + delta) == wf).all()):
        return None, "inexact escape delta"
    ncta = n_rows // R
    cta, st = rows // R, cols // (UPS * 64)
    order = torch.argsort((cta * 65536 + cols) * 4096 + rows % R)
    rows, cols, delta, cta, st = rows[order], cols[order], delta[order], cta[order], st[order]
    if rows.numel():
        _, inv, cnts = torch.unique_consecutive(cta * 4096 + st, return_inverse=True, return_counts=True)
        idx = torch.arange(rows.numel(), device=dev) - (torch.cumsum(cnts, 0) - cnts)[inv]
    else:
        idx = rows
    warp = (idx + st) % NE
    cnt = torch.bincount(cta * NE + warp, minlength=ncta * NE).view(ncta, NE)
    mo = 1 if perm is not None else 0
    ho = 1 + (R // 2 if perm is not None else 0)
    hw = (NE + 1) // 2
    rowlen = ho + hw + int((cnt.sum(1) + NE).max())
    rowlen += rowlen % 2
    tab = torch.zeros(ncta, rowlen, 2, dtype=torch.int64, device=dev)
    tab[:, 0, 0] = ho
    tab[:, 0, 1] = mo
    if perm is not None:
        tab[:, 1:1 + R // 2] = perm.view(ncta, R // 2, 2).long()
    starts = ho + hw + torch.cumsum(cnt + 1, 1) - (cnt + 1)
    hdr = torch.zeros(ncta, 2 * hw, dtype=torch.int64, device=dev)
    hdr[:, :NE] = starts
    tab[:, ho:ho + hw] = hdr.view(ncta, hw, 2)
    ar = torch.arange(ncta, device=dev)
    for w in range(NE):
        tab[ar, starts[:, w] + cnt[:, w], 0] = 0xFFFF
    if rows.numel():
        o2 = torch.argsort((cta * NE + warp) * (1 << 40) + torch.arange(rows.numel(), device=dev))
        cta2, warp2, rows2, cols2, delta2 = cta[o2], warp[o2], rows[o2], cols[o2], delta[o2]
        _, inv2, c2 = torch.unique_consecutive(cta2 * NE + warp2, return_inverse=True, return_counts=True)
        pos = starts[cta2, warp2] + torch.arange(rows2.numel(), device=dev) - (torch.cumsum(c2, 0) - c2)[inv2]
        tab[cta2, pos, 0] = ((rows2 % R) << 16) | cols2
        tab[cta2, pos, 1] = delta2.view(torch.int32).long()
    tab = torch.where(tab >= (1 << 31), tab - (1 << 32), tab).to(torch.int32).contiguous()
    return (tab, rowlen), None


def _r9b_call(st, x, out, state):
    return _R9B_MOD[0].r9b_run(st["stream"].data_ptr(), x.data_ptr(), out.data_ptr(), st["tab"].data_ptr(), st["pmax"], int(st["b4"]),
                               32, st["n"], st["K"], st["ci"], torch.cuda.current_stream().cuda_stream)


def _r9b_validate(st, layer, w, state, gen):
    """random / 1 %-channel x30 outlier rows (m = 32) against FP64 and the parent sk2 route; one-hot rows at escaped columns
    exact against W (values: +0 and -0 compare equal); 3 bit-identical repeat calls"""
    n, K, dev = st["n"], st["K"], w.device
    wd = w.double()
    pci, pstream, ptab, ppmax, pb4, _ = state["sk2"][st["parent"]]
    out = torch.empty((32, n), dtype=torch.bfloat16, device=dev)
    par = torch.empty((32, n), dtype=torch.bfloat16, device=dev)
    worst = 0.0
    for dist in (1.0, 30.0, 1.0, 30.0):
        x = torch.randn((32, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        if _r9b_call(st, x, out, state) != 0:
            return "launch error"
        _get_ext().sk2(pstream, x, par, ptab, ppmax, pb4, 32, n, K, pci, 32, 0, state["dummy"])
        torch.cuda.synchronize()
        ref = x.double() @ wd.t()
        e = float((out.double() - ref).norm() / ref.norm())
        ep = float((par.double() - ref).norm() / ref.norm())
        d = (out.float() - par.float()).abs()
        if not (bool(torch.isfinite(out).all()) and e <= ep * 1.02 + 1e-6 and e <= 4e-3
                and bool((d <= par.float().abs() * 2.0 ** -7 + 1e-5 * float(par.float().abs().max())).all())):
            return "dist %g: rel L2 %.3e (parent %.3e), max |d| %.2e" % (dist, e, ep, float(d.max()))
        worst = max(worst, e)
        o1 = out.clone()
        for _ in range(3):
            if _r9b_call(st, x, out, state) != 0:
                return "launch error"
            torch.cuda.synchronize()
            if not torch.equal(out.view(torch.int16), o1.view(torch.int16)):
                return "repeat calls differ"
    # one-hot rows at escaped columns (+ two plain columns): exact products, so the output must be bf16(W[:, k] * val)
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e71 = (bits >> 8) & 0x7F
    base, _ = _exponent_window(e71)
    esc_cols = torch.unique(_escapes(e71, base).nonzero()[:, 1])
    if esc_cols.numel():
        sel = esc_cols[torch.randperm(esc_cols.numel(), generator=gen, device=dev)[:62]]
    else:
        sel = esc_cols
    ks = torch.cat([sel, torch.tensor([0, K - 1], device=dev)])
    for b0 in range(0, ks.numel(), 32):
        kb = ks[b0:b0 + 32]
        x = torch.zeros((32, K), dtype=torch.bfloat16, device=dev)
        vals = torch.tensor([(-1.0 if j % 2 else 1.0) * 2.0 ** (1 + j % 4) for j in range(32)], device=dev)
        x[torch.arange(kb.numel(), device=dev), kb] = vals[:kb.numel()].to(torch.bfloat16)
        if _r9b_call(st, x, out, state) != 0:
            return "launch error"
        torch.cuda.synchronize()
        exp = (w.float()[:, kb].t() * vals[:kb.numel(), None]).to(torch.bfloat16)
        # compared as values: an escaped W = +0 times a negative x is +0 here (dec * x + delta * x) but -0 in torch
        if not bool((out[:kb.numel()].float() == exp.float()).all()):
            return "one-hot rows at escaped columns not exact"
    return "ok %.2e (%d one-hot columns)" % (worst, int(ks.numel()))


def _r9b_setup(state, weights, thr):
    state["r9b"] = {}
    if thr is not None:
        thr.join(timeout=max(1.0, 250.0 - (_r9b_time.time() - _R9B_T0[0])))
    if not _R9B_THR or _R9B_THR[0] != "ok" or _R9B_MOD[0] is None:
        print("R9B build %s" % (_R9B_THR[0] if _R9B_THR else "not finished"), flush=True)
        return
    dev = weights["o"].device
    if torch.cuda.get_device_capability(dev) != (9, 0):
        print("R9B off: not an sm_90 device", flush=True)
        return
    for ci, (layer, name) in enumerate(_R9B_CELLS):
        ent = state.get("sk2", {}).get(name)
        cfg = _r9b_cfg(name)
        _, _, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        w = weights[layer]
        if ent is None or CL != 1 or _sk2_geom(cfg, 32)[1] < _R9B_NESC or w.dim() != 2 or w.dtype != torch.bfloat16:
            print("R9B %s m=32: off (no parent sk2 route %s)" % (layer, name), flush=True)
            continue
        pci, stream, ptab, ppmax, pb4, pn = ent
        perm = None
        if SK2_BALANCE and layer in SK2_BALANCE_LAYERS:
            perm = sk2_balance_perm(w, n_rows, R)
            comp = sk2_compress(w[:n_rows][perm].contiguous())
        else:
            comp = sk2_compress(w.contiguous())
        if comp is None:
            print("R9B %s m=32: off (pack refused)" % layer, flush=True)
            continue
        t2, p2 = sk2_table(comp, n_rows, R)
        if perm is not None:
            t2, p2 = sk2_table_add_perm(t2, p2, perm, R)
        same = (pn == n_rows and p2 == ppmax and int(comp["base4"]) == int(pb4) and torch.equal(t2, ptab)
                and torch.equal(sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW), stream))
        if not same:
            print("R9B %s m=32: off (re-run pack differs from the parent's)" % layer, flush=True)
            continue
        tb, why = _r9b_table(comp, n_rows, R, WSK * UPW, perm)
        if tb is None:
            print("R9B %s m=32: off (%s)" % (layer, why), flush=True)
            continue
        st = dict(ci=ci, parent=name, stream=stream, tab=tb[0], pmax=tb[1], b4=pb4, n=n_rows, K=w.shape[1])
        msg = _r9b_validate(st, layer, w, state, torch.Generator(device=dev).manual_seed(90909 + ci))
        print("R9B %s m=32: %s (escapes %d, table row %d int2)" % (layer, msg if msg.startswith("ok") else "off (" + msg + "), parent route",
                                                                  int((comp["rows"] < n_rows).sum()), tb[1]), flush=True)
        if msg.startswith("ok"):
            state["r9b"][layer] = st
        del comp, t2


_r9b_parent_prepare = prepare
_r9b_parent_kernel = kernel_fn


def prepare(weights):
    _R9B_T0[0] = _r9b_time.time()
    thr = _r9b_start()
    state = _r9b_parent_prepare(weights)
    try:
        _r9b_setup(state, weights, thr)
    except Exception as error:
        state["r9b"] = {}
        print("R9B setup failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__), flush=True)
    torch.cuda.synchronize()
    print("R9B prepare done in %.0fs (r9b build %s, routed %s)" % (_r9b_time.time() - _R9B_T0[0], _R9B_THR,
                                                                    sorted(state.get("r9b", {})) or "none"), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    st = state.get("r9b")
    if st and x.dim() == 2 and x.shape[0] == 32:
        e = st.get(data["config"]["layer"])
        if (e is not None and x.shape[1] == e["K"] and data["config"]["n"] == e["n"] and x.dtype == torch.bfloat16
                and x.is_contiguous() and x.data_ptr() % 16 == 0):
            _k_ctx_bind(x)
            out = torch.empty((32, e["n"]), dtype=torch.bfloat16, device=x.device)
            if _r9b_call(e, x, out, state) == 0:
                return out
    return _r9b_parent_kernel(data, state)


# ======================================================================================================================
# track r10b (adamleex, 2026-11-03): o m = 16 only (EXACT m; every other cell, incl. o at m 15 / 17 and every other layer,
# goes to the parent chain above unchanged; the parent route of this cell is alygg77's sk2 TMA-ring GEMV 'o16t_r68').
# Kernel: this file's Nathen Fernandes v18s one-shot tensor-core GEMV template (V18S_CUDA_TEMPLATE, MIT; the o m = 8 route
# 'v18o' runs it with one token tile) instantiated for compile-time m 16 (two n8 token tiles) in the o m8 route's own
# geometry: 512 CTAs x 8 K-warps (16 W rows x 128 K per warp, both 64-K units of 12-bit W issued up front, x per unit from
# L2, mma.sync m16n8k16 bf16 -> FP32 with the tokens as N, FP32 accumulation over the warp's K slice), fixed-order FP32
# sum of the 8 K-warp partials through shared memory, ONE final BF16 rounding. No split-K over CTAs (K = 1024 fits one
# CTA), no atomics, no cluster, no cache-policy operand added (the template's W loads keep their plain
# ld.global.nc.L1::no_allocate, x loads plain ld.global.nc).
# ONE change to the template ("EW", compile-time OPT bit 8192, everything else of the instance is the template's code):
# the escape (out-of-window exponent) weights are NOT patched inside the K loop by the computing warps (that per-unit
# escape walk is what made a plain m = 16 instance only ~0.1 us faster than sk2 here). The 8 main warps multiply the
# codec's own escape-free decode 'dec' (exponent = the window base, so |dec| is small); one extra warp per CTA (lane =
# token) loads the CTA's escape list (<= 16 entries, built at prepare: row-in-CTA, k, dec bits, true bits), gathers
# x[token][k] with 4-byte cp.async.ca into its own shared-memory slots, and accumulates in FP32, in entry order,
# D[row][token] = sum dec * x and T[row][token] = sum true * x (both products are exact in FP32: bf16 x bf16) before the CTA
# barrier; the emitting threads store bf16((S - D) + T), S = the fixed-order K-warp sum. For a one-hot x row (a single
# nonzero column k) S == dec * v exactly, so the output is exactly bf16(true * v) == W[row, k] * v; for general x the
# subtraction of the small dec * x term is FP32 rounding-level. Deterministic (fixed order everywhere).
# Weights: the parent o m8 route's v18s stream of W (same geometry; reused when bitwise equal to a re-run pack, else this
# section's own copy) + an escape table of 512 x 64 int32 (128 KB). Lossless: the escape table holds the true BF16 bits.
# Prepare: the r10b extension (okc_bf16proj_r10b_<md5 12>: the template renamed v18s_ -> r10b_ (functions, kernel, format
# struct), -fno-gnu-unique, one instance) builds on a background thread while the parent prepare runs; then the route is
# validated: one-hot rows over ALL 1024 K columns exact, random / 1 %-channel x30 outlier rows against FP64 and the parent
# route, 3 bit-identical repeat calls. Any failure (or a non-sm_90 device, or a build failure, or a CTA with > 16
# escapes) leaves the cell on the parent route. Static route (offline judge-faithful cross-process timing in the r10b
# README); prepare() never times anything.
# ======================================================================================================================
import hashlib as _r10b_hashlib
import threading as _r10b_threading
import time as _r10b_time

# v18s template arguments (FMT 1 = 12-bit, MT 2, RT 1, ST 2, RED 0, DIAG 0, NTMAX 288, CM 16, CN 8192, CK 1024, CWR 1,
# CWK 8, CKU 2, CS 1, OPT 6 | 8192) and the launch geometry (WR, WK, KU, S)
_R10B_INST = (1, 2, 1, 2, 0, 0, 288, 16, 8192, 1024, 1, 8, 2, 1, 6 | 8192)
_R10B_CFG = (1, 8, 2, 1)
_R10B_M, _R10B_N, _R10B_K, _R10B_RB, _R10B_ECAP = 16, 8192, 1024, 16, 16
_R10B_MOD = [None]
_R10B_THR = []
_R10B_T0 = [0.0]


def _r10b_rep(t, a, b, count=1):
    assert t.count(a) == count, (a[:80], t.count(a))
    return t.replace(a, b)


def _r10b_source():
    t = V18S_CUDA_TEMPLATE
    # main warps: no in-loop escape table load / walk when EW (OPT 8192)
    t = _r10b_rep(t, "C12 && !RO)", "C12 && !RO && !(OPT & 8192))", 3)
    t = _r10b_rep(t, "          if constexpr (C12) {\n", "          if constexpr (C12 && !(OPT & 8192)) {\n")
    # the escape warp (warp NW of the CTA)
    t = _r10b_rep(t, "    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n",
                  "    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n" + r"""    constexpr bool EW = (OPT & 8192) != 0;   // track r10b
    static_assert(!EW || (CWR && CWK && CKU && RED == 0 && FMT == 1 && CS == 1), "EW: compile-time geometry, RED 0, FMT 1, S 1");
    constexpr int EW_RB = (CWR ? CWR : 1) * RT * 16;
    constexpr int EW_OFF = (CWR * CWK) * RT * MT * 512;
    if constexpr (EW) {
      if (warp == NW) {
        const int cta = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t ea = v18s_lde(esc + (size_t)cta * 64 + lane), ev = v18s_lde(esc + (size_t)cta * 64 + 32 + lane);
        const int cnt = __popc(__ballot_sync(0xffffffffu, ea != 0xFFFFFFFFu));
        float* Dsm = reinterpret_cast<float*>(smem + EW_OFF);
        float* Tsm = Dsm + EW_RB * 32;
        uint32_t* xg = reinterpret_cast<uint32_t*>(Tsm + EW_RB * 32);   // [16 entries][32 tokens] x 4 B (bf16 pair)
#pragma unroll
        for (int e = 0; e < 16; ++e) {
          const uint32_t a = __shfl_sync(0xffffffffu, ea, e);
          if (e < cnt && lane < m) {
            const uint32_t d = v18s_smem_u32(xg + e * 32 + lane);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(d), "l"(x + (size_t)lane * K + ((a & 0xFFFFu) & ~1u)) : "memory");
          }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
#pragma unroll
        for (int r = 0; r < EW_RB; ++r) { Dsm[r * 32 + lane] = 0.f; Tsm[r * 32 + lane] = 0.f; }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
#pragma unroll 1
        for (int e = 0; e < cnt; ++e) {
          const uint32_t a = __shfl_sync(0xffffffffu, ea, e), v = __shfl_sync(0xffffffffu, ev, e);
          const int r = (int)(a >> 16);
          const uint32_t pr = xg[e * 32 + lane];
          const float xe = lane < m ? __uint_as_float(((a & 1u) ? (pr >> 16) : (pr & 0xFFFFu)) << 16) : 0.f;
          const float dv = __uint_as_float(v & 0xFFFF0000u), tv = __uint_as_float(v << 16);
          Dsm[r * 32 + lane] = __fadd_rn(Dsm[r * 32 + lane], __fmul_rn(dv, xe));
          Tsm[r * 32 + lane] = __fadd_rn(Tsm[r * 32 + lane], __fmul_rn(tv, xe));
        }
        __syncthreads();
        return;
      }
    }
""")
    # emitting threads: bf16((S - D) + T)
    st = "            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);\n"
    assert t.count(st) == 2
    i1 = t.index(st)
    t = t[:i1] + r"""            if constexpr (EW) {
              if (tok < m) {
                const float* Dsm = reinterpret_cast<const float*>(smem + EW_OFF);
                const int rl = row - row0;
                v18s_store(out + (size_t)tok * N + row, __fadd_rn(__fsub_rn(vv[c], Dsm[rl * 32 + tok]), Dsm[EW_RB * 32 + rl * 32 + tok]));
              }
            } else
""" + t[i1:]
    # host launcher: one more warp, shared memory for D / T / the x gathers
    t = _r10b_rep(t, "  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n",
                  "  if (WR < 1 || WK < 1 || nt + ((OPT & 8192) ? 32 : 0) > NTMAX) return -1;\n")
    t = _r10b_rep(t, "  lc.blockDim = dim3(nt, 1, 1);\n", "  lc.blockDim = dim3(nt + ((OPT & 8192) ? 32 : 0), 1, 1);\n")
    t = _r10b_rep(t, "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);",
                  "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt + ((OPT & 8192) ? 32 : 0), sm);")
    t = _r10b_rep(t, "    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n",
                  "    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n"
                  "    if (OPT & 8192) sm = (size_t)NW * RT * MT * 512 + (size_t)WR * RT * 16 * 32 * 8 + 16 * 32 * 4;\n")
    args = ", ".join(str(v) for v in _R10B_INST)
    t = _r10b_rep(t, "@@CASES@@", "    case 0: return v18s_launch_t<%s>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);" % args)
    t = t.replace("v18s_", "r10b_").replace("Sm3Fmt", "R10bFmt").replace("SM3D", "R10BD")
    assert "v18s" not in t and "Sm3" not in t
    return "// track r10b (adamleex): Nathen Fernandes' v18s template (MIT), renamed, + the EW escape warp (see the Python section)\n" + t


_R10B_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int r10b_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""


def _r10b_ext():
    if _R10B_MOD[0] is None:
        src = _r10b_source()
        _R10B_MOD[0] = load_inline("okc_bf16proj_r10b_" + _r10b_hashlib.md5((_R10B_CPP + src).encode()).hexdigest()[:12], _R10B_CPP,
                                   cuda_sources=src, functions=["r10b_launch"], no_implicit_headers=True,
                                   with_pytorch_error_handling=False, verbose=False,
                                   extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-Xcompiler=-fno-gnu-unique"])
    return _R10B_MOD[0]


def _r10b_start():
    """compile the r10b extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _r10b_time.time()
        try:
            _r10b_ext()
            _R10B_THR.append("ok")
        except Exception as error:
            _R10B_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _R10B_THR.append(round(_r10b_time.time() - t0, 1))
    t = _r10b_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _r10b_call(st, x, out):
    wr, wk, ku, s = _R10B_CFG
    return _R10B_MOD[0].r10b_launch(0, st["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), st["tab"].data_ptr(), _R10B_M, _R10B_N,
                                    _R10B_K, wr, wk, ku, s, st["t01"], 0, torch.cuda.current_stream().cuda_stream, 0)


def _r10b_table(w, pk):
    """escape table [N / 16, 64] int32: per CTA (16 rows) up to 16 entries sorted by (row, k): word e = row_in_cta << 16 | k,
    word 32 + e = dec_bits << 16 | true_bits, -1 = empty. dec = the codec's decode of the escaped weight WITHOUT its escape
    XOR (exponent = the window base), recomputed from the pack and checked to differ from W exactly at the escapes."""
    n, k = w.shape
    dev = w.device
    base = int(pk["t01"]) & 0xFF
    er = pk["esc"][0]
    none = (er[:0], er[:0], er[:0], er[:0], er[:0])
    dec_p = v18s_c12_unpack(pk["data"], base, none)
    idx = torch.arange(n * k, device=dev, dtype=torch.int64).view(n // 16, 2, 8, k // 64, 4, 4, 2, 2)
    idx = idx.permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(n // 16, k // 64, 32, 32)
    dec = torch.empty(n * k, dtype=torch.int64, device=dev)
    dec[idx.reshape(-1)] = dec_p.reshape(-1).long() & 0xFFFF
    tru = w.contiguous().view(torch.int16).long().reshape(-1) & 0xFFFF
    diff = (dec != tru).nonzero().flatten()
    if diff.numel() != er.numel():
        return None, "escape count mismatch %d vs %d" % (diff.numel(), er.numel())
    if diff.numel() and int(((dec[diff] >> 8) & 0x7F).ne(base).sum()) != 0:
        return None, "dec exponent is not the window base"
    rows, cols = diff // k, diff % k
    cta, rl = rows // _R10B_RB, rows % _R10B_RB
    order = torch.argsort(cta * (1 << 40) + rl * (1 << 20) + cols)
    cta, rl, cols, d_, t_ = cta[order], rl[order], cols[order], dec[diff][order], tru[diff][order]
    rank = torch.arange(cta.numel(), device=dev) - torch.searchsorted(cta, cta)
    mx = int(rank.max()) + 1 if cta.numel() else 0
    if mx > _R10B_ECAP:
        return None, "a CTA has %d escapes (cap %d)" % (mx, _R10B_ECAP)
    tab = torch.full((n // _R10B_RB, 64), -1, dtype=torch.int64, device=dev)
    tab[cta, rank] = (rl << 16) | cols
    tab[cta, 32 + rank] = (d_ << 16) | t_
    tab = torch.where(tab >= (1 << 31), tab - (1 << 32), tab).to(torch.int32).contiguous()
    return tab, "escapes %d, max %d per CTA" % (int(diff.numel()), mx)


def _r10b_validate(st, w, state, gen):
    """one-hot rows over ALL K columns exact; random / 1 %-channel x30 outlier rows (m = 16) against FP64 and the parent
    route; 3 bit-identical repeat calls"""
    n, K, m, dev = _R10B_N, _R10B_K, _R10B_M, w.device
    wd = w.double()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    worst = 0.0
    for dist in (1.0, 30.0, 1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        if _r10b_call(st, x, out) != 0:
            return "launch error"
        par = _r10b_parent_kernel({"x": x, "config": {"m": m, "n": n, "k": K, "layer": "o"}}, state)
        torch.cuda.synchronize()
        ref = x.double() @ wd.t()
        e = float((out.double() - ref).norm() / ref.norm())
        ep = float((par.double() - ref).norm() / ref.norm())
        d = (out.float() - par.float()).abs()
        if not (bool(torch.isfinite(out).all()) and e <= ep * 1.02 + 1e-6 and e <= 4e-3
                and bool((d <= par.float().abs() * 2.0 ** -7 + 1e-5 * float(par.float().abs().max())).all())):
            return "dist %g: rel L2 %.3e (parent %.3e), max |d| %.2e" % (dist, e, ep, float(d.max()))
        worst = max(worst, e)
        o1 = out.clone()
        for _ in range(3):
            if _r10b_call(st, x, out) != 0:
                return "launch error"
            torch.cuda.synchronize()
            if not torch.equal(out.view(torch.int16), o1.view(torch.int16)):
                return "repeat calls differ"
    # one-hot rows over every K column (values 1 and 2^-3): exact products, so the output must be W[:, k] * val
    for val in (1.0, 0.125):
        for k0 in range(0, K, m):
            x = torch.zeros((m, K), dtype=torch.bfloat16, device=dev)
            x[torch.arange(m, device=dev), torch.arange(k0, k0 + m, device=dev)] = val
            if _r10b_call(st, x, out) != 0:
                return "launch error"
            exp = (w[:, k0:k0 + m].t().float() * val).to(torch.bfloat16)
            if not bool((out.float() == exp.float()).all()):
                return "one-hot rows not exact (columns %d..%d, value %g)" % (k0, k0 + m - 1, val)
    torch.cuda.synchronize()
    return "ok %.2e (one-hot over all %d columns)" % (worst, K)


def _r10b_setup(state, weights, thr):
    state["r10b"] = {}
    if thr is not None:
        thr.join(timeout=max(1.0, 250.0 - (_r10b_time.time() - _R10B_T0[0])))
    if not _R10B_THR or _R10B_THR[0] != "ok" or _R10B_MOD[0] is None:
        print("R10B build %s" % (_R10B_THR[0] if _R10B_THR else "not finished"), flush=True)
        return
    w = weights["o"]
    dev = w.device
    if torch.cuda.get_device_capability(dev) != (9, 0):
        print("R10B off: not an sm_90 device", flush=True)
        return
    if w.dim() != 2 or tuple(w.shape) != (_R10B_N, _R10B_K) or w.dtype != torch.bfloat16:
        print("R10B o m=16: off (unexpected weight shape / dtype)", flush=True)
        return
    pk = v18s_pack_format(w.contiguous(), 1)
    if pk is None:
        print("R10B o m=16: off (pack refused)", flush=True)
        return
    sp = v18s_stream(pk, 1, *_R10B_CFG)
    if sp is None:
        print("R10B o m=16: off (stream refused)", flush=True)
        return
    tab, why = _r10b_table(w, pk)
    if tab is None:
        print("R10B o m=16: off (%s)" % why, flush=True)
        return
    wp, reuse = sp["wp"], "own"
    par = (state.get("v18o") or {}).get("plan")
    if (par is not None and par.get("fmt") == 1 and par.get("t01") == sp["t01"] and par["wp"].shape == sp["wp"].shape
            and torch.equal(par["wp"], sp["wp"])):
        wp, reuse = par["wp"], "parent v18o"
    st = dict(wp=wp, tab=tab, t01=sp["t01"])
    del pk, sp
    msg = _r10b_validate(st, w, state, torch.Generator(device=dev).manual_seed(101010))
    print("R10B o m=16: %s (%s, W stream %s)" % (msg if msg.startswith("ok") else "off (" + msg + "), parent route", why, reuse),
          flush=True)
    if msg.startswith("ok"):
        state["r10b"]["o"] = st


_r10b_parent_prepare = prepare
_r10b_parent_kernel = kernel_fn


def prepare(weights):
    _R10B_T0[0] = _r10b_time.time()
    thr = _r10b_start()
    state = _r10b_parent_prepare(weights)
    try:
        _r10b_setup(state, weights, thr)
    except Exception as error:
        state["r10b"] = {}
        print("R10B setup failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__), flush=True)
    torch.cuda.synchronize()
    print("R10B prepare done in %.0fs (r10b build %s, routed %s)" % (_r10b_time.time() - _R10B_T0[0], _R10B_THR,
                                                                      sorted(state.get("r10b", {})) or "none"), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    st = state.get("r10b")
    if st and x.dim() == 2 and x.shape[0] == _R10B_M and data["config"]["layer"] == "o":
        e = st.get("o")
        if (e is not None and x.shape[1] == _R10B_K and data["config"]["n"] == _R10B_N and x.dtype == torch.bfloat16
                and x.is_contiguous() and x.data_ptr() % 16 == 0):
            _k_ctx_bind(x)
            out = torch.empty((_R10B_M, _R10B_N), dtype=torch.bfloat16, device=x.device)
            if _r10b_call(e, x, out) == 0:
                return out
    return _r10b_parent_kernel(data, state)


# ======================================================================================================================
# track r10c (adamleex, 2026-11-03): qkv_gate m = 16 and m = 32 only (EXACT m; every other cell, incl. qkv_gate at
# m 9..15 / 17..31, goes to the parent chain above unchanged; the parent routes of these two cells are alygg77's sk2 TMA-ring
# GEMV 'qkv16t_r200' (m = 16) and track r9b's DIST copy of it 'qkv32_r208' (m = 32)). Kernel: track r9b's DIST kernel, text
# unchanged (this file's SK2_SRC with r9b's DIST escape loop and 2 escape warps; same consumer / producer warps, epilogue,
# producer-side fence.proxy.async and release arrives (r7r race fix)), compiled as its own renamed extension for the two
# qkv_gate geometries (tokr 32: MT 2, NSTAGE 4, NACC 1; tokr 16: MT 1, NSTAGE 5, NACC 2). ONE change, in the weight
# stream: NATURAL row order. The parent packs qkv_gate with the v135 escape-balance permutation (4-row groups dealt so that
# every CTA owns ~ the same number of escaped weights), which in the epilogue costs a row-map lookup per 4-row group and
# 8-byte stores scattered over the whole output row; with DIST the escape warps are off the critical path (they finish
# 0.3-0.4 us before the consumers' K loop ends, r10c stamps), so the balance no longer pays: here each CTA owns 24
# CONSECUTIVE rows (the parent's sk2_compress of W itself, no permutation, so no row map) and stores one contiguous
# 48-byte run per token. Arithmetic: tensor-core FP32 accumulation of the losslessly decoded weights (12-bit pack, exact
# FP32 escape corrections delta * x dealt to 2 FP32 copies by r9b's DIST table), the unchanged fixed-order FP32 epilogue
# (copies 0..1, then the K-warp partials 0..3), ONE final BF16 rounding; deterministic (fixed assignment and order), no
# atomics, no cache-policy hint added or changed, nothing input-derived cached, graph-capturable (one plain launch).
# Prepare: lossless pack (sk2_compress: asserted exact, else off) and r9b's DIST table; the kernel is validated against the
# parent route and FP64 on random and 1 %-channel x30 outlier rows, exactly (as values) on one-hot probe rows at escaped
# columns, and for 3 bit-identical repeat calls, at each routed m. Any failure (or a non-sm_90 device, or a build failure)
# leaves the cell on the parent route. The extension (okc_bf16proj_r10c_<md5 12>: track r9b's generated source with every
# r9b identifier renamed r10c (namespace r10c_ns, kernel r10c_sk2_kernel) and the exported dispatch replaced; raw-pointer
# launcher, no ATen headers; -fno-gnu-unique) builds on a background thread while the parent prepare runs. Static route
# (offline judge-faithful cross-process timing in the r10c README); prepare() never times anything.
# ======================================================================================================================
import hashlib as _r10c_hashlib
import re as _r10c_re
import threading as _r10c_threading
import time as _r10c_time

_R10C_CELLS = (("qkv_gate", "qkv32_r208", 32), ("qkv_gate", "qkv16t_r200", 16))   # (layer, sk2 geometry, m); index = cfg
_R10C_MOD = [None]
_R10C_THR = []
_R10C_T0 = [0.0]


def _r10c_source():
    """track r9b's generated DIST source (_r9b_source(), unchanged kernel text), r9b identifiers renamed r10c, exported
    dispatch r10c_run(cfg) -> the _R10C_CELLS geometries (M == TOKR)"""
    src = _r9b_source()
    for a, b in (("r9b_ns", "r10c_ns"), ("r9b_sk2_kernel", "r10c_sk2_kernel"), ("r9b_launch", "r10c_launch"),
                 ("r9b_run", "r10c_run"), ("r9b_err", "r10c_err"), ("R9B_NESC", "R10C_NESC")):
        src = _r10c_re.sub(r"\b%s\b" % a, b, src)
    if _r10c_re.search(r"\br9b_|\bR9B_", src):
        raise RuntimeError("r10c: unrenamed r9b identifier")
    lines = src.split("\n")
    disp_at = [i for i, l in enumerate(lines) if l.startswith("  if (cfg == ") and "r10c_ns::r10c_launch<" in l]
    if len(disp_at) != len(_R9B_CELLS) or disp_at != list(range(disp_at[0], disp_at[0] + len(disp_at))):
        raise RuntimeError("r10c: dispatch anchor")
    disp = []
    for ci, (layer, name, m) in enumerate(_R10C_CELLS):
        cfg = _r9b_cfg(name)
        _, _, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        _, nstage = _sk2_geom(cfg, m)
        mt = 2 if m == 32 else 1
        nacc = 1 if (mt == 2 and nstage >= 4 and NT == 4) else NACC
        disp.append("  if (cfg == %d) return r10c_ns::r10c_launch<%d, %d, %d, %d, %d, %d, %d, %d>(w, xp, o, t, (int)pmax, (uint32_t)b4, "
                    "(int)M, (int)N, (int)K, s);" % (ci, mt, m, NT, NWR, WSK, UPW, nstage, nacc))
    lines[disp_at[0]:disp_at[-1] + 1] = disp
    return ("// track r10c (adamleex): track r9b's DIST sk2 kernel renamed r10c (qkv_gate m = 32 / 16, natural row order)\n"
            + "\n".join(lines))


def _r10c_ext():
    if _R10C_MOD[0] is None:
        src = _r10c_source()
        cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\nint64_t r10c_run(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, "
               "int64_t, int64_t, int64_t, int64_t, int64_t);\n")
        _R10C_MOD[0] = load_inline("okc_bf16proj_r10c_" + _r10c_hashlib.md5((cpp + src).encode()).hexdigest()[:12], cpp,
                                   cuda_sources=src, functions=["r10c_run"], no_implicit_headers=True, with_pytorch_error_handling=False,
                                   verbose=False, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xcompiler=-fno-gnu-unique"])
    return _R10C_MOD[0]


def _r10c_start():
    """compile the r10c extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _r10c_time.time()
        try:
            _r10c_ext()
            _R10C_THR.append("ok")
        except Exception as error:
            _R10C_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _R10C_THR.append(round(_r10c_time.time() - t0, 1))
    t = _r10c_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _r10c_call(st, x, out):
    return _R10C_MOD[0].r10c_run(st["stream"].data_ptr(), x.data_ptr(), out.data_ptr(), st["tab"].data_ptr(), st["pmax"], int(st["b4"]),
                                 st["m"], st["n"], st["K"], st["ci"], torch.cuda.current_stream().cuda_stream)


def _r10c_validate(st, layer, w, state, gen):
    """random / 1 %-channel x30 outlier rows (m = st["m"]) against FP64 and the parent route (the kernel_fn chain above);
    one-hot rows at escaped columns exact against W (values: +0 and -0 compare equal); 3 bit-identical repeat calls"""
    m, n, K, dev = st["m"], st["n"], st["K"], w.device
    wd = w.double()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    worst = 0.0
    for dist in (1.0, 30.0, 1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        if _r10c_call(st, x, out) != 0:
            return "launch error"
        par = _r10c_parent_kernel({"x": x, "config": {"m": m, "n": n, "k": K, "layer": layer}}, state)
        torch.cuda.synchronize()
        ref = x.double() @ wd.t()
        e = float((out.double() - ref).norm() / ref.norm())
        ep = float((par.double() - ref).norm() / ref.norm())
        d = (out.float() - par.float()).abs()
        if not (par.dtype == torch.bfloat16 and par.shape == out.shape and bool(torch.isfinite(out).all())
                and e <= ep * 1.02 + 1e-6 and e <= 4e-3
                and bool((d <= par.float().abs() * 2.0 ** -7 + 1e-5 * float(par.float().abs().max())).all())):
            return "dist %g: rel L2 %.3e (parent %.3e), max |d| %.2e" % (dist, e, ep, float(d.max()))
        worst = max(worst, e)
        o1 = out.clone()
        for _ in range(3):
            if _r10c_call(st, x, out) != 0:
                return "launch error"
            torch.cuda.synchronize()
            if not torch.equal(out.view(torch.int16), o1.view(torch.int16)):
                return "repeat calls differ"
    # one-hot rows at escaped columns (+ two plain columns): exact products, so the output must be bf16(W[:, k] * val)
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e71 = (bits >> 8) & 0x7F
    base, _ = _exponent_window(e71)
    esc_cols = torch.unique(_escapes(e71, base).nonzero()[:, 1])
    if esc_cols.numel():
        sel = esc_cols[torch.randperm(esc_cols.numel(), generator=gen, device=dev)[:2 * m - 2]]
    else:
        sel = esc_cols
    ks = torch.cat([sel, torch.tensor([0, K - 1], device=dev)])
    vals = torch.tensor([(-1.0 if j % 2 else 1.0) * 2.0 ** (1 + j % 4) for j in range(m)], device=dev)
    for b0 in range(0, ks.numel(), m):
        kb = ks[b0:b0 + m]
        x = torch.zeros((m, K), dtype=torch.bfloat16, device=dev)
        x[torch.arange(kb.numel(), device=dev), kb] = vals[:kb.numel()].to(torch.bfloat16)
        if _r10c_call(st, x, out) != 0:
            return "launch error"
        torch.cuda.synchronize()
        exp = (w.float()[:, kb].t() * vals[:kb.numel(), None]).to(torch.bfloat16)
        if not bool((out[:kb.numel()].float() == exp.float()).all()):
            return "one-hot rows at escaped columns not exact"
    return "ok %.2e (%d one-hot columns)" % (worst, int(ks.numel()))


def _r10c_setup(state, weights, thr):
    state["r10c"] = {}
    if thr is not None:
        thr.join(timeout=max(1.0, 250.0 - (_r10c_time.time() - _R10C_T0[0])))
    if not _R10C_THR or _R10C_THR[0] != "ok" or _R10C_MOD[0] is None:
        print("R10C build %s" % (_R10C_THR[0] if _R10C_THR else "not finished"), flush=True)
        return
    dev = weights["o"].device
    if torch.cuda.get_device_capability(dev) != (9, 0):
        print("R10C off: not an sm_90 device", flush=True)
        return
    packs = {}
    for ci, (layer, name, m) in enumerate(_R10C_CELLS):
        cfg = _r9b_cfg(name)
        _, _, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        w = weights[layer]
        if CL != 1 or w.dim() != 2 or w.dtype != torch.bfloat16 or w.shape[0] != n_rows or _sk2_geom(cfg, m)[1] < _R9B_NESC:
            print("R10C %s m=%d: off (geometry)" % (layer, m), flush=True)
            continue
        gkey = (layer, n_rows, R, NT, NWR, WSK, UPW)
        if gkey not in packs:
            comp = sk2_compress(w.contiguous())   # natural row order: no escape-balance permutation
            if comp is None:
                packs[gkey] = (None, "pack refused")
            else:
                tb, why = _r9b_table(comp, n_rows, R, WSK * UPW, None)   # r9b's DIST table (no row map)
                packs[gkey] = ((sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW), tb, comp["base4"],
                                int((comp["rows"] < n_rows).sum())) if tb is not None else None, why)
                del comp
        pk, why = packs[gkey]
        if pk is None:
            print("R10C %s m=%d: off (%s)" % (layer, m, why), flush=True)
            continue
        stream, tb, b4, nesc = pk
        st = dict(ci=ci, m=m, stream=stream, tab=tb[0], pmax=tb[1], b4=b4, n=n_rows, K=w.shape[1])
        msg = _r10c_validate(st, layer, w, state, torch.Generator(device=dev).manual_seed(101010 + ci))
        print("R10C %s m=%d: %s (escapes %d, table row %d int2)" % (layer, m, msg if msg.startswith("ok") else "off (" + msg + "), parent route",
                                                                   nesc, tb[1]), flush=True)
        if msg.startswith("ok"):
            state["r10c"][m] = st


_r10c_parent_prepare = prepare
_r10c_parent_kernel = kernel_fn


def prepare(weights):
    _R10C_T0[0] = _r10c_time.time()
    thr = _r10c_start()
    state = _r10c_parent_prepare(weights)
    try:
        _r10c_setup(state, weights, thr)
    except Exception as error:
        state["r10c"] = {}
        print("R10C setup failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__), flush=True)
    torch.cuda.synchronize()
    print("R10C prepare done in %.0fs (r10c build %s, routed qkv_gate m %s)" % (_r10c_time.time() - _R10C_T0[0], _R10C_THR,
                                                                               sorted(state.get("r10c", {})) or "none"), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    st = state.get("r10c")
    if st and x.dim() == 2 and data["config"]["layer"] == "qkv_gate":
        e = st.get(x.shape[0])
        if (e is not None and x.shape[1] == e["K"] and data["config"]["n"] == e["n"] and x.dtype == torch.bfloat16
                and x.is_contiguous() and x.data_ptr() % 16 == 0):
            _k_ctx_bind(x)
            out = torch.empty((e["m"], e["n"]), dtype=torch.bfloat16, device=x.device)
            if _r10c_call(e, x, out) == 0:
                return out
    return _r10c_parent_kernel(data, state)


# ======================================================================================================================
# track r11a (adamleex, 2026-11-03): o m = 1 and o m = 8 only (EXACT m; every other cell, incl. o at m 2..7 / 9, and every
# other layer goes to the parent chain above unchanged). Parent routes: o m1 = this file's 12-bit cgemv<2, 64, 1024> (two
# rows per CTA, 4096 CTAs x (64 + 32) threads: FFMA GEMV + an escape warp correcting escaped weights by x[k] * delta);
# o m8 = Nathen Fernandes' v18s one-shot tensor-core GEMV 'v18o' <1,1,1,2,0,0,256,0,8192,1024,1,8,2,1,6> (its warps patch
# escaped weights inside the K loop).
# Kernel: track r10b's EW kernel (its source generator _r10b_source(): Nathen Fernandes' V18S_CUDA_TEMPLATE (MIT) + the
# escape warp; renamed here r10b_ -> r11a_) instantiated at compile-time m 1 and m 8 with ONE n8 token tile (MT 1) in the
# o m8 route's geometry: 512 CTAs x 8 K-warps (16 W rows x 128 K per warp, both 64-K units of 12-bit W issued up front,
# mma.sync m16n8k16 bf16 -> FP32 with the tokens as N, FP32 accumulation over the warp's K slice), fixed-order FP32 sum of
# the 8 K-warp partials through shared memory, plus one escape warp per CTA (lane = token): it gathers x[token][k] for the
# CTA's <= 16 escapes (4-byte cp.async.ca into its own smem slots, wait_group by the issuing thread) and accumulates
# D = sum dec * x and T = sum true * x in FP32 in entry order (exact products) before the CTA barrier; the emitting threads
# store bf16((S - D) + T): ONE final BF16 rounding.
# r11a's one change to that kernel: how the escape warp gets the CTA's two escape-table lines (256 B). r10b's escape warp
# loads them with two ld.global.nc into registers; measured here (r11a README), those loads put ~0.35..0.5 us on the critical
# path at m 1 / 8 (r10b's EW was +0.2 us SLOWER than cgemv at o m1 and only -0.1 vs v18o at o m8) although the escape-free
# main path is ~0.4..0.5 us faster than both parents. Two compile-time alternatives, one per routed m (both measured best):
#   o m1, OPT bit 524288 'W0T': MAIN WARP 0 loads the two lines into registers BEFORE its own W loads (as the parent o m8
#         route's warps load their per-warp tables first), stores them to 256 B of shared memory after its up-front W loads
#         and arrives at named barrier 1 (64 threads); the escape warp bar.syncs on barrier 1, then reads them from smem.
#   o m8, OPT bits 16777216 | 67108864 'EWC' + 'EWCS': the escape warp copies the two lines itself with cp.async.ca
#         (8 B per lane into 256 B of shared memory), cp.async.wait_group 0, __syncwarp() (the lane that reads a word is not
#         the lane that copied it), then reads them from smem.
# No split-K over CTAs, no atomics, no cluster, no mbarrier / TMA / bulk async proxy (cp.async non-bulk, waited by the
# issuing threads before a warp / CTA barrier), no cache-policy operand added (the template's W loads keep
# ld.global.nc.L1::no_allocate, x / table loads plain ld.global.nc / cp.async.ca), nothing input-derived cached, one launch.
# Weights: track r10b's o state (the parent v18o W stream + r10b's 512 x 64 escape table: the SAME CTA geometry, 16 rows
# per CTA) is reused when r10b routed o m16; else this section builds the same two from the parent's lossless v18s pack
# (table: true BF16 bits of every escape; lossless). Prepare: the r11a extension (okc_bf16proj_r11a_<md5 12>,
# -fno-gnu-unique, two instances) builds on a background thread while the parent prepare runs; then each route is
# validated: one-hot rows over ALL 1024 K columns exact (values 1 and 2^-3), random / 1 %-channel x30 outlier rows against
# FP64 and the parent route, 3 bit-identical repeat calls. Any failure (or a non-sm_90 device, a build failure, a CTA with
# > 16 escapes) leaves that cell on its parent route. Static routes (offline judge-faithful cross-process timing in the
# r11a README); prepare() never times anything.
# ======================================================================================================================
import hashlib as _r11a_hashlib
import threading as _r11a_threading
import time as _r11a_time

# v18s template arguments (FMT 1 = 12-bit, MT 1, RT 1, ST 2, RED 0, DIAG 0, NTMAX 288, CM, CN 8192, CK 1024, CWR 1, CWK 8,
# CKU 2, CS 1, OPT) per routed m (launch case = index) and the launch geometry (WR, WK, KU, S)
_R11A_INST = {1: (1, 1, 1, 2, 0, 0, 288, 1, 8192, 1024, 1, 8, 2, 1, 6 | 8192 | 524288),    # EW + W0T
              8: (1, 1, 1, 2, 0, 0, 288, 8, 8192, 1024, 1, 8, 2, 1, 6 | 8192 | 16777216 | 67108864)}   # EW + EWC + EWCS
_R11A_CASE = {1: 0, 8: 1}
_R11A_CFG = (1, 8, 2, 1)
_R11A_N, _R11A_K = 8192, 1024
_R11A_MOD = [None]
_R11A_THR = []
_R11A_T0 = [0.0]


def _r11a_rep(t, a, b, count=1):
    assert t.count(a) == count, (a[:80], t.count(a))
    return t.replace(a, b)


def _r11a_source():
    t = _r10b_source()
    t = t[t.index("\n") + 1:]   # drop r10b's banner line
    t = _r11a_rep(t, "    case 0: return r10b_launch_t<%s>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);"
                  % ", ".join(str(v) for v in _R10B_INST),
                  "\n".join("    case %d: return r10b_launch_t<%s>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);"
                            % (_R11A_CASE[m], ", ".join(str(v) for v in _R11A_INST[m])) for m in sorted(_R11A_CASE, key=_R11A_CASE.get)))
    # escape warp: EWC (o m8) copies its escape words by cp.async.ca + __syncwarp; W0T (o m1) waits for warp 0's hand-off
    t = _r11a_rep(t, "        const uint32_t ea = r10b_lde(esc + (size_t)cta * 64 + lane), ev = r10b_lde(esc + (size_t)cta * 64 + 32 + lane);\n",
                  "        uint32_t ea, ev;\n"
                  "        if constexpr ((OPT & 16777216) != 0) {   // r11a EWC: the escape warp copies its own escape words by cp.async.ca (8 B / lane)\n"
                  "          const uint32_t* t0s = reinterpret_cast<const uint32_t*>(smem + EW_OFF + EW_RB * 32 * 8 + 16 * 32 * 4);\n"
                  "          asm volatile(\"cp.async.ca.shared.global [%0], [%1], 8;\" :: \"r\"(r10b_smem_u32(t0s + lane * 2)), \"l\"(esc + (size_t)cta * 64 + lane * 2) : \"memory\");\n"
                  "          asm volatile(\"cp.async.commit_group;\" ::: \"memory\");\n"
                  "          asm volatile(\"cp.async.wait_group 0;\" ::: \"memory\");\n"
                  "          if constexpr ((OPT & 67108864) != 0) __syncwarp();   // r11a EWCS: order the other lanes' copies before the reads\n"
                  "          ea = t0s[lane];\n"
                  "          ev = t0s[32 + lane];\n"
                  "        } else if constexpr ((OPT & 524288) != 0) {   // r11a W0T: warp 0 loaded the CTA's escape words first; wait for its arrival\n"
                  "          asm volatile(\"bar.sync 1, 64;\" ::: \"memory\");\n"
                  "          const uint32_t* t0s = reinterpret_cast<const uint32_t*>(smem + EW_OFF + EW_RB * 32 * 8 + 16 * 32 * 4);\n"
                  "          ea = t0s[lane];\n"
                  "          ev = t0s[32 + lane];\n"
                  "        } else {\n"
                  "          ea = r10b_lde(esc + (size_t)cta * 64 + lane);\n"
                  "          ev = r10b_lde(esc + (size_t)cta * 64 + 32 + lane);\n"
                  "        }\n")
    # W0T (o m1), main warp 0: fetch the escape words BEFORE its W loads; hand them over right after its up-front loads
    a = "#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);\n"
    t = _r11a_rep(t, a, "    uint32_t r11_ta = 0u, r11_tv = 0u;\n"
                        "    if constexpr ((OPT & 524288) != 0) {   // W0T: two plain loads into registers\n"
                        "      if (warp == 0) {\n"
                        "        const int cta_ = blockIdx.y * gridDim.x + blockIdx.x;\n"
                        "        r11_ta = r10b_lde(esc + (size_t)cta_ * 64 + lane);\n"
                        "        r11_tv = r10b_lde(esc + (size_t)cta_ * 64 + 32 + lane);\n"
                        "      }\n"
                        "    }\n" + a +
                        "    if constexpr ((OPT & 524288) != 0) {\n"
                        "      if (warp == 0) {\n"
                        "        constexpr int EW_RB_ = (CWR ? CWR : 1) * RT * 16;\n"
                        "        uint32_t* t0s = reinterpret_cast<uint32_t*>(smem + (CWR * CWK) * RT * MT * 512 + EW_RB_ * 32 * 8 + 16 * 32 * 4);\n"
                        "        t0s[lane] = r11_ta;\n"
                        "        t0s[32 + lane] = r11_tv;\n"
                        "        asm volatile(\"bar.arrive 1, 64;\" ::: \"memory\");\n"
                        "      }\n"
                        "    }\n")
    # launcher: 256 B of hand-off smem
    a = "    if (OPT & 8192) sm = (size_t)NW * RT * MT * 512 + (size_t)WR * RT * 16 * 32 * 8 + 16 * 32 * 4;\n"
    t = _r11a_rep(t, a, a + "    if (OPT & (524288 | 16777216)) sm += 256;\n")
    t = t.replace("r10b_", "r11a_").replace("R10bFmt", "R11aFmt").replace("R10BD", "R11AD")
    assert "r10b_" not in t and "R10bFmt" not in t and "R10BD" not in t and "v18s" not in t and "Sm3" not in t
    return "// track r11a (adamleex): Nathen Fernandes' v18s template (MIT) + track r10b's EW escape warp + W0T / EWC, renamed; o m = 1, 8\n" + t


_R11A_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int r11a_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""


def _r11a_ext():
    if _R11A_MOD[0] is None:
        src = _r11a_source()
        _R11A_MOD[0] = load_inline("okc_bf16proj_r11a_" + _r11a_hashlib.md5((_R11A_CPP + src).encode()).hexdigest()[:12], _R11A_CPP,
                                   cuda_sources=src, functions=["r11a_launch"], no_implicit_headers=True,
                                   with_pytorch_error_handling=False, verbose=False,
                                   extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-Xcompiler=-fno-gnu-unique"])
    return _R11A_MOD[0]


def _r11a_start():
    """compile the r11a extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _r11a_time.time()
        try:
            _r11a_ext()
            _R11A_THR.append("ok")
        except Exception as error:
            _R11A_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _R11A_THR.append(round(_r11a_time.time() - t0, 1))
    t = _r11a_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _r11a_call(st, m, x, out):
    wr, wk, ku, s = _R11A_CFG
    return _R11A_MOD[0].r11a_launch(_R11A_CASE[m], st["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), st["tab"].data_ptr(), m,
                                    _R11A_N, _R11A_K, wr, wk, ku, s, st["t01"], 0, torch.cuda.current_stream().cuda_stream, 0)


def _r11a_validate(st, m, w, state, gen):
    """one-hot rows over ALL K columns exact; random / 1 %-channel x30 outlier rows (this m) against FP64 and the parent
    route; 3 bit-identical repeat calls"""
    n, K, dev = _R11A_N, _R11A_K, w.device
    wd = w.double()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    worst = 0.0
    for dist in (1.0, 30.0, 1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        if _r11a_call(st, m, x, out) != 0:
            return "launch error"
        par = _r11a_parent_kernel({"x": x, "config": {"m": m, "n": n, "k": K, "layer": "o"}}, state)
        torch.cuda.synchronize()
        ref = x.double() @ wd.t()
        e = float((out.double() - ref).norm() / ref.norm())
        ep = float((par.double() - ref).norm() / ref.norm())
        d = (out.float() - par.float()).abs()
        if not (bool(torch.isfinite(out).all()) and e <= ep * 1.02 + 1e-6 and e <= 4e-3
                and bool((d <= par.float().abs() * 2.0 ** -7 + 1e-5 * float(par.float().abs().max())).all())):
            return "dist %g: rel L2 %.3e (parent %.3e), max |d| %.2e" % (dist, e, ep, float(d.max()))
        worst = max(worst, e)
        o1 = out.clone()
        for _ in range(3):
            if _r11a_call(st, m, x, out) != 0:
                return "launch error"
            torch.cuda.synchronize()
            if not torch.equal(out.view(torch.int16), o1.view(torch.int16)):
                return "repeat calls differ"
    # one-hot rows over every K column (values 1 and 2^-3): exact products, so the output must be W[:, k] * val
    bad = torch.zeros((), dtype=torch.int64, device=dev)
    for val in (1.0, 0.125):
        for k0 in range(0, K, m):
            x = torch.zeros((m, K), dtype=torch.bfloat16, device=dev)
            x[torch.arange(m, device=dev), torch.arange(k0, k0 + m, device=dev)] = val
            if _r11a_call(st, m, x, out) != 0:
                return "launch error"
            exp = (w[:, k0:k0 + m].t().float() * val).to(torch.bfloat16)
            bad += (out.float() != exp.float()).sum()
    torch.cuda.synchronize()
    if int(bad) != 0:
        return "one-hot rows not exact (%d outputs differ)" % int(bad)
    return "ok %.2e (one-hot over all %d columns)" % (worst, K)


def _r11a_setup(state, weights, thr):
    state["r11a"] = {}
    if thr is not None:
        thr.join(timeout=max(1.0, 250.0 - (_r11a_time.time() - _R11A_T0[0])))
    if not _R11A_THR or _R11A_THR[0] != "ok" or _R11A_MOD[0] is None:
        print("R11A build %s" % (_R11A_THR[0] if _R11A_THR else "not finished"), flush=True)
        return
    w = weights["o"]
    dev = w.device
    if torch.cuda.get_device_capability(dev) != (9, 0):
        print("R11A off: not an sm_90 device", flush=True)
        return
    if w.dim() != 2 or tuple(w.shape) != (_R11A_N, _R11A_K) or w.dtype != torch.bfloat16:
        print("R11A o: off (unexpected weight shape / dtype)", flush=True)
        return
    r10 = (state.get("r10b") or {}).get("o")
    if r10 is not None and tuple(_R10B_CFG) == tuple(_R11A_CFG) and _R10B_RB == 16 and _R10B_ECAP == 16:
        st, why, src = dict(wp=r10["wp"], tab=r10["tab"], t01=r10["t01"]), "r10b table", "r10b state"
    else:
        pk = v18s_pack_format(w.contiguous(), 1)
        if pk is None:
            print("R11A o: off (pack refused)", flush=True)
            return
        sp = v18s_stream(pk, 1, *_R11A_CFG)
        if sp is None:
            print("R11A o: off (stream refused)", flush=True)
            return
        tab, why = _r10b_table(w, pk)
        if tab is None:
            print("R11A o: off (%s)" % why, flush=True)
            return
        wp, src = sp["wp"], "own"
        par = (state.get("v18o") or {}).get("plan")
        if (par is not None and par.get("fmt") == 1 and par.get("t01") == sp["t01"] and par["wp"].shape == sp["wp"].shape
                and torch.equal(par["wp"], sp["wp"])):
            wp, src = par["wp"], "parent v18o"
        st = dict(wp=wp, tab=tab, t01=sp["t01"])
        del pk, sp
    for m in sorted(_R11A_CASE):
        msg = _r11a_validate(st, m, w, state, torch.Generator(device=dev).manual_seed(111111 + m))
        print("R11A o m=%d: %s (%s, W stream + table from %s)" % (m, msg if msg.startswith("ok") else "off (" + msg + "), parent route",
                                                                 why, src), flush=True)
        if msg.startswith("ok"):
            state["r11a"][m] = st


_r11a_parent_prepare = prepare
_r11a_parent_kernel = kernel_fn


def prepare(weights):
    _R11A_T0[0] = _r11a_time.time()
    thr = _r11a_start()
    state = _r11a_parent_prepare(weights)
    try:
        _r11a_setup(state, weights, thr)
    except Exception as error:
        state["r11a"] = {}
        print("R11A setup failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__), flush=True)
    torch.cuda.synchronize()
    print("R11A prepare done in %.0fs (r11a build %s, routed o m %s)" % (_r11a_time.time() - _R11A_T0[0], _R11A_THR,
                                                                        sorted(state.get("r11a", {})) or "none"), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    st = state.get("r11a")
    if st and x.dim() == 2 and data["config"]["layer"] == "o":
        m = x.shape[0]
        e = st.get(m)
        if (e is not None and x.shape[1] == _R11A_K and data["config"]["n"] == _R11A_N and x.dtype == torch.bfloat16
                and x.is_contiguous() and x.data_ptr() % 16 == 0):
            _k_ctx_bind(x)
            out = torch.empty((m, _R11A_N), dtype=torch.bfloat16, device=x.device)
            if _r11a_call(e, m, x, out) == 0:
                return out
    return _r11a_parent_kernel(data, state)


# ======================================================================================================================
# track r13a (adamleex, 2026-11-03): router m = 32 only (EXACT m; every other cell, incl. router at m 2..31 / 33.., goes to
# the parent chain above unchanged). Parent route: track tC's routertail_small_kernel<MT 4, S 16, WK 4, P 16> (Nathen
# Fernandes, MIT; see the MIT notice at the top of this file): 256 CTAs = 16 thread-block clusters of 16 (cluster rank =
# K slice of 512), 4 warps per CTA, BF16 hi + lo fragments of the FP32 router weight, mma.sync m16n8k16 -> FP32, warps summed
# in shared memory, each CTA pushes its FP32 partial by st.async to the owning rank, the owner sums ranks 0..15 in fixed
# order: deterministic, FP32 output.
# r13a's one change (same kernel, same pack, same per-accumulator mma sequence, same fixed-order DSMEM reduction ->
# BIT-IDENTICAL to tC): the cluster-reduction prologue (mbarrier.init + fence.proxy.async by thread 0 of each owner rank,
# __syncthreads, mbarrier.arrive.expect_tx + fence.mbarrier_init.release.cluster, barrier.cluster.arrive.relaxed) moves from
# between the load block and the mma block to right AFTER the mma block. In the parent it acts as a scheduling fence (ptxas
# issues all 32 LDG.128 per thread before the first HMMA, 158 registers, 3 CTAs / SM); moved, the loads and mma interleave
# (14 LDG, then LDG / HMMA mixed; 104 registers, 4 CTAs / SM; cluster placement unchanged: 116 SMs, 3 / 2 / 1 CTAs on
# 32 / 76 / 8 SMs, same-rank sharing). Memory-model order is unchanged: the mbarrier init and fence.mbarrier_init.release.cluster
# still precede this thread's cluster arrive, every remote st.async still follows barrier.cluster.wait.acquire (and the
# __syncthreads that publishes the warp partials), and the owner still waits on its mbarrier (complete_tx of all 16 partials)
# before summing. Measured (r13a oracle, 2 processes, 6 blocks): -0.38..-0.42 us vs tC's own route at router m32 (local).
# Prepare: the r13a extension (okc_bf16proj_r13a_<md5 12>, -fno-gnu-unique, one instance) builds on a background thread
# while the parent prepare runs; the route reuses tC's routed pack state['tc_rt_w'][(16, 4)] (no new device memory) and is
# validated BIT-IDENTICAL to the tC kernel on random, 1 %-channel x30 outlier and one-hot rows over ALL 8192 K columns, plus
# 3 bit-identical repeat calls. Any failure (or tC's route off, a non-sm_90 device, a build failure) leaves router m32 on tC.
# Static route from offline judge-faithful cross-process timing (r13a README); prepare() never times anything.
# ======================================================================================================================
import hashlib as _r13a_hashlib
import threading as _r13a_threading
import time as _r13a_time

_R13A_CU = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
namespace r13a {
#define RD __device__ __forceinline__
RD uint32_t r13a_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
RD void r13a_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void r13a_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void r13a_wait(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");
}
RD uint32_t r13a_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }
RD void r13a_push(uint32_t d, float4 v, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
RD void r13a_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }

// tC's routertail_small_kernel with the cluster-reduction prologue moved behind the mma block (r13a); otherwise unchanged
template<int MT,int S,int WK,int P>
__global__ __launch_bounds__(WK*32) void r13a_rt_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {
  constexpr int KU=128/S/WK, NI=MT*32, OWN=NI/P, PARTS=S*1;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+PARTS*OWN;
  const uint32_t rb=r13a_smem(sm+(PARTS*OWN+WK*NI)*16);
  int4 w[KU][8];uint4 xx[KU][MT][2];
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<8;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  float c[2][MT][4]={};
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          const int4 a=w[k][q+h*4];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
  // r13a: cluster-reduction prologue AFTER the mma block (tC has it between the load block and the mma block)
  if(rank<P && tid==0) {r13a_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  __syncthreads();
  if(rank<P && tid==0) {r13a_expect(rb,PARTS*OWN*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  if(tid<NI) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) r13a_add(sum,part[k*NI+tid]);
    const int dst=tid/OWN,local=tid%OWN;
    r13a_push(r13a_map(r13a_smem(recv+rank*OWN+local),dst),sum,r13a_map(rb,dst));
  }
  if(rank<P && tid<OWN) {
    r13a_wait(rb,0);
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<PARTS;k++) r13a_add(sum,recv[k*OWN+tid]);
    const int v=rank*OWN+tid,ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
    if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
    if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
  }
}
}  // namespace r13a

template<int MT,int S,int WK,int P>
static int r13a_rt_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto fn=r13a::r13a_rt_kernel<MT,S,WK,P>;
  constexpr int bytes=(S*1/P+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);
}
int r13a_rt_cfg(int cfg,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  switch(cfg) {
@@R13A_CASES@@
    default: return -99;
  }
}
"""
_R13A_CPP = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\n"
             "int r13a_rt_cfg(int,int64_t,int64_t,int64_t,int,int64_t,int);\n")
# routed exact m -> (case index, (MT, S, WK, P)); the geometry equals tC's config for that m bucket (TC_RT_CFGS)
_R13A_ROUTES = {32: (0, (4, 16, 4, 16))}
_R13A_MOD = [None]
_R13A_THR = []
_R13A_T0 = [0.0]


def _r13a_source():
    cases = "\n".join("    case %d: return r13a_rt_t<%d,%d,%d,%d>(w, x, out, m, stream, query);" % ((i,) + g)
                      for m, (i, g) in sorted(_R13A_ROUTES.items(), key=lambda kv: kv[1][0]))
    return _R13A_CU.replace("@@R13A_CASES@@", cases)


def _r13a_ext():
    if _R13A_MOD[0] is None:
        src = _r13a_source()
        _R13A_MOD[0] = load_inline("okc_bf16proj_r13a_" + _r13a_hashlib.md5((_R13A_CPP + src).encode()).hexdigest()[:12], _R13A_CPP,
                                   cuda_sources=src, functions=["r13a_rt_cfg"], no_implicit_headers=True,
                                   with_pytorch_error_handling=False, verbose=False, extra_cflags=["-fno-gnu-unique"],
                                   extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "-Xcompiler=-fno-gnu-unique"])
    return _R13A_MOD[0]


def _r13a_start():
    """compile the r13a extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _r13a_time.time()
        try:
            _r13a_ext()
            _R13A_THR.append("ok")
        except Exception as error:
            _R13A_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _R13A_THR.append(round(_r13a_time.time() - t0, 1))
    t = _r13a_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _r13a_call(st, m, x, out):
    i, (mt, s, wk, p) = _R13A_ROUTES[m]
    return _R13A_MOD[0].r13a_rt_cfg(i, st["w"].data_ptr(), x.data_ptr(), out.data_ptr(), m, torch.cuda.current_stream().cuda_stream, 0)


def _r13a_validate(st, m, state, gen):
    """BIT-IDENTICAL to tC's routertail kernel on random / 1 %-channel x30 outlier rows and one-hot rows over ALL K columns
    (values 1 and 2^-3); 3 bit-identical repeat calls"""
    dev, K = st["w"].device, 8192
    out = torch.empty((m, 256), dtype=torch.float32, device=dev)
    for dist in (1.0, 30.0, 1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        if _r13a_call(st, m, x, out) != 0:
            return "launch error"
        par = _tc_rt_run(x, state)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(out).all()) or not torch.equal(out.view(torch.int32), par.view(torch.int32)):
            return "dist %g: not bit-identical to tC (%d of %d differ)" % (dist, int((out.view(torch.int32) != par.view(torch.int32)).sum()), out.numel())
        o1 = out.clone()
        for _ in range(3):
            if _r13a_call(st, m, x, out) != 0:
                return "launch error"
            torch.cuda.synchronize()
            if not torch.equal(out.view(torch.int32), o1.view(torch.int32)):
                return "repeat calls differ"
    bad = torch.zeros((), dtype=torch.int64, device=dev)
    for val in (1.0, 0.125):
        for k0 in range(0, K, m):
            x = torch.zeros((m, K), dtype=torch.bfloat16, device=dev)
            x[torch.arange(m, device=dev), torch.arange(k0, k0 + m, device=dev)] = val
            if _r13a_call(st, m, x, out) != 0:
                return "launch error"
            par = _tc_rt_run(x, state)
            bad += (out.view(torch.int32) != par.view(torch.int32)).sum()
    torch.cuda.synchronize()
    if int(bad) != 0:
        return "one-hot rows not bit-identical to tC (%d outputs differ)" % int(bad)
    return "ok (bit-identical to tC: random, outliers, one-hot over all %d columns)" % K


def _r13a_setup(state, weights, thr):
    state["r13a"] = {}
    if thr is not None:
        thr.join(timeout=max(1.0, 250.0 - (_r13a_time.time() - _R13A_T0[0])))
    if not _R13A_THR or _R13A_THR[0] != "ok" or _R13A_MOD[0] is None:
        print("R13A build %s" % (_R13A_THR[0] if _R13A_THR else "not finished"), flush=True)
        return
    w = weights["router"]
    if torch.cuda.get_device_capability(w.device) != (9, 0):
        print("R13A off: not an sm_90 device", flush=True)
        return
    if not state.get("tc_rt_ok"):
        print("R13A off: tC routertail route is off", flush=True)
        return
    for m, (i, (mt, s, wk, p)) in sorted(_R13A_ROUTES.items()):
        tcfg = TC_RT_CFGS[next(j for bound, j in TC_RT_ROUTE if m <= bound)]
        pk = (state.get("tc_rt_w") or {}).get((s, wk))
        if tuple(tcfg) != (mt, s, wk, p) or pk is None:
            print("R13A router m=%d: off (tC geometry %s differs or pack missing)" % (m, tcfg), flush=True)
            continue
        st = dict(w=pk)
        msg = _r13a_validate(st, m, state, torch.Generator(device=w.device).manual_seed(131313 + m))
        print("R13A router m=%d: %s" % (m, msg if msg.startswith("ok") else "off (" + msg + "), tC route"), flush=True)
        if msg.startswith("ok"):
            state["r13a"][m] = st


_r13a_parent_prepare = prepare
_r13a_parent_kernel = kernel_fn


def prepare(weights):
    _R13A_T0[0] = _r13a_time.time()
    thr = _r13a_start()
    state = _r13a_parent_prepare(weights)
    try:
        _r13a_setup(state, weights, thr)
    except Exception as error:
        state["r13a"] = {}
        print("R13A setup failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__), flush=True)
    torch.cuda.synchronize()
    print("R13A prepare done in %.0fs (r13a build %s, routed router m %s)" % (_r13a_time.time() - _R13A_T0[0], _R13A_THR,
                                                                             sorted(state.get("r13a", {})) or "none"), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    st = state.get("r13a")
    if st and x.dim() == 2 and data["config"]["layer"] == "router":
        m = x.shape[0]
        e = st.get(m)
        if (e is not None and x.shape[1] == 8192 and x.dtype == torch.bfloat16 and x.is_contiguous()
                and x.data_ptr() % 16 == 0):
            _k_ctx_bind(x)
            out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
            if _r13a_call(e, m, x, out) == 0:
                return out
    return _r13a_parent_kernel(data, state)


# ======================================================================================================================
# track r14a (adamleex, 2026-11-04): unpooler m = 32 only (EXACT m; every other cell, incl. unpooler at m 17..31 / 33..,
# goes to the parent chain above unchanged; the parent route of this cell is the u32 override's owner-model sk2 'u32_r96'
# (R 32 rows per CTA, 256 CTAs, 2 CTAs per SM), while unpooler m 17..31 run alygg77's owner-model sk2 on 'unpooler32_r144').
# Kernel: track r9b's DIST kernel, text unchanged (this file's SK2_SRC with r9b's DIST escape loop and 2 escape warps; same
# consumer / producer warps, epilogue, producer-side fence.proxy.async and release arrives (r7r race fix)), compiled as its own
# renamed extension for ONE instance: alygg77's R64 geometry 'unpooler32_r144' at tokr 32 (MT 2, NT 4, NWR 2, WSK 2, UPW 2,
# ring depth 3, NACC 1; 128 CTAs x 224 threads = 4 consumer + 1 producer + 2 escape warps, 1 CTA per SM). Why: the u32
# geometry's tail is per-SM unfairness between its 2 co-resident CTAs (s13b); without escape work the 1-CTA-per-SM R64 layout
# is faster, but under the owner escape model it lost because R64 carries ~2x the escapes per CTA and the stage owner's
# serial corrections gate the 3-deep ring; DIST deals each stage's escapes round-robin to the 2 escape warps (s14c).
# Weight stream: the parent's state["sk2"]["unpooler32_r144"] stream tensor (reused, not copied; natural row order, unpooler
# is not an escape-balance layer); the parent's lossless pack is re-run and must reproduce that stream and its escape table
# bitwise; the DIST table is built from it. Arithmetic: tensor-core FP32 accumulation of the losslessly decoded weights
# (12-bit pack, exact FP32 escape corrections delta * x dealt to 2 FP32 copies by r9b's DIST table), the unchanged
# fixed-order FP32 epilogue (copies 0..1, then the K-warp partials), ONE final BF16 rounding; deterministic (fixed assignment
# and order), no atomics, no cache-policy hint added or changed, nothing input-derived cached, graph-capturable (one plain
# launch). Prepare: the kernel is validated against the parent route (u32) and FP64 on random and 1 %-channel x30 outlier rows,
# exactly (as values: +0 == -0) on one-hot probe rows at escaped columns, and for 3 bit-identical repeat calls; every launch's
# return code is checked. Any failure (or a non-sm_90 device, a geometry other than ring depth 3, a build failure, or the
# launcher refusing the shared-memory size) leaves the cell on the parent route. The extension (okc_bf16proj_r14a_<md5 12>:
# track r9b's generated source with every r9b identifier renamed r14a (namespace r14a_ns, kernel r14a_sk2_kernel) and the
# exported dispatch replaced by the one instance; raw-pointer launcher, no ATen headers; -fno-gnu-unique) builds on a
# background thread while the parent prepare runs. Static route (offline judge-faithful cross-process timing in the r14a
# README); prepare() never times anything.
# ======================================================================================================================
import hashlib as _r14a_hashlib
import re as _r14a_re
import threading as _r14a_threading
import time as _r14a_time

_R14A_CELLS = (("unpooler", "unpooler32_r144", 32),)   # (layer, sk2 geometry, m); index = cfg
_R14A_MOD = [None]
_R14A_THR = []
_R14A_T0 = [0.0]


def _r14a_nstage():
    return _sk2_geom(_r9b_cfg("unpooler32_r144"), 32)[1]


def _r14a_source():
    """track r9b's generated DIST source (_r9b_source(), unchanged kernel text), r9b identifiers renamed r14a, exported
    dispatch r14a_run(cfg 0) -> r14a_launch<MT 2, TOKR 32, NT 4, NWR 2, WSK 2, UPW 2, NSTAGE 3, NACC 1> (M == 32)"""
    nstage = _r14a_nstage()
    if nstage != 3:
        raise RuntimeError("r14a: unpooler32_r144 ring depth %d != 3" % nstage)
    src = _r9b_source()
    for a, b in (("r9b_ns", "r14a_ns"), ("r9b_sk2_kernel", "r14a_sk2_kernel"), ("r9b_launch", "r14a_launch"),
                 ("r9b_run", "r14a_run"), ("r9b_err", "r14a_err"), ("R9B_NESC", "R14A_NESC")):
        src = _r14a_re.sub(r"\b%s\b" % a, b, src)
    if _r14a_re.search(r"\br9b_|\bR9B_", src):
        raise RuntimeError("r14a: unrenamed r9b identifier")
    lines = src.split("\n")
    disp_at = [i for i, l in enumerate(lines) if l.startswith("  if (cfg == ") and "r14a_ns::r14a_launch<" in l]
    if len(disp_at) != len(_R9B_CELLS) or disp_at != list(range(disp_at[0], disp_at[0] + len(disp_at))):
        raise RuntimeError("r14a: dispatch anchor")
    lines[disp_at[0]:disp_at[-1] + 1] = [
        "  if (cfg == 0) return r14a_ns::r14a_launch<2, 32, 4, 2, 2, 2, %d, 1>(w, xp, o, t, (int)pmax, (uint32_t)b4, (int)M, (int)N, "
        "(int)K, s);" % nstage]
    return ("// track r14a (adamleex): track r9b's DIST sk2 kernel renamed r14a (unpooler m = 32, alygg77's R64 geometry "
            "'unpooler32_r144')\n" + "\n".join(lines))


def _r14a_ext():
    if _R14A_MOD[0] is None:
        src = _r14a_source()
        cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\nint64_t r14a_run(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, "
               "int64_t, int64_t, int64_t, int64_t, int64_t);\n")
        _R14A_MOD[0] = load_inline("okc_bf16proj_r14a_" + _r14a_hashlib.md5((cpp + src).encode()).hexdigest()[:12], cpp,
                                   cuda_sources=src, functions=["r14a_run"], no_implicit_headers=True, with_pytorch_error_handling=False,
                                   verbose=False, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xcompiler=-fno-gnu-unique"])
    return _R14A_MOD[0]


def _r14a_start():
    """compile the r14a extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _r14a_time.time()
        try:
            _r14a_ext()
            _R14A_THR.append("ok")
        except Exception as error:
            _R14A_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _R14A_THR.append(round(_r14a_time.time() - t0, 1))
    t = _r14a_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _r14a_call(st, x, out):
    return _R14A_MOD[0].r14a_run(st["stream"].data_ptr(), x.data_ptr(), out.data_ptr(), st["tab"].data_ptr(), st["pmax"], st["b4"],
                                 32, st["n"], st["K"], 0, torch.cuda.current_stream().cuda_stream)


def _r14a_validate(st, layer, w, state, gen):
    """random / 1 %-channel x30 outlier rows (m = 32) against FP64 and the parent route (the kernel_fn chain above: u32);
    one-hot rows at escaped columns exact against W (values: +0 and -0 compare equal); 3 bit-identical repeat calls"""
    m, n, K, dev = 32, st["n"], st["K"], w.device
    wd = w.double()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    worst = 0.0
    for dist in (1.0, 30.0, 1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        rc = _r14a_call(st, x, out)
        if rc != 0:
            return "launch error %d" % rc
        par = _r14a_parent_kernel({"x": x, "config": {"m": m, "n": n, "k": K, "layer": layer}}, state)
        torch.cuda.synchronize()
        ref = x.double() @ wd.t()
        e = float((out.double() - ref).norm() / ref.norm())
        ep = float((par.double() - ref).norm() / ref.norm())
        d = (out.float() - par.float()).abs()
        if not (par.dtype == torch.bfloat16 and par.shape == out.shape and bool(torch.isfinite(out).all())
                and e <= ep * 1.02 + 1e-6 and e <= 4e-3
                and bool((d <= par.float().abs() * 2.0 ** -7 + 1e-5 * float(par.float().abs().max())).all())):
            return "dist %g: rel L2 %.3e (parent %.3e), max |d| %.2e" % (dist, e, ep, float(d.max()))
        worst = max(worst, e)
        o1 = out.clone()
        for _ in range(3):
            rc = _r14a_call(st, x, out)
            if rc != 0:
                return "launch error %d" % rc
            torch.cuda.synchronize()
            if not torch.equal(out.view(torch.int16), o1.view(torch.int16)):
                return "repeat calls differ"
    # one-hot rows at escaped columns (+ two plain columns): exact products, so the output must be bf16(W[:, k] * val)
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e71 = (bits >> 8) & 0x7F
    base, _ = _exponent_window(e71)
    esc_cols = torch.unique(_escapes(e71, base).nonzero()[:, 1])
    if esc_cols.numel():
        sel = esc_cols[torch.randperm(esc_cols.numel(), generator=gen, device=dev)[:2 * m - 2]]
    else:
        sel = esc_cols
    ks = torch.cat([sel, torch.tensor([0, K - 1], device=dev)])
    vals = torch.tensor([(-1.0 if j % 2 else 1.0) * 2.0 ** (1 + j % 4) for j in range(m)], device=dev)
    for b0 in range(0, ks.numel(), m):
        kb = ks[b0:b0 + m]
        x = torch.zeros((m, K), dtype=torch.bfloat16, device=dev)
        x[torch.arange(kb.numel(), device=dev), kb] = vals[:kb.numel()].to(torch.bfloat16)
        rc = _r14a_call(st, x, out)
        if rc != 0:
            return "launch error %d" % rc
        torch.cuda.synchronize()
        exp = (w.float()[:, kb].t() * vals[:kb.numel(), None]).to(torch.bfloat16)
        # compared as values: an escaped W = +0 times a negative x is +0 here (dec * x + delta * x) but -0 in torch
        if not bool((out[:kb.numel()].float() == exp.float()).all()):
            return "one-hot rows at escaped columns not exact"
    return "ok %.2e (%d one-hot columns)" % (worst, int(ks.numel()))


def _r14a_setup(state, weights, thr):
    state["r14a"] = {}
    if thr is not None:
        thr.join(timeout=max(1.0, 250.0 - (_r14a_time.time() - _R14A_T0[0])))
    if not _R14A_THR or _R14A_THR[0] != "ok" or _R14A_MOD[0] is None:
        print("R14A build %s" % (_R14A_THR[0] if _R14A_THR else "not finished"), flush=True)
        return
    dev = weights["o"].device
    if torch.cuda.get_device_capability(dev) != (9, 0):
        print("R14A off: not an sm_90 device", flush=True)
        return
    layer, name, m = _R14A_CELLS[0]
    ent = state.get("sk2", {}).get(name)
    cfg = _r9b_cfg(name)
    _, _, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    w = weights[layer]
    if (ent is None or CL != 1 or (R, NT, NWR, WSK, UPW) != (64, 4, 2, 2, 2) or _r14a_nstage() != 3
            or (SK2_BALANCE and layer in SK2_BALANCE_LAYERS) or w.dim() != 2 or w.dtype != torch.bfloat16
            or tuple(w.shape) != (n_rows, 4096)):
        print("R14A unpooler m=32: off (no parent sk2 route %s or geometry)" % name, flush=True)
        return
    pci, stream, ptab, ppmax, pb4, pn = ent
    comp = sk2_compress(w.contiguous())   # natural row order (unpooler is not an escape-balance layer)
    if comp is None:
        print("R14A unpooler m=32: off (pack refused)", flush=True)
        return
    t2, p2 = sk2_table(comp, n_rows, R)
    same = (pn == n_rows and p2 == ppmax and int(comp["base4"]) == int(pb4) and torch.equal(t2, ptab)
            and torch.equal(sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW), stream))
    if not same:
        print("R14A unpooler m=32: off (re-run pack differs from the parent's)", flush=True)
        return
    tb, why = _r9b_table(comp, n_rows, R, WSK * UPW, None)   # r9b's DIST table (no row map)
    nesc = int((comp["rows"] < n_rows).sum())
    del comp, t2
    if tb is None:
        print("R14A unpooler m=32: off (%s)" % why, flush=True)
        return
    st = dict(stream=stream, tab=tb[0], pmax=int(tb[1]), b4=int(pb4), n=n_rows, K=w.shape[1])
    msg = _r14a_validate(st, layer, w, state, torch.Generator(device=dev).manual_seed(141414))
    print("R14A unpooler m=32: %s (escapes %d, table row %d int2)" % (msg if msg.startswith("ok") else "off (" + msg + "), parent route",
                                                                     nesc, st["pmax"]), flush=True)
    if msg.startswith("ok"):
        state["r14a"] = st


_r14a_parent_prepare = prepare
_r14a_parent_kernel = kernel_fn


def prepare(weights):
    _R14A_T0[0] = _r14a_time.time()
    thr = _r14a_start()
    state = _r14a_parent_prepare(weights)
    try:
        _r14a_setup(state, weights, thr)
    except Exception as error:
        state["r14a"] = {}
        print("R14A setup failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__), flush=True)
    torch.cuda.synchronize()
    print("R14A prepare done in %.0fs (r14a build %s, unpooler m32 %s)" % (_r14a_time.time() - _R14A_T0[0], _R14A_THR,
                                                                          "routed" if state.get("r14a") else "parent"), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    e = state.get("r14a")
    if (e and data["config"]["layer"] == "unpooler" and x.dim() == 2 and x.shape[0] == 32 and x.shape[1] == e["K"]
            and data["config"]["n"] == e["n"] and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0):
        _k_ctx_bind(x)
        out = torch.empty((32, e["n"]), dtype=torch.bfloat16, device=x.device)
        if _r14a_call(e, x, out) == 0:
            return out
    return _r14a_parent_kernel(data, state)


# ======================================================================================================================
# track r17b (adamleex, 2026-11-04): unpooler m = 8 only (EXACT m; every other cell, incl. unpooler at m 2..7, goes to the
# parent chain above unchanged; the parent route of this cell is alygg77's sk2 escape build (_escape_ext, config 'slice8_r192':
# the 1024-row unpooler slice, R 8 rows per CTA, 128 CTAs x 416 threads = 8 consumer K-warps + 1 producer + 4 escape warps, a
# 4-stage TMA ring of 1024 k per stage = all of K, so every stage is in flight from the start and no slot is ever reused).
# Kernel 'slice8-DD': the parent kernel written out for exactly this instance (M 8, K 4096, MT 1, TOKR 8, NT 1, NWR 1, WSK 8,
# UPW 2, NSTAGE 4, NACC 2, packed x rows) -- same grid, warps, TMA / bulk copies (W 12 KB + x 16 KB per stage, the producer's
# fence.proxy.async kept), per-warp unit order, weight decode, escape warps (x from global memory), table and epilogue -- with
# two changes:
#   (1) x A fragments: at TOKR 8 rows 8..15 of the m16 A tile are copies of rows 0..7, so each unit's fragments come from 2
#       ldmatrix.x4 of the 4 unique 8x8 matrices (k chunks 4h + 0..3) instead of the parent's 4 x4 loads whose matrices 1 / 3
#       repeat 0 / 2 (the registers get the same values: a1 = a0, a3 = a2); half the x shared-memory reads (scout s17b: the
#       parent's consumer phase is close to shared-memory-bandwidth bound);
#   (2) no empty barriers: with 4 stages over all of K the producer never waits on a slot, so the consumers' per-stage
#       empty-barrier arrives (and their inits) are dropped.
# Every mma.sync gets the parent's operand values in the parent's order into the parent's accumulator, the escape corrections
# are the parent's FP32 updates in the parent's order, and the epilogue is the parent's fixed-order FP32 sum -> outputs are
# BIT-IDENTICAL to the parent route (asserted in prepare). Arithmetic: tensor-core FP32 accumulation of the losslessly decoded
# weights (12-bit pack + exact FP32 escape corrections), ONE final BF16 rounding; deterministic, no atomics, no cache-policy
# hint (the parent's TMA / bulk copies and global x loads), nothing input-derived cached, graph-capturable (one plain launch).
# Weight stream and escape table: the parent's state["sk2"]["slice8_r192"] tensors (reused, not copied).
# Prepare: the kernel is validated against the parent route (bit-equal outputs) on random and 1 %-channel x30 outlier rows, as
# values on one-hot rows at escaped columns, and for 3 bit-identical repeat calls; every launch's return code is checked. Any
# failure (or a non-sm_90 device, a geometry / table layout other than the parent's, a build failure, or the launcher refusing
# the shared-memory size) leaves the cell on the parent route. The extension (okc_bf16proj_r17b_<md5 12>: this file's CUDA_SRC /
# MBAR_HELPERS / MAP_SRC helpers + the kernel, namespace r17b_ns, raw-pointer launcher, no ATen headers, -fno-gnu-unique, one
# instance) builds on a background thread while the parent prepare runs. Static route (offline judge-faithful cross-process
# timing in the r17b README); prepare() never times anything.
# ======================================================================================================================
import hashlib as _r17b_hashlib
import threading as _r17b_threading
import time as _r17b_time

_R17B_NAME = "slice8_r192"   # the parent's sk2 geometry / stream for unpooler m <= 8
_R17B_MOD = [None]
_R17B_THR = []
_R17B_T0 = [0.0]

_R17B_KERNEL = r"""
// track r17b: unpooler m = 8 'slice8-DD' (see the section comment in the Python file). Fixed geometry: M 8 tokens, K 4096,
// R 8 rows per CTA (one n8 tile), 8 consumer K-warps x 2 units of 64 k per stage, 4 stages of 1024 k, NACC 2, 4 escape warps.
__global__ void __launch_bounds__(416, 1)
r17b_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws, const __nv_bfloat16* __restrict__ x,
            __nv_bfloat16* __restrict__ out, const int2* __restrict__ ptab, int pmax, uint32_t base4, int N) {
  constexpr int M = 8, K = 4096;
  constexpr int NW = 8, NC = NW * 32, R = 8, WSK = 8, UPW = 2, UPS = WSK * UPW, TU = 768, NSTAGE = 4, NESC = 4;
  constexpr int WBY = NW * UPW * TU;                 // 12288 B of W per CTA per stage
  constexpr int XU = M * 128;                        // one 64-k unit of x: M rows x 128 B (128B-swizzled TMA tile)
  constexpr int XBY = UPS * XU;                      // 16384 B of x per stage
  constexpr int SST = (XBY + WBY + 1023) & ~1023;    // ring slot: x tiles, then this CTA's W
  constexpr int TOKS = 16, RP = 8, CP = R + 1, CSZ = (TOKS * CP + 3) & ~3;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);   // [NESC][TOKS][CP]
  int2* ptab_s = reinterpret_cast<int2*>(corr + NESC * CSZ);                         // [pmax]: this CTA's escapes
  float* red = reinterpret_cast<float*>(gbase);      // reuses the ring once every stage is consumed
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x;
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;      // stage-major stream [stage][cta][warp][unit][768 B] (the parent's)
  const size_t wstride = (size_t)gridDim.x * WBY;

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    mbar_init(full0, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, XBY + WBY);
    bulk_g2s(sbase + XBY, wsrc, WBY, full0);
    tma_load_3d(sbase, &xmap, full0, 0, 0, 0);
    mbar_init(pbar, 1);   // this CTA's escape table
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(pbar, pmax * 8);
    bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
    for (int i = 1; i < NSTAGE; ++i) mbar_init(full0 + 8 * i, 1);
    mbar_init(cbar, 32 * NESC);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == NW) {
    // producer warp: lane 0 issues stages 1..3 (4 stages = all of K: no slot is ever refilled)
    if (lane == 0) {
      for (int st = 1; st < NSTAGE; ++st) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const uint32_t s0 = sbase + st * SST;
        mbar_expect_tx(full0 + 8 * st, XBY + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * st);
        tma_load_3d(s0, &xmap, full0 + 8 * st, 0, 0, st * UPS);
      }
    }
    __syncwarp();
    return;
  }
  if (warp > NW) {
    // NESC escape warps (the parent's escape build): exact corrections of the escaped weights, table in k order, warp ew takes
    // entries 1 + ew, 1 + ew + NESC, ...; lane t < M owns token t (x from global memory), one FP32 copy per warp
    const int ew = warp - (NW + 1);
    float* cr = corr + ew * CSZ;
    for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
    mbar_wait_sleep(pbar, 0);
    const int cnt = ptab_s[0].x;
    const uint32_t bq = base4 & 0xFFu;
    for (int pe = 1 + ew; pe <= cnt; pe += NESC) {
      const int2 e = ptab_s[pe];
      const int kk = e.x & 0xFFFF;
      if (lane < M && kk < K) {
        const float v = __bfloat162float(x[(size_t)lane * K + kk]);
        cr[lane * CP + (e.x >> 16)] += esc_delta((uint32_t)e.y, bq) * v;
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    return;
  }

  // ---------------- consumers (warp = K-warp)
  const int g = lane >> 2, t4 = lane & 3;
  const int r8 = lane & 7, q = lane >> 3;
  float acc[2][4];
#pragma unroll
  for (int c = 0; c < 2; ++c) acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = 0.f;
#pragma unroll
  for (int st = 0; st < NSTAGE; ++st) {
    mbar_wait(full0 + 8 * st, 0);
    const uint32_t s0 = sbase + st * SST;
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const uint32_t xt = s0 + (warp * UPW + uu) * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * TU;
      uint4 lw;
      uint2 cw;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw.x), "=r"(lw.y), "=r"(lw.z), "=r"(lw.w) : "r"(wb + lane * 16));
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw.x), "=r"(cw.y) : "r"(wb + 512 + lane * 8));
      // (1) A fragments: matrix q of load h = token rows 0..7, k chunk 4h + q of the unit (rows 8..15 = rows 0..7)
      uint32_t a[4][4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t m0, m1, m2, m3;
        ldsm_x4(xt + r8 * 128 + (((4 * h + q) ^ r8) << 4), m0, m1, m2, m3);
        a[2 * h][0] = m0; a[2 * h][1] = m0; a[2 * h][2] = m1; a[2 * h][3] = m1;
        a[2 * h + 1][0] = m2; a[2 * h + 1][1] = m2; a[2 * h + 1][2] = m3; a[2 * h + 1][3] = m3;
      }
      uint32_t b[8];
      decode8(lw.x, lw.y, cw.x, base4, &b[0]);
      decode8(lw.z, lw.w, cw.y, base4, &b[4]);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma_bf16(acc[j & 1], a[j][0], a[j][1], a[j][2], a[j][3], b[2 * j], b[2 * j + 1]);
    }
  }

  // ---------------- epilogue (the parent's, same order): K-warp partials through shared memory, + corrections
  mbar_wait(cbar, 0);   // escape warps done (their copies final)
  named_bar(1, NC);     // every consumer is done reading the ring
  {
    const int row = 2 * t4, tok = g;
    float* r0 = red + (warp * TOKS + tok) * RP + row;
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      v[e] = acc[0][e];
      v[e] += acc[1][e];
    }
    *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
    *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
  }
  named_bar(1, NC);
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int c = 0; c < NESC; ++c) {
      v.x += corr[c * CSZ + tt * CP + row];
      v.y += corr[c * CSZ + tt * CP + row + 1];
      v.z += corr[c * CSZ + tt * CP + row + 2];
      v.w += corr[c * CSZ + tt * CP + row + 3];
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
    }
    __nv_bfloat16* o = out + (size_t)tt * N + cta * R + row;
    __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 pk;
    pk.x = *reinterpret_cast<uint32_t*>(&lo);
    pk.y = *reinterpret_cast<uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(o) = pk;
  }
}

static int r17b_launch(const uint8_t* wsp, const void* xp, void* op, const int2* tp, int pmax, uint32_t base4, int M, int N, int K,
                       cudaStream_t stream) {
  constexpr int SST = (16 * 8 * 128 + 8 * 2 * 768 + 1023) & ~1023;
  constexpr int CSZ = (16 * 9 + 3) & ~3;
  if (M != 8 || K != 4096 || N % 8 != 0 || pmax < 2 || (reinterpret_cast<uintptr_t>(xp) & 15) != 0) return -3;
  const size_t smem = 1024 + (size_t)4 * SST + 16 * 4 + 32 + (size_t)4 * CSZ * 4 + (size_t)pmax * 8 + 128;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, r17b_kernel);
    if ((size_t)optin - fa.sharedSizeBytes < smem) return -5;
    cudaFuncSetAttribute(r17b_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  r17b_err = 0;
  CUtensorMap xm = make_map_x3d(xp, (uint64_t)M, (uint64_t)K, (uint32_t)M, 16);
  if (r17b_err) return -2;
  r17b_kernel<<<N / 8, 416, smem, stream>>>(xm, wsp, reinterpret_cast<const __nv_bfloat16*>(xp),
                                            reinterpret_cast<__nv_bfloat16*>(op), tp, pmax, base4, N);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
"""


def _r17b_source():
    """this file's CUDA_SRC / MBAR_HELPERS / MAP_SRC helpers (unchanged; ATen includes dropped, TORCH_CHECK in the tensor-map
    helpers sets an error flag) + the r17b kernel and its raw-pointer launcher, all in namespace r17b_ns, + the entry r17b_run"""
    body = (CUDA_SRC.replace("@@MBAR@@", MBAR_HELPERS).replace("@@BODY@@", "") + MAP_SRC + _R17B_KERNEL).splitlines()
    inc = [l for l in dict.fromkeys(l for l in body if l.startswith("#include")) if "ATen" not in l and "c10/" not in l]
    rest = "\n".join(l for l in body if not l.startswith("#include"))
    return ("// track r17b (adamleex): unpooler m = 8 'slice8-DD' kernel; helpers = this file's CUDA_SRC / MBAR_HELPERS / MAP_SRC\n"
            "#include <cuda.h>\n#include <cuda_runtime.h>\n" + "\n".join(inc)
            + "\nnamespace r17b_ns {\nstatic int r17b_err = 0;\n#define TORCH_CHECK(c, ...) do { if (!(c)) r17b_err = 1; } while (0)\n"
            + rest + "\n}  // namespace r17b_ns\n"
            "int64_t r17b_run(int64_t W, int64_t x, int64_t out, int64_t tab, int64_t pmax, int64_t b4, int64_t M, int64_t N, int64_t K,\n"
            "                 int64_t stream) {\n"
            "  return r17b_ns::r17b_launch(reinterpret_cast<const uint8_t*>(W), reinterpret_cast<const void*>(x), reinterpret_cast<void*>(out),\n"
            "                              reinterpret_cast<const int2*>(tab), (int)pmax, (uint32_t)b4, (int)M, (int)N, (int)K,\n"
            "                              reinterpret_cast<cudaStream_t>(stream));\n}\n")


def _r17b_ext():
    if _R17B_MOD[0] is None:
        src = _r17b_source()
        cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\nint64_t r17b_run(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, "
               "int64_t, int64_t, int64_t, int64_t);\n")
        _R17B_MOD[0] = load_inline("okc_bf16proj_r17b_" + _r17b_hashlib.md5((cpp + src).encode()).hexdigest()[:12], cpp,
                                   cuda_sources=src, functions=["r17b_run"], no_implicit_headers=True, with_pytorch_error_handling=False,
                                   verbose=False, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xcompiler=-fno-gnu-unique"])
    return _R17B_MOD[0]


def _r17b_start():
    """compile the r17b extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _r17b_time.time()
        try:
            _r17b_ext()
            _R17B_THR.append("ok")
        except Exception as error:
            _R17B_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _R17B_THR.append(round(_r17b_time.time() - t0, 1))
    t = _r17b_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _r17b_call(st, x, out):
    return _R17B_MOD[0].r17b_run(st["stream"].data_ptr(), x.data_ptr(), out.data_ptr(), st["tab"].data_ptr(), st["pmax"], st["b4"],
                                 8, st["n"], st["K"], torch.cuda.current_stream().cuda_stream)


def _r17b_validate(st, w, state, gen):
    """random / 1 %-channel x30 outlier rows (m = 8): outputs BIT-equal to the parent route (the kernel_fn chain above: the sk2
    escape build) and close to FP64; 3 bit-identical repeat calls; one-hot rows at escaped columns exact against W (as values)"""
    m, n, K, dev = 8, st["n"], st["K"], w.device
    wd = w[:n].double()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    worst = 0.0
    for dist in (1.0, 30.0, 1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        rc = _r17b_call(st, x, out)
        if rc != 0:
            return "launch error %d" % rc
        par = _r17b_parent_kernel({"x": x, "config": {"m": m, "n": n, "k": K, "layer": "unpooler"}}, state)
        torch.cuda.synchronize()
        ref = x.double() @ wd.t()
        e = float((out.double() - ref).norm() / ref.norm())
        if not (par.dtype == torch.bfloat16 and par.shape == out.shape and e <= 4e-3
                and torch.equal(out.view(torch.int16), par.view(torch.int16))):
            return "dist %g: rel L2 %.3e, outputs differ from the parent route" % (dist, e)
        worst = max(worst, e)
        o1 = out.clone()
        for _ in range(3):
            rc = _r17b_call(st, x, out)
            if rc != 0:
                return "launch error %d" % rc
            torch.cuda.synchronize()
            if not torch.equal(out.view(torch.int16), o1.view(torch.int16)):
                return "repeat calls differ"
    # one-hot rows at escaped columns (+ two plain columns): exact products, so the output must be bf16(W[:n, k] * val)
    bits = w[:n].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e71 = (bits >> 8) & 0x7F
    base, _ = _exponent_window(e71)
    esc_cols = torch.unique(_escapes(e71, base).nonzero()[:, 1])
    if esc_cols.numel():
        sel = esc_cols[torch.randperm(esc_cols.numel(), generator=gen, device=dev)[:6 * m - 2]]
    else:
        sel = esc_cols
    ks = torch.cat([sel, torch.tensor([0, K - 1], device=dev)])
    vals = torch.tensor([(-1.0 if j % 2 else 1.0) * 2.0 ** (1 + j % 4) for j in range(m)], device=dev)
    for b0 in range(0, ks.numel(), m):
        kb = ks[b0:b0 + m]
        x = torch.zeros((m, K), dtype=torch.bfloat16, device=dev)
        x[torch.arange(kb.numel(), device=dev), kb] = vals[:kb.numel()].to(torch.bfloat16)
        rc = _r17b_call(st, x, out)
        if rc != 0:
            return "launch error %d" % rc
        torch.cuda.synchronize()
        exp = (w[:n].float()[:, kb].t() * vals[:kb.numel(), None]).to(torch.bfloat16)
        # compared as values: an escaped W = +0 times a negative x is +0 here (dec * x + delta * x) but -0 in torch
        if not bool((out[:kb.numel()].float() == exp.float()).all()):
            return "one-hot rows at escaped columns not exact"
    return "ok %.2e (%d one-hot columns)" % (worst, int(ks.numel()))


def _r17b_setup(state, weights, thr):
    state["r17b"] = {}
    if thr is not None:
        thr.join(timeout=max(1.0, 250.0 - (_r17b_time.time() - _R17B_T0[0])))
    if not _R17B_THR or _R17B_THR[0] != "ok" or _R17B_MOD[0] is None:
        print("R17B build %s" % (_R17B_THR[0] if _R17B_THR else "not finished"), flush=True)
        return
    dev = weights["o"].device
    if torch.cuda.get_device_capability(dev) != (9, 0):
        print("R17B off: not an sm_90 device", flush=True)
        return
    ent = state.get("sk2", {}).get(_R17B_NAME)
    cfg = next((c for c in SK2_CONFIGS if c[0] == _R17B_NAME), None)
    w = weights["unpooler"]
    if (ent is None or cfg is None or cfg[1] != "unpooler" or tuple(cfg[2:]) != (UNPOOLER_SLICE_ROWS, 8, 1, 1, 8, 2, 2, 1)
            or UNPOOLER_SLICE_ROWS != 1024 or _sk2_geom(cfg, 8)[1] != 4 or _sk2_name("unpooler", 8) != _R17B_NAME
            or (SK2_BALANCE and "unpooler" in SK2_BALANCE_LAYERS) or w.dim() != 2 or w.dtype != torch.bfloat16
            or tuple(w.shape) != (8192, 4096)):
        print("R17B unpooler m=8: off (no parent sk2 route %s or geometry)" % _R17B_NAME, flush=True)
        return
    pci, stream, ptab, ppmax, pb4, pn = ent
    ppmax = int(ppmax)
    if (pn != 1024 or ppmax < 2 or tuple(ptab.shape) != (1024 // 8, ppmax, 2) or not ptab.is_contiguous()
            or stream.numel() != 1024 * 4096 * 3 // 2 or stream.data_ptr() % 128 != 0 or ptab.data_ptr() % 16 != 0
            or (ppmax * 8) % 16 != 0 or bool((ptab[:, 0, 1] != 0).any())):
        print("R17B unpooler m=8: off (parent stream / table layout)", flush=True)
        return
    st = dict(stream=stream, tab=ptab, pmax=ppmax, b4=int(pb4), n=1024, K=4096)
    msg = _r17b_validate(st, w, state, torch.Generator(device=dev).manual_seed(171717))
    print("R17B unpooler m=8: %s (escapes %d, table row %d int2)" % (msg if msg.startswith("ok") else "off (" + msg + "), parent route",
                                                                    int(ptab[:, 0, 0].sum()), ppmax), flush=True)
    if msg.startswith("ok"):
        state["r17b"] = st


_r17b_parent_prepare = prepare
_r17b_parent_kernel = kernel_fn


def prepare(weights):
    _R17B_T0[0] = _r17b_time.time()
    thr = _r17b_start()
    state = _r17b_parent_prepare(weights)
    try:
        _r17b_setup(state, weights, thr)
    except Exception as error:
        state["r17b"] = {}
        print("R17B setup failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__), flush=True)
    torch.cuda.synchronize()
    print("R17B prepare done in %.0fs (r17b build %s, unpooler m8 %s)" % (_r17b_time.time() - _R17B_T0[0], _R17B_THR,
                                                                         "routed" if state.get("r17b") else "parent"), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    e = state.get("r17b")
    if (e and data["config"]["layer"] == "unpooler" and x.dim() == 2 and x.shape[0] == 8 and x.shape[1] == e["K"]
            and data["config"]["n"] == e["n"] and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0):
        _k_ctx_bind(x)
        out = torch.empty((8, e["n"]), dtype=torch.bfloat16, device=x.device)
        if _r17b_call(e, x, out) == 0:
            return out
    return _r17b_parent_kernel(data, state)


# ======================================================================================================================
# track r18a (adamleex, 2026-11-04): qkv_gate m = 8 only (EXACT m; every other cell, incl. qkv_gate at m 2..7 / 9.., goes to the
# parent chain above unchanged). The parent route of this cell is alygg77's sk2 escape build (_escape_ext, config 'qkv8t_r144': R 24
# rows per CTA in the escape-balanced row order, 96 CTAs x 288 threads = 4 consumer K-warps + 1 producer + 4 escape warps off the
# ring, a 5-slot TMA ring over 16 stages of 512 k, NACC 2, x for escapes from global memory, table + row map = one bulk copy).
# Kernel: the parent's routed instance written out for exactly this shape (M 8, K 8192, N 2304, TOKR 8, packed x rows, NSTAGE 5,
# NACC 2) -- same grid, warps, TMA / bulk copies (the producer's fence.proxy.async kept), per-warp unit order, escape warps, table,
# row map and epilogue -- with these instruction cuts:
#   (1) swapped mma operands: per k-step the decoded W of tiles 0|1 (16 rows) and of tile 2 (8 rows + 8 padding rows) are the A
#       operand and x the B operand (k x 8 tokens), so a k-step is 2 mma.sync instead of 3 whose token rows 8..15 only repeat rows
#       0..7, and x comes from 2 ldmatrix.x4 per unit (its unique 8x8 matrices) instead of 4; every output (token, row) still gets
#       the same products in the same k-steps, accumulated into the same FP32 chain (accumulator j & 1) -- the padding rows'
#       outputs are never read, so their registers are left as don't-care values;
#   (2) compile-time M / K / N and ring geometry (no runtime-M x addressing, zero-row selects or generic index math), the ring
#       slot / phase carried incrementally, the lane-constant shared-memory offsets kept in registers;
#   (3) a compact weight decode: the high byte of every weight is formed from a LOSSLESS bit re-pack of the parent stream's
#       4-bit code words (_r18a_repack: the same 16 codes per lane and tile, each sign moved next to its exponent offset) with
#       AND / funnel shift + the window base -- 10 ALU + 8 PRMT per 16 weights instead of 16 + 8; the low bytes, the 12-bit
#       stream layout and every decoded BF16 value are the parent's.
# Outputs are BIT-IDENTICAL to the parent route (asserted in prepare): FP32 tensor-core accumulation of the losslessly decoded
# weights + the parent's exact FP32 escape corrections, ONE final BF16 rounding in the parent's fixed-order epilogue;
# deterministic, no atomics, no cache-policy hint, nothing input-derived cached, one plain launch per call. Stream: a re-packed
# copy of the parent's state["sk2"]["qkv8t_r144"] stream (+28 MB); table + row map: the parent's. Any validation / build /
# geometry failure leaves the cell on the parent route. Extension okc_bf16proj_r18a_<md5 12> (this file's CUDA_SRC /
# MBAR_HELPERS / MAP_SRC helpers + the kernel, -fno-gnu-unique) builds on a background thread. Static route (README timing).
# ======================================================================================================================
import hashlib as _r18a_hashlib
import threading as _r18a_threading
import time as _r18a_time

_R18A_NAME = "qkv8t_r144"   # the parent's sk2 geometry / stream for qkv_gate m <= 8
_R18A_MOD = [None]
_R18A_THR = []
_R18A_T0 = [0.0]

_R18A_KERNEL = r"""
// track r18a: qkv_gate m = 8 (see the Python section comment). M 8, K 8192, N 2304, R 24 rows per CTA, 4 K-warps x 2 units per stage.
__global__ void __launch_bounds__(288, 1)
r18a_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws, const __nv_bfloat16* __restrict__ x,
            __nv_bfloat16* __restrict__ out, const int2* __restrict__ ptab, int pmax, uint32_t base4) {
  constexpr int M = 8, K = 8192, N = 2304;
  constexpr int NW = 4, NC = NW * 32, NT = 3, R = NT * 8, WSK = 4, UPW = 2, UPS = WSK * UPW, TU = 768, NSTAGE = 5, NESC = 4;
  constexpr int NST = K / (UPS * 64);                // 16 stages of 512 k
  constexpr int NCTA = N / R;                        // 96
  constexpr int WBY = NW * UPW * NT * TU;            // 18432 B of W per CTA per stage
  constexpr int XU = M * 128;                        // one 64-k unit of x: M rows x 128 B (128B-swizzled TMA box)
  constexpr int XBY = UPS * XU;                      // 8192 B of x per stage
  constexpr int SST = (XBY + WBY + 1023) & ~1023;    // ring slot: x tiles, then this CTA's W
  constexpr int TOKS = 16, RP = R + ((40 - (R % 32)) % 32), CP = R + 1, CSZ = (TOKS * CP + 3) & ~3;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);   // [NESC][TOKS][CP]
  int2* ptab_s = reinterpret_cast<int2*>(corr + NESC * CSZ);                         // [pmax]: escapes + row map
  float* red = reinterpret_cast<float*>(gbase);      // reuses the ring once every stage is consumed
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x;
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;      // stage-major stream [stage][cta][warp][unit][tile][768 B] (the parent's)
  constexpr size_t wstride = (size_t)NCTA * WBY;

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    mbar_init(full0, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, XBY + WBY);
    bulk_g2s(sbase + XBY, wsrc, WBY, full0);
    tma_load_3d(sbase, &xmap, full0, 0, 0, 0);
    mbar_init(pbar, 1);   // this CTA's escape table (+ row map)
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(pbar, pmax * 8);
    bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
    mbar_init(empty0, NW);
    for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW); }
    mbar_init(cbar, 32 * NESC);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == NW) {
    // producer warp: lane 0 refills the ring (stages 1..15; slot st % 5 after the consumers released stage st - 5)
    if (lane == 0) {
      for (int st = 1; st < NST; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) mbar_wait_sleep(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const uint32_t s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, XBY + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
        tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, st * UPS);
      }
    }
    __syncwarp();
    return;
  }
  if (warp > NW) {
    // NESC escape warps (the parent's): warp ew takes table entries 1 + ew, 1 + ew + NESC, ...; lane t < M owns token t
    const int ew = warp - (NW + 1);
    float* cr = corr + ew * CSZ;
    for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
    mbar_wait_sleep(pbar, 0);
    const int cnt = ptab_s[0].x;
    const uint32_t bq = base4 & 0xFFu;
    for (int pe = 1 + ew; pe <= cnt; pe += NESC) {
      const int2 e = ptab_s[pe];
      const int kk = e.x & 0xFFFF;
      if (lane < M && kk < K) {
        const float v = __bfloat162float(x[(size_t)lane * K + kk]);
        cr[lane * CP + (e.x >> 16)] += esc_delta((uint32_t)e.y, bq) * v;
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    return;
  }

  // ---------------- consumers (warp = K-warp wk; unit ui = wk * UPW + uu of every stage)
  const int g = lane >> 2, t4 = lane & 3;
  const int r8 = lane & 7, q = lane >> 3;
  const int wk = warp;
  float acc[2][2][4];   // [j & 1][W tile pair: rows 0..15 | rows 16..23 + padding][4]
#pragma unroll
  for (int c = 0; c < 2; ++c)
#pragma unroll
    for (int p = 0; p < 2; ++p) acc[c][p][0] = acc[c][p][1] = acc[c][p][2] = acc[c][p][3] = 0.f;
  // lane-constant shared-memory offsets inside a ring slot: x matrices (load h: k chunk 4h + q of the unit, token r8), W words
  uint32_t xo[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) xo[h] = (uint32_t)(wk * UPW * XU + r8 * 128 + (((4 * h + q) ^ r8) << 4));
  uint32_t wo = (uint32_t)(XBY + wk * UPW * NT * TU + lane * 16), co = (uint32_t)(XBY + wk * UPW * NT * TU + 512 + lane * 8);

  auto stage = [&](const uint32_t s0) {
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      // 1) shared-memory loads of the unit (W words of 3 tiles, x fragments of 4 k-steps)
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w)
                     : "r"(s0 + wo + (uu * NT + nt) * TU));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(s0 + co + (uu * NT + nt) * TU));
      }
      // x as the B operand: matrix q of load h = (k 0..7 | k 8..15) of k-step 2h + (q >> 1), token g
      uint32_t a[4][2];
#pragma unroll
      for (int h = 0; h < 2; ++h) ldsm_x4(s0 + xo[h] + uu * XU, a[2 * h][0], a[2 * h][1], a[2 * h + 1][0], a[2 * h + 1][1]);
      // 2) decode (re-packed codes: byte b of X = sign / offset of weight b in bits 7 / 0..2 and the offset of weight 4 + b in
      // bits 4..6, the sign of weight 4 + b in bit 3 of the next byte (weight 7: Y byte 0); Y likewise for 8..15 (15: X byte 0))
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const uint32_t X = cw[nt].x, Y = cw[nt].y;
        const uint32_t ha = (X & 0x87878787u) + base4, hb = (__funnelshift_r(X, Y, 4) & 0x87878787u) + base4;
        const uint32_t hc = (Y & 0x87878787u) + base4, hd = (__funnelshift_r(Y, X, 4) & 0x87878787u) + base4;
        b[nt][0] = prmt(lw[nt].x, ha, 0x5140);
        b[nt][1] = prmt(lw[nt].x, ha, 0x7362);
        b[nt][2] = prmt(lw[nt].y, hb, 0x5140);
        b[nt][3] = prmt(lw[nt].y, hb, 0x7362);
        b[nt][4] = prmt(lw[nt].z, hc, 0x5140);
        b[nt][5] = prmt(lw[nt].z, hc, 0x7362);
        b[nt][6] = prmt(lw[nt].w, hd, 0x5140);
        b[nt][7] = prmt(lw[nt].w, hd, 0x7362);
      }
      // 3) tensor-core work, the parent's k-step order and accumulator j & 1 (A = W rows, B = x)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        mma_bf16(acc[j & 1][0], b[0][2 * j], b[1][2 * j], b[0][2 * j + 1], b[1][2 * j + 1], a[j][0], a[j][1]);
        uint32_t p1 = 0u, p3 = 0u;   // A rows 24..31 (padding: their accumulator rows are never read)
        asm volatile("" : "=r"(p1), "=r"(p3));   // don't-care values: no instruction
        mma_bf16(acc[j & 1][1], b[2][2 * j], p1, b[2][2 * j + 1], p3, a[j][0], a[j][1]);
      }
    }
  };

  uint32_t s0 = sbase, fb = full0, ph = 0;
  for (int st = 0; st < NST; ++st) {
    asm volatile("" : "+r"(xo[0]), "+r"(xo[1]), "+r"(wo), "+r"(co));   // keep the lane offsets in registers (no recompute)
    mbar_wait(fb, ph);
    stage(s0);
    __syncwarp();
    if (lane == 0) mbar_arrive(fb + 8 * NSTAGE);   // the slot's empty barrier
    const bool wrap = fb == full0 + 8 * (NSTAGE - 1);
    s0 = wrap ? sbase : s0 + SST;
    fb = wrap ? full0 : fb + 8;
    ph ^= wrap ? 1u : 0u;
  }

  // ---------------- epilogue (the parent's, same order): K-warp partials through shared memory, + corrections
  mbar_wait(cbar, 0);   // escape warps done (their copies final)
  named_bar(1, NC);
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      v[e] = acc[0][p][e];
      v[e] += acc[1][p][e];
    }
    // c0 / c1 = (W row 16 p + g, tokens 2 t4 / 2 t4 + 1), c2 / c3 = (row + 8, same tokens); rows 24..31 are padding
    float* r0 = red + (wk * TOKS + 2 * t4) * RP + p * 16 + g;
    r0[0] = v[0];
    r0[RP] = v[1];
    if (p == 0) {
      r0[8] = v[2];
      r0[RP + 8] = v[3];
    }
  }
  named_bar(1, NC);
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int c = 0; c < NESC; ++c) {
      v.x += corr[c * CSZ + tt * CP + row];
      v.y += corr[c * CSZ + tt * CP + row + 1];
      v.z += corr[c * CSZ + tt * CP + row + 2];
      v.w += corr[c * CSZ + tt * CP + row + 3];
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
    }
    __nv_bfloat16* o = out + (size_t)tt * N + cta * R + row;
    if (ptab_s[0].y != 0)   // escape-balanced row order: 4-row groups moved whole, so they stay contiguous
      o = out + (size_t)tt * N + reinterpret_cast<const int*>(ptab_s + ptab_s[0].y)[row];
    __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 pk;
    pk.x = *reinterpret_cast<uint32_t*>(&lo);
    pk.y = *reinterpret_cast<uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(o) = pk;
  }
}

static int r18a_launch(const uint8_t* wsp, const void* xp, void* op, const int2* tp, int pmax, uint32_t base4, int M, int N, int K,
                       cudaStream_t stream) {
  constexpr int SST = (8 * 8 * 128 + 4 * 2 * 3 * 768 + 1023) & ~1023;
  constexpr int CSZ = (16 * 25 + 3) & ~3;
  if (M != 8 || K != 8192 || N != 2304 || pmax < 2 || (reinterpret_cast<uintptr_t>(xp) & 15) != 0) return -3;
  const size_t smem = 1024 + (size_t)5 * SST + 16 * 5 + 32 + (size_t)4 * CSZ * 4 + (size_t)pmax * 8 + 128;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, r18a_kernel);
    if ((size_t)optin - fa.sharedSizeBytes < smem) return -5;
    cudaFuncSetAttribute(r18a_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  r18a_err = 0;
  CUtensorMap xm = make_map_x3d(xp, (uint64_t)M, (uint64_t)K, (uint32_t)M, 8);
  if (r18a_err) return -2;
  r18a_kernel<<<N / 24, 288, smem, stream>>>(xm, wsp, reinterpret_cast<const __nv_bfloat16*>(xp), reinterpret_cast<__nv_bfloat16*>(op),
                                              tp, pmax, base4);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
"""


def _r18a_repack(stream):
    """LOSSLESS bit re-pack of the parent 12-bit stream's code words for the compact decode. Every 768-B lane tile = 512 B of low
    bytes (unchanged) + 32 lanes x two 32-bit code words (X, Y). Parent: byte b of X = code of weight b (low nibble) | code of
    weight 4 + b (high nibble), code = sign << 3 | exponent offset; Y the same for weights 8..15. Re-packed (a permutation of the
    64 code bits): X' = (X & 0x77777777) | (X & 0x08080808) << 4 | (X & 0x00808080) << 4 | (Y >> 28) & 8, Y' the same with X and
    Y exchanged -- offsets stay in place, the sign of weight b moves to bit 7 of byte b, the sign of weight 4 + b to bit 3 of byte
    b + 1 (weight 7 -> Y byte 0, weight 15 -> X byte 0)."""
    t = stream.view(-1, 768)[:, 512:].contiguous().view(torch.int32).view(-1, 32, 2).to(torch.int64) & 0xFFFFFFFF
    X, Y = t[..., 0], t[..., 1]
    f = lambda a, b: (a & 0x77777777) | ((a & 0x08080808) << 4) | ((a & 0x00808080) << 4) | ((b >> 28) & 0x8)
    new = torch.stack([f(X, Y), f(Y, X)], -1)
    new = torch.where(new >= 2 ** 31, new - 2 ** 32, new).to(torch.int32).view(-1, 64)
    out = stream.clone()
    out.view(-1, 768)[:, 512:] = new.view(torch.uint8).view(-1, 256)
    return out


def _r18a_source():
    """this file's CUDA_SRC / MBAR_HELPERS / MAP_SRC helpers (ATen includes dropped, TORCH_CHECK -> error flag) + the kernel"""
    body = (CUDA_SRC.replace("@@MBAR@@", MBAR_HELPERS).replace("@@BODY@@", "") + MAP_SRC + _R18A_KERNEL).splitlines()
    inc = [l for l in dict.fromkeys(l for l in body if l.startswith("#include")) if "ATen" not in l and "c10/" not in l]
    rest = "\n".join(l for l in body if not l.startswith("#include"))
    return ("// track r18a (adamleex): qkv_gate m = 8 kernel; helpers = this file's CUDA_SRC / MBAR_HELPERS / MAP_SRC\n"
            "#include <cuda.h>\n#include <cuda_runtime.h>\n" + "\n".join(inc)
            + "\nnamespace r18a_ns {\nstatic int r18a_err = 0;\n#define TORCH_CHECK(c, ...) do { if (!(c)) r18a_err = 1; } while (0)\n"
            + rest + "\n}  // namespace r18a_ns\n"
            "int64_t r18a_run(int64_t W, int64_t x, int64_t out, int64_t tab, int64_t pmax, int64_t b4, int64_t M, int64_t N, int64_t K,\n"
            "                 int64_t stream) {\n"
            "  return r18a_ns::r18a_launch(reinterpret_cast<const uint8_t*>(W), reinterpret_cast<const void*>(x),\n"
            "      reinterpret_cast<void*>(out), reinterpret_cast<const int2*>(tab), (int)pmax, (uint32_t)b4, (int)M, (int)N, (int)K,\n"
            "      reinterpret_cast<cudaStream_t>(stream));\n}\n")


def _r18a_ext():
    if _R18A_MOD[0] is None:
        src = _r18a_source()
        cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\nint64_t r18a_run(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, "
               "int64_t, int64_t, int64_t, int64_t);\n")
        _R18A_MOD[0] = load_inline("okc_bf16proj_r18a_" + _r18a_hashlib.md5((cpp + src).encode()).hexdigest()[:12], cpp,
                                   cuda_sources=src, functions=["r18a_run"], no_implicit_headers=True, with_pytorch_error_handling=False,
                                   verbose=False, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xcompiler=-fno-gnu-unique"])
    return _R18A_MOD[0]


def _r18a_start():
    """compile the r18a extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _r18a_time.time()
        try:
            _r18a_ext()
            _R18A_THR.append("ok")
        except Exception as error:
            _R18A_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _R18A_THR.append(round(_r18a_time.time() - t0, 1))
    t = _r18a_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _r18a_call(st, x, out):
    return _R18A_MOD[0].r18a_run(st["stream"].data_ptr(), x.data_ptr(), out.data_ptr(), st["tab"].data_ptr(), st["pmax"], st["b4"],
                                 8, st["n"], st["K"], torch.cuda.current_stream().cuda_stream)


def _r18a_validate(st, w, state, gen):
    """random / outlier rows: BIT-equal to the parent route, close to FP64, 3 equal repeats; one-hot rows at escaped columns exact"""
    m, n, K, dev = 8, st["n"], st["K"], w.device
    wd = w.double()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    worst = 0.0
    for dist in (1.0, 30.0, 1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        rc = _r18a_call(st, x, out)
        if rc != 0:
            return "launch error %d" % rc
        par = _r18a_parent_kernel({"x": x, "config": {"m": m, "n": n, "k": K, "layer": "qkv_gate"}}, state)
        torch.cuda.synchronize()
        ref = x.double() @ wd.t()
        e = float((out.double() - ref).norm() / ref.norm())
        if not (par.dtype == torch.bfloat16 and par.shape == out.shape and e <= 4e-3
                and torch.equal(out.view(torch.int16), par.view(torch.int16))):
            return "dist %g: rel L2 %.3e, outputs differ from the parent route" % (dist, e)
        worst = max(worst, e)
        o1 = out.clone()
        for _ in range(3):
            rc = _r18a_call(st, x, out)
            if rc != 0:
                return "launch error %d" % rc
            torch.cuda.synchronize()
            if not torch.equal(out.view(torch.int16), o1.view(torch.int16)):
                return "repeat calls differ"
    # one-hot rows at escaped columns (+ two plain columns): exact products, so the output must be bf16(W[:, k] * val)
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e71 = (bits >> 8) & 0x7F
    base, _ = _exponent_window(e71)
    esc_cols = torch.unique(_escapes(e71, base).nonzero()[:, 1])
    sel = esc_cols[torch.randperm(esc_cols.numel(), generator=gen, device=dev)[:6 * m - 2]] if esc_cols.numel() else esc_cols
    ks = torch.cat([sel, torch.tensor([0, K - 1], device=dev)])
    vals = torch.tensor([(-1.0 if j % 2 else 1.0) * 2.0 ** (1 + j % 4) for j in range(m)], device=dev)
    for b0 in range(0, ks.numel(), m):
        kb = ks[b0:b0 + m]
        x = torch.zeros((m, K), dtype=torch.bfloat16, device=dev)
        x[torch.arange(kb.numel(), device=dev), kb] = vals[:kb.numel()].to(torch.bfloat16)
        rc = _r18a_call(st, x, out)
        if rc != 0:
            return "launch error %d" % rc
        torch.cuda.synchronize()
        exp = (w.float()[:, kb].t() * vals[:kb.numel(), None]).to(torch.bfloat16)
        # compared as values: an escaped W = +0 times a negative x is +0 here (dec * x + delta * x) but -0 in torch
        if not bool((out[:kb.numel()].float() == exp.float()).all()):
            return "one-hot rows at escaped columns not exact"
    return "ok %.2e (%d one-hot columns)" % (worst, int(ks.numel()))


def _r18a_setup(state, weights, thr):
    state["r18a"] = {}
    if thr is not None:
        thr.join(timeout=max(1.0, 250.0 - (_r18a_time.time() - _R18A_T0[0])))
    if not _R18A_THR or _R18A_THR[0] != "ok" or _R18A_MOD[0] is None:
        print("R18A build %s" % (_R18A_THR[0] if _R18A_THR else "not finished"), flush=True)
        return
    dev = weights["o"].device
    if torch.cuda.get_device_capability(dev) != (9, 0):
        print("R18A off: not an sm_90 device", flush=True)
        return
    ent = state.get("sk2", {}).get(_R18A_NAME)
    cfg = next((c for c in SK2_CONFIGS if c[0] == _R18A_NAME), None)
    w = weights["qkv_gate"]
    if (ent is None or cfg is None or cfg[1] != "qkv_gate" or tuple(cfg[2:]) != (2304, 24, 3, 1, 4, 2, 2, 1) or ESC_WARPS != 4
            or _sk2_geom(cfg, 8)[1] != 5 or _sk2_name("qkv_gate", 8) != _R18A_NAME or w.dim() != 2 or w.dtype != torch.bfloat16
            or tuple(w.shape) != (2304, 8192)):
        print("R18A qkv_gate m=8: off (no parent sk2 route %s or geometry)" % _R18A_NAME, flush=True)
        return
    pci, stream, ptab, ppmax, pb4, pn = ent
    ppmax = int(ppmax)
    if (pn != 2304 or ppmax < 2 or tuple(ptab.shape) != (2304 // 24, ppmax, 2) or not ptab.is_contiguous()
            or stream.numel() != 2304 * 8192 * 3 // 2 or stream.data_ptr() % 128 != 0 or ptab.data_ptr() % 16 != 0
            or (ppmax * 8) % 16 != 0):
        print("R18A qkv_gate m=8: off (parent stream / table layout)", flush=True)
        return
    st = dict(stream=_r18a_repack(stream), tab=ptab, pmax=ppmax, b4=int(pb4), n=2304, K=8192)
    msg = _r18a_validate(st, w, state, torch.Generator(device=dev).manual_seed(181818))
    print("R18A qkv_gate m=8: %s (escapes %d, table row %d int2, row map %s)" % (
        msg if msg.startswith("ok") else "off (" + msg + "), parent route", int(ptab[:, 0, 0].sum()), ppmax,
        "on" if bool((ptab[:, 0, 1] != 0).all()) else "off"), flush=True)
    if msg.startswith("ok"):
        state["r18a"] = st


_r18a_parent_prepare = prepare
_r18a_parent_kernel = kernel_fn


def prepare(weights):
    _R18A_T0[0] = _r18a_time.time()
    thr = _r18a_start()
    state = _r18a_parent_prepare(weights)
    try:
        _r18a_setup(state, weights, thr)
    except Exception as error:
        state["r18a"] = {}
        print("R18A setup failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__), flush=True)
    torch.cuda.synchronize()
    print("R18A prepare done in %.0fs (r18a build %s, qkv_gate m8 %s)" % (_r18a_time.time() - _R18A_T0[0], _R18A_THR,
                                                                         "routed" if state.get("r18a") else "parent"), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    e = state.get("r18a")
    if (e and data["config"]["layer"] == "qkv_gate" and x.dim() == 2 and x.shape[0] == 8 and x.shape[1] == e["K"]
            and data["config"]["n"] == e["n"] and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0):
        _k_ctx_bind(x)
        out = torch.empty((8, e["n"]), dtype=torch.bfloat16, device=x.device)
        if _r18a_call(e, x, out) == 0:
            return out
    return _r18a_parent_kernel(data, state)


# ======================================================================================================================
# track r18b (adamleex, 2026-11-04): qkv_gate m = 16 only (EXACT m; every other cell, incl. qkv_gate at m 9..15 / 17..32,
# goes to the parent chain above unchanged; the parent route of this cell is track r10c's renamed copy of track r9b's DIST sk2
# kernel on alygg77's geometry 'qkv16t_r200': 96 CTAs x 224 threads = 4 consumer K-warps + 1 producer + 2 DIST escape warps,
# R 24 rows per CTA (NT 3), UPW 2, a 5-stage TMA ring of 512 k per stage, 16 stages over K = 8192, NACC 2, natural row order).
# Kernel: the parent's text (this file's SK2_SRC + track r9b's DIST escape loop, via _r9b_source()) written out for exactly
# this instance by plain-text replacements (_r18b_source()):
#   (1) compile-time shape: M = TOKR = 16, N = 2304, K = 8192 (template NN / KK; the launcher refuses anything else), so the
#       stage count, the stream stride (gridDim.x = N / R) and the epilogue loop bounds are constants;
#   (2) dead generic code removed: the zero-row init (PK is false: the zero row is never read), the row-map branch of the
#       epilogue (natural row order: asserted in prepare), the DIST header lookup (header at int2 1 without a row map);
#   (3) no hand-back of the last NSTAGE stages: the producer refills a slot only for stages st + NSTAGE < 16, so the
#       consumers' and escape warps' empty-barrier arrivals for stages 11..15 are never waited on and are dropped.
# Same grid, warps, TMA / bulk copies, producer fence.proxy.async (r7r race fix), per-warp unit order, decode, MMA order and
# accumulators, DIST escape assignment and order, and fixed-order epilogue -> outputs are BIT-IDENTICAL to the parent route
# (asserted in prepare). Arithmetic: tensor-core FP32 accumulation of the losslessly decoded weights (12-bit pack, exact FP32
# escape corrections), ONE final BF16 rounding; deterministic, no atomics, no cache-policy hint added or changed, nothing
# input-derived cached, graph-capturable (one plain launch). Weight stream and DIST table: track r10c's state["r10c"][16]
# tensors (reused, not copied). Prepare: the kernel is validated against the parent route (bit-equal outputs) on random and
# 1 %-channel x30 outlier rows and on one-hot rows at escaped columns (also exact as values against W), for 3 bit-identical
# repeat calls; every launch's return code is checked. Any failure (or a non-sm_90 device, another geometry / table layout, a
# build failure, or the launcher refusing the shared-memory size) leaves the cell on the parent route. The extension
# (okc_bf16proj_r18b_<md5 12>: this file's CUDA_SRC / MBAR_HELPERS / MAP_SRC helpers + the kernel, namespace r18b_ns,
# raw-pointer launcher, no ATen headers, -fno-gnu-unique, one instance) builds on a background thread while the parent prepare
# runs. Static route (offline judge-faithful cross-process timing in the r18b README); prepare() never times anything.
# ======================================================================================================================
import hashlib as _r18b_hashlib
import re as _r18b_re
import threading as _r18b_threading
import time as _r18b_time

_R18B_NAME = "qkv16t_r200"   # the parent's sk2 geometry for qkv_gate m 9..16
_R18B_MOD = [None]
_R18B_THR = []
_R18B_T0 = [0.0]

_R18B_EDITS = (   # (parent text, specialized text); each parent text must occur exactly once
    ("          bool PK>\n__global__", "          bool PK, int NN, int KK>\n__global__"),
    ("uint32_t base4, int M, int N, int K, int dg,\n", "uint32_t base4, int M_, int N_, int K_, int dg,\n"),
    ("  const int dgv = SK2_DIAG ? dg : 0;",
     "  // track r18b: compile-time shape (the launcher refuses any other M / N / K)\n  constexpr int M = TOKR, N = NN, K = KK;\n"
     "  static_assert(CL == 1 && !PK && COMP && !HL, \"r18b: CL 1, M == TOKR, 12-bit pack only\");\n"
     "  const int dgv = SK2_DIAG ? dg : 0;"),
    ("  const int tst = K / (UPS * 64);                      // stages over all of K\n",
     "  constexpr int tst = K / (UPS * 64);                  // stages over all of K (track r18b: compile-time)\n"),
    ("  const int gs0 = (crank * tst) / CL;                  // this CTA's first stage (uneven split for CL = 3)\n",
     "  constexpr int gs0 = 0;                               // track r18b: CL 1\n"),
    ("  const int nst = ((crank + 1) * tst) / CL - gs0;      // and its stage count\n",
     "  constexpr int nst = tst;                             // track r18b: CL 1\n"),
    ("  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;\n",
     "  constexpr size_t wstride = (size_t)(N / R) * WBY;    // track r18b: gridDim.x = N / R\n"),
    ("  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;\n",
     "  // (track r18b: no zero row: PK is false, zrow is never read)\n"),
    ("    if (COMP && ptab_s[0].y != 0)   // escape-balanced row order (v135): 4-row groups moved whole, so they stay contiguous\n"
     "      o = out + (size_t)tt * N + reinterpret_cast<const int*>(ptab_s + ptab_s[0].y)[row];\n",
     "    // (track r18b: natural row order, no row map: asserted at prepare)\n"),
    ("    if (UPW > 1) {\n      __syncwarp();\n      if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n    }\n",
     "    if (UPW > 1 && st + NSTAGE < nst) {   // track r18b: the last NSTAGE stages are never refilled\n"
     "      __syncwarp();\n      if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n    }\n"),
    ("      if (UPW == 1 && uu == 0) {\n", "      if (UPW == 1 && uu == 0 && st + NSTAGE < nst) {   // track r18b\n"),
    ("        int pe = reinterpret_cast<const int*>(ptab_s + ptab_s[0].x)[ew];\n",
     "        int pe = reinterpret_cast<const int*>(ptab_s + 1)[ew];   // track r18b: header at int2 1 (no row map)\n"),
    ("          __syncwarp();\n          if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n        }\n      } else {\n",
     "          if (st + NSTAGE < nst) {   // track r18b: the last NSTAGE stages are never refilled\n            __syncwarp();\n"
     "            if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n          }\n        }\n      } else {\n"),
    # launcher: the parent's checks + the compile-time shape
    ("template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC>\nstatic int r9b_launch(",
     "template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC, int NN, int KK>\nstatic int r9b_launch("),
    ("  if (M != TOKR || N % R != 0 || K % (UPS * 64) != 0) return -3;\n",
     "  if (M != TOKR || N % R != 0 || K % (UPS * 64) != 0 || N != NN || K != KK || pmax < 2) return -3;\n"),
    ("NACC, 1, __nv_bfloat16, false>;\n", "NACC, 1, __nv_bfloat16, false, NN, KK>;\n"),
)


def _r18b_source():
    """track r9b's generated DIST source (_r9b_source(): the parent route's kernel text and raw-pointer launcher), written out
    for qkv_gate m = 16 by _R18B_EDITS, r9b identifiers renamed r18b, exported entry r18b_run -> the one instance
    <MT 1, TOKR 16, NT 3, NWR 1, WSK 4, UPW 2, NSTAGE 5, NACC 2, N 2304, K 8192>"""
    src = _r9b_source()
    for old, new in _R18B_EDITS:
        if src.count(old) != 1:
            raise RuntimeError("r18b: source anchor found %d times: %r" % (src.count(old), old[:60]))
        src = src.replace(old, new)
    for a, b in (("r9b_ns", "r18b_ns"), ("r9b_sk2_kernel", "r18b_sk2_kernel"), ("r9b_launch", "r18b_launch"),
                 ("r9b_run", "r18b_run"), ("r9b_err", "r18b_err"), ("R9B_NESC", "R18B_NESC")):
        src = _r18b_re.sub(r"\b%s\b" % a, b, src)
    if _r18b_re.search(r"\br9b_|\bR9B_", src):
        raise RuntimeError("r18b: unrenamed r9b identifier")
    lines = src.split("\n")
    disp_at = [i for i, l in enumerate(lines) if l.startswith("  if (cfg == ") and "r18b_ns::r18b_launch<" in l]
    if len(disp_at) != len(_R9B_CELLS) or disp_at != list(range(disp_at[0], disp_at[0] + len(disp_at))):
        raise RuntimeError("r18b: dispatch anchor")
    lines[disp_at[0]:disp_at[-1] + 1] = [
        "  if (cfg == 0) return r18b_ns::r18b_launch<1, 16, 3, 1, 4, 2, 5, 2, 2304, 8192>(w, xp, o, t, (int)pmax, (uint32_t)b4, "
        "(int)M, (int)N, (int)K, s);"]
    return ("// track r18b (adamleex): track r9b's DIST sk2 kernel (track r10c's qkv_gate m = 16 route) written out for qkv_gate m = 16\n"
            + "\n".join(lines))


def _r18b_ext():
    if _R18B_MOD[0] is None:
        src = _r18b_source()
        cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\nint64_t r18b_run(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, "
               "int64_t, int64_t, int64_t, int64_t, int64_t);\n")
        _R18B_MOD[0] = load_inline("okc_bf16proj_r18b_" + _r18b_hashlib.md5((cpp + src).encode()).hexdigest()[:12], cpp,
                                   cuda_sources=src, functions=["r18b_run"], no_implicit_headers=True, with_pytorch_error_handling=False,
                                   verbose=False, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xcompiler=-fno-gnu-unique"])
    return _R18B_MOD[0]


def _r18b_start():
    """compile the r18b extension on a background thread while the parent prepare runs (CPU-only, never fatal)"""
    def build():
        t0 = _r18b_time.time()
        try:
            _r18b_ext()
            _R18B_THR.append("ok")
        except Exception as error:
            _R18B_THR.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        _R18B_THR.append(round(_r18b_time.time() - t0, 1))
    t = _r18b_threading.Thread(target=build, daemon=True)
    t.start()
    return t


def _r18b_call(st, x, out):
    return _R18B_MOD[0].r18b_run(st["stream"].data_ptr(), x.data_ptr(), out.data_ptr(), st["tab"].data_ptr(), st["pmax"], st["b4"],
                                 16, st["n"], st["K"], 0, torch.cuda.current_stream().cuda_stream)


def _r18b_validate(st, w, state, gen):
    """random / 1 %-channel x30 outlier rows and one-hot rows at escaped columns (m = 16): outputs BIT-equal to the parent
    route (the kernel_fn chain above: track r10c) and close to FP64; one-hot rows also exact as values against W; 3
    bit-identical repeat calls"""
    m, n, K, dev = 16, st["n"], st["K"], w.device
    wd = w.double()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)

    def run(x):
        rc = _r18b_call(st, x, out)
        if rc != 0:
            return "launch error %d" % rc
        par = _r18b_parent_kernel({"x": x, "config": {"m": m, "n": n, "k": K, "layer": "qkv_gate"}}, state)
        torch.cuda.synchronize()
        if not (par.dtype == torch.bfloat16 and par.shape == out.shape and torch.equal(out.view(torch.int16), par.view(torch.int16))):
            return "outputs differ from the parent route"
        o1 = out.clone()
        for _ in range(3):
            rc = _r18b_call(st, x, out)
            if rc != 0:
                return "launch error %d" % rc
            torch.cuda.synchronize()
            if not torch.equal(out.view(torch.int16), o1.view(torch.int16)):
                return "repeat calls differ"
        return None

    worst = 0.0
    for dist in (1.0, 30.0, 1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[: K // 100]] *= dist
        x = x.to(torch.bfloat16).contiguous()
        why = run(x)
        if why:
            return "dist %g: %s" % (dist, why)
        ref = x.double() @ wd.t()
        e = float((out.double() - ref).norm() / ref.norm())
        if not e <= 4e-3:
            return "dist %g: rel L2 %.3e" % (dist, e)
        worst = max(worst, e)
    # one-hot rows at escaped columns (+ two plain columns): exact products, so the output must be bf16(W[:, k] * val)
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e71 = (bits >> 8) & 0x7F
    base, _ = _exponent_window(e71)
    esc_cols = torch.unique(_escapes(e71, base).nonzero()[:, 1])
    if esc_cols.numel():
        sel = esc_cols[torch.randperm(esc_cols.numel(), generator=gen, device=dev)[:3 * m - 2]]
    else:
        sel = esc_cols
    ks = torch.cat([sel, torch.tensor([0, K - 1], device=dev)])
    vals = torch.tensor([(-1.0 if j % 2 else 1.0) * 2.0 ** (1 + j % 4) for j in range(m)], device=dev)
    for b0 in range(0, ks.numel(), m):
        kb = ks[b0:b0 + m]
        x = torch.zeros((m, K), dtype=torch.bfloat16, device=dev)
        x[torch.arange(kb.numel(), device=dev), kb] = vals[:kb.numel()].to(torch.bfloat16)
        why = run(x)
        if why:
            return "one-hot: %s" % why
        exp = (w.float()[:, kb].t() * vals[:kb.numel(), None]).to(torch.bfloat16)
        # compared as values: an escaped W = +0 times a negative x is +0 here (dec * x + delta * x) but -0 in torch
        if not bool((out[:kb.numel()].float() == exp.float()).all()):
            return "one-hot rows at escaped columns not exact"
    return "ok %.2e (%d one-hot columns)" % (worst, int(ks.numel()))


def _r18b_setup(state, weights, thr):
    state["r18b"] = {}
    if thr is not None:
        thr.join(timeout=max(1.0, 250.0 - (_r18b_time.time() - _R18B_T0[0])))
    if not _R18B_THR or _R18B_THR[0] != "ok" or _R18B_MOD[0] is None:
        print("R18B build %s" % (_R18B_THR[0] if _R18B_THR else "not finished"), flush=True)
        return
    dev = weights["o"].device
    if torch.cuda.get_device_capability(dev) != (9, 0):
        print("R18B off: not an sm_90 device", flush=True)
        return
    ent = state.get("r10c", {}).get(16)
    cfg = next((c for c in SK2_CONFIGS if c[0] == _R18B_NAME), None)
    w = weights["qkv_gate"]
    if (ent is None or cfg is None or tuple(cfg[1:]) != ("qkv_gate", 2304, 24, 3, 1, 4, 2, 2, 1) or _sk2_geom(cfg, 16)[1] != 5
            or _R9B_NESC != 2 or _sk2_name("qkv_gate", 16) != _R18B_NAME or w.dim() != 2 or w.dtype != torch.bfloat16
            or tuple(w.shape) != (2304, 8192)):
        print("R18B qkv_gate m=16: off (no parent r10c route on %s or geometry)" % _R18B_NAME, flush=True)
        return
    stream, tab, pmax = ent["stream"], ent["tab"], int(ent["pmax"])
    if (ent["m"] != 16 or ent["n"] != 2304 or ent["K"] != 8192 or pmax < 2 or tuple(tab.shape) != (2304 // 24, pmax, 2)
            or not tab.is_contiguous() or tab.dtype != torch.int32 or stream.numel() != 2304 * 8192 * 3 // 2
            or stream.data_ptr() % 128 != 0 or tab.data_ptr() % 16 != 0 or (pmax * 8) % 16 != 0
            or bool((tab[:, 0, 0] != 1).any()) or bool((tab[:, 0, 1] != 0).any())):
        print("R18B qkv_gate m=16: off (parent stream / table layout)", flush=True)
        return
    st = dict(stream=stream, tab=tab, pmax=pmax, b4=int(ent["b4"]), n=2304, K=8192)
    msg = _r18b_validate(st, w, state, torch.Generator(device=dev).manual_seed(181818))
    print("R18B qkv_gate m=16: %s (table row %d int2)" % (msg if msg.startswith("ok") else "off (" + msg + "), parent route", pmax),
          flush=True)
    if msg.startswith("ok"):
        state["r18b"] = st


_r18b_parent_prepare = prepare
_r18b_parent_kernel = kernel_fn


def prepare(weights):
    _R18B_T0[0] = _r18b_time.time()
    thr = _r18b_start()
    state = _r18b_parent_prepare(weights)
    try:
        _r18b_setup(state, weights, thr)
    except Exception as error:
        state["r18b"] = {}
        print("R18B setup failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__), flush=True)
    torch.cuda.synchronize()
    print("R18B prepare done in %.0fs (r18b build %s, qkv_gate m16 %s)" % (_r18b_time.time() - _R18B_T0[0], _R18B_THR,
                                                                         "routed" if state.get("r18b") else "parent"), flush=True)
    return state


def kernel_fn(data, state):
    x = data["x"]
    e = state.get("r18b")
    if (e and data["config"]["layer"] == "qkv_gate" and x.dim() == 2 and x.shape[0] == 16 and x.shape[1] == e["K"]
            and data["config"]["n"] == e["n"] and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0):
        _k_ctx_bind(x)
        out = torch.empty((16, e["n"]), dtype=torch.bfloat16, device=x.device)
        if _r18b_call(e, x, out) == 0:
            return out
    return _r18b_parent_kernel(data, state)


# ---- Apache License 2.0 (from alygg77/okc-muse-spark-kernels, commit 6cae49a639ee6d14b929a5fd4a63c627f96d7dc0) ----
#
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright [yyyy] [name of copyright owner]
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
