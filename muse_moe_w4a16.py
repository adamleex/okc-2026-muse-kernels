# okc: problem=muse_moe_w4a16
# Derived from alygg77/okc-muse-spark-kernels (https://github.com/alygg77/okc-muse-spark-kernels), commit 6cae49a639ee6d14b929a5fd4a63c627f96d7dc0,
# file kernels/muse_moe_w4a16.py (Apache-2.0, with the vjeux / Matsvei notices below retained).
# First submitted by adamleex on 2026-10-31 with this attribution header added; code changes since are listed below.
# Change set K11 (adamleex, 2026-11-02): CUDA_SRC and the helper0 twin replaced by the MoE sources of our muse_decode_layer
# entry K11 (same lineage: alygg77 + vjeux, Apache-2.0); TINY / TINY2 / MID on helper0; the MID build serves 9..32 tokens.
# Change set PB (adamleex, 2026-11-02): prepare() compiles its builds in parallel threads (each load_inline runs nvcc in
# its own process); a build that fails there is redone serially.
# Change set NJ0 (adamleex, 2026-11-02): the WGS stage-range split returns empty ranges for a launch with no local route
# (it read a stale S.jC[0] there and could hang); extension names carry a digest of their source and flags.
# Change set R3M-SC6 + SC6W (adamleex, 2026-11-02): the MID build (9..32 tokens) reads 6-bit block scales (lossless, gated,
# 8-bit fallback; see the R3M-SC6 block); its M2T split costs of 2-tile units go from 30 / 32 to 31 / 33.
# Change set R3L (adamleex, 2026-11-02): large-batch WGS MAIN build (M >= 33), each lever behind its own macro (r3l_mk.py): YBF
# (y rows in bf16 between GEMM2 and the final combine, as SGLang keeps cache3 in bf16), SC6W (lossless 6-bit block scales in the WGS
# stages for 33 <= M <= SC6_MAX_M, gated in prepare() with the 8-bit build kept; idea from keell0renz's public MoE file, Apache-2.0,
# re-implemented by hand), R3FP (final pass at M > grid: two tokens per CTA at a time), R3DS (the last 16 CTAs at the grid barrier
# skip the warm-up dry pass, which otherwise delays them past the barrier release).
# Change set R3C (adamleex, 2026-11-02): R3M-SC6 (MID build, 9..32 tokens) and R3L (WGS MAIN builds, M >= 33) combined; they touch
# different builds and dispatch bands (kernel_fn: MID -> state["mid"], 33..SC6_MAX_M -> SC6 WGS build on state["w6"], else MAIN).
# The 6-bit block-scale packing (SC6) in both follows the idea of team keell0renz's public muse_moe_w4a16 file (Apache-2.0),
# re-implemented by hand.
# Change set R4X (adamleex, 2026-11-02; ported from our muse_decode_layer entry): the WGS builds' scratch flush handshake
# drops a redundant fence.acq_rel.gpu (writers red.add, bar.sync, release; reader acquire, bar.sync; the CUTLASS
# GenericBarrier pattern, as argued by vladimir on agents.kernarc.com), and the MoE kernel stores its inter / xB / ybuf
# scratch without an L2 cache hint (was evict_last, which could keep dead scratch lines resident in L2 after the call;
# now nothing of ours stays pinned in L2 across calls; the MID build keeps the asm store form; the weights' evict_first
# copies are unchanged).
# Change set R6M (adamleex, 2026-11-02): the MID build (9..32 tokens, 6-bit scales) gets a dedicated weight-issuer
# warp: it refills the weight ring after each unit, so thread 0 no longer issues the refill on the compute path;
# release points, math and outputs are unchanged (see the r6m tail at the end).
# Change set R7-WB (adamleex, 2026-11-02): the WGS MAIN build (M > SC6_MAX_M) splits its phase-B stages over the CTAs with a
# re-fitted stage-cost table (100, 98, 106, 117, 126, 138, 150, 162; was 100, 104, ..., 174; the SC6 build keeps the old one).
# Each stage's work and math are unchanged; only which CTA runs which stages (see the r6w tail at the end).
# Change set R7-FXW (adamleex, 2026-11-02): the WGS builds (M >= 33) launch a 19th (helper) warp per CTA that finishes the
# phase-A stream-K split units of one n-tile (read-back of the partial sums, SwiGLU epilogue, doneAs release) after the
# four shares have counted in; larger split units keep the last-arriver path; same math (see the r6w FXW block at the end).
# Change set R8G (adamleex, 2026-11-02): the WGS builds' (M >= 33) phase-A x gather writes each (8-row group, 256-K
# stage) block of xA as whole 128-B lines: a warp transposes the rows through its own 4 KB of the still idle weight
# ring first (was one 16-B store per thread into 32 lines per row); padding rows are written as 0; one GPU-scope
# fence, then relaxed per-job reds; the xA data, math and outputs are unchanged (see the r8g3 tail at the end).
# Change set R10T (adamleex, 2026-11-03): the small builds (<= 32 tokens) arrive at the grid barrier right after their last join, zero the per-launch flags on a CTA without final work and add the norm sums with red.add (see the r10t tail).
# Change set R14A (adamleex, 2026-11-04): the TINY (1..4 tokens) and TINY2 (5..8 tokens) builds drop the compute warps' L1 operand prefetch hint (prefetch_x_rt) from their unit loop; the operands they load, the math and the outputs are unchanged (see the r14a tail).
# Change set R15W (adamleex, 2026-11-04): the WGS builds (33+ tokens) read each unit epilogue's per-job global weight scale and, in phase B, the post-norm weight (1 + w) from shared memory (per-job tables filled once per launch by the idle FXW helper warp; post_norm_w staged in the routing area once it is dead) instead of global memory; same values, math and outputs (see the r15w tail).
# Change set R15F (adamleex, 2026-11-04): the WGS builds (33+ tokens) release the x producer's stage-range computation as soon as routing has the job table and run it on 4 lanes (the base's existing OKC_RAE / OKC_PR4 paths); the ranges, math and outputs are unchanged (see the r15f tail).
# Change set R15S (adamleex, 2026-11-04): the MID-S6 build (9..32 tokens) re-prices its static unit split (2-tile units 27 / 30, job-start costs 15 / 17); units, math and outputs unchanged (see the r15s tail).
# Change set R15E (adamleex, 2026-11-04): the TINY2 build (5..8 tokens) deals its units to the CTAs by a per-job-count table (for some job counts one split of all phase-A + phase-B units, for some a first block of 3 jobs instead of 2); units, math and outputs unchanged (see the r15e tail).
# Change set R16D (adamleex, 2026-11-04): the TINY build (1..4 tokens) ends without a grid barrier: its phase-B epilogues write y and per-unit sums of squares into TINY-private rows that start at a sentinel, and the compute warps of 8 CTAs per token poll exactly the elements they read, form the final norm/sum (same formula, fp32, bf16 out) and restore the sentinels; TINY2 and every other build are unchanged (see the r16d tail).
# Portions (the per-thread CUDA-context bind helper; cuBLASLt plan pins) follow code posted by Matsvei (kernarc user
# 'matvey') on agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0 (llms.txt), and that
# attribution is retained here.
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
"""Routed experts (NVFP4 W4A16 MoE) as one persistent CUDA kernel.

Algorithm
---------
For each token, up to ``TOP_K`` of the 256 experts are active, and this rank
owns ``LOCAL_EXPERTS`` of them. The work is a gated MLP per expert:

1.  Routing (``route`` in the CUDA source): every CTA redundantly builds the
    same per-expert token lists from ``topk_ids`` -- counts, offsets, and a
    sorted ``(token, slot)`` entry list -- plus a job list. An expert with
    more than ``SKINNY_MAX_TOKENS`` tokens becomes one or more *fat* jobs
    (chunks of ``FAT_MAX_TOKENS`` tokens); the rest become *skinny* jobs.
    Fat jobs are listed first, largest first.
2.  Phase A (gate + up, ``GEMM1 + SwiGLU``): each job's tokens are
    multiplied by the expert's packed NVFP4 gate/up weights. A *skinny* unit
    is 16 rows x full K with warps splitting K and ``mma.sync``; a *fat*
    unit is 256 rows x full K streamed through shared memory in K-stages.
    The SwiGLU output lands in ``inter`` (bf16).
3.  Phase B (down, ``GEMM2``): the same jobs run again with the expert's
    down weights, reading ``inter``. The epilogue multiplies by
    ``1 + post_norm_w`` (the post-expert RMSNorm weight, kept bf16 as given)
    and writes fp32 rows to ``ybuf`` plus each route's sum of squares to
    ``ss`` (double-buffered per launch generation).
4.  Final pass: ``out[t] = sum_s topk_w[t, s] * rsqrt(ss[pos] / HIDDEN + eps)
    * ybuf[pos]``. A grid barrier between the phases and the final pass
    makes every CTA's phase-B writes visible to every CTA.

Scheduling: one persistent grid (one CTA per SM) runs both phases. Fat
units go first (round-robin over CTAs); skinny units are split by cost so
every CTA carries the same total cost, with a small run-time pool of tail
units claimed atomically (stream-K). With no fat jobs, phase A's last ring
refills already prefetch this CTA's first phase-B units (cross-phase
prefill).

Which kernel runs depends on the batch: ``kernel_fn`` picks one of four
compile-time variants of the same source (``MAIN`` / ``TINY`` / ``TINY2`` /
``MID``), selected by token count; the large-batch variant additionally
switches to an RS-wgmma implementation (``OKC_WGS``) for the main band.
Weights are only repacked (bit/byte permutations) in ``prepare`` -- never
requantized -- into the four layouts the kernel reads: packed
mma.sync order, fat order, unit-contiguous order, and natural-k / WGS
order.

Constraints the code preserves (OKC rules)
------------------------------------------
* One Python file; native CUDA only, via ``torch.utils.cpp_extension.
  load_inline``.
* CUDA-graph capturable: no host synchronisation in the timed path.
  ``kernel_fn`` allocates a fresh ``out`` every call, reads no tensor
  back to host, and launches only ``okc_run``. Host syncs exist only in
  ``prepare`` (one-time setup) and in the ``DIAG`` diagnostics, both
  outside the timed path.
* Fresh inputs each call, no caching across calls: the only
  module-level cache is ``_EXTS``, which maps a compile-time define
  tuple to its compiled extension module. It caches compiled code,
  never activations, weights, or outputs.
"""

from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


# ---------------------------------------------------------------- model shape
# These are the task's model dimensions, not tunables. The CUDA source has
# its own copies (HD / ID) because it is compiled standalone; keep them in
# sync by hand -- there is no mechanism to share a constant across the
# Python/CUDA boundary in a load_inline submission.
HIDDEN = 4096            # d_hidden and d_expert (the MoE is square)
TOP_K = 8                # routes per token
N_EXPERTS_TOTAL = 256    # experts across all ranks
LOCAL_START = 0          # first expert owned by this rank
LOCAL_EXPERTS = 32       # experts owned by this rank (ep8 sharding)
SHARDING = "ep8"

# ------------------------------------------------------- per-call chunking
# kernel_fn runs the kernel in slices of at most this many tokens: the
# kernel's scratch (inter / ybuf / ss) is sized for this bound.
CHUNK_TOKENS = 384
EMAX = CHUNK_TOKENS * TOP_K          # max (token, route) entries per chunk
JMAX = 32 + EMAX // 64 + 8           # max jobs per chunk (upper bound)

# ------------------------------------------------------------- WGS scratch
# Stream-K fixup area: mirror of the CUDA-side WGS_FIXF / WGS_XROWS.
WGS_SLOTS = 160                      # fixup slots per phase (>= grid + 1)
WGS_FIXW = 4096                      # floats per (slot, warpgroup) fixup
WGS_FIXF = 2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16
WGS_XROWS = EMAX + 8 * (32 + EMAX // 32 + 8)

# ------------------------------------------- compile-time variant selection
# The four variants are the same CUDA source compiled with different
# -D flags -- i.e. different tile / scheduling choices, not different code.
# Flag tuples are part of the submission's tuned behaviour; do not edit
# them without re-gating on GPU. kernel_fn picks by token count M.
MAIN_DEFINES = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1", "-DOKC_YBF=1", "-DOKC_R3FP=1", "-DOKC_R3DS=16")
# M <= TINY_MAX_M: skinny-only build; a job never has more than M tokens,
# so there are no fat units.
TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2",
                "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1",
                "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1")
TINY_MAX_M = 4
# TINY_MAX_M < M <= TINY2_MAX_M: a second skinny-only build with a shorter
# helper-warp tail.
TINY2_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_CPF=0", "-DOKC_EPI4=0",
                 "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1",
                 "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=1", "-DOKC_FINPF=1")
TINY2_MAX_M = 8
# TINY2_MAX_M < M <= MID_MAX_M: tiny-style build with experts of more than
# SKINNY_MAX_TOKENS tokens split into <= SKINNY_MAX_TOKENS-token jobs.
MID_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_CPF=0",
               "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0",
               "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2",
               "-DOKC_FINPF=1")
MID_MAX_M = 32  # K11b: the MID build also serves 17..32 tokens (measured under this problem's routing: +1.4% at 32 and
               # +2..5% at 20..28 vs the decode layer's B4FIN WGS build, which won there under its less spread routing)
# M 33..63 runs on the WGS MAIN build; M >= 64 is MAIN by default below.
# The skinny ring's slot count when the caller does not pin -DOKC_NS.
MAIN_NS = 3

# The M >= 64 build used when the scale bytes rule out the PRMT decode
# (the WGS kernel needs it): the mma.sync kernel with the same shapes.
MAIN_FALLBACK_DEFINES = ("-DOKC_CPF=0", "-DOKC_CPFF=0", "-DOKC_DYNK=7",
                         "-DOKC_EARLYN=2", "-DOKC_EPI4=0", "-DOKC_FCOSTB=35",
                         "-DOKC_FCOSTBB=36", "-DOKC_FINCOL=0")

# r3l SC6W (6-bit block scales in the WGS stages; idea from keell0renz's public muse_moe_w4a16 file, Apache-2.0,
# re-implemented by hand). A lossless repack: exact when every block-scale byte of the local experts is in 0x40..0x7E
# (the generator's scales are clamped to [8, 448] = bytes 0x50..0x7E); prepare() checks the actual bytes and the round
# trip, and without it keeps the 8-bit build only. The SC6 build serves 33 <= M <= SC6_MAX_M (it gained at 64 / 128 and
# lost at 256 in the round-2 screens: the decode's extra ALU work against fewer scale bytes).
SC6_WGS = True
SC6_MAX_M = 160
SC6_DEFINES = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1", "-DOKC_YBF=1", "-DOKC_R3FP=1", "-DOKC_R3DS=16", "-DOKC_SC6W=1")

# ------------------------------------------------------------- feature flags
# Programmatic dependent launch for embedders whose graph runs this kernel
# right after other kernels (the kernel waits on griddepcontrol.wait before
# touching their outputs). Off: one launch per call gains nothing.
PDL = False
# Scale decode with PRMT + IMAD instead of the smem LUT; exact only when
# every block-scale byte of the local experts is a positive normal e4m3
# value (0x08..0x7E). prepare() checks the actual weights and keeps the
# LUT build otherwise.
PRMT_SCALES = True
# DQ7: the bit layout whose decode is 7 ALU ops + 2 IMAD.SHL per 8 weights
# (the kernel gets -DOKC_DQ7=1 via _ext).
DQ7 = True

# ---------------------------------------------------------------- diagnostics
# All off in a submission run. DIAG makes prepare() print an H200 timing
# table (untimed); DIAG_BW additionally runs the BW_SRC microbenchmarks.
DIAG = False
DIAG_VARIANTS = {}         # name -> defines, timed in diag mode
DIAG_BW = False
DIAG_BW_MODES = (5, 8, 12, 16) + tuple(range(17, 25))

# Extension-point for out-of-tree experiments: "@name" in a defines tuple
# swaps the compiled source to ALT_SRCS[name]. Empty in a submission.
ALT_SRCS = {}

# Bit layout of a 32-bit word holding 8 E2M1 codes for one MMA k-step of
# one lane: register R (0..3) is (h, j0) = (0,0) (1,0) (0,2) (1,2): rows
# g+8h, k = base + 4s + j0 (+1 for the upper bf16). Code bits (m, e0, e1, s)
# land at output bits (6, 7, 8, 15) (+16 for the upper half) after the
# per-register rotation used by dq4().
_REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]


CUDA_SRC = r"""
// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#define OKC_DEBUG 0
#ifndef OKC_R11FIN
#define OKC_R11FIN (OKC_WGS ? (OKC_B4FIN ? 0 : 1) : 0)  // (r11c: WGS builds only)
#endif
#ifndef OKC_R11WRM
#define OKC_R11WRM (OKC_WGS ? (OKC_B4FIN ? 0 : 1) : 0)  // (r11c: WGS builds only)
#endif
#ifndef OKC_R11EPI
#define OKC_R11EPI (OKC_WGS ? (OKC_B4FIN ? 0 : 1) : 0)  // (r11c: WGS builds only)
#endif
#define OKC_USE_TMA 1
#define OKC_RLOAD 1  // routing: all topk_ids loads issued before the smem stores
#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0's barrier inits
#define OKC_WGSFINSS 1
// ---- round 3 (moe_wgs): WGS levers, each off in the B4FIN build (d32), whose code stays as it was
#ifndef OKC_PAIR
#define OKC_PAIR (OKC_B4FIN ? 0 : 1)  // phase A: whole unit 2k+1 right after whole unit 2k (same job, same CTA) releases both at once
#endif
#ifndef OKC_FCIM
#define OKC_FCIM 5  // main build: the deferred flush runs behind this commit group of the next segment's first stage
#endif
#ifndef OKC_EPW
#define OKC_EPW (OKC_B4FIN ? OKC_EPWB : 1)  // griddepcontrol.wait just before routing (prologue under PDL); the producer warps never wait
#endif
// ---- r6 (chunk2): L2CH, main build only: an expert of > 64 tokens is one job per 64-token chunk and every chunk
// streams the expert's weights; those copies get an evict_normal L2 policy (not evict_first): the other chunk's copies hit L2
#ifndef OKC_L2CH
#define OKC_L2CH (OKC_B4FIN ? 0 : 1)
#endif
#define OKC_L2CHP 0x1000000000000000ull
// ---- round 4 (r4_d32): T = 32 (B4FIN build) levers, each its own macro; the main build keeps the base code
#ifndef OKC_EPWB
#define OKC_EPWB 1  // B4FIN: EPW too (griddepcontrol.wait just before routing; the producer warps never wait)
#endif
#ifndef OKC_RAE
#define OKC_RAE (OKC_B4FIN ? 1 : 0)  // barrier 7 (x producer: stage ranges) as soon as route() has the job table
#endif
#ifndef OKC_PR4
#define OKC_PR4 (OKC_B4FIN ? 1 : 0)  // stage ranges on 4 lanes of the x producer (phase x range end), one stage_at each
#endif
#ifndef OKC_B4RK
#define OKC_B4RK 1  // B4FIN: final items + flag zeroing by barrier arrival rank, run dry while waiting (warm code)
#endif
#ifndef OKC_B4FIN
#define OKC_B4FIN 0  // bet4_claude: column-parallel final pass, fixed-order ss partials (own build at 16 < M <= 32)
#endif
#ifndef B4_EARLY_Y
#define B4_EARLY_Y 1
#endif  // WGS: the final pass forms each route's sum of squares (the phase-B epilogue only writes y)
#ifndef OKC_FINCOL
#define OKC_FINCOL 1
#endif
#define OKC_WGMIN 9
#define OKC_XPRE 1
#define FAST1_ARCH 1
#ifndef OKC_FAST1
#define OKC_FAST1 3
#endif
#define FAST1_ON (OKC_FAST1 > 0 && OKC_TINY && FAST1_ARCH && OKC_USE_TMA)
#ifndef OKC_CPFF
#define OKC_CPFF 1
#endif
#ifndef OKC_FCOSTBB
#define OKC_FCOSTBB OKC_FCOSTB
#endif
#define OKC_FCOSTSB OKC_FCOST
#define OKC_DYN 1
#ifndef OKC_DYNK
#define OKC_DYNK 4
#endif
#define OKC_TDYN 0
#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)
#define OKC_FT_T 64
#ifndef OKC_SPLIT
#define OKC_SPLIT 0  // skinny-only build with experts of > SK_T tokens split into <= SK_T-token jobs (MID, Oct 26)
#endif
#ifndef OKC_TINY
#define OKC_TINY 0
#endif
#ifndef OKC_NW
#define OKC_NW 8
#endif
#ifndef OKC_EARLYN
#define OKC_EARLYN 1
#endif
#define OKC_BIGM 64
#define OKC_XPF 0
#ifndef OKC_CPF
#define OKC_CPF 1
#endif
#define OKC_FATSTATIC 1
#define OKC_FCOST 40
#ifndef OKC_FCOSTB
#define OKC_FCOSTB 40
#endif
#ifndef OKC_WS
#define OKC_WS 0  // tiny build: one extra helper warp issues the ring's TMA copies and runs every skinny epilogue
#endif
#define OKC_WGSX 0  // dev: 1 = WGS without the wgmma, 2 = WGS phases skipped
#ifndef OKC_WGS
#define OKC_WGS 0  // tiny build on RS-wgmma: 4 consumer warpgroups split K of 64-row units, plus a TMA producer warp
#endif
#define SILU_DIV(a, b) __fdividef((a), (b))
#ifndef OKC_UWARP
#define OKC_UWARP 1  // warp index through a shuffle, so ptxas can keep warp-derived values in uniform registers
#endif
#if OKC_UWARP
#define WARP_IDX(t) __shfl_sync(0xffffffffu, (t) >> 5, 0)
#else
#define WARP_IDX(t) ((t) >> 5)
#endif
#ifndef OKC_PRMTSC
#define OKC_PRMTSC 0
#endif
#ifndef OKC_FINPF
#define OKC_FINPF 0   // skinny builds: final item's topk_w / pos before the barrier, ss + ybuf loads batched (Oct 26, bf16)
#endif
#define OKC_WSNR 4  // partial-sum buffers between the compute warps and the helper warp
#define OKC_NOLOAD 0

// ---- r3l (adamleex, 2026-11-02): WGS MAIN-build levers (M >= 33), each off unless defined
#ifndef OKC_YBF
#define OKC_YBF 0  // y rows (ybuf) in bf16: SGLang keeps its GEMM2 output (cache3) in bf16 too
#endif
#ifndef OKC_SC6W
#define OKC_SC6W 0  // 6-bit block scales in the WGS stages (768 B per warpgroup piece instead of 1024 B); prepare() gates it
#endif
#ifndef OKC_R3FP
#define OKC_R3FP 0  // final pass at M > grid: two tokens per CTA at a time (one per half CTA)
#endif
#ifndef OKC_R3DS
#define OKC_R3DS 1  // the last OKC_R3DS CTAs at the grid barrier skip the warm-up dry pass (1: only the last, as before)
#endif
#if (OKC_YBF || OKC_SC6W || OKC_R3FP || OKC_R3DS != 1) && (!OKC_WGS || OKC_B4FIN)
#error "r3l levers: WGS main build only"
#endif
#if OKC_YBF && !(OKC_R11FIN && OKC_R11WRM && OKC_WGSFINSS)
#error "r3l YBF: written for the R11FIN / R11WRM / WGSFINSS final pass"
#endif
#if OKC_R3FP && !OKC_YBF
#error "r3l FP: written for the YBF final pass"
#endif

namespace okc {

#ifndef OKC_L2H
#define OKC_L2H 1
#endif
#if OKC_L2H
// L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later in
// the same call (evict last). Hints only: the math and the memory traffic's results are unchanged.
#define L2H_SUFFIX ".L2::cache_hint"
#define L2H_OPERAND ", %4"
#define L2H_ARG , "l"(0x12F0000000000000ull)
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {
  *a = v;  // R4X: plain store, no L2 hint (was evict_last)
}
__device__ __forceinline__ void st_keep_f32(float* a, float v) {
  *a = v;  // R4X: plain store, no L2 hint (was evict_last)
}
#else
#define L2H_SUFFIX ""
#define L2H_OPERAND ""
#define L2H_ARG
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) { *a = v; }
__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }
#endif

constexpr int HD = 4096, ID = 4096;
constexpr int NW = OKC_NW, NT = NW * 32;
constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)
constexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers
#if OKC_WS || OKC_WGS
// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)
#define CTA_SYNC() asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory")
#define WS_JOIN() asm volatile("bar.sync 2, %0;\n" ::"n"(NTL) : "memory")
#else
#define CTA_SYNC() __syncthreads()
#endif
constexpr int NCH = 64 / NW;     // 64-K chunks per warp in a skinny unit (4096 / NW K per warp)
#define OKC_SKT 8
constexpr int SK_T = OKC_SKT;   // max tokens of a skinny job (8: 1 n-tile; 16: 2 n-tiles via smem x)
constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;
static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");
constexpr int FT_T = OKC_FT_T;   // max tokens of a fat job (longer jobs are split into chunks)
constexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;   // 16-row blocks
constexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;
constexpr int RB_A_LOG = 9, RB_B_LOG = 8;
static_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");
constexpr int BFLAG = 1 << 30;  // uslot: a phase-B unit prefetched at the end of phase A
// dynamic tail: the last pool units of a phase's skinny work are claimed at run time (OKC_DYNK per CTA)
__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * OKC_DYNK; return SKDYN ? (d < nS ? d : nS) : 0; }
static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
static_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)
constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
constexpr int XP_SK = HD + 8;   // skinny x row pitch (bf16)
constexpr int KS = 256;         // fat: K per x stage (4 chunks)
constexpr int XP_FT = KS + 8;   // fat x row pitch (bf16)
constexpr int NST = 3;          // fat x stages
constexpr float TWO62 = 4.611686018427388e18f;

struct Params {
  const uint4* wA; const uint4* sA; const float* gsA;
  const uint4* wB; const uint4* sB; const float* gsB;
  const uint32_t* lut; const __nv_bfloat16* normw;  // post_norm_w as given (bf16); 1 + w is formed in FP32 here
  const unsigned char* wAf; const unsigned char* wBf;  // fat-order weight copies (36 KB per k-stage)
  const unsigned char* wAu; const unsigned char* wBu;  // unit-contiguous copies (36 KB per 16-row block)
  const unsigned char* wAn; const unsigned char* wBn;  // natural-k fat copies (wgmma path)
  const unsigned char* wAs; const unsigned char* wBs;  // WGS copies: [256-row unit][stage][warpgroup][9 KB]
  __nv_bfloat16* xA; __nv_bfloat16* xB;  // WGS: per job, per stage x / inter in wgmma B layout [stage][n8][k8][8][8]
  int* xready;                           // WGS: per job, rows of xA written
  int* doneAs;                           // WGS: per (job, phase-B stage), phase-A warpgroup epilogues done (8 = ready)
  uint32_t zero;  // always 0 (opaque to the compiler)
  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;
  int M, lo, nloc;
  int* ctr;        // [8]: 0 ctrA, 1 ctrB, 2 barrier, 3 exit
  int* doneA;      // [JMAX]
  __nv_bfloat16* inter;  // [EMAX, ID]
  float* ybuf;     // [EMAX, HD]
  float* ss;       // [EMAX] copy 0 of the per-token sums of squares (copy 1: ss1); a launch uses copy gen & 1
  float* ss1;      // [EMAX]
  __nv_bfloat16* out;
  unsigned long long* dbg;
  float* sspart;  // bet4_claude B4FIN: per-(route, 16-column tile) sum-of-squares partials (WGS only)
};

struct Smem {
  uint32_t lut[256];
#if !OKC_WGS
  float red[NRB][NW][32][RED_W];  // per-unit cross-warp partial sums
#else
  float red[1][1][1][RED_W];      // (the skinny paths that use it are not run in WGS builds)
#endif
#if OKC_WS
  uint64_t wse[8], wsrf[8], wsre[8];  // WS: ring slot released by all compute warps / partials ready / buffer free
  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges
#endif
#if OKC_WGS
  uint64_t bars[24];  // WGS: 0..3 weights full, 4..7 weights empty, 8..15 x full, 16..23 x empty
  int wglast[4];      // WGS: this warpgroup completed a split unit
  int wcp[JMAX + 1];  // WGS: phase-A cost prefix per job
  int wcpB[JMAX + 1]; // WGS: phase-B cost prefix per job
  int jrow[JMAX + 1]; // WGS: first padded row of each job in xA / xB (rows padded to 8 per job)
  uint32_t wtot;      // WGS: phase-A total cost
  uint32_t wtotB;     // WGS: phase-B total cost
#else
  uint64_t bars[12];  // 0..4 skinny ring (NS <= 5), 5..11 fat ring (NSF <= 7)
#endif
#if OKC_WGS
  int wgr[4];         // WGS: this CTA's phase A and phase B unit ranges
#if OKC_WGS && !OKC_B4FIN
  int r9xf;  // r9 XF: first x copies issued
#endif
#endif
  int seq, fseq, preissued, preissuedB;
  int fcnt[32];     // OKC_SPLIT: FAST1 per-expert entry counts (the preissue's job -> expert map)
  void* xbuf_ptr;
  int cnt[64], off[64];
  int wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];
  int njF, njS, unit;
  int gen;          // the grid barrier's generation at launch start (+1 per launch): picks this launch's ss copy
  float* ssp;       // this launch's ss copy
  int uslot[8];    // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)
  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)
  int unit_next, unit_next2;
  signed char eid[EMAX];
  short ent[EMAX];   // sorted entries (token*8+slot) by expert
  short pos[EMAX];   // entry -> pair position, -1 if not local
};

// floor(a * c / G) without an integer divide (per-launch code runs from a cold instruction cache, where the divide
// subroutines cost ~1 us): a * c < 2^24, rG ~ 1 / G, then an exact +-1 fix-up
__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {
  const int p = a * c;
  int q = __float2int_rz((float)p * rG);
  if (q * G > p) --q;
  else if ((q + 1) * G <= p) ++q;
  return q;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
// a x (b.lo, b.lo) and a x (b.hi, b.hi): the broadcast folds into the HMUL2 operand (.H0_H0 / .H1_H1)
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
extern __shared__ __align__(16) unsigned char smem_raw[];
__device__ __forceinline__ float* ss_cur();  // this launch's ss copy (defined after Smem users)
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

// Dequantize one 32-bit word (8 E2M1 codes) into the 4 bf16x2 A-fragment registers (values e2m1 * 2^-126).
// Prepacked bit layout (see _pack_codes): r0 = q & M, r1 = rotr(q,6) & M, r2 = rotr(q,10) & M,
// r3 = (rotl(q,4) & 0x81808180) | (rotr(q,4) & 0x00400040), M = 0x81C081C0.
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_r(q, q, 6) & M;
  r2 = __funnelshift_r(q, q, 10) & M;
  // DQ7 bit layout (template DQ7 = True): 7 ALU + 2 IMAD.SHL per 8 weights. Register sources partition the word:
  // r0 bits as stored, r1 = word << 4, r2 = rotl 10, r3 = word << 6 (6 bits) | rotl 14 (the two e1 bits)
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  return;
  // bits kept by 0x81808180 never wrap, so the left rotate is a plain shift: as a multiply it runs on the FMA pipe
  // (IMAD.SHL) instead of the ALU pipe the rest of the decode saturates
  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
}


// ---------------------------------------------------------------- routing prologue (every CTA, identical result)
__device__ void early_issue(const Params& p, Smem& S);
__device__ void route(const Params& p, Smem& S) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int E = p.M * 8;
  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;
  const int i0 = warp * CH, i1 = min(E, i0 + CH);
  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
#if OKC_RLOAD && OKC_WGS  // (the small builds lose 1-2% with it: their E is at most 256)
  {  // 16 B loads (E = 8 M; the launcher checks the alignment): one or two per thread, all issued before the stores
    constexpr int LPT = (EMAX / 4 + NT - 1) / NT;
    int4 ev[LPT];
    const int E4 = E >> 2;
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      ev[k] = i4 < E4 ? __ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4) : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      if (i4 < E4) {
        const int v[4] = {ev[k].x - p.lo, ev[k].y - p.lo, ev[k].z - p.lo, ev[k].w - p.lo};
        char4 c;
        c.x = (signed char)((v[0] >= 0 && v[0] < p.nloc) ? v[0] : -1);
        c.y = (signed char)((v[1] >= 0 && v[1] < p.nloc) ? v[1] : -1);
        c.z = (signed char)((v[2] >= 0 && v[2] < p.nloc) ? v[2] : -1);
        c.w = (signed char)((v[3] >= 0 && v[3] < p.nloc) ? v[3] : -1);
        *reinterpret_cast<char4*>(&S.eid[i4 * 4]) = c;
      }
    }
  }
#else
#pragma unroll 8
  for (int i = tid; i < E; i += NT) {
    int e = __ldg(p.topk_ids + i) - p.lo;
    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
  }
#endif
  CTA_SYNC();
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }
  CTA_SYNC();
  if (warp == 0) {
    int c[NW];
    int tot = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) { c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0; tot += c[w]; }
    int incl = tot;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }
    int off = incl - tot;
    S.cnt[lane] = tot;
    S.off[lane] = off;
    int o2 = off;
#pragma unroll
    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }
    // jobs: fat (cnt > SK_T, chunks of FT_T) first, then skinny (0 < cnt <= SK_T); expert order within each
#if OKC_SPLIT
    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");
    const int nfj = 0;
    const int nsj = (tot + SK_T - 1) / SK_T;   // ceil(tot / 8) jobs of <= 8 tokens; a split expert re-streams its weights
#else
    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;
    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;
#endif
    int fi = nfj, si = nsj;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);
      if (lane >= d) { fi += v1; si += v2; }
    }
    const int nfTot = __shfl_sync(0xffffffffu, fi, 31), nsTot = __shfl_sync(0xffffffffu, si, 31);
    int fj = fi - nfj, sj = nfTot + si - nsj;
    for (int q = 0; q < nfj; ++q) {
      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);
    }
#if OKC_SPLIT
    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }
#else
    if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }
#endif
    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }
    __syncwarp();
#if !OKC_WGS  // (the WGS build has no fat queue)
    // fat queue order: largest n-tile count first (LPT), stable by job index
    int pos = 0;
    for (int key = FT_T / 8; key >= 1; --key) {
      for (int f0 = 0; f0 < nfTot; f0 += 32) {
        const int f = f0 + lane;
        const bool m = f < nfTot && ((S.jC[f] + 7) >> 3) == key;
        const unsigned bal = __ballot_sync(0xffffffffu, m);
        if (m) S.forder[pos + __popc(bal & ((1u << lane) - 1u))] = f;
        pos += __popc(bal);
      }
    }
#endif
  }
  CTA_SYNC();
#if OKC_WGS && OKC_RAE  // (OKC_WGSPR is defined further down: every WGS build has it)
  // r4 (RAE): the job table is complete: the x producer starts the stage ranges now (they read only jC / njF / njS),
  // while the pass below sorts the entries
  __threadfence_block();
  asm volatile("bar.arrive 7, %0;\n" ::"n"(NT + 32) : "memory");
#endif
  early_issue(p, S);
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0) {
      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));
      S.ent[q] = (short)i;
      S.pos[i] = (short)q;
    } else if (i < i1) {
      S.pos[i] = -1;
    }
    __syncwarp();
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
    __syncwarp();
  }
  CTA_SYNC();
}



// ---------------------------------------------------------------- epilogues (one 16-row block, tokens 2t,2t+1 of n-tile)
// r[0..1] rows g (gate / out row n0), r[2..3] rows g+8 (up / out row n1), tokens tk0 = 2t, 2t+1 relative to tile
__device__ __forceinline__ float probe_q(float v) {
  return v;
}
__device__ __forceinline__ void epi_A(const Params& p, int pstart, int ntok, int tok0, int col, float gsc, const float* r) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    if (tk < ntok) {
      float gt = r[h] * gsc, up = r[2 + h] * gsc;
      float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
      st_keep_bf16(p.inter + ((size_t)(pstart + tk) * ID + col), __float2bfloat16(probe_q(v)));
    }
  }
}
// returns per-token partial sum of squares (2 values) for this lane's two tokens
__device__ __forceinline__ void epi_B(const Params& p, int pstart, int ntok, int tok0, int n0, float gsc, const float* r,
                                      float* sq) {
  const float w0 = 1.f + __bfloat162float(p.normw[n0]), w1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    float v0 = r[h] * gsc * w0, v1 = r[2 + h] * gsc * w1;
    sq[h] = v0 * v0 + v1 * v1;
    if (tk < ntok) {
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0), v0);
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0 + 8), v1);
    }
  }
}
__device__ __forceinline__ void ss_reduce_add(const Params& p, int pstart, int ntok, int tok0, float* sq, int g) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 4);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 8);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 16);
  }
  if (g == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
      if (tok0 + h < ntok) atomicAdd(&ss_cur()[pstart + tok0 + h], sq[h]);
  }
}

// ---------------------------------------------------------------- the persistent kernel


// ---------------------------------------------------------------- skinny path: shared-memory weight ring
// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.
// Units of the CTA's range stream through OKC_NS smem slots (TMA bulk copies on sm_90, cp.async elsewhere).
// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.
constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;


__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#if OKC_PRMTSC
// e4m3 scale byte b of `word` -> bf16x2(scale * 2^64) in both halves with one PRMT and one IMAD: (byte << 4) puts the
// e4m3 exponent and mantissa into the bf16 fields, + 184 << 7 rebiases the exponent. Exact for positive normal bytes
// 0x08..0x7E only; prepare() checks every scale byte and uses the LUT build otherwise.
#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)
// same, with the multiplier opaque to ptxas (k16 = 16 + p.zero): an IMAD on the FMA pipe instead of a LEA on the
// ALU pipe, which the FP4 decode saturates
#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)
#endif
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
      ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
  const size_t blk = (size_t)S.jE[j] * rbs + rb;
  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;  // [32 KB codes | 4 KB scales]
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  if (threadIdx.x == 0) ring_issue<PB>(p, S, v, slot, bar);
}

template <bool PB>
__device__ __forceinline__ void load_x_regs(const Params& p, const Smem& S, int j, int lane, int warp,
                                            uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  {
    const int r = g;
    if (r < cnt) {
      const __nv_bfloat16* row;
      if (PB) row = p.inter + (size_t)(st + r) * ID;
      else row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        x[2 * c] = PB ? __ldcg(src + c * 8) : __ldg(src + c * 8);
        x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);
    }
  }
}

constexpr int X1P = 512 + 8;  // per-warp smem x rows for the 2nd n-tile (tokens 8..15), pitch in bf16
template <bool PB>
__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,
                                             __nv_bfloat16* x1s) {
  // warp-private: rows 8..15 of the job, K range [512*warp, +512): 8 rows x 64 uint4
  const int cnt = S.jC[j], st = S.jS[j];
  for (int i = lane; i < 8 * 64; i += 32) {
    const int r = i >> 6, c = i & 63;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (8 + r < cnt) {
      const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;
      v = PB ? __ldcg(src) : __ldg(src);
    }
    *reinterpret_cast<uint4*>(x1s + r * X1P + c * 8) = v;
  }
  __syncwarp();
}

#if SKDYN
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;  // static units [v0, v1); the run-time pool [poolBase, nSall) follows
  const int rsh = PB ? RB_B_LOG : RB_A_LOG;
  const int seq0 = seq;
  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // cross-phase prefill: with no fat jobs, phase A's last ring refills already fetch this CTA's first B units
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  // thread 0 owns the unit source: static units first, then pool units claimed with one atomic in flight
  int snext = v0 + pre, pend = -2, jdmin = 1 << 30, jdmax = -1;
  bool exh = poolBase >= nSall;
  int* pctr = p.ctr + (PB ? 1 : 0);
  auto take = [&]() -> int {
    if (snext < v1) return snext++;
    if (exh) return -1;
    const int cl = pend >= 0 ? pend : atomicAdd(pctr, 1);
    const int v = poolBase + cl;
    if (v >= nSall) { exh = true; pend = -2; return -1; }
    pend = atomicAdd(pctr, 1);  // the next claim; its value is first needed at the next refill
    if (!PB) {
      const int jj = njF + (v >> rsh);
      S.dync[jj] += 1;
      jdmin = min(jdmin, jj); jdmax = max(jdmax, jj);
    }
    return v;
  };
  auto refill = [&](int sl) {
    const int v = take();
    if (v >= 0) { ring_issue<PB>(p, S, v, ring + sl * UNIT_BYTES, bars + sl); S.uslot[sl] = v; }
    else if (cpf && preB < nB) {
      ring_issue<true>(p, S, v0B + preB, ring + sl * UNIT_BYTES, bars + sl);
      S.uslot[sl] = BFLAG | (v0B + preB);
      ++preB;
    } else {
      S.uslot[sl] = -1;
    }
  };
  if (tid == 0)
    for (int i = pre; i < OKC_NS; ++i) {
      const int sl = (seq0 + i) % OKC_NS;
      if (i < nU) { ring_issue<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
  if (nU < OKC_NS) CTA_SYNC();  // pool slots filled by the prologue: their unit ids must be visible
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit's rows one unit ahead
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) {
    const int u0 = nU > 0 ? v0 : (S.uslot[seq0 % OKC_NS] & ~BFLAG);
    const int rbq = (u0 < 0 ? 0 : u0) & (rbs - 1);
    nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
  }
  int nproc = 0;
  for (int i = 0;; ++i) {
    int v;
    if (i < nU) {
      v = v0 + i;  // static units: no slot lookup
    } else {
      v = S.uslot[(seq0 + i) % OKC_NS];
      if (v < 0 || (!PB && (v & BFLAG))) break;  // pool exhausted (or phase B's prefetched units follow)
      v &= ~BFLAG;
    }
    const int j = njF + (v >> rsh), rb = v & (rbs - 1);
    float nwn0 = 0.f, nwn1 = 0.f;
    int vn = -1;
    if (PB) vn = i + 1 < nU ? v + 1 : S.uslot[(seq0 + i + 1) % OKC_NS];
    if (PB && tid < 128 && vn >= 0) {
      const int rbq = (vn & ~BFLAG) & (rbs - 1);
      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
    }
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
#if OKC_NW == 16
    // a warp covers half of an 8-warp K slice: its 4 chunks' scale bytes are 8 of the slice lane's 16
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#undef MULG
#undef MULG8
    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();  // slot sl fully consumed; partials visible
    if (tid == 0) {
      if (i + OKC_NS < nU) { ring_issue<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = s0 * gsc, up = s1 * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    nwc0 = nwn0; nwc1 = nwn1;
    ++nproc;
  }
  seq = seq0 + nproc;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      if (nU > 0) {
        const int jf = njF + (v0 >> rsh), jl = njF + ((v1 - 1) >> rsh);
        for (int jj = jf; jj <= jl; ++jj) {
          const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
          atomicAdd(p.doneA + jj, b - a);
        }
      }
      for (int jj = jdmin; jj <= jdmax; ++jj) {  // pool units
        const int k = S.dync[jj];
        if (k) { atomicAdd(p.doneA + jj, k); S.dync[jj] = 0; }
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#else
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  if (nU <= 0) return;
  const int seq0 = seq;
  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // cross-phase prefill: with no fat jobs, phase A's last ring refills already fetch this CTA's first B units
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  for (int i = pre; i < OKC_NS; ++i) {
    const int sl = (seq0 + i) % OKC_NS;
    if (i < nU) ring_issue_all<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i - nU < nB) { ring_issue_all<true>(p, S, v0B + (i - nU), ring + sl * UNIT_BYTES, bars + sl); ++preB; }
  }
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit's rows one unit ahead
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) { const int rbq = v0 % rbs; nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)); }
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    float nwn0 = 0.f, nwn1 = 0.f;
    if (PB && tid < 128 && i + 1 < nU) {
      const int rbq = (v + 1) % rbs;
      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
    }
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#undef MULG
#undef MULG8
    if (!two) {
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();  // slot sl fully consumed; partials visible
    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i + OKC_NS - nU < nB) {
      ring_issue_all<true>(p, S, v0B + (i + OKC_NS - nU), ring + sl * UNIT_BYTES, bars + sl);
      ++preB;
    }
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = s0 * gsc, up = s1 * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    nwc0 = nwn0; nwc1 = nwn1;
  }
  seq = seq0 + nU;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;
      for (int jj = jf; jj <= jl; ++jj) {
        const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
        atomicAdd(p.doneA + jj, b - a);
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#endif

// ---------------------------------------------------------------- fat unit: CTA = 256 rows (warp = 2 x 16 rows) x full K, <=64 tokens
// ---------------------------------------------------------------- fat unit through the smem ring
// A fat unit is 256 rows x full K for <= 64 tokens. Its weights arrive in 16 k-stages of 36 KB (16 row blocks x
// 4 chunks + their scales, contiguous in the fat-order copy) through slots 0..2 of the ring region; the x tile
// for the stage is staged by cp.async into two buffers that live in slots 3..4.
constexpr int NSF = OKC_NSF;
template <bool PB>
__device__ __forceinline__ void fat_issue(const Params& p, int e, int ub, int ks, unsigned char* slot, uint64_t* bar, bool nat = false) {
  const int nub = PB ? UB_B : UB_A;
  const unsigned char* src = (nat ? (PB ? p.wBn : p.wAn) : (PB ? p.wBf : p.wAf)) + (((size_t)e * nub + ub) * 16 + ks) * (size_t)UNIT_BYTES;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
  }
}

// ---------------------------------------------------------------- fat units: one continuous stream per CTA
// Units come from a dynamic queue; the next unit is claimed when the current one starts, so the ring keeps
// streaming across unit boundaries (the refills of the last k-stages already fetch the next unit's weights).
// ---------------------------------------------------------------- wgmma (sm90a): A from registers, B (x) from smem
__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {  // K-major, 128 B swizzle, 8-row groups 1 KB apart
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
template <int NTT> struct WG;
template <> struct WG<1> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<2> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<3> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<4> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<5> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<6> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<7> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<8> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};

// one 36 KB k-stage of a fat unit with wgmma: A = dequantized weights (natural k order) from registers, B = x tile from
// shared memory in the K-major no-swizzle core-matrix layout [n8][k8][8 rows][16 B] (LBO 128 B along K, SBO 4 KB along N)
template <int NTT>
__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {
  // x stage: [kb 4][64 rows][128 B], 128 B swizzle (chunk c of row n at c ^ (n & 7)); k-step st of chunk cc reads
  // k block cc at byte offset 32 st
  const int g = lane >> 2;
  const uint32_t xb = smem_u32(xs);
  uint32_t A[2][8][4];  // [chunk parity][k-step x rb][fragment]; a set stays untouched until its group retired
  uint32_t keep = 0;    // folds every fragment after its group retired, so ptxas keeps the registers live (no reuse)
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const int sb = cc & 1;
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);
    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      // e4m3 -> bf16 without the LUT: bf16 bits (b << 4) hold value * 2^-120 (subnormal codes included), two exact
      // multiplies by 2^92 give value * 2^64 like the LUT; rows g | g+8 travel as one bf16x2
      const uint32_t K92 = 0x6D806D80u;  // bf16x2(2^92, 2^92)
      const uint32_t p0 = bmul2(bmul2((__byte_perm(sc0.x, sc0.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t p1 = bmul2(bmul2((__byte_perm(sc1.x, sc1.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t sg0 = __byte_perm(p0, p0, 0x1010u), sg80 = __byte_perm(p0, p0, 0x3232u);
      const uint32_t sg1 = __byte_perm(p1, p1, 0x1010u), sg81 = __byte_perm(p1, p1, 0x3232u);
#define FMG0(r) bmul2((r), sg0)
#define FMG80(r) bmul2((r), sg80)
#define FMG1(r) bmul2((r), sg1)
#define FMG81(r) bmul2((r), sg81)
      uint32_t r0, r1, r2, r3, q0, q1, q2, q3;
      dq4((&w0.x)[st], r0, r1, r2, r3);
      dq4((&w1.x)[st], q0, q1, q2, q3);
      A[sb][2 * st][0] = FMG0(r0); A[sb][2 * st][1] = FMG80(r1);
      A[sb][2 * st][2] = FMG0(r2); A[sb][2 * st][3] = FMG80(r3);
      A[sb][2 * st + 1][0] = FMG1(q0); A[sb][2 * st + 1][1] = FMG81(q1);
      A[sb][2 * st + 1][2] = FMG1(q2); A[sb][2 * st + 1][3] = FMG81(q3);
#undef FMG0
#undef FMG80
#undef FMG1
#undef FMG81
    }
    wg_fence();
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const uint64_t dk = wg_desc_sw128(xb + cc * 8192 + st * 32);
      WG<NTT>::mma(acc0, A[sb][2 * st][0], A[sb][2 * st][1], A[sb][2 * st][2], A[sb][2 * st][3], dk);
      WG<NTT>::mma(acc1, A[sb][2 * st + 1][0], A[sb][2 * st + 1][1], A[sb][2 * st + 1][2], A[sb][2 * st + 1][3], dk);
    }
    wg_commit();
    if (cc >= 1) {
      wg_wait<1>();  // the previous chunk's group retired: its register set may be rewritten
#pragma unroll
      for (int k = 0; k < 8; ++k) keep ^= A[sb ^ 1][k][0] ^ A[sb ^ 1][k][1] ^ A[sb ^ 1][k][2] ^ A[sb ^ 1][k][3];
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int k = 0; k < 8; ++k) keep ^= A[1][k][0] ^ A[1][k][1] ^ A[1][k][2] ^ A[1][k][3];
  acc0[0][0] += __uint_as_float(keep & zero);  // adds +0.0f
}

// one 36 KB k-stage of a fat unit: 2 row blocks per warp x NTT token tiles (exact tile count, no predicated MMAs)
template <int NTT>
__device__ __forceinline__ void fat_compute(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                            int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp * 32 + lane) * 8);
#if OKC_NW == 8
  const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp + 8) * 32 + lane) * 8);
#endif
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
#if OKC_NW == 8
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
#endif
#if OKC_PRMTSC
    const uint32_t sg0 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2), sg80 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2 + 1);
#if OKC_NW == 8
    const uint32_t sg1 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2), sg81 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2 + 1);
#endif
#else
    const int sh = (cc & 1) * 16;
    const uint32_t sw0 = (cc < 2 ? sc0.x : sc0.y) >> sh;
    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;
    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];
    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];
#endif
    const __nv_bfloat16* xr = xs + g * XP_FT + cc * 64 + 16 * t;
#pragma unroll
    for (int sp = 0; sp < 2; ++sp) {
      uint32_t a0[2][4], a1[2][4];
      (void)a1;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int s2 = 2 * sp + h;
        uint32_t r0, r1, r2, r3;
        dq4((&w0.x)[s2], r0, r1, r2, r3);
        a0[h][0] = bmul2(r0, sg0); a0[h][1] = bmul2(r1, sg80); a0[h][2] = bmul2(r2, sg0); a0[h][3] = bmul2(r3, sg80);
#if OKC_NW == 8
        dq4((&w1.x)[s2], r0, r1, r2, r3);
        a1[h][0] = bmul2(r0, sg1); a1[h][1] = bmul2(r1, sg81); a1[h][2] = bmul2(r2, sg1); a1[h][3] = bmul2(r3, sg81);
#endif
      }
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const uint4 bv = *reinterpret_cast<const uint4*>(xr + nt * 8 * XP_FT + 8 * sp);
        mma_bf16(acc0[nt], a0[0][0], a0[0][1], a0[0][2], a0[0][3], bv.x, bv.y);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[0][0], a1[0][1], a1[0][2], a1[0][3], bv.x, bv.y);
#endif
        mma_bf16(acc0[nt], a0[1][0], a0[1][1], a0[1][2], a0[1][3], bv.z, bv.w);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[1][0], a1[1][1], a1[1][2], a1[1][3], bv.z, bv.w);
#endif
      }
    }
  }
}

template <bool PB, bool BIG>
__device__ __forceinline__ void fat_stream(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int g = lane >> 2, t = lane & 3;
  int* ctr = p.ctr + (PB ? 1 : 0);
  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
  uint64_t* fb = S.bars + 5;
  __nv_bfloat16* xs0 = reinterpret_cast<__nv_bfloat16*>(ring + NSF * UNIT_BYTES);
  constexpr int NKS = HD / KS;  // 16
  int u = blockIdx.x;
  (void)ctr;
  if (u >= nF) return;
  constexpr int UBS = PB ? UB_B : UB_A;  // == ubs, as a constant (shift instead of a divide)
  (void)ubs;
  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };
  auto wait_A = [&](int j) {  // phase B: inter rows of job j complete
    if (PB) {
      if (tid == 0) { while (ld_acquire(p.doneA + j) < UB_A) __nanosleep(64); }
      CTA_SYNC();
    }
  };
  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };
  auto stage_x = [&](int j, int ks, int xsl) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int ntl = (cnt + 7) >> 3;
    const int nrow = ntl * 8;
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
    const int per = nrow * (KS / 8);
    if (is_nat(j)) {
      // 128 B swizzled K-major layout [kb 4][64 rows][128 B]: a warp copies one 512 B row (conflict-free)
      unsigned char* xsb = reinterpret_cast<unsigned char*>(xs);
      for (int i = tid; i < per; i += NT) {
        const int r = i >> 5, ch = i & 31;
        const bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + ch * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + ch * 8;
        cp_async16(reinterpret_cast<__nv_bfloat16*>(xsb + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4)), src, v);
      }
    } else {
      for (int i = tid; i < per; i += NT) {
        int r = i / (KS / 8), c = i - r * (KS / 8);
        bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + c * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + c * 8;
        cp_async16(xs + r * XP_FT + c * 8, src, v);
      }
    }
    cp_commit();
  };
  // BIG kernel: per-thread x staging addresses computed once per unit (a stage only adds its k offset)
  const __nv_bfloat16* xsrc[8];
  uint32_t xdst[8];
  uint32_t xval = 0;
  int xn = 0;
  auto prep_x = [&](int j) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int per = ((cnt + 7) >> 3) * 8 * (KS / 8);
    xval = 0;
    xn = per > tid ? min(8, (per - tid + NT - 1) / NT) : 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = tid + k * NT;
      const int r = i >> 5, c = i & 31;
      const bool v = i < per && r < cnt;
      if (PB) xsrc[k] = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;
      else xsrc[k] = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;
      xdst[k] = r * XP_FT + c * 8;
      if (v) xval |= 1u << k;
    }
  };
  auto stage_xp = [&](int ks, int xsl) {
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (k < xn) cp_async16(xs + xdst[k], xsrc[k] + ks * KS, (xval >> k) & 1u);
    cp_commit();
  };
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  int f = S.fseq;  // ring position of the current unit's k-stage 0
  int xq = 0;      // x stage parity counter
  int j, ub;
  job_of(u, j, ub);
#pragma unroll
  for (int k = 0; k < NSF; ++k) fat_issue<PB>(p, S.jE[j], ub, k, ring + ((f + k) % NSF) * UNIT_BYTES, fb + (f + k) % NSF, is_nat(j));
  wait_A(j);
  if (OKC_XPRE) { prep_x(j); stage_xp(0, 0); } else stage_x(j, 0, 0);
  while (true) {
    const int un = u + gridDim.x;
    int jn = 0, ubn = 0;
    if (un < nF) job_of(un, jn, ubn);
    const int e = S.jE[j], cnt = S.jC[j], st = S.jS[j];
    const int ntile = (cnt + 7) >> 3;
    const int ntc = ntile;  // exact token-tile count (rows past cnt are zero-filled)
    const bool nat = is_nat(j);
    const int rb0 = ub * 16 + warp, rb1 = rb0 + 8;
    float acc0[FT_T / 8][4], acc1[FT_T / 8][4];
#pragma unroll
    for (int i = 0; i < FT_T / 8; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) { acc0[i][k] = 0.f; acc1[i][k] = 0.f; }
    for (int ks = 0; ks < NKS; ++ks) {
      if (OKC_XPRE) {
        if (ks + 1 < NKS) stage_xp(ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); prep_x(jn); stage_xp(0, (xq + 1) & 1); }
        else cp_commit();
      } else {
        if (ks + 1 < NKS) stage_x(j, ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); stage_x(jn, 0, (xq + 1) & 1); }
        else cp_commit();
      }
      cp_wait<1>();
      const int q = f + ks, sl = q % NSF;
      mbar_wait(fb + sl, (uint32_t)((q / NSF) & 1));
      if (nat) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // cp.async x writes -> wgmma reads
      CTA_SYNC();  // x stage visible to every warp
      const unsigned char* slot = ring + sl * UNIT_BYTES;
      const __nv_bfloat16* xs = xs0 + (xq & 1) * (FT_T * XP_FT);
      if (nat) {
        switch (ntc) {
          case 1: fat_compute_wg<1>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 2: fat_compute_wg<2>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 5: fat_compute_wg<5>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 6: fat_compute_wg<6>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 7: fat_compute_wg<7>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          default: fat_compute_wg<8>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
        }
      } else {
        switch (ntc) {
          case 1: fat_compute<1>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 2: fat_compute<2>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 5: fat_compute<5>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 6: fat_compute<6>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 7: fat_compute<7>(S, slot, xs, warp, lane, acc0, acc1); break;
          default: fat_compute<8>(S, slot, xs, warp, lane, acc0, acc1); break;
        }
      }
      CTA_SYNC();  // slot sl and the x stage consumed
      ++xq;
      // refill: the stage NSF ahead, in this unit or the next one
      const int kn = ks + NSF;
      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);
      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));
    }
    // epilogue of unit u (registers -> global)
    const float gsc = (PB ? p.gsB[e] : p.gsA[e]) * TWO62;
    if (!PB) {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb0 * 8 + g, gsc, acc0[nt]);
#if OKC_NW == 8
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb1 * 8 + g, gsc, acc1[nt]);
#endif
        }
      __threadfence();
      CTA_SYNC();
      if (tid == 0) atomicAdd(p.doneA + j, 1);
    } else {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          float sq0[2], sq1[2];
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb0 * 16 + g, gsc, acc0[nt], sq0);
#if OKC_NW == 8
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb1 * 16 + g, gsc, acc1[nt], sq1);
          sq0[0] += sq1[0]; sq0[1] += sq1[1];
#else
          (void)sq1;
#endif
          ss_reduce_add(p, st, cnt, nt * 8 + 2 * t, sq0, g);
        }
    }
    f += NKS;
    if (un >= nF) break;
    u = un; j = jn; ub = ubn;
  }
  cp_wait<0>();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  if (tid == 0) S.fseq = f;
  CTA_SYNC();
}

template <bool PB>
__device__ __noinline__ void fat_stream_ni(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  fat_stream<PB, false>(p, S, xbuf, nF, ubs);
}



// issue the phase-A skinny ring prologue as soon as the job table exists (before the positions pass)
__device__ void early_issue(const Params& p, Smem& S) {
#if FAST1_ON || OKC_WGS
  return;  // the first units were issued at kernel start (FAST1), or the WGS producer warp owns the ring
#endif
  if (OKC_FATSTATIC && S.njF > 0) {  // fat units run first and own the ring region
    if (threadIdx.x == 0) S.preissued = 0;
    return;
  }
  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);  // static part (the pool tail is claimed at run time)
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);
  // only the first unit, issued by the last warp so warp 0 (routing) is not held up; the ring prologue in
  // skinny_ring issues the rest after routing, so every CTA's first unit is served ahead of the second ones
  const int n = min(OKC_EARLYN, v1 - v0);
  if (threadIdx.x == NT - 32) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }
    S.preissued = n > 0 ? n : 0;
  }
}

// skinny share [v0, v1) of CTA c when nF fat units go round robin first: every CTA ends at the same modeled cost
#if OKC_WS
// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)
// The NW compute warps only wait for data, compute, release the slot and hand their partial sums over; the helper
// warp (threads NT..NT+31) issues every TMA copy as soon as all compute warps released a slot and runs every unit's
// epilogue, so no CTA-wide barrier sits in the per-unit loop.
// the last WSBLK units keep their partial buffers until the compute warps' own epilogues: WSBLK > WSNR deadlocks the
// partial ring (tuner, Oct 26: every WSNR=2 / WSBLK=3 config hung)
static_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");
#define OKC_WSRF 0
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);
template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int seq0 = seq;
  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;                             // the last phase-B unit's epilogue constants (warp 0)
  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs;
    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {
      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);
      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);
      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);
    }
    if (j != curj) {
      curj = j;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
    }
    const int q = seq0 + i;
    const int sl = q % OKC_NS;
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        uint32_t r0, r1, r2, r3;
        dq4(ww[st], r0, r1, r2, r3);
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    // WSB: the compute warps keep the base loop's CTA barrier per unit (slot consumed + partials visible), thread 0
    // refills the slot and hands the partial buffer to the helper warp, which runs every epilogue off this loop
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
    }
  }
  if (PB && warp < OKC_WSBLK && warp < nU) {  // the CTA's last units: epilogues here, not behind the helper's queue
    const int ul = nU - 1 - warp;
    const int q = seq0 + ul, bb = q % NRB;
    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;
    const int gg = lane >> 2, tt = lane & 3;
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    const int n0 = rb * 16 + gg;
    const float gsc = lgs * TWO62;
    const float nw0 = 1.f + __uint_as_float(lnw0 << 16), nw1 = 1.f + __uint_as_float(lnw1 << 16);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = 2 * tt + h;
      const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
      float sq = v0f * v0f + v1f * v1f;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
      }
      sq += __shfl_xor_sync(0xffffffffu, sq, 4);
      sq += __shfl_xor_sync(0xffffffffu, sq, 8);
      sq += __shfl_xor_sync(0xffffffffu, sq, 16);
      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
    }
  }
  seq = seq0 + nU;
}

__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % OKC_NS;
  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = threadIdx.x & 31;
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int gg = lane >> 2, tt = lane & 3;
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
  for (int q = 0; q < Nh; ++q) {
    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q's slot once every compute warp released it
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
    }
    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again
    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    if (!PB) {
      const float gsc = p.gsA[ce] * TWO62;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gt = (h ? sm.y : sm.x) * gsc, up = (h ? sm.w : sm.z) * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }
      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
        }
      }
    } else {
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
  }
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#endif

template <bool BIG, bool PB>
__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {
  const float rG = __fdividef(1.f, (float)G);
  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;
  auto fcost = [](long long k) -> long long {
    // phase-specific cost of one fat unit in skinny-unit equivalents (phase B's skinny units are slower, so its ratio is lower)
    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);
    return k > 0 ? (long long)fc * k : 0;
  };
  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);  // total cost = G x level
  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;
  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;
  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };
  const long long WG = W(G);
  if (nF > 0 && WG > 0) {
    // split point of CTA x: floor(nS * W(x) / WG) in float (one formula for both neighbours, so the shares tile [0, nS))
    const float sc = __fdividef((float)nS, (float)WG);
    auto split = [&](int x) -> int {
      if (x >= G) return nS;
      const int v = __float2int_rz((float)W(x) * sc);
      return v < nS ? v : nS;
    };
    v0 = split(c); v1 = split(c + 1);
  } else {
    v0 = muldiv_g(nS, c, G, rG); v1 = muldiv_g(nS, c + 1, G, rG);
  }
}

#if OKC_WGS
// ---------------------------------------------------------------- WGS: RS-wgmma for every job (row split, stream-K)
// Unit = (job, 256-row block): phase A 32 per job, phase B 16, each 16 stages of 256 K. Stage = 36 KB of weights,
// for each warpgroup q its 64 rows: 8 KB codes [grp][warp][lane][4 words] (natural mma k order) + 1 KB scales
// [grp][warp][g][h][j]; the stage's x slice (the job's token rows, rounded up to 8, x 256 K as wgmma B core matrices
// [n8][32 k8][8 rows][16 B]) sits in its own ring. Warpgroup q accumulates its 64 rows over K with wgmma m64nNk16
// (A = dequantized codes in registers, N = 8 * n-tiles) and runs its own epilogue (rows g / g + 8 are gate / up).
// Work is split by cost-weighted stage ranges (stream-K): a unit cut between two CTAs adds its partial sums into a
// global buffer (indexed by the CTA boundary) and the second contributor runs the epilogue.
#define OKC_WGSNX 3                             // x ring slots
constexpr int WGS_NS = 3;                       // weight ring slots (36 KB; freed as soon as the codes are in registers)
constexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage's last wgmma retired)
constexpr int WGS_NT = FT_T / 8;                // max n-tiles of a job
constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;  // one warpgroup's weights in a stage (r3l SC6W: 8 KB codes + 768 B scales)
constexpr int WGS_STG = 4 * WGS_PIECE;             // r3l: bytes of a stage in global memory (a ring slot keeps UNIT_BYTES)
static_assert(OKC_SC6W || WGS_STG == UNIT_BYTES, "WGS stage = one ring unit");
constexpr int WGS_XS = WGS_NT * 4096;           // an x slot: up to FT_T rows x 256 K
constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices
static_assert(BX_E + WGS_NX <= 24, "WGS barriers");
constexpr int WGS_SLOTS = 160;                  // fixup slots per phase (>= grid + 1)
constexpr int WGS_FIXW = WGS_NT * 512;          // floats per (slot, warpgroup): [n-tile][128 threads][4]
constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;  // + doneAs
__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {
  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;
}
__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {
  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;
}
// per-stage cost by n-tiles (1..8), phase A and phase B (relative units, <= 255; fit on per-CTA phase times at
// bs 96..256, t_fit2.py): tuner knobs OKC_WGSA / OKC_WGSB = 8 comma-separated values each
#if OKC_B4FIN  // r9: T = 17..32 (judge routing): a steeper phase-A table (2..4 n-tiles: 1.34 / 1.47 / 1.65 x); the flat and the
// main-build tables lost (multi-tile stages carry more fixup / epilogue / release work than their stream time shows)
#define OKC_WGSA 112, 150, 165, 185, 200, 215, 230, 245
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
#else
#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207  // tuner combo_m798 on v182x (Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%)
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
#endif
constexpr unsigned long long wgs_pack8(const int (&c)[8]) {
  unsigned long long r = 0;
  for (int i = 0; i < 8; ++i) r |= (unsigned long long)(c[i] & 255) << (8 * i);
  return r;
}
constexpr int WGS_CA[8] = {OKC_WGSA};
constexpr int WGS_CB[8] = {OKC_WGSB};
constexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB);
static_assert(WGS_CA[0] > 0 && WGS_CA[7] < 256 && WGS_CB[0] > 0 && WGS_CB[7] < 256, "WGS cost tables: 1..255");
__device__ __forceinline__ int wgs_cost(int pb, int cnt) {
  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);
}
#define OKC_WGSREORD 1  // WGS: phase A starts with the trailing partial unit (see wgs_split)
#define OKC_GATHB 8  // WGS: x rows gathered per batch (loads first, then stores); 0 = one row at a time
#define OKC_WGSPR 1  // WGS: the x producer warp computes the stage ranges while the consumers gather
#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B
#define OKC_WGSFCI (OKC_B4FIN ? 3 : OKC_FCIM)  // round 3: 5 in the main build (was 3)
#define OKC_WGSGS 2  // WGS: k16 steps per wgmma commit group (4 serializes the N=16..32 variants at 96 registers: C7512)
__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {
  (void)tag;
  mbar_wait(bar, parity);
}
template <int NTT> struct WGSMMA;
template <> struct WGSMMA<1> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<2> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<3> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<4> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<5> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<6> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<7> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<8> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]), "=f"(d[7][0]), "=f"(d[7][1]), "=f"(d[7][2]), "=f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
// two producer warps: the weights warp streams every stage's 36 KB through the weight ring (TMA, lane 0), the x warp
// gathers every stage's x slice into the x ring (cp.async) and, in phase B, first waits for the job's phase-A units;
// so the weights run ahead by the whole ring even while x waits
// Phase A runs a CTA's trailing partial unit FIRST, then the rest of its range in order: the CTA that holds a unit's
// first stages otherwise reaches them last, finishes the unit (read back + epilogue) and pays it exposed at the phase
// end. Returns where the first sub-range starts: [sp, g1) then [g0, sp) (sp == g0: no reordering). Phase B keeps the
// order (its first stages wait for phase A's units in order).
__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  const int ue = ((g1 - 1) >> 4) << 4;  // first stage of the unit holding the range's last stage
  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}
__device__ void wgs_ranges(const Params& p, Smem& S);  // below (needs the cost tables)
__device__ void wgs_ranges4(const Params& p, Smem& S);  // r4 (PR4): below
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  int ws = 0, n = 0;
  uint32_t wph = 0;
#if OKC_L2CH && OKC_L2H
  int pj = -1;  // r6 (L2CH): the job the L2 policy below is for
  unsigned long long pol = 0x12F0000000000000ull;
#endif
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;  // log2 of 256-row units per job
#if OKC_L2CH && OKC_L2H
    pj = -1;  // (job ids restart per phase)
#endif
    const unsigned char* base = pb ? p.wBs : p.wAs;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
#if OKC_WGS && !OKC_B4FIN
      // r9 XF: weight stage 0, the first x copies (else ~2 us late behind 108 KB of weights), then stages 1, 2
      if (!pb && n == 1) { while (*reinterpret_cast<volatile int*>(&S.r9xf) == 0) { } }
#endif
      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(WGS_STG) : "memory");
#if OKC_L2CH && OKC_L2H
      if (j != pj) {  // once per job: a job whose expert has another chunk job (adjacent in job order) shares its weights
        pj = j;
        const int njt = S.njF + S.njS;
        const bool shared_w = (j + 1 < njt && S.jE[j + 1] == S.jE[j]) || (j > 0 && S.jE[j - 1] == S.jE[j]);
        pol = shared_w ? OKC_L2CHP : 0x12F0000000000000ull;
      }
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)), "l"(pol) : "memory");
#else
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");
#endif
      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    }
  }
}
// x producer: one bulk copy per stage (lane 0): the job's x slice for the stage is contiguous in xA (phase A, written
// by every CTA's gather pass at the start) or xB (phase B, written by the phase-A epilogues), N_pad x 512 B
constexpr int WGS_XROWS = EMAX + 8 * JMAX;  // padded rows of all jobs, upper bound
__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
  // the routing tables are ready (the consumers arrive on barrier 7 right after routing): this idle warp computes the
  // stage ranges while the consumers gather phase A's x
  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
#if OKC_PR4
  if ((threadIdx.x & 31) < 4) wgs_ranges4(p, S);
#else
  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);
#endif
  __syncwarp();
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  int xs = 0, n = 0;
  uint32_t xph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;
    const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;
    int curj = -1, npad = 8;
    const __nv_bfloat16* jb = xsrc;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl;
      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);
      if (j != curj) {
        curj = j;
        npad = ((S.jC[j] + 7) >> 3) << 3;
        jb = xsrc + (size_t)S.jrow[j] * HD;
        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }  // every row gathered
      }
      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }
      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
#if OKC_WGS && !OKC_B4FIN
      if (!pb && (n == WGS_NX - 1 || i + 1 == g1 - g0)) *reinterpret_cast<volatile int*>(&S.r9xf) = 1;  // r9 XF
#endif
      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
  }
}
// phase A's x gather (all consumer threads of every CTA, once, at the start): this CTA's share of the jobs' padded rows
// copied into xA's per-stage wgmma layout, then counted into xready (padding rows are counted, not written)
__device__ void wgs_gather(const Params& p, Smem& S) {
  const int tid = threadIdx.x, nj = S.njF + S.njS;
  const int R = S.jrow[nj];  // R * G < 2^24: exact float muldiv, no divide routine
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
  int j = 0;
  // all of a batch's row loads first, then the stores: one x row at a time was one dependent HBM round trip per row
  // (the next row's load could not move above the previous row's store: they may alias), 1.3-7 us at bs256
  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");
  constexpr int GB = OKC_GATHB;
  for (int qb = q0; qb < q1; qb += GB) {
    uint4 v[GB];
    uint4* d[GB];
#pragma unroll
    for (int k = 0; k < GB; ++k) {
      const int q = qb + k;
      d[k] = nullptr;
      if (q < q1) {
        while (S.jrow[j + 1] <= q) ++j;
        const int r = q - S.jrow[j], cnt = S.jC[j];
        if (r < cnt) {
          const int ntt = (cnt + 7) >> 3;
          d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((((tid >> 5) * ntt + (r >> 3)) * 32 + (tid & 31)) * 8 + (r & 7));
          {  // KPERM3: k8 group G = tid & 31 of stage tid >> 5 holds the bf16 pairs at stage offsets 2 B(G) + 64 t,
             // B(G) = 8 ((G >> 1) & 3) + 2 (G >> 3) + (G & 1)
            const int G = tid & 31;
            const unsigned* xs32 = reinterpret_cast<const unsigned*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD + (tid >> 5) * 256)
                                   + (8 * ((G >> 1) & 3) + 2 * (G >> 3) + (G & 1));
            v[k] = make_uint4(__ldg(xs32), __ldg(xs32 + 32), __ldg(xs32 + 64), __ldg(xs32 + 96));
          }
        }
      }
    }
#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (d[k]) *d[k] = v[k];
  }
  CTA_SYNC();
  if (tid == 0) {  // count this CTA's rows per job (release: the CTA's stores are ordered before by the barrier)
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
}
// Work left from the previous segment, done after the next segment's first wgmma group (so the stores and atomics it
// waits for have long completed): kind 1 = release doneA for job j (phase A), kind 2 = add the stage count of a split
// unit's share and, if that completes the unit, read the summed partials back and run its epilogue.
struct WgsPend { int kind, u, slot, nst; };
template <bool PB>
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];
  if (!PB) {
    // SwiGLU straight into xB (phase B's per-stage wgmma B layout) instead of row-major inter
    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;
    const float gsc = p.gsA[e] * TWO62;
    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const float gt = d4[h] * gsc, up = d4[2 + h] * gsc;
        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
                     __float2bfloat16(probe_q(v)));
      }
    }
  } else {
    // y rows only: the per-route sum of squares is formed by the final pass from these rows (no shuffles / atomics)
    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      if (tk < cnt) {
#if OKC_YBF
        __nv_bfloat16* ybh = reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0);
        st_keep_bf16(ybh, __float2bfloat16(d4[h] * w0));
        st_keep_bf16(ybh + 8, __float2bfloat16(d4[2 + h] * w1));
#else
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);
#endif
      }
    }
#if OKC_B4FIN
    {  // bet4_claude: this tile's 16-column sum of squares per token, reduced over the 8 column lanes (fixed tree)
      float q0 = 0.f, q1 = 0.f;
      if (tok0 < cnt) { const float a = d4[0] * w0, b = d4[2] * w1; q0 = a * a + b * b; }
      if (tok0 + 1 < cnt) { const float a = d4[1] * w0, b = d4[3] * w1; q1 = a * a + b * b; }
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) {
        q0 += __shfl_xor_sync(0xffffffffu, q0, o);
        q1 += __shfl_xor_sync(0xffffffffu, q1, o);
      }
      if (g == 0) {
        if (tok0 < cnt) p.sspart[(size_t)(jst + tok0) * 256 + r16] = q0;
        if (tok0 + 1 < cnt) p.sspart[(size_t)(jst + tok0 + 1) * 256 + r16] = q1;
      }
    }
#endif
  }
}
#if OKC_R11EPI
// r11 (EPI): a whole unit's epilogue with the job's values loaded once (the stores' "memory" clobber made every n-tile
// reload jC / jS / jE / jrow and gsA / gsB / normw: an L1 round trip at the head of each tile's dependency chain) and
// the per-token math unconditional (padding tokens' values are computed and dropped; only the stores are predicated:
// no branch per token). Same operations in the same order on every stored value.
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_epi_seg(const Params& p, const Smem& S, int j, int r16, const float (&d)[WGS_NT][4]) {
  static_assert(!OKC_B4FIN, "r11 EPI: main build only (B4FIN's phase-B epilogue also forms the ss partials)");
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];
  if (!PB) {
    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;
    const float gsc = p.gsA[e] * TWO62;
    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7) + (size_t)(((col >> 8) * ntt * 32 + ((col & 255) >> 3)) * 64 + 2 * t * 8);
    float v[NTT][2];
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float gt = d[nt][h] * gsc, up = d[nt][2 + h] * gsc;
        v[nt][h] = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
      }
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        if (nt * 8 + 2 * t + h < cnt) st_keep_bf16(xb + (nt * 2048 + h * 8), __float2bfloat16(probe_q(v[nt][h])));
  } else {
    const int n0 = r16 * 16 + g;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
#if OKC_YBF
    __nv_bfloat16* yb = reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + 2 * t) * HD + n0);
#else
    float* yb = p.ybuf + ((size_t)(jst + 2 * t) * HD + n0);
#endif
    float y[NTT][4];
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h) { y[nt][h] = d[nt][h] * w0; y[nt][2 + h] = d[nt][2 + h] * w1; }
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        if (nt * 8 + 2 * t + h < cnt) {
#if OKC_YBF
          st_keep_bf16(yb + (size_t)(nt * 8 + h) * HD, __float2bfloat16(y[nt][h]));
          st_keep_bf16(yb + (size_t)(nt * 8 + h) * HD + 8, __float2bfloat16(y[nt][2 + h]));
#else
          st_keep_f32(yb + (size_t)(nt * 8 + h) * HD, y[nt][h]);
          st_keep_f32(yb + (size_t)(nt * 8 + h) * HD + 8, y[nt][2 + h]);
#endif
        }
  }
}
#endif
template <bool PB>
__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {
  if (pd.kind == 0) return;
  const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  if (pd.kind == 1) {
    // release (cumulative over the warpgroup's epilogue stores ordered by the barrier); pairs with the producer's
    // acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2
#if OKC_PAIR
    // (pd.nst: 1, or 2 when unit ub - 1's release was carried into this one: both halves of stage ub / 2 at once)
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(pd.nst) : "memory");
#else
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
#endif
    pd.kind = 0;
    return;
  }
  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;
  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase
  if constexpr (DF) {
  // designated finisher (kind 3): the share processed last in time by construction (phase A: the one holding the
  // unit's last stage, run early by the reorder; phase B: the one holding stage 0, at its CTA's end). The others just
  // count with a fire-and-forget release (no atomic round trip, no drain wait); the finisher waits for their counts
  if (pd.kind == 2) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(fc), "r"(pd.nst) : "memory");
    pd.kind = 0;
    return;
  }
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  pd.kind = 0;
  } else {
  if (wt == 0) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");
    S.wglast[wg] = old + pd.nst == 16;
  }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  const bool last = S.wglast[wg];
  pd.kind = 0;
  if (!last) return;
  }
  // R4X-N2: no fence.acq_rel.gpu here: thread 0's acq_rel atomic (or acquire load) + bar.sync 3 + wg order the loads below
  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;
  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
  // two n-tiles' loads in flight before their zeroing stores and epilogues (the scout): one at a time, each load
  // waits behind the previous tile's stores (they may alias), ntt serialized L2 round trips under full HBM load
  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {
    float4 v[2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) {
        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
        const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
        wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);
      }
  }
  if (wt == 0) *fc = 0;
  if (!PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
}
// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                        int& seq, WgsPend& pd) {
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c = (OKC_SC6W ? 4u : 16u) + p.zero;  // opaque 16 (r3l SC6W: 4): the scale decode's multiply stays an IMAD (FMA pipe)
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma (no zeroing: ptxas would serialize the wgmmas, C7513)
  // k16 steps per commit group: the whole function shares one register budget (96 at 544 threads) and ptxas
  // serializes every wgmma if any pipeline does not fit (C7512), so the wide-N variants commit 2 steps at a time
  constexpr int GS = OKC_WGSGS;
  uint32_t af[2][GS][4];
  int prev = -1;  // x slot the last committed group may still read
  // slot-0 descriptor and code pointer, once per segment: a stage adds its slot's offset (the address field cannot
  // carry: shared addresses >> 4 stay below 2^14) instead of rebuilding the descriptor (~8 uniform ops per stage)
  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);
  const unsigned char* pc0 = ring + wg * WGS_PIECE;
  auto stage = [&](auto first) {
    constexpr bool FIRST = decltype(first)::value;
    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
    // codes and scales one 4-step group at a time (fewer live registers)
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
    // KPERM3: K is permuted inside the stage so that step j of every load group is scale block 4 t + j for this lane:
    // the 4 scale pairs ([j][h] bytes) come from one 8-byte load per stage and are built once per stage
#if OKC_SC6W
    // r3l SC6W (idea from keell0renz's public muse_moe_w4a16 file, Apache-2.0; written here by hand): this thread's 8 scale
    // bytes of the stage ([j][h]: j = k16 step of a load group, h = rows g / g + 8) are kept as v = byte - 0x40 (6 bits)
    // in 6 bytes: plane W (4 B per thread) byte n = 4 v_n | c_n (n < 4), plane H (2 B per thread) byte n = 4 v_(4+n) |
    // c_(4+n) (n < 2), with c_0..c_2 the 2-bit pieces of v_6 and c_3..c_5 those of v_7. Each pair below holds (4 v_a,
    // 4 v_b) in the low bytes of its halves, and * 4 + 0x6000 per half == (0x40 + v) * 16 + 0x5C00: the 8-bit decode's
    // value, bit for bit. The 4 pairs are built once per stage.
    uint32_t sgs[4];
    {
      const int tt = wp * 32 + lane;
      const uint32_t W = *reinterpret_cast<const uint32_t*>(pc + 8192 + tt * 4);
      const uint32_t H = (uint32_t)*reinterpret_cast<const unsigned short*>(pc + 8192 + 512 + tt * 2);
      const uint32_t X = W & 0xFCFCFCFCu, Hm = H & 0xFCFCu;
      const uint32_t p6 = (W & 0x00030303u) * 0x04104000u;                    // byte 3 = 4 v_6; bits 0..13 zero
      const uint32_t p7 = ((H & 0x0303u) | (W & 0x03000000u)) * 0x10400004u;  // byte 3 = 4 v_7
      sgs[0] = __byte_perm(X, 0u, 0x4140u) * k16c + 0x60006000u;
      sgs[1] = __byte_perm(X, 0u, 0x4342u) * k16c + 0x60006000u;
      sgs[2] = __byte_perm(Hm, 0u, 0x4140u) * k16c + 0x60006000u;
      sgs[3] = __byte_perm(p6, p7, 0x0703u) * k16c + 0x60006000u;            // (selector 0 = p6 byte 0 = 0)
    }
#else
    const uint2 scs = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);
#endif
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cwn;
#if !OKC_SC6W
      const uint2 scg = scs;
#endif
      if (lg < 3) {
        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
      } else if (lane == 0) {
        mbar_arrive1(S.bars + BW_E + ws);  // every code and scale of this stage is in registers: weights slot free
      }
#pragma unroll
      for (int h = 0; h < 4 / GS; ++h) {
        const int ci = lg * (4 / GS) + h;  // commit group index in the stage
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          const int js = h * GS + jj;      // k16 step within the 4-step load group
          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;
          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
          // scales stored [j][h] (rows g, g+8 of step j side by side): one PRMT + one IMAD build both rows' scales in
          // one bf16x2, and the multiplies broadcast its halves (HMUL2 .H0_H0 / .H1_H1): 2 instructions per k16 fewer
#if OKC_SC6W
          const uint32_t sgp = sgs[js];
#else
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
#endif
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);
        }
        wg_fence();
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
          else
            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
        }
        wg_commit();
        wg_wait<1>();
        if (ci == 0) {
          if (prev >= 0) {  // the previous stage's last group just retired: its x slot is free
            if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
            prev = -1;
          }
        }
        // the previous segment's deferred work, behind group OKC_WGSFCI of the first stage (scout: CI 3 lets this WG's
        // own red.adds / epilogue stores drain a little longer before the count / release atomics: +0.5-1%)
        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);
      }
    }
    prev = xs;
    ++seq;
  };
  stage(std::true_type{});
#pragma unroll 1
  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});
  wg_wait<0>();
  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
  if (OKC_WGSX & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }
  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
    // kind 3: this share finishes the unit (designated finisher, OKC_WGSDFIN): phase A the share holding stage 15,
    // phase B the one holding stage 0
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#if OKC_R11EPI
  wgs_epi_seg<PB, NTT>(p, S, j, r16, d);
#else
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
#endif
#if OKC_PAIR
  if (!PB) pd = WgsPend{1, u, -1, 1};  // the doneA release waits until the stores are done: deferred (nst: units)
#else
  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred
#endif
}
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {
  // (__forceinline__: outlined by NVVM when it grows, a CALL that makes ptxas serialize every wgmma, C7510)
  constexpr int UBL = PB ? 4 : 5;
  WgsPend pd{0, 0, 0, 0};
#pragma unroll 1
  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {
    if (gs >= ge) {  // [sp, g1) done: then [g0, sp); the bounds come back from smem (no extra live state)
      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);
      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;
    }
    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
    // a unit split between CTAs: its fixup slot is the CTA holding its first stage
    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));
    const int ntt = (S.jC[u >> UBL] + 7) >> 3;
#if OKC_PAIR
    // whole unit 2k + 1 right after whole unit 2k (same job: 32 units per job): unit 2k's release is not flushed at
    // this segment's start but carried, and goes out with this unit's (phase-B stage k waits for both anyway): one
    // release (MEMBAR.GPU, ~0.9 us of warpgroup stall) fewer per pair, no added dependency delay
    int carry = 0;
    if (!PB && slot < 0 && pd.kind == 1 && pd.u + 1 == u && !(pd.u & 1)) { carry = pd.nst; pd.kind = 0; }
#endif
    switch (ntt) {
      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;
    }
#if OKC_PAIR
    pd.nst += carry;  // (carry != 0: this segment was a whole phase-A unit, pd = its kind-1 release)
#endif
    gs += s1 - s0;
  }
  wgs_flush<PB>(p, S, pd);
}
// cost-weighted stage ranges: per-stage cost by the job's n-tiles, a prefix per job and phase (S.wcp, S.wcpB)
// all 32-bit (tot * c < 2^32: tot <= 136 jobs x 512 stages x 255 x 132): no 64-bit divide subroutine calls, which run
// from a cold instruction cache at the start of every launch
__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {
  uint32_t acc = 0, accB = 0;
  for (int j = 0; j < nj; ++j) {
    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;
    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;
    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;
  }
  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;
  S.wtot = acc; S.wtotB = accB;
}
// stage at cost position x (phase pb), the stage whose cost interval holds x
__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  if (nj <= 0) return 0;  // NJ0: no local route in this launch: empty ranges (the cost lookup below read a stale S.jC[0])
  const int* pre = pb ? S.wcpB : S.wcp;
  int lo = 0, hi = nj;  // job: last j with prefix(j) <= x
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);
  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);
}
__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);
  g1 = c == G - 1 ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c + 1) / (uint32_t)G);
}
// this CTA's stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same
__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {
  const int nj = S.njF + S.njS;
  wgs_prefix(S, nj);
  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);
  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);
}
#if OKC_PR4
// r4 (PR4): the same ranges (same 32-bit formulas) on lanes 0..3: lane bit 0 = phase, bit 1 = range end (c or c + 1).
// Lanes 0 / 1 write the phase prefixes and totals; every lane runs one stage_at (the single-lane routine ran 4 of them
// and 2 prefix passes back to back, and held 4 inlined copies of the search + divide: cold instruction fetch)
__device__ __noinline__ void wgs_ranges4(const Params& p, Smem& S) {
  const int ln = threadIdx.x & 3, pb = ln & 1, hi = ln >> 1, sh = 9 - pb;
  const int nj = S.njF + S.njS;
  const unsigned long long tab = pb ? WGS_PB : WGS_PA;
  int* pre = pb ? S.wcpB : S.wcp;
  uint32_t acc = 0;
  for (int j = 0; j < nj; ++j) {
    if (!hi) pre[j] = (int)acc;
    acc += (uint32_t)((tab >> (8 * (((S.jC[j] + 7) >> 3) - 1))) & 255ull) << sh;
  }
  if (!hi) {
    pre[nj] = (int)acc;
    if (pb) S.wtotB = acc; else S.wtot = acc;
  }
  __syncwarp(0xFu);  // lanes 2 / 3 search the prefix lanes 0 / 1 wrote
  const int G = gridDim.x, c = blockIdx.x + hi;
  int g;
  if (c == G) {
    g = nj << sh;
  } else {
    const uint32_t x = acc * (uint32_t)c / (uint32_t)G;
    int lo = 0, up = nj;  // job: last j with prefix(j) <= x
    while (up - lo > 1) { const int mid = (lo + up) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else up = mid; }
    const uint32_t cs = (uint32_t)((tab >> (8 * (((S.jC[lo] + 7) >> 3) - 1))) & 255ull);
    g = (lo << sh) + (int)((x - (uint32_t)pre[lo]) / cs);
  }
  S.wgr[2 * pb + hi] = g;
}
#endif
// the CTA whose range holds stage s (the fixup slot of a split unit is the CTA holding its first stage)
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  const int j = s >> (9 - pb);
  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);
  // largest c with floor(tot * c / G) < xe
  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);
  if (c > G - 1) c = G - 1;
  while (c + 1 < G && tot * (uint32_t)(c + 1) / (uint32_t)G < xe) ++c;
  while (c > 0 && tot * (uint32_t)c / (uint32_t)G >= xe) --c;
  return c;
}
#endif

template <bool PB, bool BIG>
__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {
  const int njF = S.njF, njS = S.njS;
  const int ubs = PB ? UB_B : UB_A, rbs = PB ? RB_B : RB_A;
  const int nS = njS * rbs, nF = njF * ubs;
  // fat units first (round robin over CTAs), then a skinny share sized so every CTA carries the same cost
#if OKC_TINY
  if (nF > 0) __trap();  // tiny build (M <= 8): a job never has more than 8 tokens, so there are no fat units
#else
  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }
#endif
  const int G = gridDim.x, c = blockIdx.x;
  int v0, v1;
  const int nSs = nS - dyn_pool(nS, G);  // static skinny units; [nSs, nS) is the run-time pool
  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);
  // cross-phase prefill: a CTA without phase-B fat units starts phase B with its skinny share, so phase A's last ring
  // refills already fetch those units (same split function as phase B uses)
  int cB0 = 0, cBn = 0;
  bool cpf = false;
  if (!PB) {
    const int nFB = njF * UB_B;
    if (c >= nFB) {
      int b1;
      skinny_split<BIG, true>(njS * RB_B - dyn_pool(njS * RB_B, G), nFB, G, c, cB0, b1);
      cBn = b1 - cB0;
      cpf = OKC_CPFF || njF == 0;
    }
  }
#if OKC_WS
  (void)cpf; (void)cBn;
  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);
#else
  skinny_ring<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq,
                  reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<unsigned char*>(xbuf) + OKC_NS * UNIT_BYTES), cpf, cB0, cBn, nS, nSs);
#endif
}

// a volatile shared load at every use: hoisted, the pointer would stay live across the accumulate loops (96-register
// cap: spills)
__device__ __forceinline__ float* ss_cur() {
#if OKC_WGS
  unsigned long long v;
  asm volatile("ld.volatile.shared.u64 %0, [%1];\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));
  return reinterpret_cast<float*>(v);
#else
  return reinterpret_cast<const Smem*>(smem_raw)->ssp;  // the skinny / fat builds have registers to spare: let it hoist
#endif
}
#if OKC_WGS && OKC_B4FIN && OKC_B4RK
// r4 (B4RK): 32-bit global store under a run-time predicate: the instruction is always there, so the dry pass of the
// tail goes through the same instructions as the real one
__device__ __forceinline__ void st_pred_b32(void* a, uint32_t v, int dry) {
  asm volatile("{\n .reg .pred q;\n setp.eq.s32 q, %2, 0;\n @q st.global.b32 [%0], %1;\n}\n" ::"l"(a), "r"(v), "r"(dry) : "memory");
}
#endif
#if OKC_R11WRM
// r11 (WRM): global stores under a run-time predicate (dry pass: off); the instruction is always there
__device__ __forceinline__ void wrm_st_b32(void* a, uint32_t v, int dry) {
  asm volatile("{\n .reg .pred q;\n setp.eq.s32 q, %2, 0;\n @q st.global.b32 [%0], %1;\n}\n" ::"l"(a), "r"(v), "r"(dry) : "memory");
}
__device__ __forceinline__ void wrm_st_v4(void* a, uint4 v, int dry) {
  asm volatile("{\n .reg .pred q;\n setp.eq.s32 q, %5, 0;\n @q st.global.v4.b32 [%0], {%1, %2, %3, %4};\n}\n"
               ::"l"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(dry) : "memory");
}
#endif
#if OKC_YBF
// r3l YBF: 8 bf16 y values (one 16 B load) as fp32 (exact: a bf16's bits are its fp32's upper half)
__device__ __forceinline__ float r3l_bl(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float r3l_bh(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ float r3l_sumsq8(uint4 u) {
  const float a0 = r3l_bl(u.x), a1 = r3l_bh(u.x), a2 = r3l_bl(u.y), a3 = r3l_bh(u.y);
  const float b0 = r3l_bl(u.z), b1 = r3l_bh(u.z), b2 = r3l_bl(u.w), b3 = r3l_bh(u.w);
  return a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
}
__device__ __forceinline__ void r3l_fma8(float* acc, float r, uint4 u) {
  acc[0] += r * r3l_bl(u.x); acc[1] += r * r3l_bh(u.x); acc[2] += r * r3l_bl(u.y); acc[3] += r * r3l_bh(u.y);
  acc[4] += r * r3l_bl(u.z); acc[5] += r * r3l_bh(u.z); acc[6] += r * r3l_bl(u.w); acc[7] += r * r3l_bh(u.w);
}
__device__ __forceinline__ uint4 r3l_pack8(const float* acc) {
  uint4 o;
  uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    ow[i] = *reinterpret_cast<uint32_t*>(&v);
  }
  return o;
}
#endif
template <bool BIG>
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);
  const int tid = threadIdx.x;
  DBG(0);
  // programmatic dependent launch: everything below reads the previous kernel's outputs (routing, x) or writes
  // scratch, so wait for it here; without the launch attribute this returns at once
#if !(OKC_WGS && OKC_EPW)
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
#if OKC_WS
  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif
#if OKC_WGS
  if (tid >= NT) {
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;
  }
#endif
#if FAST1_ON
  // tiny build (M <= 8): every touched expert is one skinny job, in expert order, so this CTA's first units follow
  // from the touched-expert mask alone; the last warp issues them before routing, off the cold-start critical path
  if (tid >= NT - 32) {
    const int ln = tid & 31;
#if OKC_SPLIT
    // per-expert entry counts -> jobs per expert (ceil(c / 8)) -> inclusive prefix over experts (lane = expert)
    S.fcnt[ln] = 0;
    __syncwarp();
    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {
      const int i = i0 + ln;
      int e = -1;
      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));
    }
    __syncwarp();
    int jincl = (S.fcnt[ln] + SK_T - 1) / SK_T;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, jincl, d); if (ln >= d) jincl += v; }
    const int njob = __shfl_sync(0xffffffffu, jincl, 31);
    const int nS1 = njob * RB_A - dyn_pool(njob * RB_A, gridDim.x);
    const float rG1 = __fdividef(1.f, (float)gridDim.x);
    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);
    const int n1 = min(OKC_FAST1, v11 - v01);
    int fex[OKC_FAST1 > 0 ? OKC_FAST1 : 1];
#pragma unroll
    for (int i = 0; i < OKC_FAST1; ++i) {
      const int jj = (v01 + i) >> RB_A_LOG;   // (n1 is warp-uniform: every lane runs the same ballots)
      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;
    }
#elif OKC_F1EARLY
    // the topk loads go out first (E <= 64 in the tiny builds: two per lane at most) and lane 0 initializes the
    // barriers while they are in flight (the inits used to wait behind the cold loads)
    uint32_t m = 0;
    const int E8 = p.M * 8;
    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;
    if (ln == 0) {
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (ln < E8) { const int e = ev0 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    if (ln + 32 < E8) { const int e = ev1 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    for (int i = ln + 64; i < E8; i += 32) {  // (never in the tiny builds: M <= 8)
      const int e = __ldg(p.topk_ids + i) - p.lo;
      if (e >= 0 && e < p.nloc) m |= 1u << e;
    }
    m = __reduce_or_sync(0xffffffffu, m);
#endif
    if (ln == 0) {
#if OKC_SPLIT || !OKC_F1EARLY
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
#if OKC_SPLIT
      const int v0 = v01, n = n1;
#else
      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);  // static part (pool tail claimed at run time)
      const float rG = __fdividef(1.f, (float)gridDim.x);
      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
      const int n = min(OKC_FAST1, v1 - v0);
#endif
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#if OKC_SPLIT
#pragma unroll
      for (int i = 0; i < OKC_FAST1; ++i) {
        if (i >= n) break;
        const int v = v0 + i;
        const int e = fex[i];
#else
      for (int i = 0; i < n; ++i) {
        const int v = v0 + i;
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);
#endif
        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;
        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");
      }
      S.preissued = n > 0 ? n : 0;
    }
  }
#endif
  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later
  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));
  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code -> bf16x2(value * 2^64), computed (no global load)
    const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;
    uint32_t b;
    if (ex == 15 && mt == 7) {
      b = 0xFFFFu;
    } else {
      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);
      v *= 0x1p64f;
      b = __float_as_uint(sgn ? -v : v) >> 16;
    }
    S.lut[i] = b | (b << 16);
  }
  if (tid == 0) {
#if !FAST1_ON
#if OKC_WGS
    // full slots (TMA tx), empty slots (one arrival per consumer warp), partials ready (every consumer thread),
    // partial buffer free (the epilogue warpgroup)
    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);
#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
#endif
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    S.preissued = 0;
#endif
    S.seq = 0;
    S.fseq = 0;
    S.preissuedB = 0;
#if OKC_WGS && !OKC_B4FIN
    S.r9xf = 0;
#endif
    S.xbuf_ptr = xbuf;
    // the generation can only change at this launch's grid barrier, which every CTA reaches after this point
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
#if SKDYN
  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;
#endif
#if OKC_WGS && OKC_EPW
  // (EPW: the barrier / LUT / generation prologue above reads nothing the previous kernel writes, so under PDL it ran
  // while that kernel drained; the producer warps returned above without waiting: they read weights and this launch's
  // own xA / flags only)
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  route(p, S);
#if OKC_WGS && OKC_RAE && !OKC_WGSPR
#error "OKC_RAE: route() releases barrier 7 for the x producer's ranges (OKC_WGSPR)"
#endif
#if OKC_WGS && OKC_WGSPR && !OKC_RAE
  __threadfence_block();
  asm volatile("bar.arrive 7, %0;\n" ::"n"(NT + 32) : "memory");  // the x producer computes the stage ranges
#endif
  DBG(1);
  {  // the previous launch's ss copy and barrier counter are the next launch's: zero them now (this CTA's share)
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
#if OKC_WS
  if (tid == 0) {  // this CTA's unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)
    const int G = gridDim.x, c = blockIdx.x;
    const float rG = __fdividef(1.f, (float)G);
    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
#endif
#if OKC_WGS
  if (tid < 32) {  // padded-row base of every job in xA / xB (warp scan)
    const int nj = S.njF + S.njS;
    int carry = 0;
    for (int j0 = 0; j0 < nj; j0 += 32) {
      const int j = j0 + tid;
      const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;
      int incl = v;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) { const int t = __shfl_up_sync(0xffffffffu, incl, d); if (tid >= d) incl += t; }
      if (j < nj) S.jrow[j] = carry + incl - v;
      carry += __shfl_sync(0xffffffffu, incl, 31);
    }
    if (tid == 0) S.jrow[nj] = carry;
  }
  CTA_SYNC();
  // phase A's x into xA (this CTA's share of the rows), before the producers start: its x reads do not queue behind
  // every CTA's first weight stages
  wgs_gather(p, S);
  DBG(7);
  WS_JOIN();
  DBG(6);
  {
    unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
    int seq = 0;
    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);
    DBG(2);
    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);
    DBG(3);
  }
#else
  run_phase<false, BIG>(p, S, xbuf);
  DBG(2);
  run_phase<true, BIG>(p, S, xbuf);
  DBG(3);
#endif
#if OKC_WS
  WS_JOIN();  // the helper's last epilogue stores are visible
#endif
  // programmatic dependent launch: the next kernel may start its own prologue now (its griddepcontrol.wait still
  // waits for this whole grid to finish and its writes to be visible)
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  // grid barrier on this launch's counter (two, by generation: the other one was zeroed by this launch for the next);
  // the last arriver bumps the generation, which no CTA of this launch reads again
#if OKC_WGS && OKC_B4FIN && OKC_B4RK
  {
    // r4 (B4RK): the tail by grid-barrier arrival rank. Rank r < min(M * 4, grid) runs final item r (then r + G, ...;
    // the B4FIN item: token it / 4, 1024 columns, route sums of squares from the fixed-order epilogue partials) and a
    // share of the per-launch flag zeroing; the last arrivers have nothing to do and exit at the release. The working
    // CTAs (but the very last) first run the same loop body dry (stores off) while the stragglers finish: that warms
    // the instruction cache (one inline code copy, 2 passes; out of line it costs a 304-byte ABI stack and ~12 us)
    static_assert(NT == 512 && NW == 16, "B4FIN: 16 warps = 8 routes x 2 halves, 512 threads x 2 columns");
    constexpr int SEG = 1024, NSEG = HD / SEG;
    CTA_SYNC();
    int* bc = p.ctr + 4 + (S.gen & 1);
    int* rk = reinterpret_cast<int*>(xbuf) + 64;  // the ring is free now (red uses floats 0..15)
    if (tid == 0) {
      __threadfence();
      const int old = atomicAdd(bc, 1);
      if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
      *reinterpret_cast<volatile int*>(rk) = old;
    }
    CTA_SYNC();
    const int rank = *reinterpret_cast<volatile int*>(rk);
    const int G = gridDim.x, nit = p.M * NSEG, nz = nit < G ? nit : G;
    const bool work = rank < nz;
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
#pragma unroll 1
    for (int pass = (work && rank < G - 1) ? 0 : 1; pass < 2; ++pass) {
      const int dry = 1 - pass;
      if (pass == 1) {
        if (tid == 0) { while (ld_acquire(bc) < G) __nanosleep(64); }
        CTA_SYNC();
        DBG(4);
        if (!work) break;
      }
#pragma unroll 1
      for (int it = rank; it < nit; it += nz) {
        const int tk = it / NSEG, col = (it % NSEG) * SEG + tid * 2;
        float2 yv[8];
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) {
          const int q = S.pos[tk * 8 + sl];
          yv[sl] = q >= 0 ? __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col)) : make_float2(0.f, 0.f);
        }
        {
          const int sl = warp >> 1, hh = warp & 1;
          const int q = S.pos[tk * 8 + sl];
          float v = 0.f;
          if (q >= 0) {
            const float4 a = __ldcg(reinterpret_cast<const float4*>(p.sspart + (size_t)q * 256 + hh * 128 + lane * 4));
            v = (a.x + a.y) + (a.z + a.w);
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
          if (lane == 0) red[warp] = v;
        }
        CTA_SYNC();
        float2 acc = make_float2(0.f, 0.f);
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) {
          const int q = S.pos[tk * 8 + sl];
          if (q >= 0) {
            const float r = p.topk_w[tk * 8 + sl] * rsqrtf((red[2 * sl] + red[2 * sl + 1]) * (1.f / HD) + 1e-8f);
            acc.x += r * yv[sl].x; acc.y += r * yv[sl].y;
          }
        }
        __nv_bfloat162 o = __floats2bfloat162_rn(acc.x, acc.y);
        st_pred_b32(p.out + (size_t)tk * HD + col, *reinterpret_cast<uint32_t*>(&o), dry);
        CTA_SYNC();  // red is reused by the next item
      }
      if (work) {  // this rank's share of the flag zeroing (phases A and B are over everywhere after the release)
        const int i0 = rank * NT + tid;
        if (i0 < 2) st_pred_b32(p.ctr + i0, 0u, dry);  // dynamic pool counters
        for (int i = i0; i < JMAX; i += nz * NT) { st_pred_b32(p.doneA + i, 0u, dry); st_pred_b32(p.xready + i, 0u, dry); }
        for (int i = i0; i < JMAX * 16; i += nz * NT) st_pred_b32(p.doneAs + i, 0u, dry);
      }
    }
    DBG(5);
    return;
  }
#endif
#if OKC_FINPF && OKC_FINCOL && !OKC_WGS
  // the CTA's first final item: its routing (pos) and route weights are inputs, so they load before the barrier
  constexpr int FSEGP = HD / (NT * 4);
  const int itp = blockIdx.x;
  const bool hasp = itp < p.M * FSEGP;
  const int tkp = hasp ? itp / FSEGP : 0;
  int qvp[8];
  float wvp[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;
    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;
  }
#endif
#if OKC_R11FIN
  // r11 (FIN): the first final token's route weights (an input) into L1 while the grid barrier waits
  if (tid < 8 && (int)blockIdx.x < p.M) asm volatile("prefetch.global.L1 [%0];\n" ::"l"(p.topk_w + blockIdx.x * 8 + tid));
#if OKC_R3FP
  if (tid >= 8 && tid < 16 && (int)(blockIdx.x + gridDim.x) < p.M)  // r3l FP: the CTA's second token's too
    asm volatile("prefetch.global.L1 [%0];\n" ::"l"(p.topk_w + (blockIdx.x + gridDim.x) * 8 + tid - 8));
#endif
#endif
#if OKC_R11WRM
  // r11 (WRM): every CTA but the last to arrive runs the post-barrier code once dry (stores off, the same instructions)
  // while the stragglers finish: the zeroing and final-pass code (executed once per launch, after the judge's L2 flush
  // fetched from DRAM at ~0.1 us per line on the critical tail) is then warm when the barrier releases (as B4RK)
  static_assert(OKC_WGS && OKC_WGSFINSS && !OKC_B4FIN, "r11 WRM: main WGS build");
  CTA_SYNC();
  int* const bcw = p.ctr + 4 + (S.gen & 1);
  volatile int* const rkw = reinterpret_cast<volatile int*>(xbuf) + 192;  // (red uses floats 0 .. NW * 8 + 7)
  if (tid == 0) {
    __threadfence();
    const int old = atomicAdd(bcw, 1);
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    *rkw = old;
  }
  CTA_SYNC();
#pragma unroll 1
  for (int pass = *rkw < (int)gridDim.x - OKC_R3DS ? 0 : 1; pass < 2; ++pass) {  // (r3l DS: near-last arrivers skip the dry pass)
  const int dry = 1 - pass;
  if (pass == 1) {
    if (tid == 0) { while (ld_acquire(bcw) < (int)gridDim.x) __nanosleep(64); }
    CTA_SYNC();
    DBG(4);
  }
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) wrm_st_b32(p.ctr + i0, 0u, dry);  // dynamic pool counters
    for (int i = i0; i < JMAX; i += G * NT) wrm_st_b32(p.doneA + i, 0u, dry);
    for (int i = i0; i < JMAX; i += G * NT) wrm_st_b32(p.xready + i, 0u, dry);
    for (int i = i0; i < JMAX * 16; i += G * NT) wrm_st_b32(p.doneAs + i, 0u, dry);
  }
#else
  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    const int old = atomicAdd(bc, 1);
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA's share)
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
#endif
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if OKC_WGS && OKC_B4FIN
  // bet4_claude: item = (token, 1024-column segment), thread = 2 columns: 4 x M items over all CTAs. Each route's sum
  // of squares = fixed-order sum of its 256 epilogue partials (warp w: route w / 2, half w & 1; xor tree; 2 halves)
  {
    static_assert(NT == 512 && NW == 16, "B4FIN: 16 warps = 8 routes x 2 halves, 512 threads x 2 columns");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    constexpr int SEG = 1024, NSEG = HD / SEG;
    for (int it = blockIdx.x; it < p.M * NSEG; it += gridDim.x) {
      const int tk = it / NSEG, col = (it % NSEG) * SEG + tid * 2;
#if B4_EARLY_Y
      float2 yv[8];  // A2: the 8 routes' y loads go out before the ss partial loads and the reduction
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        const int q = S.pos[tk * 8 + sl];
        yv[sl] = q >= 0 ? __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col)) : make_float2(0.f, 0.f);
      }
#endif
      {
        const int sl = warp >> 1, hh = warp & 1;
        const int q = S.pos[tk * 8 + sl];
        float v = 0.f;
        if (q >= 0) {
          const float4 a = __ldcg(reinterpret_cast<const float4*>(p.sspart + (size_t)q * 256 + hh * 128 + lane * 4));
          v = (a.x + a.y) + (a.z + a.w);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[warp] = v;
      }
      CTA_SYNC();
      float2 acc = make_float2(0.f, 0.f);
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        const int q = S.pos[tk * 8 + sl];
        if (q >= 0) {
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf((red[2 * sl] + red[2 * sl + 1]) * (1.f / HD) + 1e-8f);
#if B4_EARLY_Y
          const float2 v = yv[sl];
#else
          const float2 v = __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col));
#endif
          acc.x += r * v.x; acc.y += r * v.y;
        }
      }
      *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc.x, acc.y);
      CTA_SYNC();  // red is reused by the next item
    }
  }
#elif OKC_WGS && OKC_WGSFINSS && OKC_YBF
  // r3l YBF: the y rows are bf16 (a thread's 8 columns = one 16 B load per route). The same two passes as the fp32
  // version below (pass 1: each local route's sum of squares over the row; pass 2: the weighted sum, from the thread's
  // row values kept in the free ring smem), on the bf16 values.
  {
    static_assert(HD == NT * 8 && NW == 16, "r3l YBF final pass: 512 threads x 8 columns");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    const __nv_bfloat16* ybh = reinterpret_cast<const __nv_bfloat16*>(p.ybuf);
    uint4* rc = reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(xbuf) + 4096);  // own entries only
#if OKC_R3FP
    if (p.M > (int)gridDim.x) {
      // r3l FP (M > grid): tokens tb and tb + G at once, one per half CTA (threads 0..255 / 256..511, 16 columns per
      // thread): the second token's dependent L2 round trips run beside the first's instead of after them
      const int G = gridDim.x, hf = tid >> 8, ht = tid & 255;
      const int tbe = dry ? min(p.M, (int)blockIdx.x + 1) : p.M;  // (the dry pass: the CTA's first pair only)
      for (int tb = blockIdx.x; tb < tbe; tb += 2 * G) {
        const int tk = tb + hf * G;
        const bool on = tk < p.M;  // (uniform over the half)
        int qv[8];
        float wv[8], ps[8];
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) qv[sl] = on ? (int)S.pos[tk * 8 + sl] : -1;
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) wv[sl] = qv[sl] >= 0 ? p.topk_w[tk * 8 + sl] : 0.f;
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) {
          ps[sl] = 0.f;
          if (qv[sl] >= 0) {
            const uint4* src = reinterpret_cast<const uint4*>(ybh + (size_t)qv[sl] * HD) + ht * 2;
            const uint4 u0 = __ldcg(src), u1 = __ldcg(src + 1);
            rc[(sl * 2) * NT + tid] = u0; rc[(sl * 2 + 1) * NT + tid] = u1;
            ps[sl] = r3l_sumsq8(u0) + r3l_sumsq8(u1);
          }
        }
#pragma unroll
        for (int sl = 0; sl < 8; ++sl)
          if (qv[sl] >= 0) {  // uniform over the half (one token)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);
          }
        if (lane == 0) {
#pragma unroll
          for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
        }
        CTA_SYNC();
        if (ht < 8) {
          float tsum = 0.f;
          for (int w = 0; w < NW / 2; ++w) tsum += red[(hf * (NW / 2) + w) * 8 + ht];
          red[NW * 8 + hf * 8 + ht] = tsum;
        }
        CTA_SYNC();
        float acc[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
        for (int sl = 0; sl < 8; ++sl)
          if (qv[sl] >= 0) {
            const float r = wv[sl] * rsqrtf(red[NW * 8 + hf * 8 + sl] * (1.f / HD) + 1e-8f);
            r3l_fma8(acc, r, rc[(sl * 2) * NT + tid]);
            r3l_fma8(acc + 8, r, rc[(sl * 2 + 1) * NT + tid]);
          }
        if (on) {
          wrm_st_v4(p.out + (size_t)tk * HD + ht * 16, r3l_pack8(acc), dry);
          wrm_st_v4(p.out + (size_t)tk * HD + ht * 16 + 8, r3l_pack8(acc + 8), dry);
        }
        CTA_SYNC();  // red / rc are reused by the next pair
      }
    } else
#endif
    {
      const int tke = dry ? min(p.M, (int)blockIdx.x + 1) : p.M;  // (the dry pass: the CTA's first token only)
      for (int tk = blockIdx.x; tk < tke; tk += gridDim.x) {
        int qv[8];
        float wv[8], ps[8];
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) wv[sl] = qv[sl] >= 0 ? p.topk_w[tk * 8 + sl] : 0.f;
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) {
          ps[sl] = 0.f;
          if (qv[sl] >= 0) {
            const uint4 u = __ldcg(reinterpret_cast<const uint4*>(ybh + (size_t)qv[sl] * HD) + tid);
            rc[sl * NT + tid] = u;
            ps[sl] = r3l_sumsq8(u);
          }
        }
#pragma unroll
        for (int sl = 0; sl < 8; ++sl)
          if (qv[sl] >= 0) {  // uniform over the CTA (one token)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);
          }
        if (lane == 0) {
#pragma unroll
          for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
        }
        CTA_SYNC();
        if (tid < 8) {
          float tsum = 0.f;
          for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
          red[NW * 8 + tid] = tsum;
        }
        CTA_SYNC();
        float acc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
        for (int sl = 0; sl < 8; ++sl)
          if (qv[sl] >= 0) {
            const float r = wv[sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
            r3l_fma8(acc, r, rc[sl * NT + tid]);
          }
        wrm_st_v4(p.out + (size_t)tk * HD + tid * 8, r3l_pack8(acc), dry);
        CTA_SYNC();  // red / rc are reused by the next token
      }
    }
  }
  }  // (pass)
#elif OKC_WGS && OKC_WGSFINSS
  // item = one token's whole row (thread = 8 columns): pass 1 sums each local route's y^2 over the row (warp
  // shuffles, then the 16 warps' partials through smem: the ring is free now), pass 2 re-reads the rows (L2 hits)
  {
    static_assert(HD == NT * 8, "8 columns per thread");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
#if OKC_R11WRM
    const int tke = dry ? min(p.M, (int)blockIdx.x + 1) : p.M;  // (the dry pass: the CTA's first token only)
    for (int tk = blockIdx.x; tk < tke; tk += gridDim.x) {
#else
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
#endif
      int qv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
#if OKC_R11FIN
      // r11 (FIN): the route weights load with the rows (inputs: no dependence on the reduction), and pass 1 keeps each
      // thread's 8 row values per route in the (free) ring smem for pass 2, which re-read them from L2: one dependent
      // L2 round trip less per token on the kernel's critical tail. Same values, same operations.
      float wv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) wv[sl] = qv[sl] >= 0 ? p.topk_w[tk * 8 + sl] : 0.f;
      float4* rc = reinterpret_cast<float4*>(reinterpret_cast<unsigned char*>(xbuf) + 4096);  // [8][2][NT] (own entries only)
#endif
      float ps[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
        if (qv[sl] >= 0) {
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
#if OKC_R11FIN
          rc[(sl * 2) * NT + tid] = a; rc[(sl * 2 + 1) * NT + tid] = b;
#endif
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
      }
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {  // uniform over the CTA (one token)
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);
        }
      if (lane == 0) {
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
      }
      CTA_SYNC();
      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        red[NW * 8 + tid] = tsum;
      }
      CTA_SYNC();
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {
#if OKC_R11FIN
          const float r = wv[sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
          const float4 a = rc[(sl * 2) * NT + tid], b = rc[(sl * 2 + 1) * NT + tid];
#else
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
#endif
          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;
          acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;
        }
      uint4 o;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
        ow[i] = *reinterpret_cast<uint32_t*>(&v);
      }
#if OKC_R11WRM
      wrm_st_v4(p.out + (size_t)tk * HD + tid * 8, o, dry);
#else
      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
#endif
      CTA_SYNC();  // red is reused by the next token
    }
  }
#if OKC_R11WRM
  }  // (pass)
#endif
#elif OKC_FINCOL
  // column-parallel: item = (token, 1024-column segment); thread = 4 columns
  constexpr int FSEG = HD / (NT * 4);  // column segments per token (thread = 4 columns)
  int it0 = blockIdx.x;
#if OKC_FINPF
  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
    const int col = (itp % FSEG) * (NT * 4) + tid * 4;
    float ssv[8];
    float4 yv[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      ssv[s] = qvp[s] >= 0 ? __ldcg(ss_cur() + qvp[s]) : 0.f;
      yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (qvp[s] >= 0) {
        const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);
        acc.x += r * yv[s].x; acc.y += r * yv[s].y; acc.z += r * yv[s].z; acc.w += r * yv[s].w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;
  }
  it0 += gridDim.x;
#endif
  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));
        acc.x += r * v.x; acc.y += r * v.y; acc.z += r * v.z; acc.w += r * v.w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;
  }
#else
  for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float4 v = __ldcg(src + i);
          acc[4 * i] += r * v.x; acc[4 * i + 1] += r * v.y; acc[4 * i + 2] += r * v.z; acc[4 * i + 3] += r * v.w;
        }
      }
    }
    uint4 o[2];
    uint32_t* ow = reinterpret_cast<uint32_t*>(o);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
      ow[i] = *reinterpret_cast<uint32_t*>(&v);
    }
    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 16);
    dst[0] = o[0]; dst[1] = o[1];
  }
#endif
  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered
}

}  // namespace okc

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static size_t okc_smem_bytes() {
  size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;
#if OKC_WGS
  return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;
#endif
  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);
  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;
  return base + (xa > xb ? xa : xb);
}

static int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)

void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }

void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {
  okc::Params p;
  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();
  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();
  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();
  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();
  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();
  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }
  else { p.wAn = p.wAf; p.wBn = p.wBf; }  // no natural copies: the wgmma path must stay off (OKC_WGMIN 9)
#if OKC_WGS
  TORCH_CHECK(w.size() > 13, "OKC_WGS needs the WGS weight copies (w[12], w[13])");
  TORCH_CHECK((size_t)scratch[1].numel() >= (size_t)okc::EMAX + (size_t)okc::EMAX * okc::HD + okc::WGS_FIXF,
              "OKC_WGS needs the stream-K fixup area after ybuf (scratch[1])");
  p.wAs = (const unsigned char*)w[12].data_ptr(); p.wBs = (const unsigned char*)w[13].data_ptr();
#else
  p.wAs = nullptr; p.wBs = nullptr;
#endif
  p.zero = 0;
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr();
  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;
#if OKC_WGS && OKC_RLOAD
  TORCH_CHECK(((uintptr_t)ids.data_ptr() & 15) == 0 && ids.is_contiguous(), "topk_ids must be 16 B aligned and contiguous");
#endif
  int* ib = (int*)scratch[0].data_ptr();
  p.ctr = ib; p.doneA = ib + 8;
  p.ss = (float*)scratch[1].data_ptr();
  TORCH_CHECK(scratch[0].numel() >= 8 + 1024 + okc::EMAX, "scratch[0] needs the second ss copy");
  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);
  p.ybuf = p.ss + okc::EMAX;
#if OKC_WGS && OKC_B4FIN
  // B4FIN runs only at M <= 32: ybuf rows (sorted routes) stay below M * 8 <= 256, so its last 64 rows (1024 routes x 256 partials) are free
  TORCH_CHECK(x.size(0) <= 32 && x.size(0) * 8 <= okc::EMAX - 64, "B4FIN build is for M <= 32");
  p.sspart = p.ybuf + (size_t)(okc::EMAX - 64) * okc::HD;
#else
  p.sspart = nullptr;
#endif
  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
#if OKC_WGS
  TORCH_CHECK((size_t)scratch[2].numel() >= ((size_t)okc::EMAX + 2 * (size_t)okc::WGS_XROWS) * okc::HD,
              "OKC_WGS needs xA / xB after inter (scratch[2])");
  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;
  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;
  p.xready = ib + 8 + 512;
  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;
#else
  p.xA = p.xB = nullptr;
  p.xready = nullptr;
  p.doneAs = nullptr;
#endif
  p.out = (__nv_bfloat16*)out.data_ptr();
  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;
  static bool attr = false;
  size_t sm = okc_smem_bytes();
  if (!attr) {
    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    (void)e0; (void)e1;
    attr = true;
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  // large batches run the variant with the fat path inlined (and wgmma); small ones keep it out of line, which
  // leaves the skinny loop's code and registers as tuned
  if (g_okc_pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)grid);
    cfg.blockDim = dim3(okc::NTL);
    cfg.dynamicSmemBytes = sm;
    cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);
    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);
    return;
  }
  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);
  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);
}


"""

BW_SRC = r'''
// H200 memory-pipeline microbenchmarks (diagnostics only).
#include <cuda_runtime.h>
#include <stdint.h>

namespace bw {






// TMA bulk (cp.async.bulk) ring: thread 0 issues 16 KB bulk copies into NS stages, full/empty mbarriers.
template <int NS>
__global__ void __launch_bounds__(256, 1) tma_kernel(const uint4* __restrict__ src, size_t n16, unsigned* out) {
  extern __shared__ __align__(128) uint4 sm[];
  __shared__ __align__(8) uint64_t full[NS];
  const int tid = threadIdx.x;
  constexpr int ST16 = 1024;
  const size_t stages = n16 / ST16;
  const size_t per = (stages + gridDim.x - 1) / gridDim.x;
  const size_t s0 = blockIdx.x * per, s1 = s0 + per < stages ? s0 + per : stages;
  if (tid == 0) {
    for (int k = 0; k < NS; ++k) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[k]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(b));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
  }
  __syncthreads();
  auto issue = [&](size_t s, int slot) {
    if (s < s1) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(b), "r"(16384));
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(d), "l"(src + s * ST16), "r"(16384), "r"(b) : "memory");
    }
  };
  if (tid == 0)
    for (int k = 0; k < NS; ++k) issue(s0 + k, k);
  unsigned acc = 0;
  for (size_t s = s0; s < s1; ++s) {
    const int slot = (int)((s - s0) % NS);
    const uint32_t phase = (uint32_t)(((s - s0) / NS) & 1);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
        ::"r"(b), "r"(phase));
    const uint4* st = sm + slot * ST16;
#pragma unroll
    for (int k = 0; k < 4; ++k) { uint4 v = st[k * 256 + tid]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    __syncthreads();
    if (tid == 0) issue(s + NS, slot);
  }
  if (acc == 0x12345678u) out[0] = acc;
}


template <int SB, int NS, int P, bool MIX>
__global__ void __launch_bounds__(256) tma2_kernel(const uint4* __restrict__ src, size_t n16, unsigned* out) {
  extern __shared__ __align__(128) uint4 sm[];
  __shared__ __align__(8) uint64_t full[NS];
  const int tid = threadIdx.x;
  constexpr int ST16 = SB / 16;
  const size_t stages = n16 / ST16;
  const size_t per = (stages + gridDim.x - 1) / gridDim.x;
  const size_t s0 = blockIdx.x * per, s1 = s0 + per < stages ? s0 + per : stages;
  if (tid == 0) {
    for (int k = 0; k < NS; ++k) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[k]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(b), "r"(MIX ? 257 : 1));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
  }
  __syncthreads();
  auto issue = [&](size_t s, int slot) {
    if (s < s1) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      constexpr int TB = MIX ? SB / 2 : SB;  // bytes via TMA
      if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(b), "r"(TB));
        for (int k = 0; k < P; ++k) {
          uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16 + k * (TB / 16 / P));
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                       ::"r"(d), "l"(src + s * ST16 + k * (TB / 16 / P)), "r"(TB / P), "r"(b) : "memory");
        }
      }
      if (MIX) {
        for (int i = tid; i < (SB - TB) / 16; i += 256) {
          uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16 + TB / 16 + i);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src + s * ST16 + TB / 16 + i));
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(b) : "memory");
      }
    } else if (MIX) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      (void)b;
    }
  };
  for (int k = 0; k < NS; ++k) issue(s0 + k, k);
  unsigned acc = 0;
  for (size_t s = s0; s < s1; ++s) {
    const int slot = (int)((s - s0) % NS);
    const uint32_t phase = (uint32_t)(((s - s0) / NS) & 1);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
        ::"r"(b), "r"(phase));
    const uint4* st = sm + slot * ST16;
    for (int k = tid; k < ST16; k += 256) { uint4 v = st[k]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    __syncthreads();
    issue(s + NS, slot);
  }
  if (acc == 0x12345678u) out[0] = acc;
}



}  // namespace bw

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void bw_run(torch::Tensor src, torch::Tensor out, int64_t mode, int64_t grid) {
  const uint4* p = (const uint4*)src.data_ptr();
  size_t n16 = src.numel() * src.element_size() / 16;
  unsigned* o = (unsigned*)out.data_ptr();
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  switch (mode) {
    case 5: {
      size_t sm = 6 * 16384;
      cudaFuncSetAttribute(bw::tma_kernel<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma_kernel<6><<<(int)grid, 256, sm, st>>>(p, n16, o); break;
    }
    case 8: { constexpr int SB = 32768, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 12: { constexpr int SB = 16384, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid * 2, 256, sm, st>>>(p, n16, o); break; }
    case 16: { constexpr int SB = 36864, NS = 5; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 17: { constexpr int SB = 36864, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 18: { constexpr int SB = 36864, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 2, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 19: { constexpr int SB = 36864, NS = 3; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 20: { constexpr int SB = 32768, NS = 5; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 21: { constexpr int SB = 16384, NS = 8; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 22: { constexpr int SB = 16384, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 23: { constexpr int SB = 16384, NS = 6; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 24: { constexpr int SB = 36864, NS = 2; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid * 2, 256, sm, st>>>(p, n16, o); break; }
  }
}

'''

CPP_SRC = r'''
#include <torch/extension.h>
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void okc_set_pdl(int64_t on);
'''


# ------------------------------------------------------------------ extension
# Compiled extensions, keyed by their define tuple. This caches compiled
# code only -- never weights, activations, or outputs.
_EXTS = {}


def _ext_name(name, src, defs):
    """Extension module name: a readable prefix plus a digest of the source and flags (a changed source never reuses
    a stale build of the same name); CPython looks up PyInit_<name> with the name cut to 200 characters."""
    import hashlib
    dig = hashlib.md5((CPP_SRC + src + repr(tuple(defs)) + repr(MAIN_NS)).encode()).hexdigest()[:16]
    return name[:150] + "_" + dig


def _ext(defines=None):
    """Compile (once per define tuple) and return the extension module."""
    defines = tuple(MAIN_DEFINES if defines is None else defines)
    if DQ7 and "-DOKC_DQ7=1" not in defines:
        defines = defines + ("-DOKC_DQ7=1",)
    if defines not in _EXTS:
        src = CUDA_SRC
        defs = [d for d in defines if not d.startswith("@")]
        if torch.cuda.get_device_capability()[0] < 9:
            # Small-smem dev GPUs: keep the ring at 1 slot.
            defs = [d for d in defs if not d.startswith("-DOKC_NS=") and not d.startswith("-DOKC_NSF=")]
        for d in defines:
            if d.startswith("@"):
                src = ALT_SRCS[d[1:]]
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major == 9 else "")
        tag = "_".join(d.replace("=", "").replace("-D", "").replace("@", "src").lower() for d in defines) or "main"
        _EXTS[defines] = load_inline(
            name=_ext_name(_R4X_PFX.get(src, "okc_moe_w4a16_") + f"{tag}_sm{arch}", src, defs),
            cpp_sources=CPP_SRC,
            cuda_sources=src,
            functions=["okc_run", "okc_set_pdl"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                               *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={MAIN_NS if major >= 9 else 1}"]),
                               *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],
            verbose=False,
        )
        _EXTS[defines].okc_set_pdl(1 if PDL else 0)
    return _EXTS[defines]


# ------------------------------------------------------------- weight packing
def _scales_prmt_exact(*sfs):
    """True if every scale byte is a positive normal e4m3 value (0x08..0x7E),
    which is what makes the PRMT scale decode exact."""
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x08) | (b > 0x7E)).any()):
            return False
    return True


def _stored_pos():
    """(register, upper-half, code-bit) -> source bit of the packed word.

    The packers scatter each 4-bit code's bits into a 32-bit word at these
    positions; the kernel's dq4() rotates them back. The positions must be
    a permutation of 0..31, which the assert checks.
    """
    pos = {}
    for R in range(4):
        for up in range(2):
            outb = {0: 6 + 16 * up, 1: 7 + 16 * up, 2: 8 + 16 * up, 3: 15 + 16 * up}  # code bit -> output bit
            for b, o in outb.items():
                if DQ7:
                    # register R reads source bit (o + d) mod 32: r0 as stored, r1 = word << 4, r2 = rotl 10,
                    # r3 = word << 6 except the e1 bits (output 8 / 24), which come from rotl 14
                    d = (0, 28, 22, 18 if b == 2 else 26)[R]
                elif R == 0:
                    d = 0
                elif R == 1:
                    d = 6
                elif R == 2:
                    d = 10
                else:
                    d = 4 if b == 0 else 28   # m via rotr 4, (e0, e1, s) via rotl 4 == rotr 28
                pos[(R, up, b)] = (o + d) % 32
    assert sorted(pos.values()) == list(range(32))
    return pos


_POS = _stored_pos()


def _split_nibbles(codes, rows):
    """Packed uint8 codes (2 per byte) for the given rows -> (nibbles, R).

    nibbles is [R, K]: one 4-bit E2M1 code per entry, low nibble first.
    """
    c = codes.index_select(0, rows)
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(c.shape[0], -1)
    return nib, c.shape[0]


def _scatter_code_bits(word, code, reg, up):
    """OR one code's 4 bits into `word` at the positions for (reg, up)."""
    for b in range(4):
        word |= ((code >> b) & 1) << _POS[(reg, up, b)]
    return word


def _pack_codes(codes, rows):
    """codes uint8 [R0, K/2] (one expert), rows: LongTensor row order (R). Returns int32 words
    laid out [rb][w][c][g][t][s] (uint4 per lane per chunk)."""
    nib, R = _split_nibbles(codes, rows)
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 4, 4)  # rb, h, g, w, c, t, s, j
    n = n.permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)  # rb, w, c, g, t, s, h, j
    word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
    for Rg, (h, j0) in enumerate(_REG_HJ):
        for up in range(2):
            code = n[..., h, j0 + up]
            word = _scatter_code_bits(word, code, Rg, up)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _pack_scales(sf, rows):
    """sf fp8 [R0, K/16] -> bytes [rb][w][g][t][c][h]."""
    s = sf.view(torch.uint8).index_select(0, rows)
    R = s.shape[0]
    nb = R // 16
    s = s.view(nb, 2, 8, 8, 8, 4)  # rb, h, g, w, c, t
    s = s.permute(0, 3, 2, 5, 4, 1)  # rb, w, g, t, c, h
    return s.contiguous().view(-1)


def _fat_order(codes, scales):
    """Packed per-expert codes (int32 [rb][w][c][lane][4]) and scales (uint8 [rb][w][lane][c][h]) -> fat order:
    [ub (256-row unit)][ks (16 k-stages of 256)][32 KB codes [rb16][cc][lane][16 B] + 4 KB scales [rb16][lane][cc][h]]."""
    nub = codes.numel() // (16 * 8 * 8 * 32 * 4)
    c = codes.view(nub, 16, 8, 2, 4, 32, 4).permute(0, 2, 3, 1, 4, 5, 6).contiguous()
    c = c.view(torch.uint8).view(nub, 16, 32768)
    s = scales.view(nub, 16, 8, 32, 2, 4, 2).permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(nub, 16, 4096)
    return torch.cat([c, s], dim=2).contiguous().view(-1)


def _unit_order(codes, scales):
    """Per 16-row block: its 32 KB of codes followed by its 4 KB of scales (one TMA copy per skinny unit)."""
    nblk = codes.numel() // (8 * 8 * 32 * 4)
    c = codes.contiguous().view(torch.uint8).view(nblk, 32768)
    s = scales.contiguous().view(nblk, 4096)
    return torch.cat([c, s], dim=1).contiguous().view(-1)


def _pack_codes_nat(codes, rows):
    """Like _pack_codes but with the natural mma k order inside each 16-k step (for the wgmma fat path):
    word (rb, w, c, g, t, s) holds k = 64c + 16s + 8q + 2t + e of rows g (h=0) / g+8 (h=1)."""
    nib, R = _split_nibbles(codes, rows)
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)  # rb, h, g, w, c, s, q, t, e
    n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)  # rb, w, c, g, t, s, h, q, e
    word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
    for Rg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for up in range(2):
            code = n[..., h, q, up]
            word = _scatter_code_bits(word, code, Rg, up)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _kperm_src():
    """KPERM3: stage position p -> stage column k. p = 16 s + c (k16 step s = 4 lg + j, fragment column c of lane
    t = (c >> 1) & 3, i = (c & 1) | ((c >> 3) << 1)); k = 16 (4 t + j) + 4 lg + i, so lane t's columns of step j in all
    four load groups are exactly scale block 4 t + j."""
    p = torch.arange(256)
    s, c = p >> 4, p & 15
    lg, j, t, i = s >> 2, s & 3, (c >> 1) & 3, (c & 1) | ((c >> 3) << 1)
    k = 16 * (4 * t + j) + 4 * lg + i
    assert sorted(k.tolist()) == list(range(256))
    return k


def _kperm_codes(codes):
    """codes uint8 [R, K/2] (low nibble first) -> the same codes with K permuted inside every 256-K stage."""
    R, K = codes.shape[0], codes.shape[1] * 2
    nib = torch.stack((codes & 0xF, codes >> 4), -1).reshape(R, K // 256, 256)
    nib = nib.index_select(2, _kperm_src().to(codes.device)).reshape(R, K // 2, 2)
    return (nib[..., 0] | (nib[..., 1] << 4)).contiguous()


def _kperm_rows(rowsA):
    """Gate/up row order whose 16-row block r16, lane g holds inter column colmap(P), P = 8 r16 + g, where
    colmap(P) = 256 (P >> 8) + _kperm_src()[P & 255] is the column phase B reads at K position P (the phase-A
    epilogue then stores at its usual position)."""
    r = rowsA.view(-1, 2, 8)
    gate, up = r[:, 0, :].reshape(-1), r[:, 1, :].reshape(-1)
    P = torch.arange(gate.numel(), device=rowsA.device)
    cm = (P >> 8) * 256 + _kperm_src().to(rowsA.device)[P & 255]
    return torch.stack((gate.index_select(0, cm).view(-1, 8), up.index_select(0, cm).view(-1, 8)), 1).reshape(-1)


def _wgs_copy(codes, sf, rows):
    """WGS copy (RS-wgmma tiny band, moe -DOKC_WGS=1): [256-row unit][stage (16)][warpgroup q (4)][9216 B]. Warpgroup q
    owns rows 64q..64q+63 of the unit; stage st holds k16 steps st*16 + (0..15): codes [grp (4)][warp (4)][lane (32)]
    [4 words] (word (grp, j) = k16 step st*16 + grp*4 + j, natural mma k order), then scales [grp][warp][g][j][h] (the kernel's OKC_WGSSC2 layout)."""
    wn = _pack_codes_nat(_kperm_codes(codes), rows)
    R = rows.numel()
    nb, nub = R // 16, R // 256
    w = wn.view(nb, 8, 8, 8, 4, 4)                                    # rb16, w8, c8, g, t, s
    w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)   # rb16, k16 step (w8, c8, s), lane (g, t)
    w = w.view(nub, 4, 4, 16, 4, 4, 32)                               # ub, q, warp, st, grp, j, lane
    w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()                   # ub, st, q, grp, warp, lane, j
    cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
    s = sf.view(torch.uint8).index_select(0, rows)                    # [R, 256]: one byte per row per 16 K
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)                             # ub, q, warp, h, g, st, t, j (block 16 st + 4 t + j)
    s = s.permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 16, 4, 1024)  # ub, st, q, warp, g, t, j, h (KPERM3)
    return torch.cat([cb, s], dim=3).contiguous().view(-1)


def _main_defines(sfx):
    """The MAIN variant's defines for this weight set: the WGS build needs
    the PRMT scale decode, so without it fall back to the mma.sync build."""
    if not sfx and "-DOKC_WGS=1" in tuple(MAIN_DEFINES):
        return MAIN_FALLBACK_DEFINES
    return MAIN_DEFINES


def _wgs_on():
    """True if any compiled variant uses the WGS (RS-wgmma) kernel, which
    needs the extra WGS weight copies in state["w"]."""
    return any("-DOKC_WGS=1" in tuple(d) for d in (MAIN_DEFINES, TINY_DEFINES, TINY2_DEFINES, MID_DEFINES))


# ------------------------------------------------------- r3l SC6W weight copies
def _sc6_copy(wgs):
    """A WGS copy ([..][9216 B] pieces: 8192 B codes, then 1024 B scales = 128 threads x 8 bytes [j][h]) -> SC6 pieces
    [..][8960 B]: the codes, plane W (thread t: 4 B, byte n = 4 v_n | c_n), plane H (thread t: 2 B, byte n = 4 v_(4+n) |
    c_(4+n)); v = byte - 0x40; c_0..c_2 = the 2-bit pieces of v_6, c_3..c_5 those of v_7. None unless every v fits 6 bits
    and the round trip gives every byte back."""
    x = wgs.view(-1, 9216)
    v = x[:, 8192:].reshape(-1, 128, 8).to(torch.int16) - 0x40
    if int(v.min()) < 0 or int(v.max()) > 63:
        return None
    v6, v7 = v[..., 6], v[..., 7]
    c = [v6 & 3, (v6 >> 2) & 3, (v6 >> 4) & 3, v7 & 3, (v7 >> 2) & 3, (v7 >> 4) & 3]
    wb = torch.stack([(v[..., n] << 2) | c[n] for n in range(4)], -1)          # [pieces, 128, 4]
    hb = torch.stack([(v[..., 4 + n] << 2) | c[4 + n] for n in range(2)], -1)  # [pieces, 128, 2]
    back = torch.stack([wb[..., 0] >> 2, wb[..., 1] >> 2, wb[..., 2] >> 2, wb[..., 3] >> 2, hb[..., 0] >> 2, hb[..., 1] >> 2,
                        (wb[..., 0] & 3) | ((wb[..., 1] & 3) << 2) | ((wb[..., 2] & 3) << 4),
                        (wb[..., 3] & 3) | ((hb[..., 0] & 3) << 2) | ((hb[..., 1] & 3) << 4)], -1)
    if not torch.equal(back, v):
        return None
    out = torch.cat([x[:, :8192], wb.to(torch.uint8).reshape(-1, 512), hb.to(torch.uint8).reshape(-1, 256)], 1)
    return out.contiguous().view(-1)


# ---------------------------------------------------------------------- setup
def prepare(weights):
    """Repack weights (bit/byte permutations only, never requantized) into
    the layouts the kernel reads, allocate scratch, and compile the
    variants kernel_fn may pick. Called once; not on the timed path."""
    dev = weights["w13"].device
    lo, cnt = LOCAL_START, LOCAL_EXPERTS
    I = weights["w13"].shape[1] // 2   # gate+up rows: w13 stacks [gate; up]
    H = weights["w2"].shape[1]         # down rows
    assert I == HIDDEN and H == HIDDEN
    assert weights["post_norm_w"].dtype == torch.bfloat16  # kept as given (TASK.md: BF16 weights stay BF16)
    assert weights["w13_gs"].dtype == torch.float32 and weights["w2_gs"].dtype == torch.float32  # stored as given
    # rowsA: gate rows and up rows interleaved in 8-row groups (the mma
    # row-group size); rowsB: down rows in order.
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=dev)
    wA, sA, wB, sB = [], [], [], []
    for e in range(lo, lo + cnt):
        wA.append(_pack_codes(weights["w13"][e], rowsA))
        sA.append(_pack_scales(weights["w13_sf"][e], rowsA))
        wB.append(_pack_codes(weights["w2"][e], rowsB))
        sB.append(_pack_scales(weights["w2_sf"][e], rowsB))
    # LUT: fp8 e4m3 byte -> bf16x2(value * 2^64), packed into one int32.
    lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float()
           * 2.0**64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    lut = (lut | (lut << 16)).to(torch.int32)
    state = {
        # w: [packed gate/up codes, scales, group scales,
        #     packed down codes, scales, group scales,
        #     e4m3 LUT, post_norm_w (bf16 as given; kernel forms 1 + w in FP32),
        #     fat-order gate/up, fat-order down,
        #     unit-order gate/up, unit-order down,
        #     (+ WGS copies appended below, w[12] / w[13], when any variant uses WGS)]
        "w": [torch.cat(wA), torch.cat(sA), weights["w13_gs"][lo:lo + cnt].contiguous(),
              torch.cat(wB), torch.cat(sB), weights["w2_gs"][lo:lo + cnt].contiguous(),
              lut.contiguous(), weights["post_norm_w"].contiguous(),
              torch.cat([_fat_order(a, b) for a, b in zip(wA, sA)]), torch.cat([_fat_order(a, b) for a, b in zip(wB, sB)]),
              torch.cat([_unit_order(a, b) for a, b in zip(wA, sA)]), torch.cat([_unit_order(a, b) for a, b in zip(wB, sB)])],
        "lo": lo, "cnt": cnt, "scratch": _make_scratch(dev),
        "grid": torch.cuda.get_device_properties(dev).multi_processor_count,
    }
    if _wgs_on():  # w[12], w[13]: the WGS copies (one more copy of the weights, laid out per 64-row unit)
        state["w"].append(torch.cat([_wgs_copy(weights["w13"][e], weights["w13_sf"][e], _kperm_rows(rowsA)) for e in range(lo, lo + cnt)]))
        state["w"].append(torch.cat([_wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB) for e in range(lo, lo + cnt)]))
    prmt = PRMT_SCALES and _scales_prmt_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])
    state["sfx"] = ("-DOKC_PRMTSC=1",) if prmt else ()
    # R3M-SC6: 6-bit scale unit copies (w[14], w[15]) for the MID build when every local scale byte allows it
    sc6 = bool(prmt) and _wgs_on() and len(state["w"]) == 14 and _r3m_sc6_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])
    state["mid"] = tuple(MID_S6_DEFINES) if sc6 else tuple(MID_DEFINES)
    if sc6:
        state["w"].append(torch.cat([_r3m_unit_order6(a, b) for a, b in zip(wA, sA)]))
        state["w"].append(torch.cat([_r3m_unit_order6(a, b) for a, b in zip(wB, sB)]))
    mods = [tuple(_main_defines(state["sfx"])) + state["sfx"]]
    for d in (TINY_DEFINES, TINY2_DEFINES, state["mid"]):
        if d and (state["sfx"] or "-DOKC_WGS=1" not in tuple(d)):  # the WGS build needs the PRMT scale decode
            mods.append(tuple(d) + state["sfx"])
    state["sc6"] = False
    if SC6_WGS and state["sfx"] and "-DOKC_WGS=1" in tuple(MAIN_DEFINES):  # r3l SC6W (gated, lossless)
        a6, b6 = _sc6_copy(state["w"][12]), _sc6_copy(state["w"][13])
        if a6 is not None and b6 is not None:
            state["w6"] = state["w"][:12] + [a6, b6]
            state["sc6"] = True
            mods.append(tuple(SC6_DEFINES) + state["sfx"])
    _ext_parallel(mods)
    for d in mods:
        _ext(d)
    torch.cuda.synchronize()
    if DIAG:
        _diag(state, weights)
    return state


def _ext_parallel(mods):
    """Compile the not-yet-built variants in parallel threads (each load_inline runs nvcc in its own process). An error
    here is ignored: prepare() calls _ext serially right after, which rebuilds or reloads whatever is missing."""
    import concurrent.futures
    todo = [d for d in mods if (tuple(d) + (("-DOKC_DQ7=1",) if DQ7 and "-DOKC_DQ7=1" not in tuple(d) else ())) not in _EXTS]
    if len(todo) < 2:
        return
    def one(d):
        try:
            _ext(d)
        except Exception as ex:  # redone serially below
            print(f"[prepare] parallel build failed, redone serially: {type(ex).__name__}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(todo))) as ex:
        list(ex.map(one, todo))


def _make_scratch(dev):
    """Per-call scratch, sized for CHUNK_TOKENS tokens. Layout is fixed by
    the CUDA launcher (okc_run), which indexes scratch[0]/[1]/[2] as:
    [0] int32: 8 counters + 1024 doneA slots + the second ss copy
    (ss1, EMAX floats viewed as int32 -- the launcher takes its address
    as ib + 8 + 1024);
    [1] fp32: ss (EMAX) + ybuf (EMAX x HIDDEN) + WGS stream-K fixups;
    [2] bf16: inter (EMAX x HIDDEN) + WGS xA / xB (2 x WGS_XROWS x HIDDEN).
    Zeroed here so the first call starts from a clean grid barrier state;
    the kernel re-zeroes what it needs between launches.
    """
    return [torch.zeros(8 + 1024 + EMAX, dtype=torch.int32, device=dev),
            torch.zeros(EMAX + EMAX * HIDDEN + WGS_FIXF, dtype=torch.float32, device=dev),
            torch.zeros((EMAX + 2 * WGS_XROWS) * HIDDEN, dtype=torch.bfloat16, device=dev)]


# ------------------------------------------------------------------ timed path
def kernel_fn(data, state):
    """Run one MoE layer. Fresh output every call; no host syncs; the only
    per-call work is picking the compiled variant by token count and
    launching it in CHUNK_TOKENS-token slices."""
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    assert cfg["local_start"] == state["lo"] and cfg["local_count"] == state["cnt"]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    if ids.data_ptr() & 15:  # the WGS routing loads topk_ids 16 B at a time (an offset view gets an aligned copy)
        ids = ids.clone()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=x.device)
    sfx = state.get("sfx", ())
    wl = state["w"]
    if TINY_DEFINES and M <= TINY_MAX_M:
        ext = _ext(tuple(TINY_DEFINES) + sfx)
    elif TINY2_DEFINES and M <= TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(TINY2_DEFINES)):
        ext = _ext(tuple(TINY2_DEFINES) + sfx)  # (the WGS build needs the PRMT scale decode: else MID/MAIN)
    elif MID_DEFINES and M <= MID_MAX_M:
        ext = _ext(tuple(state.get("mid", MID_DEFINES)) + sfx)
    elif state.get("sc6") and M <= SC6_MAX_M:  # r3l SC6W: 33 <= M <= SC6_MAX_M, the 6-bit-scale WGS copies
        ext = _ext(tuple(SC6_DEFINES) + sfx)
        wl = state["w6"]
    else:
        ext = _ext(tuple(_main_defines(sfx)) + sfx)
    for m0 in range(0, M, CHUNK_TOKENS):
        m1 = min(M, m0 + CHUNK_TOKENS)
        ext.okc_run(wl, state["scratch"], x[m0:m1], ids[m0:m1], tw[m0:m1], out[m0:m1],
                    state["lo"], state["cnt"], state["grid"])
    return out


# ---------------------------------------------------------------------------------------------- diagnostics
# Everything below runs only when DIAG is set (prepare) or when called
# directly. It is development tooling, not part of the submission's timed
# path, and it is the only place host synchronisation appears besides
# prepare()'s one-time compile wait.
def _gen_input(bs, seed, dev):
    """Same distribution as the task's generate_input (normal dist, busiest-GPU relabel), written here."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, N_EXPERTS_TOTAL), generator=gen, device=dev)
    z = torch.randn((N_EXPERTS_TOTAL,), generator=gen, device=dev)
    u = torch.rand((N_EXPERTS_TOTAL,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(TOP_K, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=TOP_K)
    ids = (ids - LOCAL_EXPERTS * pairs.argmax()) % N_EXPERTS_TOTAL
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": HIDDEN, "d_expert": HIDDEN, "n_experts": N_EXPERTS_TOTAL, "local_start": LOCAL_START,
                       "local_count": LOCAL_EXPERTS, "top_k": TOP_K, "sharding": SHARDING}}


def _time_graph(fn, reps=40):
    """Median / min CUDA-graph replay time of fn in microseconds."""
    dev = torch.device("cuda")
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.add_(1)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    ts.sort()
    del g
    return ts[len(ts) // 2], ts[0]


def _diag_bw(state):
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major == 9 else "")
    ext = load_inline(name=f"okc_bw_v216x_sm{arch}", cpp_sources="#include <torch/extension.h>\nvoid bw_run(torch::Tensor src, torch::Tensor out, int64_t mode, int64_t grid);",
                      cuda_sources=BW_SRC, functions=["bw_run"],
                      extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"], verbose=False)
    src = state["w"][0]
    nbytes = src.numel() * src.element_size()
    out = torch.zeros(4, dtype=torch.int32, device=src.device)
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=src.device)
    names = {5: "tma 6x16KB", 8: "tma 4x32KB", 12: "tma 4x16KB 2cta/sm", 16: "tma 5x36KB",
             17: "tma 4x36KB", 18: "tma 4x36KB as 2x18KB", 19: "tma 3x36KB", 20: "tma 5x32KB", 21: "tma2 8x16KB",
             22: "tma2 4x16KB", 23: "tma2 6x16KB", 24: "tma 2x36KB 2cta/sm"}
    for mode in DIAG_BW_MODES:
        if mode >= 5 and mode != 7 and major < 9:
            continue
        ts = []
        for _ in range(5):
            flush.add_(1)
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            ext.bw_run(src, out, mode, state["grid"])
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort()
        print(f"[diag-bw] {names[mode]:18s}: {ts[2]:8.1f} us  {nbytes / ts[2] / 1e6:6.2f} TB/s", flush=True)


def _diag(state, weights):
    import time
    dev = weights["w13"].device
    t0 = time.time()
    if DIAG_BW:
        try:
            _diag_bw(state)
        except Exception as ex:  # diagnostics must never break the run
            print(f"[diag-bw] failed: {ex}", flush=True)
    shapes = [1, 16, 32, 8, 128, 256]
    variants = {"main": tuple(MAIN_DEFINES)}
    variants.update(DIAG_VARIANTS)
    # compile the variants concurrently (separate extension dirs; nvcc runs in subprocesses)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_ext, [d for d in variants.values() if tuple(d) not in _EXTS]))
    print(f"[diag] compiled {len(variants)} variants in {time.time() - t0:.0f}s", flush=True)
    lines = []
    for bs in shapes:
        d = _gen_input(bs, 1234 + bs, dev)
        ids = d["topk_ids"]
        touched = int(torch.unique(ids[(ids >= 0) & (ids < LOCAL_EXPERTS)]).numel())
        row = [f"bs={bs:3d} touched={touched:2d}"]
        for name, defs in variants.items():
            ext = _ext(defs)
            x = d["hidden_states"]
            idt = ids.to(torch.int32).contiguous()
            tw = d["topk_weights"].float().contiguous()
            out = torch.empty((bs, HIDDEN), dtype=torch.bfloat16, device=dev)

            def fn():
                ext.okc_run(state["w"], state["scratch"], x, idt, tw, out, LOCAL_START, LOCAL_EXPERTS, state["grid"])

            med, mn = _time_graph(fn)
            # 28.31 MB: approx. weight bytes streamed per touched expert
            # (NVFP4 codes + scales for gate/up + down), diagnostic only.
            row.append(f"{name}: {med:7.1f}us ({touched * 28.31 / med:4.2f} TB/s)")
        lines.append("  ".join(row))
        print("[diag] " + lines[-1], flush=True)
    # timeline of the main variant with debug timestamps
    for dbgname in [k for k, v in variants.items() if "-DOKC_DEBUG=1" in v]:
        ext = _ext(variants[dbgname])
        G = state["grid"]
        dbg = torch.zeros(G * 8 + G * 8 * 4, dtype=torch.int64, device=dev)
        sc = list(state["scratch"]) + [dbg]
        names = ["start", "routed", "A done", "B done", "barrier", "final"]
        for bs in shapes:
            d = _gen_input(bs, 1234 + bs, dev)
            x = d["hidden_states"]
            idt = d["topk_ids"].to(torch.int32).contiguous()
            tw = d["topk_weights"].float().contiguous()
            out = torch.empty((bs, HIDDEN), dtype=torch.bfloat16, device=dev)
            flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
            for _ in range(3):
                flush.add_(1)
                dbg.zero_()
                torch.cuda.synchronize()
                ext.okc_run(state["w"], sc, x, idt, tw, out, LOCAL_START, LOCAL_EXPERTS, G)
                torch.cuda.synchronize()
            tt = dbg[:G * 8].view(G, 8).cpu().double()
            t00 = tt[:, 0][tt[:, 0] > 0].min()
            parts = []
            for k, n in enumerate(names):
                col = tt[:, k][tt[:, k] > 0]
                if col.numel():
                    parts.append(f"{n} {(col.min() - t00) / 1000:.1f}-{(col.max() - t00) / 1000:.1f}")
            print(f"[diag] timeline {dbgname} bs={bs}: " + "  ".join(parts), flush=True)
    print(f"[diag] total {time.time() - t0:.0f}s", flush=True)

# helper0 twin: our muse_decode_layer K11 copy (helper warp zero built in; TINY, TINY2 and MID use it).
ALT_SRCS["helper0"] = '\n// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.\n#include <cuda_bf16.h>\n#include <type_traits>\n#include <cuda_runtime.h>\n#include <stdint.h>\n#define OKC_DEBUG 0\n#define OKC_USE_TMA 1\n#define OKC_RLOAD 1  // routing: all topk_ids loads issued before the smem stores\n#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0\'s barrier inits\n#define OKC_WGSFINSS 1  // WGS: the final pass forms each route\'s sum of squares (the phase-B epilogue only writes y)\n#ifndef OKC_FINCOL\n#define OKC_FINCOL 1\n#endif\n#define OKC_WGMIN 9\n#define OKC_XPRE 1\n#define FAST1_ARCH 1\n#define FAST1_ON (OKC_FAST1 > 0 && OKC_TINY && FAST1_ARCH && OKC_USE_TMA)\n#ifndef OKC_CPFF\n#define OKC_CPFF 1\n#endif\n#ifndef OKC_FCOSTBB\n#define OKC_FCOSTBB OKC_FCOSTB\n#endif\n#define OKC_FCOSTSB OKC_FCOST\n#define OKC_DYN 1\n#ifndef OKC_DYNK\n#define OKC_DYNK 4\n#endif\n#define OKC_TDYN 0\n#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)\n#define OKC_FT_T 64\n#ifndef OKC_SPLIT\n#define OKC_SPLIT 0  // skinny-only build with experts of > SK_T tokens split into <= SK_T-token jobs (MID, Oct 26)\n#endif\n#ifndef OKC_EARLYN\n#define OKC_EARLYN 1\n#endif\n#define OKC_BIGM 64\n#define OKC_XPF 0\n#ifndef OKC_CPF\n#define OKC_CPF 1\n#endif\n#define OKC_FATSTATIC 1\n#define OKC_FCOST 40\n#ifndef OKC_FCOSTB\n#define OKC_FCOSTB 40\n#endif\n#ifndef OKC_WGS\n#define OKC_WGS 0  // tiny build on RS-wgmma: 4 consumer warpgroups split K of 64-row units, plus a TMA producer warp\n#endif\n#define SILU_DIV(a, b) __fdividef((a), (b))\n#if OKC_UWARP\n#else\n#define WARP_IDX(t) ((t) >> 5)\n#endif\n#ifndef OKC_PRMTSC\n#define OKC_PRMTSC 0\n#endif\n#define OKC_WSNR (OKC_SPLIT ? 4 : 8)  // partial-sum buffers between the compute warps and the helper warp (r8: 8 in TINY / TINY2)\n#define OKC_NOLOAD 0\n#ifndef OKC_HOFW\n#define OKC_HOFW (OKC_WS && !OKC_SPLIT && FAST1_ON)  // r8: poller warp (TINY / TINY2, with OKC_HOF)\n#endif\n\nnamespace okc {\n\n#ifndef OKC_L2H\n#define OKC_L2H 1\n#endif\n#if OKC_L2H\n// L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later in\n// the same call (evict last). Hints only: the math and the memory traffic\'s results are unchanged.\n#define L2H_SUFFIX ".L2::cache_hint"\n#define L2H_OPERAND ", %4"\n#define L2H_ARG , "l"(0x12F0000000000000ull)\n__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {\n  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");\n}\n__device__ __forceinline__ void st_keep_f32(float* a, float v) {\n  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");\n}\n#else\n#define L2H_SUFFIX ""\n#define L2H_OPERAND ""\n#define L2H_ARG\n__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) { *a = v; }\n__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }\n#endif\n\nconstexpr int HD = 4096, ID = 4096;\nconstexpr int NW = OKC_NW, NT = NW * 32;\nconstexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? (OKC_HOFW ? 64 : 32) : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)\n#define TIDX ((int)((threadIdx.x + NT) % NTL))\nconstexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers\n#if OKC_WS || OKC_WGS\n// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)\n#define CTA_SYNC() asm volatile("bar.sync 1, %0;\\n" ::"n"(NT) : "memory")\n#define WS_JOIN() asm volatile("bar.sync 2, %0;\\n" ::"n"(NTL) : "memory")\n#if OKC_SPLIT && FAST1_ON  // r8 MR7: route() on warps 0..NW-2; FAST1\'s warp skips it\n#define RT_N (NT - 32)\n#define RT_SYNC() asm volatile("bar.sync 3, %0;\\n" ::"n"(RT_N) : "memory")\n#else\n#define RT_N NT\n#define RT_SYNC() CTA_SYNC()\n#endif\n#endif\nconstexpr int NCH = 64 / NW;     // 64-K chunks per warp in a skinny unit (4096 / NW K per warp)\n#ifndef OKC_M2T\n#define OKC_M2T 1  // MID (OKC_SPLIT + OKC_WS): jobs of up to 16 tokens, 2-n-tile units for 9..16 tokens (r3 moe_small)\n#endif\n#define M2T_ON (OKC_M2T && OKC_SPLIT && OKC_WS)\n#ifndef OKC_HOFK2\n#define OKC_HOFK2 4  // r8: X block of 2 jobs from this many jobs on\n#endif\n#ifndef OKC_HOF\n#define OKC_HOF (OKC_WS && !M2T_ON && !OKC_SPLIT && FAST1_ON)  // r8: last job first, dealt evenly (TINY / TINY2)\n#endif\n#ifndef OKC_M2W1\n#define OKC_M2W1 20  // M2T static split: cost of a 1-tile unit\n#endif\n#ifndef OKC_M2W2\n#define OKC_M2W2 30  // M2T static split: cost of a phase-A 2-tile unit (r9: 30 / 32 beat 26 / 28 and 34 / 36 at d16, judge routing)\n#endif\n#ifndef OKC_M2W2B\n#define OKC_M2W2B 32  // M2T static split: cost of a phase-B 2-tile unit (r9, with OKC_M2W2)\n#endif\n#ifndef OKC_M2TF\n#define OKC_M2TF 0  // M2T debug: every job runs 2-tile units (zero tokens 8-15 past the count): measures the unit cost\n#endif\n#if M2T_ON\n#define OKC_SKT 16\n#else\n#define OKC_SKT 8\n#endif\nconstexpr int SK_T = OKC_SKT;   // max tokens of a skinny job (8: 1 n-tile; 16: 2 n-tiles via smem x)\nconstexpr int RED_W = (OKC_SKT > 8 && !M2T_ON) ? 8 : 4;\nstatic_assert(!OKC_WS || OKC_SKT == 8 || M2T_ON, "OKC_WS assumes single-tile skinny jobs");\nconstexpr int FT_T = OKC_FT_T;   // max tokens of a fat job (longer jobs are split into chunks)\nconstexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;   // 16-row blocks\nconstexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;\nconstexpr int RB_A_LOG = 9, RB_B_LOG = 8;\nstatic_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");\nconstexpr int BFLAG = 1 << 30;  // uslot: a phase-B unit prefetched at the end of phase A\n// dynamic tail: the last pool units of a phase\'s skinny work are claimed at run time (OKC_DYNK per CTA)\n__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * OKC_DYNK; return SKDYN ? (d < nS ? d : nS) : 0; }\nstatic_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");\nstatic_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)\nconstexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;\nconstexpr int XP_SK = HD + 8;   // skinny x row pitch (bf16)\nconstexpr int KS = 256;         // fat: K per x stage (4 chunks)\nconstexpr int XP_FT = KS + 8;   // fat x row pitch (bf16)\nconstexpr int NST = 3;          // fat x stages\nconstexpr float TWO62 = 4.611686018427388e18f;\n\nstruct Params {\n  const uint4* wA; const uint4* sA; const float* gsA;\n  const uint4* wB; const uint4* sB; const float* gsB;\n  const uint32_t* lut; const __nv_bfloat16* normw;  // post_norm_w as given (bf16); 1 + w is formed in FP32 here\n  const unsigned char* wAf; const unsigned char* wBf;  // fat-order weight copies (36 KB per k-stage)\n  const unsigned char* wAu; const unsigned char* wBu;  // unit-contiguous copies (36 KB per 16-row block)\n  const unsigned char* wAn; const unsigned char* wBn;  // natural-k fat copies (wgmma path)\n  const unsigned char* wAs; const unsigned char* wBs;  // WGS copies: [256-row unit][stage][warpgroup][9 KB]\n  __nv_bfloat16* xA; __nv_bfloat16* xB;  // WGS: per job, per stage x / inter in wgmma B layout [stage][n8][k8][8][8]\n  int* xready;                           // WGS: per job, rows of xA written\n  int* doneAs;                           // WGS: per (job, phase-B stage), phase-A warpgroup epilogues done (8 = ready)\n  uint32_t zero;  // always 0 (opaque to the compiler)\n  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;\n  int M, lo, nloc;\n  int* ctr;        // [8]: 0 ctrA, 1 ctrB, 2 barrier, 3 exit\n  int* doneA;      // [JMAX]\n  __nv_bfloat16* inter;  // [EMAX, ID]\n  float* ybuf;     // [EMAX, HD]\n  float* ss;       // [EMAX] copy 0 of the per-token sums of squares (copy 1: ss1); a launch uses copy gen & 1\n  float* ss1;      // [EMAX]\n  __nv_bfloat16* out;\n  unsigned long long* dbg;\n};\n\nstruct Smem {\n  uint32_t lut[256];\n#if !OKC_WGS\n  float red[NRB][NW][32][RED_W];  // per-unit cross-warp partial sums\n#endif\n#if OKC_WS\n  uint64_t wse[8], wsrf[8], wsre[8];  // WS: ring slot released by all compute warps / partials ready / buffer free\n  int wsr[4];                          // WS: this CTA\'s phase A and phase B unit ranges\n#endif\n#if M2T_ON\n  float red2[NRB][NW][32][4];          // M2T: the second n-tile\'s (tokens 8-15) partial sums\n  int m2any;                           // M2T: a job of this launch has more than 8 tokens (FAST1 wrote wsr)\n  int f1v0;                            // M2T: FAST1\'s first unit (debug check against wsr[0])\n#endif\n#if OKC_WGS\n#else\n  uint64_t bars[12];  // 0..4 skinny ring (NS <= 5), 5..11 fat ring (NSF <= 7)\n#endif\n  int seq, fseq, preissued, preissuedB;\n#if OKC_HOF\n  int hof[8];       // r8 HOF: per phase {X segment start, X units, rest start, total units} of this CTA\'s sequence\n#if OKC_HOFW\n  uint64_t hrdy[8];  // r8: the poller saw the doneA of this CTA\'s k-th phase-B job complete\n  uint64_t hxd, had;  // r8: the helper wrote every X / every phase-A epilogue of this CTA\n#endif\n#endif\n  int fcnt[32];     // OKC_SPLIT: FAST1 per-expert entry counts (the preissue\'s job -> expert map)\n  void* xbuf_ptr;\n  int cnt[64], off[64];\n  int wc[NW][32];\n  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];\n  int njF, njS, unit;\n  int gen;          // the grid barrier\'s generation at launch start (+1 per launch): picks this launch\'s ss copy\n  float* ssp;       // this launch\'s ss copy\n  int uslot[8];    // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)\n  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)\n  int unit_next, unit_next2;\n  signed char eid[EMAX];\n  short ent[EMAX];   // sorted entries (token*8+slot) by expert\n  short pos[EMAX];   // entry -> pair position, -1 if not local\n};\n\n// floor(a * c / G) without an integer divide (per-launch code runs from a cold instruction cache, where the divide\n// subroutines cost ~1 us): a * c < 2^24, rG ~ 1 / G, then an exact +-1 fix-up\n__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {\n  const int p = a * c;\n  int q = __float2int_rz((float)p * rG);\n  if (q * G > p) --q;\n  else if ((q + 1) * G <= p) ++q;\n  return q;\n}\n#if M2T_ON\n__device__ __forceinline__ int m2t_muldiv(int a, int c, int G, float rG) {\n  const int p = a * c;\n  int q = __float2int_rz((float)p * rG);\n  while (q * G > p) --q;\n  while ((q + 1) * G <= p) ++q;\n  return q;\n}\n#endif\n__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {\n  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));\n  return *reinterpret_cast<uint32_t*>(&r);\n}\n// a x (b.lo, b.lo) and a x (b.hi, b.hi): the broadcast folds into the HMUL2 operand (.H0_H0 / .H1_H1)\n__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {\n  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));\n  return *reinterpret_cast<uint32_t*>(&r);\n}\n__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {\n  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));\n  return *reinterpret_cast<uint32_t*>(&r);\n}\n\n__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,\n                                         uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {\n  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);\n  int sz = valid ? 16 : 0;\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(s), "l"(gmem), "r"(sz));\n}\n__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::); }\ntemplate <int N>\n__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N)); }\n__device__ __forceinline__ unsigned long long gtimer() {\n  unsigned long long t;\n  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));\n  return t;\n}\n#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)\nextern __shared__ __align__(16) unsigned char smem_raw[];\n__device__ __forceinline__ float* ss_cur();  // this launch\'s ss copy (defined after Smem users)\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p));\n  return v;\n}\n\n// Dequantize one 32-bit word (8 E2M1 codes) into the 4 bf16x2 A-fragment registers (values e2m1 * 2^-126).\n// Prepacked bit layout (see _pack_codes): r0 = q & M, r1 = rotr(q,6) & M, r2 = rotr(q,10) & M,\n// r3 = (rotl(q,4) & 0x81808180) | (rotr(q,4) & 0x00400040), M = 0x81C081C0.\n__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {\n  constexpr uint32_t M = 0x81C081C0u;\n  r0 = q & M;\n  r1 = __funnelshift_r(q, q, 6) & M;\n  r2 = __funnelshift_r(q, q, 10) & M;\n  // DQ7 bit layout (template DQ7 = True): 7 ALU + 2 IMAD.SHL per 8 weights. Register sources partition the word:\n  // r0 bits as stored, r1 = word << 4, r2 = rotl 10, r3 = word << 6 (6 bits) | rotl 14 (the two e1 bits)\n  r1 = (q * 16u) & M;\n  r2 = __funnelshift_l(q, q, 10) & M;\n  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);\n  return;\n  // bits kept by 0x81808180 never wrap, so the left rotate is a plain shift: as a multiply it runs on the FMA pipe\n  // (IMAD.SHL) instead of the ALU pipe the rest of the decode saturates\n  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);\n}\n\n\n// ---------------------------------------------------------------- routing prologue (every CTA, identical result)\n__device__ void early_issue(const Params& p, Smem& S);\n__device__ void route(const Params& p, Smem& S) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const int E = p.M * 8;\n#if OKC_SPLIT && FAST1_ON\n  const int CH = ((E + (NW - 1) * 32 - 1) / ((NW - 1) * 32)) * 32;\n#else\n  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;\n#endif\n  const int i0 = warp * CH, i1 = min(E, i0 + CH);\n#if OKC_SPLIT && FAST1_ON\n  for (int i = tid; i < NW * 32; i += RT_N) S.wc[i >> 5][i & 31] = 0;\n#else\n  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;\n#endif\n#if OKC_RLOAD && OKC_WGS  // (the small builds lose 1-2% with it: their E is at most 256)\n#else\n#pragma unroll 8\n  for (int i = tid; i < E; i += RT_N) {\n    int e = __ldg(p.topk_ids + i) - p.lo;\n    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);\n  }\n#endif\n  RT_SYNC();\n  for (int base = i0; base < i1; base += 32) {\n    const int i = base + lane;\n    const int e = (i < i1) ? (int)S.eid[i] : -1;\n    const unsigned peers = __match_any_sync(0xffffffffu, e);\n    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);\n  }\n  RT_SYNC();\n  if (warp == 0) {\n    int c[NW];\n    int tot = 0;\n#pragma unroll\n    for (int w = 0; w < NW; ++w) { c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0; tot += c[w]; }\n    int incl = tot;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }\n    int off = incl - tot;\n    S.cnt[lane] = tot;\n    S.off[lane] = off;\n    int o2 = off;\n#pragma unroll\n    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }\n    // jobs: fat (cnt > SK_T, chunks of FT_T) first, then skinny (0 < cnt <= SK_T); expert order within each\n#if OKC_SPLIT\n    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");\n    const int nfj = 0;\n    const int nsj = (tot + SK_T - 1) / SK_T;   // ceil(tot / 8) jobs of <= 8 tokens; a split expert re-streams its weights\n#else\n    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;\n    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;\n#endif\n    int fi = nfj, si = nsj;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);\n      if (lane >= d) { fi += v1; si += v2; }\n    }\n    const int nfTot = __shfl_sync(0xffffffffu, fi, 31), nsTot = __shfl_sync(0xffffffffu, si, 31);\n    int fj = fi - nfj, sj = nfTot + si - nsj;\n    for (int q = 0; q < nfj; ++q) {\n      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);\n    }\n#if OKC_SPLIT\n    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }\n#else\n    if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }\n#endif\n    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }\n    __syncwarp();\n#if !OKC_WGS  // (the WGS build has no fat queue)\n    // fat queue order: largest n-tile count first (LPT), stable by job index\n    int pos = 0;\n    for (int key = FT_T / 8; key >= 1; --key) {\n      for (int f0 = 0; f0 < nfTot; f0 += 32) {\n        const int f = f0 + lane;\n        const bool m = f < nfTot && ((S.jC[f] + 7) >> 3) == key;\n        const unsigned bal = __ballot_sync(0xffffffffu, m);\n        if (m) S.forder[pos + __popc(bal & ((1u << lane) - 1u))] = f;\n        pos += __popc(bal);\n      }\n    }\n#endif\n  }\n  RT_SYNC();\n  early_issue(p, S);\n  for (int base = i0; base < i1; base += 32) {\n    const int i = base + lane;\n    const int e = (i < i1) ? (int)S.eid[i] : -1;\n    const unsigned peers = __match_any_sync(0xffffffffu, e);\n    if (e >= 0) {\n      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));\n      S.ent[q] = (short)i;\n      S.pos[i] = (short)q;\n    } else if (i < i1) {\n      S.pos[i] = -1;\n    }\n    __syncwarp();\n    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);\n    __syncwarp();\n  }\n  RT_SYNC();\n}\n\n\n\n// ---------------------------------------------------------------- epilogues (one 16-row block, tokens 2t,2t+1 of n-tile)\n// r[0..1] rows g (gate / out row n0), r[2..3] rows g+8 (up / out row n1), tokens tk0 = 2t, 2t+1 relative to tile\n__device__ __forceinline__ float probe_q(float v) {\n  return v;\n}\n__device__ __forceinline__ void epi_A(const Params& p, int pstart, int ntok, int tok0, int col, float gsc, const float* r) {\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    int tk = tok0 + h;\n    if (tk < ntok) {\n      float gt = r[h] * gsc, up = r[2 + h] * gsc;\n      float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n      st_keep_bf16(p.inter + ((size_t)(pstart + tk) * ID + col), __float2bfloat16(probe_q(v)));\n    }\n  }\n}\n// returns per-token partial sum of squares (2 values) for this lane\'s two tokens\n__device__ __forceinline__ void epi_B(const Params& p, int pstart, int ntok, int tok0, int n0, float gsc, const float* r,\n                                      float* sq) {\n  const float w0 = 1.f + __bfloat162float(p.normw[n0]), w1 = 1.f + __bfloat162float(p.normw[n0 + 8]);\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    int tk = tok0 + h;\n    float v0 = r[h] * gsc * w0, v1 = r[2 + h] * gsc * w1;\n    sq[h] = v0 * v0 + v1 * v1;\n    if (tk < ntok) {\n      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0), v0);\n      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0 + 8), v1);\n    }\n  }\n}\n__device__ __forceinline__ void ss_reduce_add(const Params& p, int pstart, int ntok, int tok0, float* sq, int g) {\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 4);\n    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 8);\n    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 16);\n  }\n  if (g == 0) {\n#pragma unroll\n    for (int h = 0; h < 2; ++h)\n      if (tok0 + h < ntok) atomicAdd(&ss_cur()[pstart + tok0 + h], sq[h]);\n  }\n}\n\n// ---------------------------------------------------------------- the persistent kernel\n\n\n// ---------------------------------------------------------------- skinny path: shared-memory weight ring\n// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.\n// Units of the CTA\'s range stream through OKC_NS smem slots (TMA bulk copies on sm_90, cp.async elsewhere).\n// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.\nconstexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;\n\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n#if OKC_PRMTSC\n// e4m3 scale byte b of `word` -> bf16x2(scale * 2^64) in both halves with one PRMT and one IMAD: (byte << 4) puts the\n// e4m3 exponent and mantissa into the bf16 fields, + 184 << 7 rebiases the exponent. Exact for positive normal bytes\n// 0x08..0x7E only; prepare() checks every scale byte and uses the LUT build otherwise.\n#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)\n// same, with the multiplier opaque to ptxas (k16 = 16 + p.zero): an IMAD on the FMA pipe instead of a LEA on the\n// ALU pipe, which the FP4 decode saturates\n#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)\n// r14 G1: scale bytes b and b + 1 -> bf16x2(s_b * 2^64, s_b+1 * 2^64) with one PRMT + one IMAD (no carry crosses the\n// halves: (byte << 4) + (184 << 7) < 2^16); each HMUL2 takes its scale as a broadcast half (.H0_H0 / .H1_H1)\n#define SCP(word, b, k16) (__byte_perm((word), 0u, 0x4140u + ((b) * 0x0101u)) * (k16) + 0x5C005C00u)\n#define BML(a, s) bmul2l((a), (s))\n#define BMH(a, s) bmul2h((a), (s))\n#else\n#define BML(a, s) bmul2((a), (s))\n#define BMH(a, s) bmul2((a), (s))\n#endif\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(count));\n}\n__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" ::"r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\n  asm volatile(\n      "{\\n .reg .pred p;\\n WAIT_%=:\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n @!p bra WAIT_%=;\\n}\\n"\n      ::"r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n\ntemplate <bool PB>\n__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;\n  const size_t blk = (size_t)S.jE[j] * rbs + rb;\n  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;  // [32 KB codes | 4 KB scales]\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n}\n\ntemplate <bool PB>\n__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {\n  if (TIDX == 0) ring_issue<PB>(p, S, v, slot, bar);\n}\n\ntemplate <bool PB>\n__device__ __forceinline__ void load_x_regs(const Params& p, const Smem& S, int j, int lane, int warp,\n                                            uint4 (&x)[2 * NCH]) {\n  const int g = lane >> 2, t = lane & 3;\n  const int cnt = S.jC[j], st = S.jS[j];\n  {\n    const int r = g;\n    if (r < cnt) {\n      const __nv_bfloat16* row;\n      if (PB) row = p.inter + (size_t)(st + r) * ID;\n      else row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;\n      const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);\n#pragma unroll\n      for (int c = 0; c < NCH; ++c) {\n        x[2 * c] = PB ? __ldcg(src + c * 8) : __ldg(src + c * 8);\n        x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1);\n      }\n    } else {\n#pragma unroll\n      for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);\n    }\n  }\n}\n\nconstexpr int X1P = 512 + 8;  // per-warp smem x rows for the 2nd n-tile (tokens 8..15), pitch in bf16\ntemplate <bool PB>\n__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,\n                                             __nv_bfloat16* x1s) {\n  // warp-private: rows 8..15 of the job, K range [512*warp, +512): 8 rows x 64 uint4\n  const int cnt = S.jC[j], st = S.jS[j];\n  for (int i = lane; i < 8 * 64; i += 32) {\n    const int r = i >> 6, c = i & 63;\n    uint4 v = make_uint4(0, 0, 0, 0);\n    if (8 + r < cnt) {\n      const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD;\n      const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;\n      v = PB ? __ldcg(src) : __ldg(src);\n    }\n    *reinterpret_cast<uint4*>(x1s + r * X1P + c * 8) = v;\n  }\n  __syncwarp();\n}\n\n#if SKDYN\n#else\ntemplate <bool PB>\n__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,\n                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)\n  (void)k16;\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int nU = v1 - v0;\n  if (nU <= 0) return;\n  const int seq0 = seq;\n  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes\n  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  // cross-phase prefill: with no fat jobs, phase A\'s last ring refills already fetch this CTA\'s first B units\n  int preB = 0;\n  const bool cpf = OKC_CPF && !PB && cpf_ok;\n  const int v0B = cB0, nB = cBn;\n  const int pre = PB ? S.preissuedB : S.preissued;\n  for (int i = pre; i < OKC_NS; ++i) {\n    const int sl = (seq0 + i) % OKC_NS;\n    if (i < nU) ring_issue_all<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl);\n    else if (cpf && i - nU < nB) { ring_issue_all<true>(p, S, v0B + (i - nU), ring + sl * UNIT_BYTES, bars + sl); ++preB; }\n  }\n  uint4 x0[2 * NCH];\n  int curj = -1;\n  bool two = false;\n  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;\n  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit\'s rows one unit ahead\n  float nwc0 = 0.f, nwc1 = 0.f;\n  const int egg = (tid & 31) >> 2;\n  if (PB && tid < 128) { const int rbq = v0 % rbs; nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)); }\n  for (int i = 0; i < nU; ++i) {\n    const int v = v0 + i;\n    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;\n    float nwn0 = 0.f, nwn1 = 0.f;\n    if (PB && tid < 128 && i + 1 < nU) {\n      const int rbq = (v + 1) % rbs;\n      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));\n    }\n    if (j != curj) {\n      curj = j;\n      two = (OKC_SKT > 8) && S.jC[j] > 8;\n      if (PB) {\n        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);\n      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);\n    }\n    const int sl = (seq0 + i) % OKC_NS;\n    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);\n    mbar_wait(bars + sl, par);\n    const unsigned char* slot = ring + sl * UNIT_BYTES;\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCP(swd, (c & 1) * 2, k16), sg8 = sg;\n#define MULG(r) bmul2l((r), sg)\n#define MULG8(r) bmul2h((r), sg8)\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#endif\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n      if (!two) {\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);\n        }\n      } else {\n        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);\n        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);\n        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);\n          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);\n          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);\n        }\n      }\n    }\n#undef MULG\n#undef MULG8\n    if (!two) {\n#pragma unroll\n      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n    }\n    float* rd = &S.red[i & 1][warp][lane][0];\n    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    CTA_SYNC();  // slot sl fully consumed; partials visible\n    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);\n    else if (cpf && i + OKC_NS - nU < nB) {\n      ring_issue_all<true>(p, S, v0B + (i + OKC_NS - nU), ring + sl * UNIT_BYTES, bars + sl);\n      ++preB;\n    }\n    if (tid < (two ? 128 : 64)) {\n      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;\n      const int gg = ln >> 2, tt = ln & 3;\n      float s0 = 0.f, s1 = 0.f;\n#pragma unroll\n      for (int ww = 0; ww < NW; ++ww) {\n        const float* q = &S.red[i & 1][ww][ln][nt * 4];\n        s0 += q[h];\n        s1 += q[2 + h];\n      }\n      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n      const int tk = nt * 8 + 2 * tt + h;\n      if (!PB) {\n        if (tk < cnt) {\n          const float gsc = p.gsA[ce] * TWO62;\n          const float gt = s0 * gsc, up = s1 * gsc;\n          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));\n        }\n      } else {\n        const float gsc = p.gsB[ce] * TWO62;\n        const int n0 = rb * 16 + gg;\n        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;\n        float sq = v0f * v0f + v1f * v1f;\n        if (tk < cnt) {\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n        }\n        sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n    nwc0 = nwn0; nwc1 = nwn1;\n  }\n  seq = seq0 + nU;\n  if (!PB) {\n    __threadfence();\n    CTA_SYNC();\n    if (tid == 0) {\n      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;\n      for (int jj = jf; jj <= jl; ++jj) {\n        const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);\n        atomicAdd(p.doneA + jj, b - a);\n      }\n    }\n  }\n  if (!PB && tid == 0) S.preissuedB = preB;\n  if (PB && tid == 0) S.preissuedB = 0;\n  CTA_SYNC();\n}\n#endif\n\n// ---------------------------------------------------------------- fat unit: CTA = 256 rows (warp = 2 x 16 rows) x full K, <=64 tokens\n// ---------------------------------------------------------------- fat unit through the smem ring\n// A fat unit is 256 rows x full K for <= 64 tokens. Its weights arrive in 16 k-stages of 36 KB (16 row blocks x\n// 4 chunks + their scales, contiguous in the fat-order copy) through slots 0..2 of the ring region; the x tile\n// for the stage is staged by cp.async into two buffers that live in slots 3..4.\nconstexpr int NSF = OKC_NSF;\ntemplate <bool PB>\n__device__ __forceinline__ void fat_issue(const Params& p, int e, int ub, int ks, unsigned char* slot, uint64_t* bar, bool nat = false) {\n  const int nub = PB ? UB_B : UB_A;\n  const unsigned char* src = (nat ? (PB ? p.wBn : p.wAn) : (PB ? p.wBf : p.wAf)) + (((size_t)e * nub + ub) * 16 + ks) * (size_t)UNIT_BYTES;\n  if (TIDX == 0) {\n    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n  }\n}\n\n// ---------------------------------------------------------------- fat units: one continuous stream per CTA\n// Units come from a dynamic queue; the next unit is claimed when the current one starts, so the ring keeps\n// streaming across unit boundaries (the refills of the last k-stages already fetch the next unit\'s weights).\n// ---------------------------------------------------------------- wgmma (sm90a): A from registers, B (x) from smem\n__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {  // K-major, 128 B swizzle, 8-row groups 1 KB apart\n  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);\n}\n__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {\n  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);\n}\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory"); }\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory"); }\ntemplate <int N>\n__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\\n" ::"n"(N) : "memory"); }\ntemplate <int NTT> struct WG;\ntemplate <> struct WG<1> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<2> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<3> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<4> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<5> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<6> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<7> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<8> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\n\n// one 36 KB k-stage of a fat unit with wgmma: A = dequantized weights (natural k order) from registers, B = x tile from\n// shared memory in the K-major no-swizzle core-matrix layout [n8][k8][8 rows][16 B] (LBO 128 B along K, SBO 4 KB along N)\ntemplate <int NTT>\n__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,\n                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {\n  // x stage: [kb 4][64 rows][128 B], 128 B swizzle (chunk c of row n at c ^ (n & 7)); k-step st of chunk cc reads\n  // k block cc at byte offset 32 st\n  const int g = lane >> 2;\n  const uint32_t xb = smem_u32(xs);\n  uint32_t A[2][8][4];  // [chunk parity][k-step x rb][fragment]; a set stays untouched until its group retired\n  uint32_t keep = 0;    // folds every fragment after its group retired, so ptxas keeps the registers live (no reuse)\n#pragma unroll\n  for (int cc = 0; cc < 4; ++cc) {\n    const int sb = cc & 1;\n    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);\n    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);\n    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);\n    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);\n#pragma unroll\n    for (int st = 0; st < 4; ++st) {\n      // e4m3 -> bf16 without the LUT: bf16 bits (b << 4) hold value * 2^-120 (subnormal codes included), two exact\n      // multiplies by 2^92 give value * 2^64 like the LUT; rows g | g+8 travel as one bf16x2\n      const uint32_t K92 = 0x6D806D80u;  // bf16x2(2^92, 2^92)\n      const uint32_t p0 = bmul2(bmul2((__byte_perm(sc0.x, sc0.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);\n      const uint32_t p1 = bmul2(bmul2((__byte_perm(sc1.x, sc1.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);\n      const uint32_t sg0 = __byte_perm(p0, p0, 0x1010u), sg80 = __byte_perm(p0, p0, 0x3232u);\n      const uint32_t sg1 = __byte_perm(p1, p1, 0x1010u), sg81 = __byte_perm(p1, p1, 0x3232u);\n#define FMG0(r) bmul2((r), sg0)\n#define FMG80(r) bmul2((r), sg80)\n#define FMG1(r) bmul2((r), sg1)\n#define FMG81(r) bmul2((r), sg81)\n      uint32_t r0, r1, r2, r3, q0, q1, q2, q3;\n      dq4((&w0.x)[st], r0, r1, r2, r3);\n      dq4((&w1.x)[st], q0, q1, q2, q3);\n      A[sb][2 * st][0] = FMG0(r0); A[sb][2 * st][1] = FMG80(r1);\n      A[sb][2 * st][2] = FMG0(r2); A[sb][2 * st][3] = FMG80(r3);\n      A[sb][2 * st + 1][0] = FMG1(q0); A[sb][2 * st + 1][1] = FMG81(q1);\n      A[sb][2 * st + 1][2] = FMG1(q2); A[sb][2 * st + 1][3] = FMG81(q3);\n#undef FMG0\n#undef FMG80\n#undef FMG1\n#undef FMG81\n    }\n    wg_fence();\n#pragma unroll\n    for (int st = 0; st < 4; ++st) {\n      const uint64_t dk = wg_desc_sw128(xb + cc * 8192 + st * 32);\n      WG<NTT>::mma(acc0, A[sb][2 * st][0], A[sb][2 * st][1], A[sb][2 * st][2], A[sb][2 * st][3], dk);\n      WG<NTT>::mma(acc1, A[sb][2 * st + 1][0], A[sb][2 * st + 1][1], A[sb][2 * st + 1][2], A[sb][2 * st + 1][3], dk);\n    }\n    wg_commit();\n    if (cc >= 1) {\n      wg_wait<1>();  // the previous chunk\'s group retired: its register set may be rewritten\n#pragma unroll\n      for (int k = 0; k < 8; ++k) keep ^= A[sb ^ 1][k][0] ^ A[sb ^ 1][k][1] ^ A[sb ^ 1][k][2] ^ A[sb ^ 1][k][3];\n    }\n  }\n  wg_wait<0>();\n#pragma unroll\n  for (int k = 0; k < 8; ++k) keep ^= A[1][k][0] ^ A[1][k][1] ^ A[1][k][2] ^ A[1][k][3];\n  acc0[0][0] += __uint_as_float(keep & zero);  // adds +0.0f\n}\n\n// one 36 KB k-stage of a fat unit: 2 row blocks per warp x NTT token tiles (exact tile count, no predicated MMAs)\ntemplate <int NTT>\n__device__ __forceinline__ void fat_compute(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,\n                                            int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4]) {\n  const int g = lane >> 2, t = lane & 3;\n  const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp * 32 + lane) * 8);\n#if OKC_NW == 8\n  const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp + 8) * 32 + lane) * 8);\n#endif\n#pragma unroll\n  for (int cc = 0; cc < 4; ++cc) {\n    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);\n#if OKC_NW == 8\n    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);\n#endif\n#if OKC_PRMTSC\n    const uint32_t sg0 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2), sg80 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2 + 1);\n#if OKC_NW == 8\n    const uint32_t sg1 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2), sg81 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2 + 1);\n#endif\n#else\n    const int sh = (cc & 1) * 16;\n    const uint32_t sw0 = (cc < 2 ? sc0.x : sc0.y) >> sh;\n    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;\n    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];\n    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];\n#endif\n    const __nv_bfloat16* xr = xs + g * XP_FT + cc * 64 + 16 * t;\n#pragma unroll\n    for (int sp = 0; sp < 2; ++sp) {\n      uint32_t a0[2][4], a1[2][4];\n      (void)a1;\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int s2 = 2 * sp + h;\n        uint32_t r0, r1, r2, r3;\n        dq4((&w0.x)[s2], r0, r1, r2, r3);\n        a0[h][0] = bmul2(r0, sg0); a0[h][1] = bmul2(r1, sg80); a0[h][2] = bmul2(r2, sg0); a0[h][3] = bmul2(r3, sg80);\n#if OKC_NW == 8\n        dq4((&w1.x)[s2], r0, r1, r2, r3);\n        a1[h][0] = bmul2(r0, sg1); a1[h][1] = bmul2(r1, sg81); a1[h][2] = bmul2(r2, sg1); a1[h][3] = bmul2(r3, sg81);\n#endif\n      }\n#pragma unroll\n      for (int nt = 0; nt < NTT; ++nt) {\n        const uint4 bv = *reinterpret_cast<const uint4*>(xr + nt * 8 * XP_FT + 8 * sp);\n        mma_bf16(acc0[nt], a0[0][0], a0[0][1], a0[0][2], a0[0][3], bv.x, bv.y);\n#if OKC_NW == 8\n        mma_bf16(acc1[nt], a1[0][0], a1[0][1], a1[0][2], a1[0][3], bv.x, bv.y);\n#endif\n        mma_bf16(acc0[nt], a0[1][0], a0[1][1], a0[1][2], a0[1][3], bv.z, bv.w);\n#if OKC_NW == 8\n        mma_bf16(acc1[nt], a1[1][0], a1[1][1], a1[1][2], a1[1][3], bv.z, bv.w);\n#endif\n      }\n    }\n  }\n}\n\ntemplate <bool PB, bool BIG>\n__device__ __forceinline__ void fat_stream(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const int g = lane >> 2, t = lane & 3;\n  int* ctr = p.ctr + (PB ? 1 : 0);\n  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);\n  uint64_t* fb = S.bars + 5;\n  __nv_bfloat16* xs0 = reinterpret_cast<__nv_bfloat16*>(ring + NSF * UNIT_BYTES);\n  constexpr int NKS = HD / KS;  // 16\n  int u = blockIdx.x;\n  (void)ctr;\n  if (u >= nF) return;\n  constexpr int UBS = PB ? UB_B : UB_A;  // == ubs, as a constant (shift instead of a divide)\n  (void)ubs;\n  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };\n  auto wait_A = [&](int j) {  // phase B: inter rows of job j complete\n    if (PB) {\n      if (tid == 0) { while (ld_acquire(p.doneA + j) < UB_A) __nanosleep(64); }\n      CTA_SYNC();\n    }\n  };\n  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };\n  auto stage_x = [&](int j, int ks, int xsl) {\n    const int cnt = S.jC[j], st = S.jS[j];\n    const int ntl = (cnt + 7) >> 3;\n    const int nrow = ntl * 8;\n    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);\n    const int per = nrow * (KS / 8);\n    if (is_nat(j)) {\n      // 128 B swizzled K-major layout [kb 4][64 rows][128 B]: a warp copies one 512 B row (conflict-free)\n      unsigned char* xsb = reinterpret_cast<unsigned char*>(xs);\n      for (int i = tid; i < per; i += NT) {\n        const int r = i >> 5, ch = i & 31;\n        const bool v = r < cnt;\n        const __nv_bfloat16* src;\n        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + ch * 8;\n        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + ch * 8;\n        cp_async16(reinterpret_cast<__nv_bfloat16*>(xsb + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4)), src, v);\n      }\n    } else {\n      for (int i = tid; i < per; i += NT) {\n        int r = i / (KS / 8), c = i - r * (KS / 8);\n        bool v = r < cnt;\n        const __nv_bfloat16* src;\n        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + c * 8;\n        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + c * 8;\n        cp_async16(xs + r * XP_FT + c * 8, src, v);\n      }\n    }\n    cp_commit();\n  };\n  // BIG kernel: per-thread x staging addresses computed once per unit (a stage only adds its k offset)\n  const __nv_bfloat16* xsrc[8];\n  uint32_t xdst[8];\n  uint32_t xval = 0;\n  int xn = 0;\n  auto prep_x = [&](int j) {\n    const int cnt = S.jC[j], st = S.jS[j];\n    const int per = ((cnt + 7) >> 3) * 8 * (KS / 8);\n    xval = 0;\n    xn = per > tid ? min(8, (per - tid + NT - 1) / NT) : 0;\n#pragma unroll\n    for (int k = 0; k < 8; ++k) {\n      const int i = tid + k * NT;\n      const int r = i >> 5, c = i & 31;\n      const bool v = i < per && r < cnt;\n      if (PB) xsrc[k] = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;\n      else xsrc[k] = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;\n      xdst[k] = r * XP_FT + c * 8;\n      if (v) xval |= 1u << k;\n    }\n  };\n  auto stage_xp = [&](int ks, int xsl) {\n    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);\n#pragma unroll\n    for (int k = 0; k < 8; ++k)\n      if (k < xn) cp_async16(xs + xdst[k], xsrc[k] + ks * KS, (xval >> k) & 1u);\n    cp_commit();\n  };\n  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  int f = S.fseq;  // ring position of the current unit\'s k-stage 0\n  int xq = 0;      // x stage parity counter\n  int j, ub;\n  job_of(u, j, ub);\n#pragma unroll\n  for (int k = 0; k < NSF; ++k) fat_issue<PB>(p, S.jE[j], ub, k, ring + ((f + k) % NSF) * UNIT_BYTES, fb + (f + k) % NSF, is_nat(j));\n  wait_A(j);\n  if (OKC_XPRE) { prep_x(j); stage_xp(0, 0); } else stage_x(j, 0, 0);\n  while (true) {\n    const int un = u + gridDim.x;\n    int jn = 0, ubn = 0;\n    if (un < nF) job_of(un, jn, ubn);\n    const int e = S.jE[j], cnt = S.jC[j], st = S.jS[j];\n    const int ntile = (cnt + 7) >> 3;\n    const int ntc = ntile;  // exact token-tile count (rows past cnt are zero-filled)\n    const bool nat = is_nat(j);\n    const int rb0 = ub * 16 + warp, rb1 = rb0 + 8;\n    float acc0[FT_T / 8][4], acc1[FT_T / 8][4];\n#pragma unroll\n    for (int i = 0; i < FT_T / 8; ++i)\n#pragma unroll\n      for (int k = 0; k < 4; ++k) { acc0[i][k] = 0.f; acc1[i][k] = 0.f; }\n    for (int ks = 0; ks < NKS; ++ks) {\n      if (OKC_XPRE) {\n        if (ks + 1 < NKS) stage_xp(ks + 1, (xq + 1) & 1);\n        else if (un < nF) { wait_A(jn); prep_x(jn); stage_xp(0, (xq + 1) & 1); }\n        else cp_commit();\n      } else {\n        if (ks + 1 < NKS) stage_x(j, ks + 1, (xq + 1) & 1);\n        else if (un < nF) { wait_A(jn); stage_x(jn, 0, (xq + 1) & 1); }\n        else cp_commit();\n      }\n      cp_wait<1>();\n      const int q = f + ks, sl = q % NSF;\n      mbar_wait(fb + sl, (uint32_t)((q / NSF) & 1));\n      if (nat) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");  // cp.async x writes -> wgmma reads\n      CTA_SYNC();  // x stage visible to every warp\n      const unsigned char* slot = ring + sl * UNIT_BYTES;\n      const __nv_bfloat16* xs = xs0 + (xq & 1) * (FT_T * XP_FT);\n      if (nat) {\n        switch (ntc) {\n          case 1: fat_compute_wg<1>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 2: fat_compute_wg<2>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 5: fat_compute_wg<5>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 6: fat_compute_wg<6>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 7: fat_compute_wg<7>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          default: fat_compute_wg<8>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n        }\n      } else {\n        switch (ntc) {\n          case 1: fat_compute<1>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 2: fat_compute<2>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 5: fat_compute<5>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 6: fat_compute<6>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 7: fat_compute<7>(S, slot, xs, warp, lane, acc0, acc1); break;\n          default: fat_compute<8>(S, slot, xs, warp, lane, acc0, acc1); break;\n        }\n      }\n      CTA_SYNC();  // slot sl and the x stage consumed\n      ++xq;\n      // refill: the stage NSF ahead, in this unit or the next one\n      const int kn = ks + NSF;\n      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);\n      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));\n    }\n    // epilogue of unit u (registers -> global)\n    const float gsc = (PB ? p.gsB[e] : p.gsA[e]) * TWO62;\n    if (!PB) {\n#pragma unroll\n      for (int nt = 0; nt < FT_T / 8; ++nt)\n        if (nt < ntile) {\n          epi_A(p, st, cnt, nt * 8 + 2 * t, rb0 * 8 + g, gsc, acc0[nt]);\n#if OKC_NW == 8\n          epi_A(p, st, cnt, nt * 8 + 2 * t, rb1 * 8 + g, gsc, acc1[nt]);\n#endif\n        }\n      __threadfence();\n      CTA_SYNC();\n      if (tid == 0) atomicAdd(p.doneA + j, 1);\n    } else {\n#pragma unroll\n      for (int nt = 0; nt < FT_T / 8; ++nt)\n        if (nt < ntile) {\n          float sq0[2], sq1[2];\n          epi_B(p, st, cnt, nt * 8 + 2 * t, rb0 * 16 + g, gsc, acc0[nt], sq0);\n#if OKC_NW == 8\n          epi_B(p, st, cnt, nt * 8 + 2 * t, rb1 * 16 + g, gsc, acc1[nt], sq1);\n          sq0[0] += sq1[0]; sq0[1] += sq1[1];\n#endif\n          ss_reduce_add(p, st, cnt, nt * 8 + 2 * t, sq0, g);\n        }\n    }\n    f += NKS;\n    if (un >= nF) break;\n    u = un; j = jn; ub = ubn;\n  }\n  cp_wait<0>();\n  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  if (tid == 0) S.fseq = f;\n  CTA_SYNC();\n}\n\ntemplate <bool PB>\n__device__ __noinline__ void fat_stream_ni(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {\n  fat_stream<PB, false>(p, S, xbuf, nF, ubs);\n}\n\n\n\n// issue the phase-A skinny ring prologue as soon as the job table exists (before the positions pass)\n__device__ void early_issue(const Params& p, Smem& S) {\n#if FAST1_ON || OKC_WGS\n  return;  // the first units were issued at kernel start (FAST1), or the WGS producer warp owns the ring\n#endif\n  if (OKC_FATSTATIC && S.njF > 0) {  // fat units run first and own the ring region\n    if (TIDX == 0) S.preissued = 0;\n    return;\n  }\n  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);  // static part (the pool tail is claimed at run time)\n  const float rG = __fdividef(1.f, (float)gridDim.x);\n  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);\n  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);\n  // only the first unit, issued by the last warp so warp 0 (routing) is not held up; the ring prologue in\n  // skinny_ring issues the rest after routing, so every CTA\'s first unit is served ahead of the second ones\n  const int n = min(OKC_EARLYN, v1 - v0);\n  if (TIDX == NT - 32) {\n    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }\n    S.preissued = n > 0 ? n : 0;\n  }\n}\n\n// skinny share [v0, v1) of CTA c when nF fat units go round robin first: every CTA ends at the same modeled cost\n#if OKC_WS\n// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)\n// The NW compute warps only wait for data, compute, release the slot and hand their partial sums over; the helper\n// warp (threads NT..NT+31) issues every TMA copy as soon as all compute warps released a slot and runs every unit\'s\n// epilogue, so no CTA-wide barrier sits in the per-unit loop.\n// the last WSBLK units keep their partial buffers until the compute warps\' own epilogues: WSBLK > WSNR deadlocks the\n// partial ring (tuner, Oct 26: every WSNR=2 / WSBLK=3 config hung)\nstatic_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");\n#define OKC_WSRF 0\n__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);\ntemplate <bool PB>\n__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)\n  (void)k16;\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n#if OKC_HOF\n  const int nU = S.hof[4 * PB + 3], hx0 = S.hof[4 * PB], hnx = S.hof[4 * PB + 1], hr0 = S.hof[4 * PB + 2];\n  (void)v1;\n#else\n  const int nU = v1 - v0;\n#endif\n  const int seq0 = seq;\n  uint4 x0[2 * NCH];\n  int curj = -1;\n#if OKC_HOFW\n  int kbB = -1;  // index of the current job in this CTA\'s phase-B job sequence\n#endif\n  float lgs = 0.f;                             // the last phase-B unit\'s epilogue constants (warp 0)\n  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)\n  for (int i = 0; i < nU; ++i) {\n#if OKC_HOF\n    const int v = i < hnx ? hx0 + i : hr0 + (i - hnx);\n#else\n    const int v = v0 + i;\n#endif\n    const int j = njF + v / rbs;\n    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {\n      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);\n      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);\n      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);\n    }\n    if (j != curj) {\n      curj = j;\n      if (PB) {\n#if OKC_HOFW\n        ++kbB;\n        if (kbB < 8) mbar_wait(S.hrdy + kbB, 0u);  // acquire.cta: the poller\'s acquire of doneA happens-before\n        else {\n#endif\n        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n#if OKC_HOFW\n        }\n#endif\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);\n    }\n    const int q = seq0 + i;\n    const int sl = q % OKC_NS;\n    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n    const unsigned char* slot = ring + sl * UNIT_BYTES;\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCP(swd, (c & 1) * 2, k16), sg8 = sg;\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#endif\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n#pragma unroll\n      for (int st = 0; st < 4; ++st) {\n        uint32_t r0, r1, r2, r3;\n        dq4(ww[st], r0, r1, r2, r3);\n        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, BML(r0, sg), BMH(r1, sg8), BML(r2, sg), BMH(r3, sg8), b[2 * st], b[2 * st + 1]);\n      }\n    }\n#pragma unroll\n    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n    // WSB: the compute warps keep the base loop\'s CTA barrier per unit (slot consumed + partials visible), thread 0\n    // refills the slot and hands the partial buffer to the helper warp, which runs every epilogue off this loop\n    const int bb = q % NRB;\n    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer\'s previous unit\n    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    CTA_SYNC();\n    if (tid == 0) {\n      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);\n      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);\n      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1\n    }\n  }\n  if (PB && warp < OKC_WSBLK && warp < nU) {  // the CTA\'s last units: epilogues here, not behind the helper\'s queue\n    const int ul = nU - 1 - warp;\n    const int q = seq0 + ul, bb = q % NRB;\n#if OKC_HOF\n    const int v = ul < hnx ? hx0 + ul : hr0 + (ul - hnx);\n#else\n    const int v = v0 + ul;\n#endif\n    const int j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;\n    const int gg = lane >> 2, tt = lane & 3;\n    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int w = 0; w < NW; ++w) {\n      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);\n      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;\n    }\n    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n    const int n0 = rb * 16 + gg;\n    const float gsc = lgs * TWO62;\n    const float nw0 = 1.f + __uint_as_float(lnw0 << 16), nw1 = 1.f + __uint_as_float(lnw1 << 16);\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const int tk = 2 * tt + h;\n      const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;\n      float sq = v0f * v0f + v1f * v1f;\n      if (tk < cnt) {\n        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n      }\n      sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n      sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n      sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n    }\n  }\n  seq = seq0 + nU;\n}\n\n#if M2T_ON\n// r8 MID (M2T) lean unified loop. The CTA\'s sequence q = 0 .. nA + nB - 1 (phase A then phase B, the phase a run-time\n// flag) runs as job segments: a short prologue per segment (phase-B readiness, operand loads, the WSBL epilogue\n// constants) and a counters-only inner loop per tile variant, so phase B runs the same (already warm) bodies as phase A.\n// Per unit only the data slot / parity rotate and thread 0\'s ring refill walks a register cursor (global unit pointer and\n// units left in its contiguous run). Operand rows past a job\'s token count are left as they are (they only feed MMA\n// columns whose results are never stored).\n__device__ __forceinline__ void m2u_load_x(const Params& p, const Smem& S, int j, bool PB, int lane, int warp, uint4 (&x)[2 * NCH]) {\n  const int g = lane >> 2, t = lane & 3;\n  if (g < S.jC[j]) {\n    const int st = S.jS[j];\n    const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + g) * ID : p.x + (size_t)(S.ent[st + g] >> 3) * HD;\n    const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) { x[2 * c] = __ldca(src + c * 8); x[2 * c + 1] = __ldca(src + c * 8 + 1); }\n  }\n}\n// rows 8..15 of job j, K range [512 * warp, +512), into the warp\'s smem x1 tile (cp.async, zero-filled past the count)\n__device__ __forceinline__ void m2u_load_x1(const Params& p, const Smem& S, int j, bool PB, int lane, int warp, __nv_bfloat16* x1s) {\n  const int r = lane >> 2, c0 = lane & 3, st = S.jS[j];\n  const bool ok = 8 + r < S.jC[j];\n  const __nv_bfloat16* row = ok ? (PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD) : p.x;\n  const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c0;\n  __nv_bfloat16* dst = x1s + r * X1P + c0 * 8;\n#pragma unroll\n  for (int k = 0; k < 16; ++k) cp_async16(dst + k * 32, src + k * 4, ok);  // 16 B chunks c0, c0 + 4, ..., c0 + 60 of the row\n  cp_commit();\n}\nstruct M2uCur { int sl; uint32_t par; int rq, rleft; const unsigned char* rsrc; };\ntemplate <bool TWO>\n__device__ __forceinline__ void m2u_unit(const Params& p, Smem& S, int q, int N, int nA, int a0, int b0, int qarr, unsigned char* ring,\n                                         const uint4 (&x0)[2 * NCH], const __nv_bfloat16* x1s, M2uCur& cu) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)\n  (void)k16; (void)x1s;\n  mbar_wait(S.bars + cu.sl, cu.par);\n  const unsigned char* slot = ring + cu.sl * UNIT_BYTES;\n  float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n  float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n  if (TWO) {\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCP(swd, (c & 1) * 2, k16), sg8 = sg;\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#endif\n      const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);\n      const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n      const uint32_t bq[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};\n#pragma unroll\n      for (int st = 0; st < 4; ++st) {\n        uint32_t r0, r1, r2, r3;\n        dq4(ww[st], r0, r1, r2, r3);\n        const uint32_t a0 = BML(r0, sg), a1 = BMH(r1, sg8), a2 = BML(r2, sg), a3 = BMH(r3, sg8);\n        mma_bf16((st & 1) ? acc2 : acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);\n        mma_bf16((st & 1) ? acc4 : acc3, a0, a1, a2, a3, bq[2 * st], bq[2 * st + 1]);\n      }\n    }\n#pragma unroll\n    for (int k = 0; k < 4; ++k) { acc[k] += acc2[k]; acc3[k] += acc4[k]; }\n  } else {\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCP(swd, (c & 1) * 2, k16), sg8 = sg;\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#endif\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n#pragma unroll\n      for (int st = 0; st < 4; ++st) {\n        uint32_t r0, r1, r2, r3;\n        dq4(ww[st], r0, r1, r2, r3);\n        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, BML(r0, sg), BMH(r1, sg8), BML(r2, sg), BMH(r3, sg8), b[2 * st], b[2 * st + 1]);\n      }\n    }\n#pragma unroll\n    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n  }\n  const int bb = q % NRB;\n  mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer\'s previous unit\n  *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n  if (TWO) *reinterpret_cast<float4*>(&S.red2[bb][warp][lane][0]) = make_float4(acc3[0], acc3[1], acc3[2], acc3[3]);\n  CTA_SYNC();\n  if (tid == 0) {\n    if (cu.rq < N) {  // unit rq (= q + OKC_NS) into the slot unit q just released\n      if (cu.rleft == 0) {  // a new contiguous run of one job (and phase)\n        const bool rP = cu.rq >= nA;\n        const int rv = rP ? b0 + (cu.rq - nA) : a0 + cu.rq, rlg = rP ? RB_B_LOG : RB_A_LOG, rvj = rv >> rlg;\n        cu.rleft = min((rP ? N : nA) - cu.rq, ((rvj + 1) << rlg) - rv);\n        cu.rsrc = (rP ? p.wBu : p.wAu) + ((size_t)S.jE[S.njF + rvj] * (1 << rlg) + (rv - (rvj << rlg))) * (size_t)UNIT_BYTES;\n      }\n      uint64_t* bar = S.bars + cu.sl;\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                   ::"r"(smem_u32(slot)), "l"(cu.rsrc), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n      cu.rsrc += UNIT_BYTES; --cu.rleft; ++cu.rq;\n    }\n    if (!OKC_WSBL || q < qarr) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1\n  }\n  cu.sl = cu.sl == OKC_NS - 1 ? 0 : cu.sl + 1;\n  cu.par ^= cu.sl == 0 ? 1u : 0u;\n}\n__device__ __forceinline__ void skinny_ring_m2u(const Params& p, Smem& S, __nv_bfloat16* xbuf) {\n  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const int njF = S.njF;\n  const int a0 = S.wsr[0], nA = S.wsr[1] - a0, b0 = S.wsr[2], N = nA + (S.wsr[3] - b0), nB = N - nA;\n  uint4 x0[2 * NCH];\n#pragma unroll\n  for (int c = 0; c < 2 * NCH; ++c) x0[c] = make_uint4(0, 0, 0, 0);\n  __nv_bfloat16* x1s = reinterpret_cast<__nv_bfloat16*>(ring + OKC_NS * UNIT_BYTES) + warp * 8 * X1P;\n  float lgs = 0.f;                             // the last phase-B unit\'s epilogue constants (warp 0)\n  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)\n  const int qlgs = warp < OKC_WSBLK ? N - 1 - warp : -1;  // this warp\'s own-epilogue unit (WSBL)\n  const int qarr = N - min(OKC_WSBLK, nB);                  // units from here keep their partial buffers\n  M2uCur cu{0, 0u, OKC_NS, 0, nullptr};\n  int q = 0;\n  while (q < N) {\n    const bool PB = q >= nA;\n    const int v = PB ? b0 + (q - nA) : a0 + q, lg = PB ? RB_B_LOG : RB_A_LOG, vj = v >> lg, j = njF + vj;\n    const int qend = q + min((PB ? N : nA) - q, ((vj + 1) << lg) - v);\n    if (PB) {\n      if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n      __syncwarp();\n    }\n    const bool two = OKC_M2TF || S.jC[j] > 8;\n    if (two) m2u_load_x1(p, S, j, PB, lane, warp, x1s);  // async, overlapped with the x0 register loads\n    m2u_load_x(p, S, j, PB, lane, warp, x0);\n    if (qlgs >= q && qlgs < qend) {  // (phase B: the CTA\'s last units) epilogue constants of this warp\'s own unit\n      const int n0l = ((v + (qlgs - q)) & (RB_B - 1)) * 16 + (lane >> 2);\n      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);\n      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);\n    }\n    if (two) {\n      cp_wait<0>();\n      __syncwarp();\n      for (; q < qend; ++q) m2u_unit<true>(p, S, q, N, nA, a0, b0, qarr, ring, x0, x1s, cu);\n    } else {\n      for (; q < qend; ++q) m2u_unit<false>(p, S, q, N, nA, a0, b0, qarr, ring, x0, x1s, cu);\n    }\n  }\n  if (warp < OKC_WSBLK && warp < nB) {  // the CTA\'s last units: epilogues here, not behind the helper\'s queue\n    const int ul = nB - 1 - warp;\n    const int q = N - 1 - warp, bb = q % NRB;\n    const int v = b0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;\n    const int gg = lane >> 2, tt = lane & 3;\n    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int w = 0; w < NW; ++w) {\n      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);\n      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;\n    }\n    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n    const int n0 = rb * 16 + gg;\n    const float gsc = lgs * TWO62;\n    const float nw0 = 1.f + __uint_as_float(lnw0 << 16), nw1 = 1.f + __uint_as_float(lnw1 << 16);\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const int tk = 2 * tt + h;\n      const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;\n      float sq = v0f * v0f + v1f * v1f;\n      if (tk < cnt) {\n        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n      }\n      sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n      sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n      sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n    }\n    if (OKC_M2TF || cnt > 8) {  // tokens 8-15 of a 2-tile job\n      float4 s2 = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n      for (int w = 0; w < NW; ++w) {\n        const float4 t = *reinterpret_cast<const float4*>(&S.red2[bb][w][lane][0]);\n        s2.x += t.x; s2.y += t.y; s2.z += t.z; s2.w += t.w;\n      }\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int tk = 8 + 2 * tt + h;\n        const float v0f = (h ? s2.y : s2.x) * gsc * nw0, v1f = (h ? s2.w : s2.z) * gsc * nw1;\n        float sq = v0f * v0f + v1f * v1f;\n        if (tk < cnt) {\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n        }\n        sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n  }\n  S.seq = N;\n}\n#endif\n#if OKC_HOF\n// r8 HOF: unit of sequence position i of phase PB (X segment first, then the rest)\n__device__ __forceinline__ int hof_unit(const Smem& S, int PB, int i) {\n  const int* h = S.hof + 4 * PB;\n  return i < h[1] ? h[0] + i : h[2] + (i - h[1]);\n}\n#endif\n__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {\n  const int sl = q % OKC_NS;\n#if OKC_HOF\n  (void)a0; (void)b0;\n  if (q < nA) ring_issue<false>(p, S, hof_unit(S, 0, q), ring + sl * UNIT_BYTES, S.bars + sl);\n  else ring_issue<true>(p, S, hof_unit(S, 1, q - nA), ring + sl * UNIT_BYTES, S.bars + sl);\n#else\n  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);\n  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);\n#endif\n}\n\n__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n  const int lane = TIDX & 31;\n  WS_JOIN();  // routing tables and this CTA\'s ranges are ready\n  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];\n  const int nA = a1 - a0, N = nA + (b1 - b0);\n  const int njF = S.njF;\n  if (lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n  const int gg = lane >> 2, tt = lane & 3;\n  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;\n  for (int q = 0; q < Nh; ++q) {\n    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q\'s slot once every compute warp released it\n      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));\n      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);\n    }\n    const bool PB = q >= nA;\n#if OKC_HOF\n    const int v = PB ? hof_unit(S, 1, q - nA) : hof_unit(S, 0, q);\n#else\n    const int v = PB ? b0 + (q - nA) : a0 + q;\n#endif\n    const int rbs = PB ? RB_B : RB_A;\n    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;\n    const int bb = q % NRB;\n    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));\n    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int w = 0; w < NW; ++w) {\n      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);\n      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;\n    }\n#if M2T_ON\n    const bool two = OKC_M2TF || S.jC[j] > 8;\n    float4 s2 = make_float4(0.f, 0.f, 0.f, 0.f);\n    if (two) {\n#pragma unroll\n      for (int w = 0; w < NW; ++w) {\n        const float4 t = *reinterpret_cast<const float4*>(&S.red2[bb][w][lane][0]);\n        s2.x += t.x; s2.y += t.y; s2.z += t.z; s2.w += t.w;\n      }\n    }\n#endif\n    __syncwarp();\n    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again\n    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)\n    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n    if (!PB) {\n      const float gsc = p.gsA[ce] * TWO62;\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int tk = 2 * tt + h;\n        if (tk < cnt) {\n          const float gt = (h ? sm.y : sm.x) * gsc, up = (h ? sm.w : sm.z) * gsc;\n          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));\n        }\n      }\n#if M2T_ON\n      if (two) {\n#pragma unroll\n        for (int h = 0; h < 2; ++h) {\n          const int tk = 8 + 2 * tt + h;\n          if (tk < cnt) {\n            const float gt = (h ? s2.y : s2.x) * gsc, up = (h ? s2.w : s2.z) * gsc;\n            st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));\n          }\n        }\n      }\n#endif\n#if OKC_HOF\n      if (q == S.hof[1] - 1 && q < nA - 1) {  // r8 HOF: this CTA\'s X segment is done: credit it now\n#if OKC_HOFW\n        __syncwarp();\n        if (lane == 0) mbar_arrive1(&S.hxd);  // the poller credits X\n#endif\n      }\n#endif\n      if (q == nA - 1) {  // this CTA\'s phase A is done: credit its units to the jobs\' counters\n#if OKC_HOFW\n        __syncwarp();\n        if (lane == 0) mbar_arrive1(&S.had);  // the poller fences and credits\n        if (false) {\n#else\n        __threadfence();\n        __syncwarp();\n        if (lane == 0) {\n#endif\n#if OKC_HOF\n          const int r0 = S.hof[2], r1 = r0 + (nA - S.hof[1]);  // the rest segment (X was credited at its end)\n          if (r1 > r0) {\n          const int jf = njF + r0 / RB_A, jl = njF + (r1 - 1) / RB_A;\n          for (int jj = jf; jj <= jl; ++jj) {\n            const int a = max(r0, (jj - njF) * RB_A), b = min(r1, (jj - njF + 1) * RB_A);\n            atomicAdd(p.doneA + jj, b - a);\n          }\n          }\n#else\n          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;\n          for (int jj = jf; jj <= jl; ++jj) {\n            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);\n            atomicAdd(p.doneA + jj, b - a);\n          }\n#endif\n        }\n      }\n    } else {\n      const float gsc = p.gsB[ce] * TWO62;\n      const int n0 = rb * 16 + gg;\n      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int tk = 2 * tt + h;\n        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;\n        float sq = v0f * v0f + v1f * v1f;\n        if (tk < cnt) {\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n        }\n        sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n#if M2T_ON\n      if (two) {\n#pragma unroll\n        for (int h = 0; h < 2; ++h) {\n          const int tk = 8 + 2 * tt + h;\n          const float v0f = (h ? s2.y : s2.x) * gsc * nw0, v1f = (h ? s2.w : s2.z) * gsc * nw1;\n          float sq = v0f * v0f + v1f * v1f;\n          if (tk < cnt) {\n            st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n            st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n          }\n          sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n          sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n          sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n          if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n        }\n      }\n#endif\n    }\n  }\n  __threadfence();\n  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n}\n#endif\n\n#if OKC_HOFW\n// r8 poller warp (threads NT+32..NT+63): lane 0 credits X once the helper wrote this CTA\'s X epilogues (release fence at\n// gpu scope here, not on the helper), then waits for X\'s GEMM1 everywhere and arrives on S.hrdy[0]; lane 1 credits the\n// rest of this CTA\'s phase-A units once the helper wrote them, then waits for each later phase-B job of this CTA (in\n// sequence order) and arrives on S.hrdy[k]. Lanes 0 and 1 run independently (divergent spin loops).\n__device__ void ws_poller(const Params& p, Smem& S) {\n  const int lane = TIDX & 31;\n  WS_JOIN();  // routing tables and this CTA\'s ranges are ready\n  const int njF = S.njF, nA = S.hof[3], nxa = S.hof[1];\n  if (nA > 0 && lane == 0) {\n    if (nxa > 0 && nxa < nA) {\n      mbar_wait(&S.hxd, 0u);\n      asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");\n      const int x0u = S.hof[0], x1u = x0u + nxa;\n      for (int jj = njF + x0u / RB_A; jj <= njF + (x1u - 1) / RB_A; ++jj) {\n        const int a = max(x0u, (jj - njF) * RB_A), b = min(x1u, (jj - njF + 1) * RB_A);\n        atomicAdd(p.doneA + jj, b - a);\n      }\n    }\n    if (S.hof[5] > 0) {\n      int k = 0;\n      for (int jj = njF + S.hof[4] / RB_B; jj <= njF + (S.hof[4] + S.hof[5] - 1) / RB_B; ++jj, ++k) {\n        while (ld_acquire(p.doneA + jj) < RB_A) __nanosleep(32);\n        mbar_arrive1(S.hrdy + k);\n      }\n    }\n  } else if (nA > 0 && lane == 1) {\n    mbar_wait(&S.had, 0u);\n    asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");\n    {\n      const int r0 = S.hof[2], r1 = r0 + (nA - nxa);\n      if (r1 > r0) {\n        const int jf = njF + r0 / RB_A, jl = njF + (r1 - 1) / RB_A;\n        for (int jj = jf; jj <= jl; ++jj) {\n          const int a = max(r0, (jj - njF) * RB_A), b = min(r1, (jj - njF + 1) * RB_A);\n          atomicAdd(p.doneA + jj, b - a);\n        }\n      }\n    }\n    int k = S.hof[5] > 0 ? (S.hof[4] + S.hof[5] - 1) / RB_B - S.hof[4] / RB_B + 1 : 0;  // after the X-B jobs\n    const int rn = S.hof[7] - S.hof[5];\n    if (rn > 0) {\n      const int jl = njF + (S.hof[6] + rn - 1) / RB_B;\n      for (int jj = njF + S.hof[6] / RB_B; jj <= jl && k < 8; ++jj, ++k) {\n        while (ld_acquire(p.doneA + jj) < RB_A) __nanosleep(32);\n        mbar_arrive1(S.hrdy + k);\n      }\n    }\n  }\n  __syncwarp();\n  WS_JOIN();  // the end-of-kernel join\n}\n#endif\n\n#if OKC_HOFW\n// operand of job j for either phase (run-time PB): x rows (phase A) or inter rows (phase B), L2 loads\n__device__ __forceinline__ void load_x_rt(const Params& p, const Smem& S, int j, bool PB, int lane, int warp, uint4 (&x)[2 * NCH]) {\n  const int g = lane >> 2, t = lane & 3;\n  const int cnt = S.jC[j], st = S.jS[j];\n  if (g < cnt) {\n    const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + g) * ID : p.x + (size_t)(S.ent[st + g] >> 3) * HD;\n    const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);\n#pragma unroll\n    if (PB) {\n#pragma unroll\n      for (int c = 0; c < NCH; ++c) { x[2 * c] = __ldca(src + c * 8); x[2 * c + 1] = __ldca(src + c * 8 + 1); }\n    } else {\n#pragma unroll\n      for (int c = 0; c < NCH; ++c) { x[2 * c] = __ldg(src + c * 8); x[2 * c + 1] = __ldg(src + c * 8 + 1); }\n    }\n  } else {\n#pragma unroll\n    for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);\n  }\n}\n// HUNI4: operand rows of job j for this lane (rows past the job\'s token count are left as they are)\n__device__ __forceinline__ void load_x_rt4(const Params& p, const Smem& S, int j, bool PB, int lane, int warp, uint4 (&x)[2 * NCH]) {\n  const int g = lane >> 2, t = lane & 3;\n  if (g < S.jC[j]) {\n    const int st = S.jS[j];\n    const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + g) * ID : p.x + (size_t)(S.ent[st + g] >> 3) * HD;\n    const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) { x[2 * c] = __ldca(src + c * 8); x[2 * c + 1] = __ldca(src + c * 8 + 1); }\n  }\n}\n// L1 prefetch of the operand lines load_x_rt will read for job j (this lane: lines t and t + 4 of its row)\n__device__ __forceinline__ void prefetch_x_rt(const Params& p, const Smem& S, int j, bool PB, int lane, int warp) {\n  const int g = lane >> 2, t = lane & 3;\n  if (g < S.jC[j]) {\n    const int st = S.jS[j];\n    const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + g) * ID : p.x + (size_t)(S.ent[st + g] >> 3) * HD;\n    const char* b = reinterpret_cast<const char*>(row + warp * (64 * NCH));\n    asm volatile("prefetch.global.L1 [%0];\\n" ::"l"(b + t * 128));\n    asm volatile("prefetch.global.L1 [%0];\\n" ::"l"(b + (t + 4) * 128));\n  }\n}\n__device__ __forceinline__ bool mbar_test(uint64_t* bar, uint32_t parity) {  // r8: non-blocking (acquire.cta when complete)\n  uint32_t r;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(r) : "r"(smem_u32(bar)), "r"(parity) : "memory");\n  return r != 0;\n}\n// r8 HUNI2 compute warps (TINY / TINY2): the CTA\'s sequence q = 0 .. nA + nB - 1 (HOF order in both phases)\n__device__ __forceinline__ void skinny_ring_uni2(const Params& p, Smem& S, __nv_bfloat16* xbuf) {\n  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);\n  uint64_t* bars = S.bars;\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)\n  (void)k16;\n  const int njF = S.njF;\n  const int nA = S.hof[3], N = nA + S.hof[7];\n  const int hA0 = S.hof[0], hAn = S.hof[1], hAr = S.hof[2], hB0 = S.hof[4], hBn = S.hof[5], hBr = S.hof[6];\n  uint4 x0[2 * NCH];\n#pragma unroll\n  for (int c = 0; c < 2 * NCH; ++c) x0[c] = make_uint4(0, 0, 0, 0);\n  int xk = -1;    // key (j | PB << 16) of the job whose operand x0 holds\n  int kbB = -1;   // index of the current job in this CTA\'s phase-B job sequence\n  float lgs = 0.f;                             // the last phase-B unit\'s epilogue constants (warp 0)\n  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)\n  const int qlgs = warp < OKC_WSBLK ? N - 1 - warp : -1;          // this warp\'s own-epilogue unit (WSBL)\n  const int qarr = N - min(OKC_WSBLK, N - nA);                     // units from here keep their partial buffers\n  // thread 0\'s refill cursor: the unit at sequence position rq (= q + OKC_NS), its global address, units left in its run\n  int rq = OKC_NS, rleft = 0;\n  const unsigned char* rsrc = nullptr;\n  int sl = 0;                                   // data slot of unit q (q % OKC_NS) and its mbarrier parity\n  uint32_t par = 0u;\n  int q = 0;\n  while (q < N) {\n    // ---- a job segment starts at q\n    const bool PB = q >= nA;\n    const int i = PB ? q - nA : q;\n    const int hnx = PB ? hBn : hAn;\n    const int v = i < hnx ? (PB ? hB0 : hA0) + i : (PB ? hBr : hAr) + (i - hnx);\n    const int lg = PB ? RB_B_LOG : RB_A_LOG;\n    const int vj = v >> lg, j = njF + vj, key = j | (PB ? 0x10000 : 0);\n    const int qend = q + min((i < hnx ? hnx : (PB ? N - nA : nA)) - i, ((vj + 1) << lg) - v);\n    if (PB) ++kbB;\n    if (xk != key) {\n      if (PB) {\n        if (kbB < 8) mbar_wait(S.hrdy + kbB, 0u);  // acquire.cta: the poller\'s acquire of doneA happens-before\n        else { if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); } __syncwarp(); }\n      }\n      load_x_rt4(p, S, j, PB, lane, warp, x0);\n      xk = key;\n    }\n    // the next segment (its operand lines are prefetched at the start of this segment\'s last unit)\n    int qpre = -1, jn = 0, kn = -1;\n    bool PBn = false;\n    if (qend < N) {\n      PBn = qend >= nA;\n      const int i2 = PBn ? qend - nA : qend, hnx2 = PBn ? hBn : hAn;\n      const int v2 = i2 < hnx2 ? (PBn ? hB0 : hA0) + i2 : (PBn ? hBr : hAr) + (i2 - hnx2);\n      jn = njF + (v2 >> (PBn ? RB_B_LOG : RB_A_LOG)); qpre = qend - 1; kn = PBn ? kbB + 1 : -1;\n    }\n    if (qlgs >= q && qlgs < qend) {  // (phase B: the CTA\'s last units) epilogue constants of this warp\'s own unit\n      const int vl = v + (qlgs - q), rbl = vl & (RB_B - 1), n0l = rbl * 16 + (lane >> 2);\n      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);\n      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);\n    }\n    for (; q < qend; ++q) {\n      if (q == qpre || q == 0) {  // (q == 0: warms this block during the first unit\'s data wait)\n        const int jj = q == qpre ? jn : j;\n        const bool pp = q == qpre ? PBn : false;\n        if (!pp || (kn < 8 && __all_sync(0xffffffffu, mbar_test(S.hrdy + kn, 0u)))) prefetch_x_rt(p, S, jj, pp, lane, warp);\n      }\n      mbar_wait(bars + sl, par);\n      const unsigned char* slot = ring + sl * UNIT_BYTES;\n      const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n      float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n      float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n#pragma unroll\n      for (int c = 0; c < NCH; ++c) {\n        const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n        const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n        const uint32_t sg = SCP(swd, (c & 1) * 2, k16), sg8 = sg;\n#else\n        const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n        const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#endif\n        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n        const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                               x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, BML(r0, sg), BMH(r1, sg8), BML(r2, sg), BMH(r3, sg8), b[2 * st], b[2 * st + 1]);\n        }\n      }\n#pragma unroll\n      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n      const int bb = q % NRB;\n      mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer\'s previous unit\n      *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n      CTA_SYNC();\n      if (tid == 0) {\n        if (rq < N) {  // unit rq into the slot unit q just released (rq % OKC_NS == sl)\n          if (rleft == 0) {  // a new contiguous run of one job (and phase)\n            const bool rP = rq >= nA;\n            const int ri = rP ? rq - nA : rq, rnx = rP ? hBn : hAn;\n            const int rv = ri < rnx ? (rP ? hB0 : hA0) + ri : (rP ? hBr : hAr) + (ri - rnx);\n            const int rlg = rP ? RB_B_LOG : RB_A_LOG, rvj = rv >> rlg;\n            rleft = min((ri < rnx ? rnx : (rP ? N - nA : nA)) - ri, ((rvj + 1) << rlg) - rv);\n            rsrc = (rP ? p.wBu : p.wAu) + ((size_t)S.jE[njF + rvj] * (1 << rlg) + (rv - (rvj << rlg))) * (size_t)UNIT_BYTES;\n          }\n          uint64_t* bar = bars + sl;\n          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                       ::"r"(smem_u32(slot)), "l"(rsrc), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n          rsrc += UNIT_BYTES; --rleft; ++rq;\n        }\n        if (!OKC_WSBL || q < qarr) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1\n      }\n      sl = sl == OKC_NS - 1 ? 0 : sl + 1;\n      par ^= sl == 0 ? 1u : 0u;\n    }\n  }\n  const int nB = N - nA;\n  if (warp < OKC_WSBLK && warp < nB) {  // the CTA\'s last units: epilogues here, not behind the helper\'s queue\n    const int ul = nB - 1 - warp;\n    const int sq = N - 1 - warp, bb = sq % NRB;\n    const int v = ul < hBn ? hB0 + ul : hBr + (ul - hBn);\n    const int j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;\n    const int gg = lane >> 2, tt = lane & 3;\n    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int w = 0; w < NW; ++w) {\n      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);\n      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;\n    }\n    const int cnt = S.jC[j], st = S.jS[j];\n    const int n0 = rb * 16 + gg;\n    const float gsc = lgs * TWO62;\n    const float nw0 = 1.f + __uint_as_float(lnw0 << 16), nw1 = 1.f + __uint_as_float(lnw1 << 16);\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const int tk = 2 * tt + h;\n      const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;\n      float sq2 = v0f * v0f + v1f * v1f;\n      if (tk < cnt) {\n        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n      }\n      sq2 += __shfl_xor_sync(0xffffffffu, sq2, 4);\n      sq2 += __shfl_xor_sync(0xffffffffu, sq2, 8);\n      sq2 += __shfl_xor_sync(0xffffffffu, sq2, 16);\n      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq2);\n    }\n  }\n  S.seq = N;\n}\n#endif\n\ntemplate <bool BIG, bool PB>\n__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {\n  const float rG = __fdividef(1.f, (float)G);\n  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;\n  auto fcost = [](long long k) -> long long {\n    // phase-specific cost of one fat unit in skinny-unit equivalents (phase B\'s skinny units are slower, so its ratio is lower)\n    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);\n    return k > 0 ? (long long)fc * k : 0;\n  };\n  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);  // total cost = G x level\n  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;\n  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;\n  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };\n  const long long WG = W(G);\n  if (nF > 0 && WG > 0) {\n    // split point of CTA x: floor(nS * W(x) / WG) in float (one formula for both neighbours, so the shares tile [0, nS))\n    const float sc = __fdividef((float)nS, (float)WG);\n    auto split = [&](int x) -> int {\n      if (x >= G) return nS;\n      const int v = __float2int_rz((float)W(x) * sc);\n      return v < nS ? v : nS;\n    };\n    v0 = split(c); v1 = split(c + 1);\n  } else {\n    v0 = muldiv_g(nS, c, G, rG); v1 = muldiv_g(nS, c + 1, G, rG);\n  }\n}\n\n\ntemplate <bool PB, bool BIG>\n__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {\n  const int njF = S.njF, njS = S.njS;\n  const int ubs = PB ? UB_B : UB_A, rbs = PB ? RB_B : RB_A;\n  const int nS = njS * rbs, nF = njF * ubs;\n  // fat units first (round robin over CTAs), then a skinny share sized so every CTA carries the same cost\n#if OKC_TINY\n  if (nF > 0) __trap();  // tiny build (M <= 8): a job never has more than 8 tokens, so there are no fat units\n#endif\n  const int G = gridDim.x, c = blockIdx.x;\n  int v0, v1;\n  const int nSs = nS - dyn_pool(nS, G);  // static skinny units; [nSs, nS) is the run-time pool\n  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);\n  // cross-phase prefill: a CTA without phase-B fat units starts phase B with its skinny share, so phase A\'s last ring\n  // refills already fetch those units (same split function as phase B uses)\n  int cB0 = 0, cBn = 0;\n  bool cpf = false;\n  if (!PB) {\n    const int nFB = njF * UB_B;\n    if (c >= nFB) {\n      int b1;\n      skinny_split<BIG, true>(njS * RB_B - dyn_pool(njS * RB_B, G), nFB, G, c, cB0, b1);\n      cBn = b1 - cB0;\n      cpf = OKC_CPFF || njF == 0;\n    }\n  }\n#if OKC_WS\n  (void)cpf; (void)cBn;\n#if !M2T_ON\n  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n#endif\n#endif\n}\n\n// a volatile shared load at every use: hoisted, the pointer would stay live across the accumulate loops (96-register\n// cap: spills)\n__device__ __forceinline__ float* ss_cur() {\n#if OKC_WGS\n#else\n  return reinterpret_cast<const Smem*>(smem_raw)->ssp;  // the skinny / fat builds have registers to spare: let it hoist\n#endif\n}\ntemplate <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {\n  Smem& S = *reinterpret_cast<Smem*>(smem_raw);\n  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);\n  const int tid = TIDX;\n  DBG(0);\n  // programmatic dependent launch: everything below reads the previous kernel\'s outputs (routing, x) or writes\n  // scratch, so wait for it here; without the launch attribute this returns at once\n  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n#if OKC_WS\n#if OKC_HOFW\n  if (tid >= NT + 32) { ws_poller(p, S); return; }\n#endif\n  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n#endif\n#if FAST1_ON\n  // tiny build (M <= 8): every touched expert is one skinny job, in expert order, so this CTA\'s first units follow\n  // from the touched-expert mask alone; the last warp issues them before routing, off the cold-start critical path\n  if (tid >= NT - 32) {\n    const int ln = tid & 31;\n#if OKC_SPLIT\n    // per-expert entry counts -> jobs per expert (ceil(c / 8)) -> inclusive prefix over experts (lane = expert)\n    S.fcnt[ln] = 0;\n    int ev4[4];  // r8 MF1B: this lane\'s first four topk entries (all of them for M <= 16), loaded together\n#pragma unroll\n    for (int k = 0; k < 4; ++k) ev4[k] = ln + 32 * k < p.M * 8 ? __ldg(p.topk_ids + ln + 32 * k) : -1;\n    __syncwarp();\n    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {\n      const int i = i0 + ln;\n      int e = -1;\n      if (i < p.M * 8) { e = (i0 < 128 ? ev4[i0 >> 5] : __ldg(p.topk_ids + i)) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }\n      const unsigned peers = __match_any_sync(0xffffffffu, e);\n      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));\n    }\n    __syncwarp();\n    int jincl = (S.fcnt[ln] + SK_T - 1) / SK_T;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, jincl, d); if (ln >= d) jincl += v; }\n    const int njob = __shfl_sync(0xffffffffu, jincl, 31);\n    const int nS1 = njob * RB_A - dyn_pool(njob * RB_A, gridDim.x);\n    const float rG1 = __fdividef(1.f, (float)gridDim.x);\n#if M2T_ON\n    int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);\n    const bool any2 = OKC_M2TF || __any_sync(0xffffffffu, S.fcnt[ln] > 8);\n    // with 2-tile jobs: u(x) = first unit whose cost prefix reaches x (unit costs OKC_M2W1 / OKC_M2W2 per job)\n    const int fc = S.fcnt[ln], fr = fc & 15, je = (fc + SK_T - 1) / SK_T;\n    const int n1j = (!OKC_M2TF && fr >= 1 && fr <= 8) ? 1 : 0, n2j = je - n1j;\n    int cwe = n2j * OKC_M2W2 + n1j * OKC_M2W1, w2 = OKC_M2W2;\n    int cwi = 0;\n    auto scan = [&]() {\n      cwi = cwe;\n#pragma unroll\n      for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, cwi, d); if (ln >= d) cwi += v; }\n    };\n    auto unit_at = [&](int c, int rbs) -> int {\n      if (c >= (int)gridDim.x) return njob * rbs;\n      const int x = m2t_muldiv(__shfl_sync(0xffffffffu, cwi, 31) * rbs, c, gridDim.x, rG1);\n      const unsigned bal = __ballot_sync(0xffffffffu, cwi * rbs > x);\n      const int le = __ffs(bal) - 1;\n      const int cexcl = __shfl_sync(0xffffffffu, (cwi - cwe) * rbs, le);\n      const int uexcl = __shfl_sync(0xffffffffu, (jincl - je) * rbs, le);\n      const int n2 = __shfl_sync(0xffffffffu, n2j, le);\n      const int d = x - cexcl, c2 = n2 * rbs * w2;\n      return uexcl + (d <= c2 ? (w2 == OKC_M2W2 ? (d + OKC_M2W2 - 1) / OKC_M2W2 : (d + OKC_M2W2B - 1) / OKC_M2W2B) : n2 * rbs + (d - c2 + OKC_M2W1 - 1) / OKC_M2W1);\n    };\n    if (any2) {\n      scan();\n      v01 = unit_at(blockIdx.x, RB_A); v11 = unit_at(blockIdx.x + 1, RB_A);\n    }\n#endif\n    const int n1 = min(OKC_FAST1, v11 - v01);\n    int fex[OKC_FAST1 > 0 ? OKC_FAST1 : 1];\n#pragma unroll\n    for (int i = 0; i < OKC_FAST1; ++i) {\n      const int jj = (v01 + i) >> RB_A_LOG;   // (n1 is warp-uniform: every lane runs the same ballots)\n      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;\n    }\n#elif OKC_F1EARLY\n    // the topk loads go out first (E <= 64 in the tiny builds: two per lane at most) and lane 0 initializes the\n    // barriers while they are in flight (the inits used to wait behind the cold loads)\n    uint32_t m = 0;\n    const int E8 = p.M * 8;\n    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;\n    if (ln == 0) {\n      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);\n      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);\n      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }\n#if OKC_HOFW\n      for (int k = 0; k < 8; ++k) mbar_init(S.hrdy + k, 1);\n      mbar_init(&S.hxd, 1); mbar_init(&S.had, 1);\n#endif\n      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n    }\n    if (ln < E8) { const int e = ev0 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }\n    if (ln + 32 < E8) { const int e = ev1 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }\n    for (int i = ln + 64; i < E8; i += 32) {  // (never in the tiny builds: M <= 8)\n      const int e = __ldg(p.topk_ids + i) - p.lo;\n      if (e >= 0 && e < p.nloc) m |= 1u << e;\n    }\n    m = __reduce_or_sync(0xffffffffu, m);\n#endif\n    if (ln == 0) {\n#if OKC_SPLIT || !OKC_F1EARLY\n      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);\n      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);\n      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }\n      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n#endif\n#if OKC_SPLIT\n      const int v0 = v01, n = n1;\n#else\n      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);  // static part (pool tail claimed at run time)\n      const float rG = __fdividef(1.f, (float)gridDim.x);\n#if OKC_HOF\n      int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);\n      if (__popc(m) >= 2) {  // r8 HOF: this CTA\'s share of the last job(s)\' units first\n        const int kx = __popc(m) >= OKC_HOFK2 ? 2 : 1;\n        const int xo = (__popc(m) - kx) * RB_A;\n        v0 = xo + muldiv_g(kx * RB_A, blockIdx.x, gridDim.x, rG); v1 = xo + muldiv_g(kx * RB_A, blockIdx.x + 1, gridDim.x, rG);\n      }\n#endif\n      const int n = min(OKC_FAST1, v1 - v0);\n#endif\n      asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n#if OKC_SPLIT\n#pragma unroll\n      for (int i = 0; i < OKC_FAST1; ++i) {\n        if (i >= n) break;\n        const int v = v0 + i;\n        const int e = fex[i];\n#else\n      for (int i = 0; i < n; ++i) {\n        const int v = v0 + i;\n        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);\n#endif\n        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;\n        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;\n        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");\n        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");\n      }\n      S.preissued = n > 0 ? n : 0;\n    }\n#if M2T_ON && OKC_SPLIT\n    if (any2) {  // the rest of this CTA\'s cost-weighted ranges, off the first TMA\'s path (tid 0 skips its own split)\n      cwe = n2j * OKC_M2W2B + n1j * OKC_M2W1; w2 = OKC_M2W2B;  // phase-B unit costs\n      scan();\n      const int b0r = unit_at(blockIdx.x, RB_B), b1r = unit_at(blockIdx.x + 1, RB_B);\n      if (ln == 0) { S.wsr[0] = v01; S.wsr[1] = v11; S.wsr[2] = b0r; S.wsr[3] = b1r; }\n    }\n    if (ln == 0) {\n      if (!any2) {  // r8 MR7: the even split (tid 0 used to write it after route)\n        S.wsr[0] = muldiv_g(njob * RB_A, blockIdx.x, gridDim.x, rG1); S.wsr[1] = muldiv_g(njob * RB_A, blockIdx.x + 1, gridDim.x, rG1);\n        S.wsr[2] = muldiv_g(njob * RB_B, blockIdx.x, gridDim.x, rG1); S.wsr[3] = muldiv_g(njob * RB_B, blockIdx.x + 1, gridDim.x, rG1);\n      }\n      S.m2any = 1; S.f1v0 = v01;\n    }\n#endif\n  }\n#endif\n  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later\n  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)\n    asm volatile("prefetch.global.L2 [%0];\\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));\n  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code -> bf16x2(value * 2^64), computed (no global load)\n    const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;\n    uint32_t b;\n    if (ex == 15 && mt == 7) {\n      b = 0xFFFFu;\n    } else {\n      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);\n      v *= 0x1p64f;\n      b = __float_as_uint(sgn ? -v : v) >> 16;\n    }\n    S.lut[i] = b | (b << 16);\n  }\n  if (tid == 0) {\n    S.seq = 0;\n    S.fseq = 0;\n    S.preissuedB = 0;\n    S.xbuf_ptr = xbuf;\n    // the generation can only change at this launch\'s grid barrier, which every CTA reaches after this point\n    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)\n    S.gen = gen;\n    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n  }\n#if OKC_SPLIT && FAST1_ON\n  if (tid < RT_N) {  // r8 MR7: FAST1\'s warp skips route() and this zeroing (S.gen is ordered by route()\'s barriers)\n    route(p, S);\n    float* other = (S.gen & 1) ? p.ss : p.ss1;\n    for (int i = blockIdx.x * RT_N + tid; i < EMAX; i += gridDim.x * RT_N) other[i] = 0.f;\n    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;\n  }\n  {\n#else\n  route(p, S);\n  DBG(1);\n  {  // the previous launch\'s ss copy and barrier counter are the next launch\'s: zero them now (this CTA\'s share)\n    float* other = (S.gen & 1) ? p.ss : p.ss1;\n    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;\n    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;\n#endif\n  }\n#if OKC_WS\n  if (tid == 0) {  // this CTA\'s unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)\n    const int G = gridDim.x, c = blockIdx.x;\n    const float rG = __fdividef(1.f, (float)G);\n    const int nA = S.njS * RB_A, nB = S.njS * RB_B;\n#if M2T_ON\n    if (false) {  // r8 MR7: FAST1 wrote every M2T range\n#endif\n    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);\n    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n#if M2T_ON\n    }\n#endif\n#if OKC_HOF\n    {  // r8 HOF: X = the last job; its units dealt evenly over the CTAs and run first; the rest keeps the base order\n      int xa0 = 0, nxa = 0, ra0 = S.wsr[0], xb0 = 0, nxb = 0, rb0 = S.wsr[2];\n      if (S.njS >= 2) {\n        const int kx = S.njS >= OKC_HOFK2 ? 2 : 1;  // r8 HOFK: the X block\'s jobs\n        const int ka = muldiv_g(kx * RB_A, c, G, rG), kb = muldiv_g(kx * RB_B, c, G, rG);\n        xa0 = (S.njS - kx) * RB_A + ka; nxa = muldiv_g(kx * RB_A, c + 1, G, rG) - ka; ra0 = S.wsr[0] - ka;\n        xb0 = (S.njS - kx) * RB_B + kb; nxb = muldiv_g(kx * RB_B, c + 1, G, rG) - kb; rb0 = S.wsr[2] - kb;\n      }\n      S.hof[0] = xa0; S.hof[1] = nxa; S.hof[2] = ra0; S.hof[3] = S.wsr[1] - S.wsr[0];\n      S.hof[4] = xb0; S.hof[5] = nxb; S.hof[6] = rb0; S.hof[7] = S.wsr[3] - S.wsr[2];\n    }\n#endif\n  }\n  WS_JOIN();\n#endif\n#if OKC_WGS\n#else\n#if OKC_HOFW\n  skinny_ring_uni2(p, S, xbuf);\n#elif M2T_ON\n  skinny_ring_m2u(p, S, xbuf);\n#else\n  run_phase<false, BIG>(p, S, xbuf);\n#endif\n  DBG(2);\n#if !OKC_HOFW && !M2T_ON\n  run_phase<true, BIG>(p, S, xbuf);\n#endif\n  DBG(3);\n#endif\n#if OKC_WS\n  WS_JOIN();  // the helper\'s last epilogue stores are visible\n#endif\n  // programmatic dependent launch: the next kernel may start its own prologue now (its griddepcontrol.wait still\n  // waits for this whole grid to finish and its writes to be visible)\n  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n  // grid barrier on this launch\'s counter (two, by generation: the other one was zeroed by this launch for the next);\n  // the last arriver bumps the generation, which no CTA of this launch reads again\n#if OKC_FINPF && OKC_FINCOL && !OKC_WGS\n  // the CTA\'s first final item: its routing (pos) and route weights are inputs, so they load before the barrier\n  constexpr int FSEGP = HD / (NT * 4);\n  const int itp = blockIdx.x;\n  const bool hasp = itp < p.M * FSEGP;\n  const int tkp = hasp ? itp / FSEGP : 0;\n  int qvp[8];\n  float wvp[8];\n#pragma unroll\n  for (int s = 0; s < 8; ++s) {\n    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;\n    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;\n  }\n#endif\n  CTA_SYNC();\n  if (tid == 0) {\n    int* bc = p.ctr + 4 + (S.gen & 1);\n    __threadfence();\n    const int old = atomicAdd(bc, 1);\n    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);\n    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);\n  }\n  CTA_SYNC();\n  DBG(4);\n  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA\'s share)\n  {\n    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;\n    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters\n    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;\n  }\n  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]\n#if OKC_WGS && OKC_WGSFINSS\n#elif OKC_FINCOL\n  // column-parallel: item = (token, 1024-column segment); thread = 4 columns\n  constexpr int FSEG = HD / (NT * 4);  // column segments per token (thread = 4 columns)\n  int it0 = blockIdx.x;\n#if OKC_FINPF\n  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math\n    const int col = (itp % FSEG) * (NT * 4) + tid * 4;\n    float ssv[8];\n    float4 yv[8];\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      ssv[s] = qvp[s] >= 0 ? __ldcg(ss_cur() + qvp[s]) : 0.f;\n      yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col)) : make_float4(0.f, 0.f, 0.f, 0.f);\n    }\n    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      if (qvp[s] >= 0) {\n        const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);\n        acc.x += r * yv[s].x; acc.y += r * yv[s].y; acc.z += r * yv[s].z; acc.w += r * yv[s].w;\n      }\n    }\n    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);\n    uint2 o;\n    o.x = *reinterpret_cast<uint32_t*>(&o0);\n    o.y = *reinterpret_cast<uint32_t*>(&o1);\n    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;\n  }\n  it0 += gridDim.x;\n#endif\n  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {\n    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;\n    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      const int q = S.pos[tk * 8 + s];\n      if (q >= 0) {\n        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);\n        const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));\n        acc.x += r * v.x; acc.y += r * v.y; acc.z += r * v.z; acc.w += r * v.w;\n      }\n    }\n    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);\n    uint2 o;\n    o.x = *reinterpret_cast<uint32_t*>(&o0);\n    o.y = *reinterpret_cast<uint32_t*>(&o1);\n    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;\n  }\n#endif\n  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered\n}\n\n}  // namespace okc\n\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n\nstatic size_t okc_smem_bytes() {\n  size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;\n  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);\n  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;\n  return base + (xa > xb ? xa : xb);\n}\n\nstatic int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)\n\nvoid okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }\n\nvoid okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {\n  okc::Params p;\n  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();\n  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();\n  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();\n  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();\n  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();\n  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }\n  else { p.wAn = p.wAf; p.wBn = p.wBf; }  // no natural copies: the wgmma path must stay off (OKC_WGMIN 9)\n#if OKC_WGS\n#else\n  p.wAs = nullptr; p.wBs = nullptr;\n#endif\n  p.zero = 0;\n  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr();\n  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;\n  int* ib = (int*)scratch[0].data_ptr();\n  p.ctr = ib; p.doneA = ib + 8;\n  p.ss = (float*)scratch[1].data_ptr();\n  TORCH_CHECK(scratch[0].numel() >= 8 + 1024 + okc::EMAX, "scratch[0] needs the second ss copy");\n  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);\n  p.ybuf = p.ss + okc::EMAX;\n  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();\n#if OKC_WGS\n#else\n  p.xA = p.xB = nullptr;\n  p.xready = nullptr;\n  p.doneAs = nullptr;\n#endif\n  p.out = (__nv_bfloat16*)out.data_ptr();\n  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;\n  static bool attr = false;\n  size_t sm = okc_smem_bytes();\n  if (!attr) {\n    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);\n    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);\n    (void)e0; (void)e1;\n    attr = true;\n  }\n  cudaStream_t st = at::cuda::getCurrentCUDAStream();\n  // large batches run the variant with the fat path inlined (and wgmma); small ones keep it out of line, which\n  // leaves the skinny loop\'s code and registers as tuned\n  if (g_okc_pdl) {\n    cudaLaunchConfig_t cfg = {};\n    cfg.gridDim = dim3((unsigned)grid);\n    cfg.blockDim = dim3(okc::NTL);\n    cfg.dynamicSmemBytes = sm;\n    cfg.stream = st;\n    cudaLaunchAttribute at[1];\n    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n    at[0].val.programmaticStreamSerializationAllowed = 1;\n    cfg.attrs = at;\n    cfg.numAttrs = 1;\n    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);\n    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);\n    return;\n  }\n  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);\n  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);\n}\n\n\n'
MID_DEFINES += ("@helper0",)
# K11: TINY and TINY2 also run on the helper0 twin (decode K11: handoff var2 lean A/B loop, G1 paired scales)
TINY_DEFINES += ("@helper0",)
TINY2_DEFINES += ("@helper0",)


def _r3m_rp(s, old, new, n=1):
    """exact-count text replacement (asserted)"""
    assert s.count(old) == n, (old[:80], s.count(old))
    return s.replace(old, new)


# ---- Change set R3M-SC6 (adamleex, 2026-11-02): the MID build (9..32 tokens) streams 6-bit block scales.
# The generated weights' block scales are log-normal (log2 ~ N(7, 0.4), clamped to [8, 448]), so every local scale byte is
# a positive normal e4m3 value of at least 2.0 (bits 7:6 == 01) and v = byte - 0x40 is an exact 6-bit code. The MID
# build's skinny units then carry 3 KB of scales instead of 4 KB (35,840 B instead of 36,864 B per unit: -2.8% of its
# weight bytes). The kernel decode is bit-identical to the PRMT decode of the 8-bit byte ((v << 4) + 0x6000 ==
# (byte << 4) + 0x5C00 per bf16 half), so outputs do not change. prepare() checks every local scale byte (0x40..0x7E)
# and keeps K11's 8-bit MID build (MID_DEFINES) when any byte is outside: a pure bit repack of the given bytes, gated.
# Packing of 8 six-bit codes into 6 bytes (6 codes in the high bits, two codes spread over the 2-bit low pieces): a hand
# port of the SC6 idea of team keell0renz's public muse_moe_w4a16 file (Apache-2.0); written here from scratch.
def _r3m_sc6_exact(*sfs):
    """True if every scale byte is in 0x40..0x7E, so byte - 0x40 is an exact 6-bit code (R3M-SC6)."""
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x40) | (b > 0x7E)).any()):
            return False
    return True


def _r3m_unit_order6(codes, scales):
    """R3M-SC6 unit copy: per 16-row block its 32 KB of codes, then 3 KB of 6-bit scales = plane W [thread (256)]
    [group (2)][4 B] | plane H [thread][group][2 B]. Thread t's 16 scale bytes s_0..s_15 (the [c][h] order of
    _pack_scales) become v = s - 0x40; group g holds v_8g+n (n = 0..7): W byte n (n < 4) = v_n << 2 | piece_n,
    H byte n (n < 2) = v_4+n << 2 | piece_4+n, with pieces 0..2 / 3..5 the 2-bit pieces (low first) of v_6 / v_7."""
    nblk = codes.numel() // (8 * 8 * 32 * 4)
    c = codes.contiguous().view(torch.uint8).view(nblk, 32768)
    v = scales.contiguous().view(nblk, 256, 2, 8).to(torch.int32) - 0x40
    assert int(v.min()) >= 0 and int(v.max()) <= 63
    v6, v7 = v[..., 6], v[..., 7]
    pc = (v6 & 3, (v6 >> 2) & 3, (v6 >> 4) & 3, v7 & 3, (v7 >> 2) & 3, (v7 >> 4) & 3)
    wb = torch.stack([(v[..., n] << 2) | pc[n] for n in range(4)], -1).to(torch.uint8)       # [blk, t, g, 4]
    hb = torch.stack([(v[..., 4 + n] << 2) | pc[4 + n] for n in range(2)], -1).to(torch.uint8)  # [blk, t, g, 2]
    return torch.cat([c, wb.reshape(nblk, 2048), hb.reshape(nblk, 1024)], dim=1).contiguous().view(-1)


_R3M_SC6_DEC = r"""
// R3M-SC6: a group of 8 six-bit codes v_n = s_n - 0x40 is word W (byte n < 4: v_n << 2 | piece_n) and half H (byte n:
// v_4+n << 2 | piece_4+n), pieces 0..2 / 3..5 the 2-bit pieces of v_6 / v_7. Pair k -> bf16x2(s_2k * 2^64,
// s_2k+1 * 2^64), bit-identical to SCP on the 8-bit bytes: (v << 4) + 0x6000 == (s << 4) + 0x5C00; k4 = 4 kept opaque
// (the multiply stays an IMAD). The two multiplies gather each spread code's three pieces into byte 3.
__device__ __forceinline__ void r3m_sc6_dec(uint32_t W, uint32_t H, uint32_t k4, uint32_t* s) {
  const uint32_t X = W & 0xFCFCFCFCu, Hm = H & 0xFCFCu;
  const uint32_t p6 = (W & 0x00030303u) * 0x04104000u;                  // byte 3 = v_6 << 2 (pieces at bits 0, 8, 16)
  const uint32_t p7 = ((H & 0x0303u) | (W & 0x03000000u)) * 0x10400004u;  // byte 3 = v_7 << 2 (pieces at bits 24, 0, 8)
  s[0] = __byte_perm(X, 0u, 0x4140u) * k4 + 0x60006000u;
  s[1] = __byte_perm(X, 0u, 0x4342u) * k4 + 0x60006000u;
  s[2] = __byte_perm(Hm, 0u, 0x4140u) * k4 + 0x60006000u;
  s[3] = __byte_perm(p6, p7, 0x0703u) * k4 + 0x60006000u;
}
"""


def _r3m_sc6_src(s):
    rp = _r3m_rp
    s = rp(s, "#ifndef OKC_PRMTSC\n#define OKC_PRMTSC 0\n#endif\n",
           "#ifndef OKC_PRMTSC\n#define OKC_PRMTSC 0\n#endif\n#ifndef OKC_SC6\n"
           "#define OKC_SC6 0  // R3M-SC6: the MID (M2T) build's skinny units carry 6-bit block scales (3 KB per unit)\n#endif\n")
    s = rp(s, "constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;\n",
           "constexpr int UNIT_CODES = 32768, UNIT_SC = OKC_SC6 ? 3072 : 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;\n"
           "static_assert(!OKC_SC6 || (M2T_ON && OKC_PRMTSC && OKC_TINY), \"R3M-SC6: MID (M2T) build on the PRMT decode only\");\n"
           + _R3M_SC6_DEC)
    a = s.index("template <bool TWO>\n__device__ __forceinline__ void m2u_unit(")
    b = s.index("__device__ __forceinline__ void skinny_ring_m2u(")
    f = s[a:b]
    f = rp(f, "  const unsigned char* slot = ring + cu.sl * UNIT_BYTES;\n",
           "  const unsigned char* slot = ring + cu.sl * UNIT_BYTES;\n"
           "#if OKC_SC6\n"
           "  uint32_t sg6[8];  // R3M-SC6: this thread's 8 scale pairs of the unit, decoded once\n"
           "  {\n"
           "    const int t8 = warp * 32 + lane;\n"
           "    const uint2 w6 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + t8 * 8);\n"
           "    const uint32_t h6 = *reinterpret_cast<const uint32_t*>(slot + UNIT_CODES + 2048 + t8 * 4);\n"
           "    const uint32_t k4 = 4u + p.zero;\n"
           "    r3m_sc6_dec(w6.x, h6, k4, sg6);\n"
           "    r3m_sc6_dec(w6.y, h6 >> 16, k4, sg6 + 4);\n"
           "  }\n"
           "#endif\n")
    f = rp(f, "    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n",
           "#if !OKC_SC6\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n#endif\n", n=2)
    f = rp(f, "#if OKC_PRMTSC\n      const uint32_t swd",
           "#if OKC_SC6\n      const uint32_t sg = sg6[c], sg8 = sg;\n#elif OKC_PRMTSC\n      const uint32_t swd", n=2)
    s = s[:a] + f + s[b:]
    s = rp(s, "  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();\n",
           "#if OKC_SC6\n"
           "  TORCH_CHECK(w.size() >= 16, \"R3M-SC6 build: needs the 6-bit unit copies w[14], w[15]\");\n"
           "  p.wAu = (const unsigned char*)w[14].data_ptr(); p.wBu = (const unsigned char*)w[15].data_ptr();\n"
           "#else\n"
           "  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();\n"
           "#endif\n")
    return s


ALT_SRCS["helper0s6"] = _r3m_sc6_src(ALT_SRCS["helper0"])
# MID build on 6-bit scales (used when prepare()'s R3M-SC6 gate passes); MID_DEFINES stays K11's 8-bit build (fallback)
MID_S6_DEFINES = tuple("@helper0s6" if d == "@helper0" else d for d in MID_DEFINES) + ("-DOKC_SC6=1",)
# R3M-SC6W: 6-bit units are 2.8% shorter but 2-tile units stay compute-bound: their split cost goes up (31 / 33 vs K11 30 / 32)
MID_S6_DEFINES = MID_S6_DEFINES + ("-DOKC_M2W2=31", "-DOKC_M2W2B=33")



# Per-thread primary-context bind (hunk copied verbatim from fc85fa2f): the outer kernel_fn binds the device primary
# context in the calling thread first, then runs the unchanged kernel_fn.
import ctypes as _C
import threading as _ContextThreading

_CONTEXT_TLS = _ContextThreading.local()
_CONTEXT_API = None


def _context_api():
    global _CONTEXT_API
    if _CONTEXT_API is None:
        api = _C.CDLL("libcuda.so.1")
        for name, args in (
            ("cuCtxGetCurrent", [_C.POINTER(_C.c_void_p)]),
            ("cuCtxGetDevice", [_C.POINTER(_C.c_int)]),
            ("cuDeviceGet", [_C.POINTER(_C.c_int), _C.c_int]),
            ("cuDevicePrimaryCtxRetain", [_C.POINTER(_C.c_void_p), _C.c_int]),
            ("cuCtxSetCurrent", [_C.c_void_p]),
        ):
            function = getattr(api, name)
            function.argtypes, function.restype = args, _C.c_int
        _CONTEXT_API = api
    return _CONTEXT_API


def _context_check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with CUDA driver status {status}")


def _bind_context(x):
    index = x.device.index if x.device.index is not None else 0
    if getattr(_CONTEXT_TLS, "device", None) == index:
        return
    api = _context_api()
    context = _C.c_void_p()
    _context_check(api.cuCtxGetCurrent(_C.byref(context)), "cuCtxGetCurrent")
    current_device = _C.c_int(-1)
    if context.value:
        _context_check(api.cuCtxGetDevice(_C.byref(current_device)), "cuCtxGetDevice")
    if not context.value or current_device.value != index:
        device, primary = _C.c_int(), _C.c_void_p()
        _context_check(api.cuDeviceGet(_C.byref(device), index), "cuDeviceGet")
        _context_check(api.cuDevicePrimaryCtxRetain(_C.byref(primary), device), "cuDevicePrimaryCtxRetain")
        _context_check(api.cuCtxSetCurrent(primary), "cuCtxSetCurrent")
    _CONTEXT_TLS.device = index


_context_parent_kernel = kernel_fn


def kernel_fn(data, state):
    _bind_context(data["hidden_states"])
    return _context_parent_kernel(data, state)


# ---------------------------------------------------------------- change set R4X (adamleex, 2026-11-02): N2 + PS + MS
# The OKC_L2H branch of every MoE source stored the inter / xB / ybuf scratch with an evict_last L2 cache hint
# (st_keep_bf16 / st_keep_f32, policy 0x14F0000000000000). That scratch is dead once the call's final pass has read it,
# and evict_last lines can stay resident through a cold-L2 read of a larger buffer, i.e. into the next timed replay.
# The builds converted below compile a derived copy of their '@' source with the hint removed ("asm": the same
# st.global instructions without a policy; "plain": C stores, the OKC_L2H = 0 body); every other build keeps its source.
_R4X_ID = "r4xc"
_R4X_PFX = {}  # source text -> extension-name prefix: only the builds whose code R4X changes get a new name


def _r4x_rp(s, old, new, n=1):
    """exact-count text replacement (asserted)"""
    assert s.count(old) == n, ("R4X", old[:80], s.count(old), n)
    return s.replace(old, new)


def _r4x_stores(src, form):
    """src with st_keep_bf16 / st_keep_f32 stored without the evict_last hint (form "asm" or "plain")"""
    pol = '"l"(0x14F0000000000000ull) : "memory");\n'
    old_bf = ('  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\\n" ::"l"(a), '
              '"h"(*reinterpret_cast<unsigned short*>(&v)), ' + pol)
    old_f = '  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\\n" ::"l"(a), "f"(v), ' + pol
    if form == "asm":
        new_bf = '  asm volatile("st.global.b16 [%0], %1;\\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)) : "memory");\n'
        new_f = '  asm volatile("st.global.b32 [%0], %1;\\n" ::"l"(a), "f"(v) : "memory");\n'
    else:
        new_bf = new_f = "  *a = v;  // R4X: plain store, no L2 hint (was evict_last)\n"
    src = _r4x_rp(_r4x_rp(src, old_bf, new_bf), old_f, new_f)
    assert "0x14F0000000000000" not in src, "R4X: an evict_last policy is left in the source"
    return src


def _r4x_band(defs, form, tag):
    """defs with its '@' source swapped for the derived copy ALT_SRCS[<key>_<_R4X_ID><tag>] (made once, renamed)"""
    at = [d for d in defs if d.startswith("@")]
    assert len(at) == 1, ("R4X: expected one '@' source", defs)
    key = at[0][1:] + "_" + _R4X_ID + tag
    if key not in ALT_SRCS:
        ALT_SRCS[key] = _r4x_stores(ALT_SRCS[at[0][1:]], form)
        _R4X_PFX[ALT_SRCS[key]] = "okc_moe_" + _R4X_ID + tag + "_"
    return tuple("@" + key if d == at[0] else d for d in defs)


_R4X_PFX[CUDA_SRC] = "okc_moe_" + _R4X_ID + "np_"  # CUDA_SRC builds (MAIN / fallback / SC6W): edited in place above
MID_DEFINES = _r4x_band(MID_DEFINES, "asm", "m")  # MID: asm stores, no hint
MID_S6_DEFINES = _r4x_band(MID_S6_DEFINES, "asm", "m")  # MID: asm stores, no hint
TINY2_DEFINES = _r4x_band(TINY2_DEFINES, "plain", "p")  # TINY2: plain stores


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


# ---------------------------------------------------------------- r6m tail (round 6 Track M, 2026-11-02): MID-S6 build only
_R6M_TAG, _R6M_HD, _R6M_W = ('hd', True, None)
def _r6c_rp(s, old, new, n=1):
    assert s.count(old) == n, ("r6c", old[:80], s.count(old), n)
    return s.replace(old, new)


_R6C_REFILL = r"""
      {
        if (r6left == 0) {  // r6 cons: a new contiguous run of one job (and phase)
          const bool rP = r6q >= nA;
          const int rv = rP ? b0 + (r6q - nA) : a0 + r6q, rlg = rP ? RB_B_LOG : RB_A_LOG, rvj = rv >> rlg;
          r6left = min((rP ? N : nA) - r6q, ((rvj + 1) << rlg) - rv);
          r6src = (rP ? p.wBu : p.wAu) + ((size_t)S.jE[njF + rvj] * (1 << rlg) + (rv - (rvj << rlg))) * (size_t)UNIT_BYTES;
        }
        const int r6sl = r6q % OKC_NS;
        uint64_t* r6bar = S.bars + r6sl;
        unsigned char* r6slot = ring + r6sl * UNIT_BYTES;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(r6bar)), "r"(UNIT_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                     ::"r"(smem_u32(r6slot)), "l"(r6src), "r"(UNIT_BYTES), "r"(smem_u32(r6bar)) L2H_ARG : "memory");
        r6src += UNIT_BYTES; --r6left; ++r6q;
      }
"""


def _r6c_mid(s):
    rp = _r6c_rp
    s = rp(s, "#define OKC_HOFW (OKC_WS && !OKC_SPLIT && FAST1_ON)  // r8: poller warp (TINY / TINY2, with OKC_HOF)\n#endif\n",
           "#define OKC_HOFW (OKC_WS && !OKC_SPLIT && FAST1_ON)  // r8: poller warp (TINY / TINY2, with OKC_HOF)\n#endif\n"
           "#ifndef OKC_R6HR\n#define OKC_R6HR 0  // r6 cons: MID ring refills by 1 the helper warp, 2 an issuer warp\n#endif\n"
           "#ifndef OKC_R6ER\n#define OKC_R6ER 0  // r6 cons: MID early slot release (codes to registers, then S.wse)\n#endif\n")
    s = rp(s, "#define OKC_WSRF 0\n",
           "#define OKC_WSRF 0\n"
           "static_assert(!OKC_R6ER || OKC_R6HR, \"r6 cons: early release needs an issuer\");\n"
           "static_assert(!OKC_R6HR || (M2T_ON && !OKC_HOFW), \"r6 cons: MID (M2T) build only\");\n")
    s = rp(s, "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? (OKC_HOFW ? 64 : 32) : 0);",
           "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? ((OKC_HOFW || OKC_R6HR == 2) ? 64 : 32) : 0);")
    a = s.index("template <bool TWO>\n__device__ __forceinline__ void m2u_unit(")
    b = s.index("__device__ __forceinline__ void skinny_ring_m2u(")
    f = s[a:b]
    f = rp(f, "  float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n  if (TWO) {\n",
           "  float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n"
           "#if OKC_R6ER\n"
           "  uint4 r6w[NCH];  // r6 cons ER: this warp's 8 code words of the unit, loaded at once\n"
           "#pragma unroll\n"
           "  for (int c = 0; c < NCH; ++c) r6w[c] = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n"
           "  __syncwarp();\n"
           "  if (lane == 0) mbar_arrive1(S.wse + cu.sl);  // this warp is done reading the slot (release.cta orders the loads)\n"
           "#endif\n"
           "  if (TWO) {\n")
    f = rp(f, "      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n",
           "#if OKC_R6ER\n      const uint4 w = r6w[c];\n#else\n"
           "      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#endif\n", n=2)
    f = rp(f, "  CTA_SYNC();\n  if (tid == 0) {\n    if (cu.rq < N) {", "  CTA_SYNC();\n  if (tid == 0) {\n#if !OKC_R6HR\n    if (cu.rq < N) {")
    f = rp(f, "      cu.rsrc += UNIT_BYTES; --cu.rleft; ++cu.rq;\n    }\n", "      cu.rsrc += UNIT_BYTES; --cu.rleft; ++cu.rq;\n    }\n#endif\n")
    s = s[:a] + f + s[b:]
    # helper warp (OKC_R6HR == 1)
    a = s.index("__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {")
    b = s.index("#if OKC_HOFW\n// r8 poller warp")
    h = s[a:b]
    h = rp(h, "  if (lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n",
           "  if (lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n"
           "#if OKC_R6HR == 1\n"
           "  int r6q = OKC_NS, r6left = 0;  // r6 cons: refill cursor (lane 0): next unit to issue, units left in its run\n"
           "  const unsigned char* r6src = nullptr;\n"
           "#if OKC_R6ER\n"
           "  if (lane == 0 && OKC_NS < N) {  // unit NS into unit 0's slot once every compute warp loaded unit 0\n"
           "    mbar_wait(S.wse + 0, 0u);\n" + _R6C_REFILL +
           "  }\n"
           "#endif\n"
           "#endif\n")
    h = rp(h, "    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));\n",
           "#if OKC_R6HR == 1 && OKC_R6ER\n"
           "    if (lane == 0 && q == nA && q > 0 && q + OKC_NS < N) {  // deferred at the A -> B boundary (see below)\n"
           "      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));\n" + _R6C_REFILL +
           "    }\n"
           "#endif\n"
           "    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));\n"
           "#if OKC_R6HR == 1 && !OKC_R6ER\n"
           "    if (lane == 0 && q + OKC_NS < N)  // unit q passed the per-unit CTA barrier: its slot is free\n" + _R6C_REFILL +
           "#endif\n")
    h = rp(h, "    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again\n",
           "    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again\n"
           "#if OKC_R6HR == 1 && OKC_R6ER\n"
           "    // unit q + 1 + NS into unit q + 1's slot once every compute warp loaded unit q + 1; not across the A -> B\n"
           "    // boundary (unit nA waits for doneA, which may need this CTA's phase-A credit below)\n"
           "    if (lane == 0 && q + 1 + OKC_NS < N && q + 1 != nA) {\n"
           "      mbar_wait(S.wse + (q + 1) % OKC_NS, (uint32_t)(((q + 1) / OKC_NS) & 1));\n" + _R6C_REFILL +
           "    }\n"
           "#endif\n")
    s = s[:a] + h + s[b:]
    # issuer warp (OKC_R6HR == 2)
    s = rp(s, "#if OKC_HOFW\n// r8 poller warp",
           "#if OKC_R6HR == 2\n"
           "// r6 cons: issuer warp (threads NT+32..NT+63): lane 0 refills ring slot q % NS with unit q + NS as soon as unit q\n"
           "// released it (OKC_R6ER: every compute warp loaded its codes, S.wse; else: unit q passed the per-unit barrier)\n"
           "__device__ void r6_issuer(const Params& p, Smem& S, unsigned char* ring) {\n"
           "  const int lane = TIDX & 31;\n"
           "  WS_JOIN();  // routing tables and this CTA's ranges are ready\n"
           "  const int a0 = S.wsr[0], nA = S.wsr[1] - a0, b0 = S.wsr[2], N = nA + (S.wsr[3] - b0), njF = S.njF;\n"
           "  if (lane == 0) {\n"
           "    int r6q = OKC_NS, r6left = 0;\n"
           "    const unsigned char* r6src = nullptr;\n"
           "    for (int q = 0; q + OKC_NS < N; ++q) {\n"
           "#if OKC_R6ER\n"
           "      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));\n"
           "#else\n"
           "      mbar_wait(S.wsrf + q % NRB, (uint32_t)((q / NRB) & 1));\n"
           "#endif\n" + _R6C_REFILL +
           "    }\n"
           "  }\n"
           "  __syncwarp();\n"
           "  WS_JOIN();  // the end-of-kernel join\n"
           "}\n"
           "#endif\n"
           "#if OKC_HOFW\n// r8 poller warp")
    s = rp(s, "#if OKC_HOFW\n  if (tid >= NT + 32) { ws_poller(p, S); return; }\n#endif\n",
           "#if OKC_HOFW\n  if (tid >= NT + 32) { ws_poller(p, S); return; }\n#endif\n"
           "#if OKC_R6HR == 2\n  if (tid >= NT + 32) { r6_issuer(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n#endif\n")
    return s

_r6mk = [d[1:] for d in MID_S6_DEFINES if d.startswith("@")]
assert len(_r6mk) == 1 and _r6mk[0] == "helper0s6_r4xcm", MID_S6_DEFINES
_r6mk = _r6mk[0]
_r6m_src = (_r6c_mid(ALT_SRCS[_r6mk]) if _R6M_HD else ALT_SRCS[_r6mk]) + "\n// r6m " + _R6M_TAG + "\n"
ALT_SRCS[_r6mk + "_r6m"] = _r6m_src
_R4X_PFX[_r6m_src] = "okc_moe_r6m" + _R6M_TAG + "m_"
_r6md = tuple("@" + _r6mk + "_r6m" if d == "@" + _r6mk else d for d in MID_S6_DEFINES)
if _R6M_W is not None:  # MID-S6 2-tile split weights (phase A, phase B); OKC_M2W1 = 20 unchanged
    assert _r6md.count("-DOKC_M2W2=31") == 1 and _r6md.count("-DOKC_M2W2B=33") == 1, _r6md
    _r6md = tuple("-DOKC_M2W2=%d" % _R6M_W[0] if d == "-DOKC_M2W2=31" else ("-DOKC_M2W2B=%d" % _R6M_W[1] if d == "-DOKC_M2W2B=33" else d)
                  for d in _r6md)
if _R6M_HD:  # r6 cons hd: the issuer warp (OKC_R6ER stays 0: late slot release)
    _r6md = _r6md + ("-DOKC_R6HR=2",)
MID_S6_DEFINES = _r6md

# ---------------------------------------------------------------- r7 (round 7 stack, adamleex 2026-11-02, tools/r7_mk.py):
# levers W3 + F3A below = the f3a tail of tools/r6w6b_mk.py verbatim except its MAIN defines (+ -DOKC_R6WB=1: lever W3, whose
# macro text is part of this edit set). Its banner's "everything above is r4xc" reads here: r4xc + the r6m tail above.
# =====================================================================================================================
# r6w TAIL (moe_probe_20261102, round 6 track W, 2026-11-02; generated by tools/r6w6b_mk.py (FXW v3: hybrid by unit size)). Everything above this banner
# is r4xc (md5 ca721f5c) unchanged. Exact-count asserted text edits of the WGS source (CUDA_SRC: the MAIN and SC6W builds);
# each lever is behind its own macro, off unless the build's defines set it. The MID / TINY / TINY2 builds are untouched.
_R6W_TAG, _R6W_EDITS, _R6W_MAIN, _R6W_SC6 = ('f3a', ('hdr', 'pr4', 'gr', 'fxw', 'fx3o', 'wb'), ('-DOKC_FXW=1', '-DOKC_FXWN=1', '-DOKC_R6WB=1'), ('-DOKC_FXW=1', '-DOKC_FXWN=1'))


def _r6w_rp(s, old, new, n=1):
    c = s.count(old)
    assert c == n, ("r6w", c, n, old[:120])
    return s.replace(old, new)


_R6W_HDR = r"""
// ---- r6w (round 6 track W, 2026-11-02): WGS levers, each off unless defined
#ifndef OKC_GR
#define OKC_GR 0  // rolled x gather (cp.async rows staged in the free rings, KPERM3 from smem) + one fence for the xready reds
#endif
#ifndef OKC_FXW
#define OKC_FXW 0  // the phase-A split-unit fixup on a 19th (helper) warp; the consumer warpgroups only count their stages
#endif
#ifndef OKC_FXWN
#define OKC_FXWN 2  // FXW v3: only units of <= OKC_FXWN n-tiles go to the helper
#endif
#ifndef R6W_FX3O
#define R6W_FX3O 0  // TIMING-ONLY oracle (never shipped): the phase-A finisher share only releases doneAs (wrong results)
#endif
#ifndef OKC_R6WB
#define OKC_R6WB 0  // MAIN build only (bs > 160): re-fit phase-B stage-cost table (SC6W keeps r4xc's)
#endif
#if (OKC_GR || OKC_FXW || R6W_FX3O) && (!OKC_WGS || OKC_B4FIN)
#error "r6w levers: WGS main / SC6W builds only"
#endif
#if OKC_FXW && (R6W_FX3O || !OKC_R11WRM)
#error "r6w FXW: written for the R11WRM grid barrier, not with the FX3O oracle"
#endif
"""

_R6W_GR = r"""#if OKC_GR
// r6w GR (round 6 track W; finding of the r6 transitions designer): the gather above ran once per launch from a cold
// instruction cache, and its unrolled GB = 8 batch is one ~0x330 B code copy per row (~0.6-0.8 us per real row at bs128 /
// bs256), plus one red.release (MEMBAR.GPU) per job. Here: the lanes of every warp find the rows of a GR_W-row window in
// parallel (binary search over jrow); loop 1 copies each real row's 8 KB into the weight / x rings (free until WS_JOIN:
// the producers wait there), one 16 B cp.async per thread (warp w copies stage w's 512 B, so it reads back only its own
// copies: wait_all + __syncwarp); loop 2 picks this k8 group's KPERM3 words from the warp's slice (4 LDS.32) and stores
// the same 16 B into xA as before. Both loops are rolled (one copy of the code). Then one fence.acq_rel.gpu by thread 0
// and relaxed reds of the per-job row counts (a release pattern per red; it pairs with the x producer's ld.acquire).
#ifndef R6W_GRNI
#define R6W_GRNI 0  // (G0 probe) the GR gather out of line
#endif
#ifndef R6W_GRPF
#define R6W_GRPF 1  // fence.proxy.async after the staging (generic accesses of the rings before their TMA writes)
#endif
constexpr int GR_W = 24;  // padded rows per window: 24 x 8 KB staged fit the rings (3 x 36 KB + 3 x 32 KB)
static_assert(GR_W * HD * 2 <= WGS_NS * UNIT_BYTES + WGS_NX * WGS_XS && GR_W <= 32, "r6w GR: staging fits the rings");
#if R6W_GRNI
__device__ __noinline__ void wgs_gather(const Params& p, Smem& S) {
#else
__device__ void wgs_gather(const Params& p, Smem& S) {
#endif
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nj = S.njF + S.njS;
  const int R = S.jrow[nj];  // R * G < 2^24: exact float muldiv, no divide routine
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");
  unsigned char* const stg = smem_raw + ((sizeof(Smem) + 127) / 128) * 128;  // xbuf: the weight ring, then the x ring
  // KPERM3: k8 group G = lane of stage warp holds the bf16 pairs at stage words B(G) + 32 c, B(G) = 8 ((G >> 1) & 3) +
  // 2 (G >> 3) + (G & 1)
  const int bw = 8 * ((lane >> 1) & 3) + 2 * (lane >> 3) + (lane & 1);
  int jq0 = 0;  // job of row q0 (thread 0: where the release loop starts)
#pragma unroll 1
  for (int w0 = q0; w0 < q1; w0 += GR_W) {
    const int q = w0 + lane;
    int j = 0, tok = 0, dst = 0;
    bool real = false;
    if (lane < GR_W && q < q1) {
      int lo = 0, hi = nj;  // last j with jrow[j] <= q (jrow is strictly increasing: every job has >= 8 padded rows)
      while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (S.jrow[mid] <= q) lo = mid; else hi = mid; }
      j = lo;
      const int jr = S.jrow[j], r = q - jr, cnt = S.jC[j];
      if (r < cnt) {  // a real row (padding rows are counted, not written)
        real = true;
        tok = S.ent[S.jS[j] + r] >> 3;
        dst = jr * (HD / 8) + (warp * ((cnt + 7) >> 3) + (r >> 3)) * 256 + (r & 7);  // xA uint4 index, + lane * 8
      }
    }
    if (w0 == q0) jq0 = j;  // (thread 0 is lane 0: its j is row q0's job)
    const unsigned m = __ballot_sync(0xffffffffu, real);
    int k = 0;
#pragma unroll 1
    for (unsigned b = m; b; b &= b - 1u, ++k) {  // loop 1: the real rows' 8 KB, 16 B per thread
      const int t = __shfl_sync(0xffffffffu, tok, __ffs(b) - 1);
      cp_async16(stg + k * (HD * 2) + tid * 16, p.x + (size_t)t * HD + tid * 8, true);
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncwarp();
    k = 0;
#pragma unroll 1
    for (unsigned b = m; b; b &= b - 1u, ++k) {  // loop 2: KPERM3 from the warp's own 512 B slice, one 16 B store
      const int dd = __shfl_sync(0xffffffffu, dst, __ffs(b) - 1);
      const unsigned* s32 = reinterpret_cast<const unsigned*>(stg + k * (HD * 2) + warp * 512) + bw;
      reinterpret_cast<uint4*>(p.xA)[dd + lane * 8] = make_uint4(s32[0], s32[32], s32[64], s32[96]);
    }
    __syncwarp();  // (the next window's copies reuse the staging)
  }
#if R6W_GRPF
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // the rings' generic accesses before their TMA writes
#endif
  CTA_SYNC();
  if (tid == 0 && q0 < q1) {  // count this CTA's rows per job: one fence (cumulative over the CTA's stores via the barrier)
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
    int jj = jq0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
      asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
}
#else
"""

_R6W_FXH = r"""#if OKC_FXW
// r6w FXW (round 6 track W; lever of the r6 transitions designer): the phase-A split unit whose stage 15 lies in this
// CTA's range (its leading partial unit, started by an earlier CTA) is finished by this 19th warp instead of by the 4
// consumer warpgroups of the share that counted last (atomic round trip, read-back, zeroing, SwiGLU, release: a multi-us
// stall of all 4 warpgroups once per CTA). Every share, this CTA's too, only counts its stages (wgs_flush, fire-and-
// forget release); this warp waits until the unit's 4 warpgroup counters reach 16, then runs the same read-back / zero /
// epilogue per (warpgroup, warp) lane mapping, resets the counters, releases doneAs (+4: the 4 warpgroups' shares), and
// arrives on R6W_FXB, which the consumers wait for before the grid barrier (doneAs is zeroed after it).
__device__ void wgs_fxhelper(const Params& p, Smem& S) {
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  const int lane = threadIdx.x & 31;
  const int g0 = S.wgr[0], g1 = S.wgr[1], u = g0 >> 4;
  if ((g0 & 15) != 0 && ((u + 1) << 4) <= g1) {
    const int slot = wgs_cta_of(S, 0, u << 4, gridDim.x);  // (the consumers' slot: the CTA holding the unit's first stage)
    const int j = u >> 5, ub = u & 31, ntt = (S.jC[j] + 7) >> 3;
    int* fc = wgs_cnt(p, 0) + slot * 4;
    if (ntt <= OKC_FXWN) {  // r6w FXW v3: larger units stay with the consumers (last arriver)
    if (lane < 4) { while (ld_acquire(fc + lane) < 16) __nanosleep(64); }
    __syncwarp();
    // v2: the unit's constants once (no reload after each store); tiles (n-tile nt, virtual consumer warp vw = warpgroup
    // vw >> 2, warp vw & 3) flattened n-tile-major, read back 8 at a time; per value the expression of wgs_epi_tile<false>
    const int cnt = S.jC[j], g = lane >> 2, t2 = 2 * (lane & 3);
    const float gsc = p.gsA[S.jE[j]] * TWO62;
    __nv_bfloat16* const xbj = p.xB + (size_t)S.jrow[j] * HD;
    float* const fb0 = wgs_fix(p, 0) + (size_t)slot * 4 * WGS_FIXW + lane * 4;
    const int ntile = ntt << 4;
#pragma unroll 1
    for (int t0 = 0; t0 < ntile; t0 += 8) {
      float4 v[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int ti = t0 + k, vw = ti & 15, nt = ti >> 4;
        if (ti < ntile) v[k] = __ldcg(reinterpret_cast<const float4*>(fb0 + (size_t)(vw >> 2) * WGS_FIXW + (vw & 3) * 128 + nt * 512));
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int ti = t0 + k, vw = ti & 15, nt = ti >> 4;
        if (ti < ntile) {
          *reinterpret_cast<float4*>(fb0 + (size_t)(vw >> 2) * WGS_FIXW + (vw & 3) * 128 + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
          const int col = (ub * 16 + vw) * 8 + g;
          const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int tk = nt * 8 + t2 + h;
            if (tk < cnt) {
              const float gt = d4[h] * gsc, up = d4[2 + h] * gsc;
              const float val = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
              st_keep_bf16(xbj + (col & 7) + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
                           __float2bfloat16(probe_q(val)));
            }
          }
        }
      }
    }
    __syncwarp();
    if (lane < 4) fc[lane] = 0;
    __syncwarp();
    if (lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(4) : "memory");
    }
  }
  __syncwarp();
  if (lane == 0) mbar_arrive1(S.bars + R6W_FXB);
}
#endif
"""


def _r6w_src(s, ed):
    rp = _r6w_rp
    if "hdr" in ed:
        s = rp(s, '#if OKC_R3FP && !OKC_YBF\n#error "r3l FP: written for the YBF final pass"\n#endif\n',
               '#if OKC_R3FP && !OKC_YBF\n#error "r3l FP: written for the YBF final pass"\n#endif\n' + _R6W_HDR)
    if "pr4" in ed:  # NJ0 guard (as wgs_stage_at): no local route -> empty ranges, no stale S.jC[0] cost lookup / divide
        s = rp(s, "  if (c == G) {\n    g = nj << sh;\n  } else {",
               "  if (c == G || nj <= 0) {  // r6w NJ0 guard (wgs_stage_at's): nj = 0 -> empty ranges, no stale S.jC[0] read\n"
               "    g = nj << sh;\n  } else {")
    if "gr" in ed:
        s = rp(s, "__device__ void wgs_gather(const Params& p, Smem& S) {", _R6W_GR + "__device__ void wgs_gather(const Params& p, Smem& S) {")
        s = rp(s, "}\n// Work left from the previous segment", "}\n#endif  // OKC_GR\n// Work left from the previous segment")
    if "fxw" in ed or "fx3o" in ed:
        s = rp(s, "  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;\n  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase\n",
               "  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;\n  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase\n"
               "#if OKC_FXW || R6W_FX3O\n"
               "  if (!PB && (R6W_FX3O || ((S.jC[j] + 7) >> 3) <= OKC_FXWN)) {  // r6w FXW v3: a phase-A split share of <= OKC_FXWN n-tiles only counts (fire-and-forget); the CTA holding\n"
               "    // the unit's stage 15 finishes it on its helper warp (FX3O, timing only: that share just releases doneAs)\n"
               "    if (wt == 0) {\n"
               "      if (R6W_FX3O && pd.kind == 3)\n"
               "        asm volatile(\"red.release.gpu.global.add.s32 [%0], %1;\\n\" ::\"l\"(p.doneAs + j * 16 + (ub >> 1)), \"r\"(1) : \"memory\");\n"
               "      else\n"
               "        asm volatile(\"red.release.gpu.global.add.s32 [%0], %1;\\n\" ::\"l\"(fc), \"r\"(pd.nst) : \"memory\");\n"
               "    }\n    pd.kind = 0;\n    return;\n  }\n#endif\n")
    if "fx3o" in ed:
        s = rp(s, "    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};",
               "    pd = WgsPend{((((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) || (R6W_FX3O && !PB)) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};")
    if "fxw" in ed:
        s = rp(s, "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);",
               "constexpr int NTL = NT + (OKC_WGS ? (OKC_FXW ? 96 : 64) : OKC_WS ? 32 : 0);  // (r6w FXW: + the fixup helper warp)")
        s = rp(s, "constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices\n",
               "constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices\n"
               "constexpr int R6W_FXB = BX_E + WGS_NX;  // r6w FXW: the helper warp's phase-A fixup is done (1 arrival)\n")
        s = rp(s, "  return c;\n}\n#endif\n\ntemplate <bool PB, bool BIG>", "  return c;\n}\n" + _R6W_FXH + "#endif\n\ntemplate <bool PB, bool BIG>")
        s = rp(s, "    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n",
               "    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n"
               "#if OKC_FXW\n    else if (tid >= NT + 64) wgs_fxhelper(p, S);  // r6w FXW\n#endif\n"
               "    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n")
        s = rp(s, "    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);\n",
               "    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);\n"
               "#if OKC_FXW\n    mbar_init(S.bars + R6W_FXB, 1);  // r6w FXW: the helper warp's fixup done\n#endif\n")
        s = rp(s, '  static_assert(OKC_WGS && OKC_WGSFINSS && !OKC_B4FIN, "r11 WRM: main WGS build");\n  CTA_SYNC();\n',
               '  static_assert(OKC_WGS && OKC_WGSFINSS && !OKC_B4FIN, "r11 WRM: main WGS build");\n'
               '#if OKC_FXW\n  if (tid == 0) mbar_wait(S.bars + R6W_FXB, 0);  // r6w FXW: the helper\'s doneAs release precedes the barrier (zeroing)\n#endif\n'
               '  CTA_SYNC();\n')
    if "wb" in ed:
        s = rp(s, "#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174\n#endif\n",
               "#if OKC_R6WB && !OKC_SC6W  // r6w WB (r6 sched designer's re-fit to measured per-CTA phase-B ends): MAIN (bs > 160) only\n"
               "#define OKC_WGSB 100, 98, 106, 117, 126, 138, 150, 162\n#else\n"
               "#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174\n#endif\n#endif\n")
    return s


CUDA_SRC = _r6w_src(CUDA_SRC, _R6W_EDITS)
_R4X_PFX[CUDA_SRC] = "okc_moe_r6w6" + _R6W_TAG + "_"
MAIN_DEFINES = tuple(MAIN_DEFINES) + tuple(_R6W_MAIN)
SC6_DEFINES = tuple(SC6_DEFINES) + tuple(_R6W_SC6)

# ---------------------------------------------------------------- r9c (round 9 stack C4, adamleex 2026-11-03, tools/r9c_mk.py):
# lever GU below = the r8g3 tail of tools/r8g3_mk.py (tag u) verbatim. Its banner's "everything above this banner is r6m_hd"
# reads here: r6m_hd + the r7 W3 + F3A tail above. Its CUDA_SRC edits apply on top of F3A's (r9c_mk.py checks: every anchor
# matches once, F3A's "gr" block nests inside GU's #else); MAIN and SC6 also get -DOKC_R8GT=1.

# ---------------------------------------------------------------- r8g3 tail (round 8 track g, adamleex 2026-11-02, tools/r8g3_mk.py)
# lever GU (OKC_R8GT, default off): WGS phase-A x gather with whole-line xA writes; only wgs_gather's body changes. Everything
# above this banner is r6m_hd (md5 f764e4cd) [+ the r6a stamp tail, stamped tag only]. -DOKC_R8GT=1 goes to MAIN / SC6 only.
_R8G3_TAG, _R8G3_DEFS, _R8G3_GATHER = ('u', ('-DOKC_R8GT=1',), '#if OKC_R8GT\n// r8g GU (round 8, adamleex 2026-11-02): phase A\'s x gather by (8-row group, 256-K stage) items: item it = padded rows\n// 8 (it >> 4) .. +7 (one n8 group of a job) at stage it & 15; this CTA\'s items are [2 q0, 2 q1) (its padded rows\' share, as\n// before, so its count per job is even); one warp per item. The warp loads the 8 rows\' KPERM3 words as before (lane G: the bf16\n// pairs at stage offsets 2 B(G) + 64 t), transposes them through its own 4 KB of the still idle weight ring ([G][row ^ (G & 7)]:\n// conflict-free both ways) and writes the job\'s (stage, n-tile) 4 KB block of xA as 8 warp-wide 512-B runs (whole 128-B lines);\n// the group\'s padding rows are written as 0. xready still counts rows (items / 2). One GPU-scope fence, then relaxed reds.\n__device__ void wgs_gather(const Params& p, Smem& S) {\n  const int tid = threadIdx.x, lane = tid & 31, nj = S.njF + S.njS;\n  const int R = S.jrow[nj];  // R * G < 2^24: exact float muldiv, no divide routine\n  const float rG = __fdividef(1.f, (float)gridDim.x);\n  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);\n  uint4* tb = reinterpret_cast<uint4*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128) + (tid >> 5) * 256;  // (the ring = xbuf)\n  const int boff = 2 * (8 * ((lane >> 1) & 3) + 2 * (lane >> 3) + (lane & 1));  // bf16 offset of word B(lane) in a stage\n  int j = 0;\n  for (int it = 2 * q0 + (tid >> 5); it < 2 * q1; it += NW) {\n    const int q8 = (it >> 4) << 3, w = it & 15;\n    while (S.jrow[j + 1] <= q8) ++j;\n    const int r0 = q8 - S.jrow[j], cnt = S.jC[j], ntt = (cnt + 7) >> 3, e0 = S.jS[j] + r0;\n    uint4 v[8];\n#pragma unroll\n    for (int rr = 0; rr < 8; ++rr) {\n      v[rr] = make_uint4(0u, 0u, 0u, 0u);\n      if (r0 + rr < cnt) {\n        const unsigned* xs32 = reinterpret_cast<const unsigned*>(p.x + (size_t)(S.ent[e0 + rr] >> 3) * HD + w * 256 + boff);\n        v[rr] = make_uint4(__ldg(xs32), __ldg(xs32 + 32), __ldg(xs32 + 64), __ldg(xs32 + 96));\n      }\n    }\n#pragma unroll\n    for (int rr = 0; rr < 8; ++rr) tb[lane * 8 + (rr ^ (lane & 7))] = v[rr];\n    __syncwarp();\n    uint4* dst = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + (w * ntt + (r0 >> 3)) * 256 + lane;\n#pragma unroll\n    for (int m = 0; m < 8; ++m) {\n      const int g2 = 4 * m + (lane >> 3);\n      dst[32 * m] = tb[g2 * 8 + ((lane & 7) ^ (g2 & 7))];\n    }\n    __syncwarp();  // (the next item reuses tb)\n  }\n  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");  // the staging accesses before the ring\'s TMA writes\n  CTA_SYNC();\n  if (tid == 0) {  // count this CTA\'s rows per job (the CTA\'s stores are ordered before the fence by the barrier)\n    int jj = 0;\n    asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");\n    for (int q = q0; q < q1;) {\n      while (S.jrow[jj + 1] <= q) ++jj;\n      const int e = min(q1, S.jrow[jj + 1]);\n      asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");\n      q = e;\n    }\n  }\n}\n#else\n')


def _r8g3_rp(s, old, new, n=1):
    c = s.count(old)
    assert c == n, ("r8g3", c, n, old[:120])
    return s.replace(old, new)


def _r8g3_src(s):
    rp = _r8g3_rp
    a = s.index("// phase A's x gather (all consumer threads of every CTA, once, at the start): this CTA's share of the jobs' padded rows\n")
    b = s.index("// Work left from the previous segment", a)
    assert s.count("// Work left from the previous segment") == 1
    s = s[:a] + _R8G3_GATHER + s[a:b] + "#endif  // OKC_R8GT\n" + s[b:]
    s = rp(s, "#define OKC_GATHB 8  // WGS: x rows gathered per batch (loads first, then stores); 0 = one row at a time\n",
           "#define OKC_GATHB 8  // WGS: x rows gathered per batch (loads first, then stores); 0 = one row at a time\n"
           "#ifndef OKC_R8GT\n#define OKC_R8GT 0  // r8g GU: phase A's x gather by (8-row group, stage) items, whole-line xA writes\n#endif\n")
    return s


CUDA_SRC = _r8g3_src(CUDA_SRC)
_R4X_PFX[CUDA_SRC] = "okc_moe_r8g3" + _R8G3_TAG + "_"
MAIN_DEFINES = tuple(MAIN_DEFINES) + tuple(_R8G3_DEFS)
SC6_DEFINES = tuple(SC6_DEFINES) + tuple(_R8G3_DEFS)


# ---------------------------------------------------------------- change set R10T (adamleex, 2026-11-03): small-build barrier tail
# TINY / TINY2 / MID / MID-S6 only (every other build keeps its source and name). Generated by tools/r10t_mk.py.
#   T2 arrive first: right after the post-loop WS_JOIN (bar.sync 2 over every compute, helper and poller / issuer thread;
#      each thread's phase-B global stores and atomics precede its WS_JOIN in program order), thread 0 fences at gpu scope
#      and arrives; the fence is cumulative over the CTA barrier (the cooperative-groups grid.sync pattern), so no CTA_SYNC
#      precedes the arrival. The first final item's prefetch (FINPF) runs between the arrival and the poll, only on CTAs
#      that own a final item. T1: the helper warp's trailing fence before that WS_JOIN is redundant and removed.
#   T5 the per-launch flag zeroing (doneA, pool counters) runs on the last CTA, which owns no final item when
#      M * FSEG < grid (every M <= 32 at grid 132); otherwise every CTA zeroes its share as before.
#   T4 ss accumulations are red.relaxed.gpu.global.add.f32 (the same relaxed float atomic add, no returned value).
_R10T_TAG, _R10T_EDITS = ('tall', ('T2', 'T5', 'T1', 'T4'))


def _r10t_rp(s, old, new, n=1):
    c = s.count(old)
    assert c == n, ("r10t", c, n, old[:120])
    return s.replace(old, new)


_R10T_SSADD = r"""
__device__ __forceinline__ void r10t_ssadd(float* a, float v) {  // r10t T4: ss accumulation without a returned value
  asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;\n" ::"l"(__cvta_generic_to_global(a)), "f"(v) : "memory");
}
"""
_R10T_JOIN = "#if OKC_WS\n  WS_JOIN();  // the helper's last epilogue stores are visible\n#endif\n"
_R10T_T2 = r"""#if OKC_WS && OKC_FINPF && OKC_FINCOL && !OKC_WGS
  WS_JOIN();  // the helper's last epilogue stores are visible
  // r10t T2: arrive first (no CTA_SYNC before the arrival: WS_JOIN above joined every thread of the CTA after its last
  // phase-B store or atomic, and thread 0's gpu-scope fence after that barrier is cumulative)
  int* const bc = p.ctr + 4 + (S.gen & 1);
#if !R10T_BRK
  if (tid == 0) {
    __threadfence();
    const int old = atomicAdd(bc, 1);
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
  }
#endif
  // programmatic dependent launch (a no-op without the launch attribute): after the arrival
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  // the CTA's first final item: its routing (pos) and route weights are inputs, so they load before the release;
  // a CTA without a final item skips this (CTA-uniform branch)
  constexpr int FSEGP = HD / (NT * 4);
  const int itp = blockIdx.x;
  const bool hasp = itp < p.M * FSEGP;
  const int tkp = hasp ? itp / FSEGP : 0;
  int qvp[8];
  float wvp[8];
  if (hasp) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;
      wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;
    }
  }
  if (tid == 0) {
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
#else
"""
_R10T_ZERO_OLD = ("  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA's share)\n"
                  "  {\n    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;\n    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters\n"
                  "    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;\n  }\n")
_R10T_ZERO_NEW = r"""  // phases A and B are over everywhere: zero the per-launch flags for the next launch
#if OKC_FINCOL && !OKC_WGS
  const bool r10tz = p.M * (HD / (NT * 4)) < (int)gridDim.x;  // r10t T5: the last CTA owns no final item
#else
  const bool r10tz = false;
#endif
  if (r10tz) {
    if (blockIdx.x == gridDim.x - 1) {
      if (tid < 2) p.ctr[tid] = 0;  // dynamic pool counters
      for (int i = tid; i < JMAX; i += NT) p.doneA[i] = 0;
    }
  } else {  // (this CTA's share)
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
  }
"""
_R10T_HFENCE_OLD = "  __threadfence();\n  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n}"
_R10T_HFENCE_NEW = ("  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier (r10t T1: no trailing fence here;\n"
                    "              // thread 0's gpu-scope fence after the post-loop WS_JOIN is cumulative)\n}")


def _r10t_src(s, edits):
    rp = _r10t_rp
    if "T4" in edits:
        s = rp(s, "__device__ __forceinline__ float* ss_cur();  // this launch's ss copy (defined after Smem users)\n",
               "__device__ __forceinline__ float* ss_cur();  // this launch's ss copy (defined after Smem users)\n" + _R10T_SSADD)
        s = rp(s, "atomicAdd(&ss_cur()[", "r10t_ssadd(&ss_cur()[", n=8)
        assert "atomicAdd(&ss_cur()" not in s
    if "T1" in edits:
        s = rp(s, _R10T_HFENCE_OLD, _R10T_HFENCE_NEW)
    if "T2" in edits:
        a = s.index(_R10T_JOIN)
        assert s.count(_R10T_JOIN) == 1
        b = s.index("  CTA_SYNC();\n  DBG(4);\n", a) + len("  CTA_SYNC();\n")
        old = s[a:b]
        # the base block: join, launch_dependents, FINPF prefetch, CTA_SYNC, thread 0 fence + arrive + poll, CTA_SYNC
        assert old.count("CTA_SYNC();") == 2 and old.count("__threadfence();") == 1 and old.count("griddepcontrol.launch_dependents") == 1, old
        assert old.count("#if OKC_FINPF && OKC_FINCOL && !OKC_WGS\n") == 1 and old.count("while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);") == 1
        assert ("    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;\n    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;\n") in old
        s = s[:a] + _R10T_T2 + old + "#endif\n" + s[b:]
    if "T5" in edits:
        s = rp(s, _R10T_ZERO_OLD, _R10T_ZERO_NEW)
    if "BRK" in edits:  # race positive control: thread 0 arrives at the unit-loop end, before the own-WSBL epilogues
        s = "#define R10T_BRK 1\n" + s
        brk = ("  if (TIDX == 0) {  // r10t BRK (race positive control, broken on purpose): arrive before the own epilogues\n"
               "    int* bc = p.ctr + 4 + (S.gen & 1);\n    __threadfence();\n    const int old = atomicAdd(bc, 1);\n"
               "    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);\n  }\n")
        n0 = s.count("  if (warp < OKC_WSBLK && warp < nB) {  // the CTA's last units: epilogues here, not behind the helper's queue\n")
        assert n0 == 2, n0  # skinny_ring_m2u and skinny_ring_uni2
        s = s.replace("  if (warp < OKC_WSBLK && warp < nB) {  // the CTA's last units: epilogues here, not behind the helper's queue\n",
                      brk + "  if (warp < OKC_WSBLK && warp < nB) {  // the CTA's last units: epilogues here, not behind the helper's queue\n")
    return s


# T5 relies on M * FSEG <= 128 < grid 132 for every M these builds serve (else the base zeroing runs, still correct)
assert TINY_MAX_M <= 32 and TINY2_MAX_M <= 32 and MID_MAX_M <= 32, (TINY_MAX_M, TINY2_MAX_M, MID_MAX_M)
for _r10tn, _r10tb, _r10te in (("TINY_DEFINES", "t", "helper0"), ("TINY2_DEFINES", "p", "helper0_r4xcp"),
                               ("MID_DEFINES", "m", "helper0_r4xcm"), ("MID_S6_DEFINES", "s", "helper0s6_r4xcm_r6m")):
    _r10td = globals()[_r10tn]
    _r10ta = [d for d in _r10td if d.startswith("@")]
    assert len(_r10ta) == 1 and _r10ta[0] == "@" + _r10te, (_r10tn, _r10td)
    _r10tk = _r10te + "_r10t" + _R10T_TAG
    ALT_SRCS[_r10tk] = _r10t_src(ALT_SRCS[_r10te], _R10T_EDITS) + "\n// r10t " + _R10T_TAG + " " + _r10tb + "\n"
    _R4X_PFX[ALT_SRCS[_r10tk]] = "okc_moe_r10t" + _R10T_TAG + _r10tb + "_"
    globals()[_r10tn] = tuple("@" + _r10tk if d == _r10ta[0] else d for d in _r10td)


# ---------------------------------------------------------------- change set R14A (adamleex, 2026-11-04): NOPF in TINY + TINY2
# The small builds named in _R14A_BUILDS (TINY: 1..4 tokens; TINY2: 5..8 tokens) no longer issue the compute warps' L1 operand
# prefetch (prefetch_x_rt: a CCTL prefetch hint at a segment's first unit and at each segment's last unit) in their unit loop
# skinny_ring_uni2. A prefetch is only a cache hint: every operand is still loaded by the same load_x_rt4 at the same point
# (after the same mbarrier / acquire waits), so the loaded values, the units, their order, the math, the epilogues and the
# outputs are unchanged. Measured: the hint was a net cost (probably because the poller's acquire spins invalidate L1, so
# the prefetched lines do not survive while the prefetches still occupy the L1 path). Each edited source gets its own extension
# prefix; every other build (MID, MID-S6, SC6, MAIN, fallbacks) and the dispatch bands are unchanged.
_R14A_VARIANT = 'r14a'
_R14A_BUILDS = (("TINY_DEFINES", "helper0_r10ttall", "t"), ("TINY2_DEFINES", "helper0_r4xcp_r10ttall", "p"))
_R14A_PF_OLD = "        if (!pp || (kn < 8 && __all_sync(0xffffffffu, mbar_test(S.hrdy + kn, 0u)))) prefetch_x_rt(p, S, jj, pp, lane, warp);\n"
_R14A_PF_NEW = "        (void)jj; (void)pp;  // R14A NOPF: no operand prefetch\n"
_R14A_UNI_A = "__device__ __forceinline__ void skinny_ring_uni2("
_R14A_UNI_B = "template <bool BIG, bool PB>\n__device__ __forceinline__ void skinny_split("


def _r14a_src(s, edit):
    """skinny_ring_uni2 of s with its one prefetch_x_rt call removed (edit) or unchanged (placebo)"""
    assert s.count(_R14A_UNI_A) == 1 and s.count(_R14A_UNI_B) == 1, ("r14a", s.count(_R14A_UNI_A), s.count(_R14A_UNI_B))
    a = s.index(_R14A_UNI_A)
    b = s.index(_R14A_UNI_B, a)
    u = s[a:b]
    c = u.count(_R14A_PF_OLD)
    assert c == 1, ("r14a: prefetch line count in skinny_ring_uni2", c)
    assert u.count("prefetch_x_rt(") == 1, ("r14a: prefetch_x_rt calls in skinny_ring_uni2", u.count("prefetch_x_rt("))
    if edit:
        u = u.replace(_R14A_PF_OLD, _R14A_PF_NEW)
        assert "prefetch_x_rt(" not in u
    return s[:a] + u + s[b:]


for _r14an, _r14ae, _r14ab in _R14A_BUILDS:
    _r14ad = globals()[_r14an]
    _r14aa = [d for d in _r14ad if d.startswith("@")]
    assert len(_r14aa) == 1 and _r14aa[0] == "@" + _r14ae, (_r14an, _r14ad)
    _r14ak = _r14ae + "_" + _R14A_VARIANT
    assert _r14ak not in ALT_SRCS, _r14ak
    ALT_SRCS[_r14ak] = _r14a_src(ALT_SRCS[_r14ae], _R14A_VARIANT != "r14p") + "\n// " + _R14A_VARIANT + " nopf " + _r14ab + "\n"
    _R4X_PFX[ALT_SRCS[_r14ak]] = "okc_moe_" + _R14A_VARIANT + _r14ab + "_"
    globals()[_r14an] = tuple("@" + _r14ak if d == _r14aa[0] else d for d in _r14ad)




# =====================================================================================================================
# ---------------------------------------------------------------- change set R15W (adamleex, 2026-11-04): WGS epilogue scales
# The large-batch WGS builds named in _R15W_CFG (SC6-WGS: 33..160 tokens; MAIN-WGS: > 160 tokens) read each unit epilogue's
# per-job global weight scale (gsA / gsB of the job's expert, times 2^62) from a per-job table in shared memory, filled once per
# launch by the FXW helper warp (idle at the kernel front: it joins barrier 7, i.e. waits for the routing job table, and fills
# the tables before WS_JOIN, which every consumer passes before its first epilogue), instead of loading the expert id from smem
# and the scale from global memory after the MMA drain of every unit; right after WS_JOIN the same warp copies post_norm_w (8 KB)
# into the Smem eid / ent area (dead after the phase-A gather) and publishes it on its own mbarrier, which the consumers wait on
# once at the start of phase B, so the phase-B epilogues read (1 + w[n]) from smem (shipped form 'jN': helper fill + NW below;
# the x-producer-lane fill path is kept only as the generator's screen form). The stored values are the same floats, so the
# epilogues, the outputs and every other build
# (TINY, TINY2, MID, MID-S6, the non-WGS fallback) are unchanged. Each edited source gets its own extension prefix.
_R15W_TAG, _R15W_CFG = ('jN', {'SC6': {'epis': ['end', 'AB', True, True], 'nw': None}, 'MAIN': {'epis': ['end', 'AB', True, True], 'nw': None}})


def _r15w_rp(s, old, new, n=1):
    c = s.count(old)
    assert c == n, ("r15w", c, n, old[:100])
    return s.replace(old, new)


_R15W_GSA = "const float gsc = p.gsA[e] * TWO62;"
_R15W_GSB = "const float gsc = p.gsB[e] * TWO62;"
_R15W_FXG = "p.gsA[S.jE[j]] * TWO62"
_R15W_POS = "  short pos[EMAX];   // entry -> pair position, -1 if not local\n};\n"
_R15W_XP = ("#if OKC_PR4\n  if ((threadIdx.x & 31) < 4) wgs_ranges4(p, S);\n#else\n  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);\n#endif\n"
            "  __syncwarp();\n")
_R15W_CHK_AT = "__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {"
_R15W_CHK = ("static_assert(((sizeof(Smem) + 127) / 128) * 128 + (size_t)WGS_NS * UNIT_BYTES + (size_t)WGS_NX * WGS_XS <= 232448,\n"
             "              \"r15w: WGS dynamic shared memory over 227 KB\");\n")


def _r15w_src(s, place, phases, fx, helper=False):
    rp = _r15w_rp
    assert place == "end", place
    decl = ""
    if "A" in phases or fx: decl += "  float jgA[JMAX];  // r15w: per-job phase-A global scale x 2^62 (WGS unit epilogues)\n"
    if "B" in phases: decl += "  float jgB[JMAX];  // r15w: per-job phase-B global scale x 2^62 (WGS unit epilogues)\n"
    s = rp(s, _R15W_POS, _R15W_POS.replace("\n};\n", "\n" + decl + "};\n"))
    if not helper:
        fill = ("  if ((threadIdx.x & 31) >= 4) {  // r15w: the idle lanes stage the per-job global scales for the unit epilogues\n"
                "    const int nj = S.njF + S.njS;\n"
                "    for (int j = (threadIdx.x & 31) - 4; j < nj; j += 28) {\n"
                "      const int e = S.jE[j];\n")
        if "A" in phases or fx: fill += "      S.jgA[j] = p.gsA[e] * TWO62;\n"
        if "B" in phases: fill += "      S.jgB[j] = p.gsB[e] * TWO62;\n"
        fill += "    }\n  }\n"
        s = rp(s, _R15W_XP, _R15W_XP.replace("  __syncwarp();\n", fill + "  __syncwarp();\n"))
    else:
        # the FXW helper warp (idle until WS_JOIN) joins barrier 7 ("routing done": the consumers arrive, the x producer syncs)
        # and stages the tables; WS_JOIN (bar.sync 2 over all NTL threads) then orders them before every consumer's epilogue
        s = rp(s, 'asm volatile("bar.arrive 7, %0;\\n" ::"n"(NT + 32) : "memory");', 'asm volatile("bar.arrive 7, %0;\\n" ::"n"(NT + 64) : "memory");', 2)
        s = rp(s, 'asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 32) : "memory");', 'asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 64) : "memory");', 1)
        fill = ("  static_assert(OKC_FXW, \"r15w: barrier 7 counts the FXW helper warp\");\n"
                "  asm volatile(\"bar.sync 7, %0;\\n\" ::\"n\"(NT + 64) : \"memory\");  // r15w: routing tables ready\n"
                "  {  // r15w: stage the per-job global scales for the WGS unit epilogues\n"
                "    const int nj = S.njF + S.njS;\n"
                "    for (int j = (int)(threadIdx.x & 31); j < nj; j += 32) {\n"
                "      const int e = S.jE[j];\n")
        if "A" in phases or fx: fill += "      S.jgA[j] = p.gsA[e] * TWO62;\n"
        if "B" in phases: fill += "      S.jgB[j] = p.gsB[e] * TWO62;\n"
        fill += "    }\n    __syncwarp();\n  }\n"
        s = rp(s, "__device__ void wgs_fxhelper(const Params& p, Smem& S) {\n  WS_JOIN();", "__device__ void wgs_fxhelper(const Params& p, Smem& S) {\n" + fill + "  WS_JOIN();")
    if "A" in phases: s = rp(s, _R15W_GSA, "const float gsc = S.jgA[j];  // r15w", 2)
    if "B" in phases: s = rp(s, _R15W_GSB, "const float gsc = S.jgB[j];  // r15w", 2)
    if fx: s = rp(s, _R15W_FXG, "S.jgA[j]")
    return rp(s, _R15W_CHK_AT, _R15W_CHK + _R15W_CHK_AT)


_R15W_NW_OLD = ("const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;")
_R15W_FXJ = "  WS_JOIN();  // routing tables and this CTA's stage ranges are ready\n  const int lane = threadIdx.x & 31;\n"


def _r15w_nw(s, sel):
    """NW: post_norm_w (HD bf16 = 8 KB) staged into the Smem eid / ent area by the FXW helper warp right after WS_JOIN (every
    reader of eid / ent, i.e. route() and the phase-A x gather, precedes WS_JOIN in the consumers' program order), published on
    mbarrier R6W_FXB + 1; the consumer warps wait on it once at the start of phase B; the phase-B unit epilogues then read
    (1 + w[n]) from smem instead of two __ldg per thread and unit."""
    rp = _r15w_rp
    s = rp(s, "    mbar_init(S.bars + R6W_FXB, 1);", "    mbar_init(S.bars + R6W_FXB, 1);\n    mbar_init(S.bars + R6W_FXB + 1, 1);  // r15w NW: post_norm_w staged in smem\n")
    s = rp(s, _R15W_FXJ, _R15W_FXJ +
           "  static_assert(OKC_FXW && R6W_FXB + 2 <= 24 && 3 * EMAX >= 2 * HD && offsetof(Smem, ent) == offsetof(Smem, eid) + EMAX,\n"
           "                \"r15w NW: post_norm_w fits the eid / ent area\");\n"
           "  {  // r15w NW: post_norm_w into the eid / ent area (dead after the gather; every reader precedes WS_JOIN)\n"
           "    const uint4* nsrc = reinterpret_cast<const uint4*>(p.normw);\n"
           "    uint32_t* ndst = reinterpret_cast<uint32_t*>(&S.eid[0]);\n"
           "#pragma unroll 4\n"
           "    for (int i = lane; i < HD / 8; i += 32) {\n"
           "      const uint4 v = __ldg(nsrc + i);\n"
           "      ndst[4 * i] = v.x; ndst[4 * i + 1] = v.y; ndst[4 * i + 2] = v.z; ndst[4 * i + 3] = v.w;\n"
           "    }\n"
           "    __syncwarp();\n"
           "    if (lane == 0) mbar_arrive1(S.bars + R6W_FXB + 1);  // (release.cta: the stores above)\n"
           "  }\n")
    s = rp(s, "  WgsPend pd{0, 0, 0, 0};\n", "  WgsPend pd{0, 0, 0, 0};\n  if (PB) mbar_wait(S.bars + R6W_FXB + 1, 0);  // r15w NW: post_norm_w is in smem (acquire.cta)\n")
    if sel is None:
        new = ("const __nv_bfloat16* nws = reinterpret_cast<const __nv_bfloat16*>(&S.eid[0]);  // r15w NW\n"
               "    const float w0 = (1.f + __bfloat162float(nws[n0])) * gsc, w1 = (1.f + __bfloat162float(nws[n0 + 8])) * gsc;")
    else:
        new = ("const __nv_bfloat16* nws = p.zero == R15WSEL ? reinterpret_cast<const __nv_bfloat16*>(&S.eid[0]) : p.normw;  // r15w NW twin\n"
               "    const float w0 = (1.f + __bfloat162float(nws[n0])) * gsc, w1 = (1.f + __bfloat162float(nws[n0 + 8])) * gsc;")
        s = "\n#define R15WSEL " + sel + "  // r15w NW same-codegen twin selector (p.zero is 0)\n" + s
    return rp(s, _R15W_NW_OLD, new, 2)


for _r15wb, _r15wc in sorted(_R15W_CFG.items()):
    _r15wn = _r15wb + "_DEFINES"
    _r15wd = tuple(globals()[_r15wn])
    assert not any(d.startswith("@") for d in _r15wd), (_r15wn, _r15wd)  # (the WGS builds compile CUDA_SRC itself)
    assert "-DOKC_WGS=1" in _r15wd, (_r15wn, _r15wd)
    _r15wk = "r15w_" + _R15W_TAG + "_" + _r15wb.lower()
    assert _r15wk not in ALT_SRCS, _r15wk
    _r15ws = CUDA_SRC
    if _r15wc.get("epis"):
        _r15ws = _r15w_src(_r15ws, *_r15wc["epis"])
    if "nw" in _r15wc:
        _r15ws = _r15w_nw(_r15ws, _r15wc["nw"])
    ALT_SRCS[_r15wk] = _r15ws + "\n// r15w " + _R15W_TAG + " " + _r15wb + "\n"
    _R4X_PFX[ALT_SRCS[_r15wk]] = "okc_moe_r15w" + _R15W_TAG + _r15wb.lower()[0] + "_"
    for _r15wx in _r15wc.get("defs", []):
        assert not any(d.split("=")[0] == _r15wx.split("=")[0] for d in _r15wd), (_r15wn, _r15wx)
    globals()[_r15wn] = _r15wd + tuple(_r15wc.get("defs", [])) + ("@" + _r15wk,)


# ---------------------------------------------------------------- change set R15F (adamleex, 2026-11-04): WGS front (bs >= 33)
# The WGS builds (CUDA_SRC: SC6-WGS 33..SC6_MAX_M tokens, MAIN-WGS above) compile the base's existing OKC_RAE / OKC_PR4 paths (no
# code change): barrier 7 (the x producer's stage-range computation) is released inside route() as soon as the job table is
# complete (the ranges read only jC / njF / njS), and the ranges run on lanes 0..3 (wgs_ranges4: one stage_at per lane, the same
# 32-bit formulas and NJ0 guard, hence the same ranges, stage order, math and outputs). The CTA join (gather done + ranges ready)
# is then gated by the gather alone, so the weight and x producers start ~0.5 / 0.25 us (bs128 / bs256) earlier. Own extension
# prefix; TINY / TINY2 / MID-S6 / MID and the non-WGS fallback keep their sources, names and .so files.
assert "-DOKC_WGS=1" in MAIN_DEFINES and "-DOKC_SC6W=1" in SC6_DEFINES, (MAIN_DEFINES, SC6_DEFINES)
assert not any(d.startswith(("-DOKC_RAE", "-DOKC_PR4")) for d in tuple(MAIN_DEFINES) + tuple(SC6_DEFINES)), (MAIN_DEFINES, SC6_DEFINES)
CUDA_SRC = CUDA_SRC + "\n// r15f rp\n"
_R4X_PFX[CUDA_SRC] = "okc_moe_r15frp_"
MAIN_DEFINES = tuple(MAIN_DEFINES) + ("-DOKC_RAE=1", "-DOKC_PR4=1")
SC6_DEFINES = tuple(SC6_DEFINES) + ("-DOKC_RAE=1", "-DOKC_PR4=1")


# ---------------------------------------------------------------- change set R15S (adamleex, 2026-11-04): MID-S6 static unit split
# The MID-S6 build (9..32 tokens, 6-bit scales) splits each launch's phase-A / phase-B weight units over the CTAs by a static cost
# prefix computed by the FAST1 warp (1-tile unit = OKC_M2W1 = 20; 2-tile unit = OKC_M2W2 / OKC_M2W2B). Measured on r14a (stamped
# MID-S6, judge-faithful dirty-L2 timing): a 2-tile unit takes 1.31 / 1.32 x a 1-tile unit (priced 1.55 / 1.65 x), and every extra
# job segment in a CTA's range costs +0.7..0.9 us that the split did not charge (the slowest CTA was a job-boundary CTA far more
# often than chance). This change set (1) prices 2-tile units at 27 / 30 (build constants) and (2) charges every job 15 (phase A)
# / 17 (phase B) extra cost units at its first unit, in both the even split (no 2-tile job) and the 2-tile split; with zero job
# costs the new split reproduces the old ranges exactly. Only the FAST1 prologue arithmetic of the MID-S6 build changes: every
# unit, its weights, its math and the outputs are unchanged - only which CTA processes which contiguous unit range moves (the
# per-job doneA credits count whatever range a CTA holds). Own extension prefix; every other build and the dispatch unchanged.
_R15S_TAG = "dj2730"

_R15S_J_HELPER_AT = 'template <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {\n'
_R15S_J_HELPER = "// r15s J: the M2T static split with a job-start cost WJ (cost units) on every job's first unit. Even split (no 2-tile job):\n// the last unit u whose cost prefix k * (R * OKC_M2W1 + WJ) + (i ? WJ + i * OKC_M2W1 : 0) <= x = floor(total * c / G); WJ = 0\n// gives floor(njob * R * c / G), the base's even split, exactly.\ntemplate <int R, int WJ>\n__device__ __forceinline__ int r15j_even1(int njob, int c, int G, float rG) {\n  constexpr int L = R * OKC_M2W1 + WJ;\n  if (c >= G) return njob * R;\n  const int x = m2t_muldiv(njob * L, c, G, rG);\n  const int k = x / L, r = x - k * L;\n  return k * R + min(R, max(0, (r - WJ) / OKC_M2W1));\n}\ntemplate <int R, int WJ>\n__device__ __forceinline__ int r15j_even(int njob, int c, int G, float rG, bool on) {\n  return on ? r15j_even1<R, WJ>(njob, c, G, rG) : r15j_even1<R, 0>(njob, c, G, rG);\n}\n"
_R15S_J_OLD = ('    int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);\n', '      const int x = m2t_muldiv(__shfl_sync(0xffffffffu, cwi, 31) * rbs, c, gridDim.x, rG1);\n      const unsigned bal = __ballot_sync(0xffffffffu, cwi * rbs > x);\n      const int le = __ffs(bal) - 1;\n      const int cexcl = __shfl_sync(0xffffffffu, (cwi - cwe) * rbs, le);\n      const int uexcl = __shfl_sync(0xffffffffu, (jincl - je) * rbs, le);\n      const int n2 = __shfl_sync(0xffffffffu, n2j, le);\n      const int d = x - cexcl, c2 = n2 * rbs * w2;\n      return uexcl + (d <= c2 ? (w2 == OKC_M2W2 ? (d + OKC_M2W2 - 1) / OKC_M2W2 : (d + OKC_M2W2B - 1) / OKC_M2W2B) : n2 * rbs + (d - c2 + OKC_M2W1 - 1) / OKC_M2W1);\n', '        S.wsr[0] = muldiv_g(njob * RB_A, blockIdx.x, gridDim.x, rG1); S.wsr[1] = muldiv_g(njob * RB_A, blockIdx.x + 1, gridDim.x, rG1);\n        S.wsr[2] = muldiv_g(njob * RB_B, blockIdx.x, gridDim.x, rG1); S.wsr[3] = muldiv_g(njob * RB_B, blockIdx.x + 1, gridDim.x, rG1);\n')
_R15S_J_NEW = ('    static_assert(!SKDYN, "r15s J: no dynamic pool in the M2T build");\n    const bool r15jon = p.M > R15J_MT;  // r15s J: job-start costs R15J_WA / R15J_WB in the split (else 0: the base split)\n    int v01 = r15j_even<RB_A, R15J_WA>(njob, blockIdx.x, gridDim.x, rG1, r15jon), v11 = r15j_even<RB_A, R15J_WA>(njob, blockIdx.x + 1, gridDim.x, rG1, r15jon);\n', "      const bool pa = rbs == RB_A;  // (phase A: w2 == OKC_M2W2; phase B: w2 == OKC_M2W2B)\n      const int wj = r15jon ? (pa ? R15J_WA : R15J_WB) : 0;\n      const int x = m2t_muldiv(__shfl_sync(0xffffffffu, cwi * rbs + jincl * wj, 31), c, gridDim.x, rG1);\n      const unsigned bal = __ballot_sync(0xffffffffu, cwi * rbs + jincl * wj > x);\n      const int le = __ffs(bal) - 1;\n      const int cexcl = __shfl_sync(0xffffffffu, (cwi - cwe) * rbs + (jincl - je) * wj, le);\n      const int uexcl = __shfl_sync(0xffffffffu, (jincl - je) * rbs, le);\n      const int n2 = __shfl_sync(0xffffffffu, n2j, le);\n      const int L2 = rbs * w2 + wj, d = x - cexcl, c2 = n2 * L2;\n      if (d <= c2) {  // inside the expert's 2-tile jobs: job k, then the first unit whose prefix reaches x\n        const int k = r15jon ? (pa ? d / (RB_A * OKC_M2W2 + R15J_WA) : d / (RB_B * OKC_M2W2B + R15J_WB))\n                             : (pa ? d / (RB_A * OKC_M2W2) : d / (RB_B * OKC_M2W2B));\n        const int r = d - k * L2;\n        return uexcl + k * rbs + (r <= 0 ? 0 : max(1, pa ? (r - wj + OKC_M2W2 - 1) / OKC_M2W2 : (r - wj + OKC_M2W2B - 1) / OKC_M2W2B));\n      }\n      const int r = d - c2;  // the expert's 1-tile job (at most one, after its 2-tile jobs)\n      return uexcl + n2 * rbs + (r <= 0 ? 0 : max(1, (r - wj + OKC_M2W1 - 1) / OKC_M2W1));\n", "        S.wsr[0] = v01; S.wsr[1] = v11;  // (r15s J: the pre-issue's split)\n        S.wsr[2] = r15j_even<RB_B, R15J_WB>(njob, blockIdx.x, gridDim.x, rG1, r15jon); S.wsr[3] = r15j_even<RB_B, R15J_WB>(njob, blockIdx.x + 1, gridDim.x, rG1, r15jon);\n")


def _r15s_rp(s, old, new, n=1):
    c = s.count(old)
    assert c == n, ("r15s", c, n, old[:120])
    return s.replace(old, new)


def _r15s_src(s):
    s = _r15s_rp(s, _R15S_J_HELPER_AT, _R15S_J_HELPER + _R15S_J_HELPER_AT)
    for o, n_ in zip(_R15S_J_OLD, _R15S_J_NEW):
        s = _r15s_rp(s, o, n_)
    return s


_r15s_at = [d for d in MID_S6_DEFINES if d.startswith("@")]
assert len(_r15s_at) == 1, MID_S6_DEFINES
assert MID_S6_DEFINES.count("-DOKC_M2W2=31") == 1 and MID_S6_DEFINES.count("-DOKC_M2W2B=33") == 1, MID_S6_DEFINES
_r15s_k = _r15s_at[0][1:] + "_r15s" + _R15S_TAG
assert _r15s_k not in ALT_SRCS, _r15s_k
ALT_SRCS[_r15s_k] = _r15s_src(ALT_SRCS[_r15s_at[0][1:]]) + "\n// r15s " + _R15S_TAG + "\n"
_R4X_PFX[ALT_SRCS[_r15s_k]] = "okc_moe_r15s" + _R15S_TAG + "m_"
MID_S6_DEFINES = tuple("@" + _r15s_k if d == _r15s_at[0] else d for d in MID_S6_DEFINES) + ("-DR15J_MT=0", "-DR15J_WA=15", "-DR15J_WB=17")
MID_S6_DEFINES = tuple("-DOKC_M2W2=27" if d == "-DOKC_M2W2=31" else ("-DOKC_M2W2B=30" if d == "-DOKC_M2W2B=33" else d) for d in MID_S6_DEFINES)



# ---------------------------------------------------------------- change set R15E (adamleex, 2026-11-04): TINY2 (5..8 tokens) BALTZ
# Only the TINY2 build (5 <= M <= 8; ALT_SRCS["helper0_r4xcp_r10ttall_r14a"], r14a NOPF already in it) is edited; every other build
# (TINY, MID, MID-S6, SC6, MAIN, fallbacks), the dispatch bands, prepare() and kernel_fn are unchanged. Exact-count asserted text
# replacements; work assignment only: the units, their math, their order within a CTA's phase and every output element are
# unchanged.
#   balt the CTA unit deal per job count E (= the call's distinct local experts; the deal geometry over the grid depends on E
#        only): for E with its bit set in _R15E_BALM the bal split below (else the separate A / B splits), for E with its bit
#        set in _R15E_KX3M an HOF X block (jobs dealt evenly and run first in both phases) of the last 3 jobs instead of 2;
#        the bits were chosen per E from measured bs8 timings (BAL on E = 6, 7, 9..; X block of 3 at E = 7, 10, 11, 12).
#        bal = each CTA's phase-B unit range is its share of ONE split of all nA + nB units minus its phase-A range (per-CTA
#        A + B totals differ by at most one). Each mask is written as (p.zero == 0u ? mask : 0) (p.zero = the kernel's opaque
#        zero): the same behaviour, with the unit-loop schedule that measured faster than the plain-constant form
_R15E_VARIANT = 'r15et'
_R15E_EDITS = ('baltz',)
_R15E_KEY = "helper0_r4xcp_r10ttall_r14a"
_R15E_BALM = '0xfffffffffffffec0'  # balt: BAL for the job counts E whose bit is set (E = the call's distinct local experts)
_R15E_KX3M = '0x1c80'  # balt: an X block of 3 jobs for the job counts E whose bit is set (else 2, from 4 jobs on)


def _r15e_rp(s, old, new, n=1):
    c = s.count(old)
    assert c == n, ("r15e", c, n, old[:120])
    return s.replace(old, new)


def _r15e_uni(s, fn):
    a_mark, b_mark = "__device__ __forceinline__ void skinny_ring_uni2(", "template <bool BIG, bool PB>\n__device__ __forceinline__ void skinny_split("
    assert s.count(a_mark) == 1 and s.count(b_mark) == 1
    a = s.index(a_mark)
    b = s.index(b_mark, a)
    return s[:a] + fn(s[a:b]) + s[b:]


def _r15e_src(s, edits):
    rp = _r15e_rp
    assert s.count("(void)jj; (void)pp;  // R14A NOPF: no operand prefetch") == 1
    if "balt" in edits or "baltz" in edits:
        bm, km = "%sull" % _R15E_BALM, "%sull" % _R15E_KX3M
        if "baltz" in edits:  # the masks behind the opaque p.zero (always 0): the form whose unit-loop schedule measured best
            bm, km = "(p.zero == 0u ? %s : 0ull)" % bm, "(p.zero == 0u ? %s : 0ull)" % km
        s = rp(s, "    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);\n"
                  "    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n",
               "    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);\n"
               "    if ((" + bm + " >> min(S.njS, 63)) & 1ull) {  // r15e BALT: BAL for these job counts (one split of all nA + nB units)\n"
               "      S.wsr[2] = muldiv_g(nA + nB, c, G, rG) - S.wsr[0]; S.wsr[3] = muldiv_g(nA + nB, c + 1, G, rG) - S.wsr[1];\n"
               "    } else {\n"
               "      S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n"
               "    }\n")
        s = rp(s, "        const int kx = __popc(m) >= OKC_HOFK2 ? 2 : 1;",
               "        const int kx = __popc(m) >= OKC_HOFK2 ? ((((" + km + ") >> min(__popc(m), 63)) & 1ull) ? 3 : 2) : 1;  // r15e BALT")
        s = rp(s, "        const int kx = S.njS >= OKC_HOFK2 ? 2 : 1;",
               "        const int kx = S.njS >= OKC_HOFK2 ? ((((" + km + ") >> min(S.njS, 63)) & 1ull) ? 3 : 2) : 1;  // r15e BALT")
    return s


_r15e_d = TINY2_DEFINES
_r15e_at = [d for d in _r15e_d if d.startswith("@")]
assert len(_r15e_at) == 1 and _r15e_at[0] == "@" + _R15E_KEY, _r15e_d
if "wb2" in _R15E_EDITS:
    assert list(_r15e_d).count("-DOKC_WSBLK=1") == 1, _r15e_d
    _r15e_d = tuple("-DOKC_WSBLK=2" if d == "-DOKC_WSBLK=1" else d for d in _r15e_d)
_r15e_k = _R15E_KEY + "_" + _R15E_VARIANT
assert _r15e_k not in ALT_SRCS, _r15e_k
ALT_SRCS[_r15e_k] = _r15e_src(ALT_SRCS[_R15E_KEY], _R15E_EDITS) + "\n// " + _R15E_VARIANT + " " + ",".join(_R15E_EDITS) + "\n"
_R4X_PFX[ALT_SRCS[_r15e_k]] = "okc_moe_" + _R15E_VARIANT + "p_"
TINY2_DEFINES = tuple("@" + _r15e_k if d == _r15e_at[0] else d for d in _r15e_d)


# =====================================================================================================================
# r16d TAIL (moe_probe_20261102, 2026-11-04): A/B copy of a bs1 barrier-free kernel end (tools/r16d_mk.py). Everything above
# this banner is the base file unchanged.
_R16D_TAG, _R16D_KIND = ('j', 'W8Pur')


def _r16d_rp(s, old, new, n=1):
    c = s.count(old)
    assert c == n, ("r16d", _R16D_KIND, c, n, old[:160])
    return s.replace(old, new)


def _r16d_slice(s, a_mark, b_mark, fn):
    a = s.index(a_mark)
    b = s.index(b_mark, a)
    return s[:a] + fn(s[a:b]) + s[b:]


_R16D_HLP_A = "__device__ void ws_helper("
_R16D_HLP_B = "#if OKC_HOFW\n// r8 poller warp"
_R16D_POL_A = "__device__ void ws_poller(const Params& p, Smem& S) {"
_R16D_POL_B = "#if OKC_HOFW\n// operand of job j"
_R16D_END_A = "#if OKC_WS && OKC_FINPF && OKC_FINCOL && !OKC_WGS\n  WS_JOIN();  // the helper's last epilogue stores are visible\n  // r10t T2"
_R16D_END_B = "  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered\n}\n"
_R16D_HEND = ("  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier (r10t T1: no trailing fence here;\n"
              "              // thread 0's gpu-scope fence after the post-loop WS_JOIN is cumulative)\n}\n")
_R16D_PSTART = "__device__ void ws_poller(const Params& p, Smem& S) {\n  const int lane = TIDX & 31;\n  WS_JOIN();  // routing tables and this CTA's ranges are ready\n"
_R16D_DBG_OLD = "  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;\n"
_R16D_STK_OLD = "__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }\n"
_R16D_STK_NEW = ("__device__ __forceinline__ void st_keep_f32(float* a, float v) {  // r16d: strong relaxed (TINY's y rows are polled)\n"
                 "  asm volatile(\"st.relaxed.gpu.global.f32 [%0], %1;\\n\" ::\"l\"(__cvta_generic_to_global(a)), \"f\"(v));\n}\n")

_R16D_DEV = r"""
// r16d: TINY-private state at the end of scratch[2] (prepare: the int words zero, the y rows and the per-unit partials 0xFFFFFFFF =
// a NaN payload GPU arithmetic never produces, the ss sink zero)
constexpr int R16_FS = @@FS@@;     // final CTAs per token = column segments (HD / R16_FS columns each) = partial copies
constexpr int R16_NPOS = 32;       // TINY routes (M <= 4 tokens x 8)
constexpr int R16_TGW = 0, R16_DA0 = 64, R16_DAN = 64, R16_CTR = 256;  // int words: launch generation, 2 doneA copies, ctr block
constexpr size_t R16_YOFF = 2048;
constexpr size_t R16_POFF = R16_YOFF + (size_t)R16_NPOS * HD * 4;            // partials [copy][route][unit]
constexpr size_t R16_KOFF = R16_POFF + (size_t)R16_FS * R16_NPOS * RB_B * 4;  // ss sink (EMAX floats, never read)
constexpr size_t R16_BYTES = R16_KOFF + (size_t)EMAX * 4;
constexpr uint32_t R16_SENT = 0xFFFFFFFFu;
static_assert(R16_FS >= 2 && R16_FS <= 8 && HD % (R16_FS * 128) == 0 && (HD / R16_FS) % NT == 0, "r16d: final segment shape");
static_assert(NW == 8 && RB_B == 256, "r16d: warp = route slot, 256 partials per route");
__device__ __forceinline__ unsigned char* r16_priv(const Params& p) { return reinterpret_cast<unsigned char*>(p.ybuf) - R16_YOFF; }
__device__ __forceinline__ void r16_st(float* a, float v) {
  asm volatile("st.relaxed.gpu.global.f32 [%0], %1;\n" ::"l"(__cvta_generic_to_global(a)), "f"(v));
}
__device__ __forceinline__ void r16_sent4(const float* a) {
  asm volatile("st.relaxed.gpu.global.v4.b32 [%0], {%1, %1, %1, %1};\n" ::"l"(__cvta_generic_to_global(a)), "r"(R16_SENT) : "memory");
}
__device__ __forceinline__ uint4 r16_ld4(const float* a) {
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.b32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(__cvta_generic_to_global(a)) : "memory");
  return v;
}
__device__ __forceinline__ unsigned r16_bad(const uint4 v) {
  return (unsigned)(v.x == R16_SENT) | (unsigned)(v.y == R16_SENT) | (unsigned)(v.z == R16_SENT) | (unsigned)(v.w == R16_SENT);
}
// r16d final on the compute warps (tid < NT) of CTA c < M * R16_FS, after the post-loop WS_JOIN: token tk = c / R16_FS, column
// segment sg = c % R16_FS (CW columns). Warp w = route slot w of tk: lane polls the YU float4 of y at columns sg * CW + (u * 32 +
// lane) * 4 and the PU float4 of partial copy sg at units (u * 32 + lane) * 4 (each element: one writer per launch, this lane
// its only reader) until no sentinel is left, restores the sentinels, forms ss = the sum of the route's 256 partials (warp
// reduce), stages raw y and r = w * rsqrt(ss / HD + 1e-8) in the idle weight ring; after one CTA_SYNC each thread forms its CPT
// columns of out = sum over the token's local slots (slot order, as the base final) of r * y, in bf16.
__device__ __forceinline__ void r16_fin(const Params& p, Smem& S, int tid) {
  constexpr int CW = HD / R16_FS, YU = CW / 128, PU = RB_B / 128, CPT = CW / NT;
@@SKIP@@@@ITEM@@  const int lane = tid & 31, w = tid >> 5;
  unsigned char* const pv = r16_priv(p);
  float* const stage = reinterpret_cast<float*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);  // [8 slots][CW] y, then [8] r
  float* const rs = stage + 8 * CW;
  const int q = S.pos[tk * 8 + w];  // (warp-uniform)
  if (q >= 0) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // (the ring was last written by TMA)
    const float wt = __ldg(p.topk_w + tk * 8 + w);
    const float* const ya = reinterpret_cast<const float*>(pv + R16_YOFF) + (size_t)q * HD + sg * CW + lane * 4;
    const float* const pa = reinterpret_cast<const float*>(pv + R16_POFF) + ((size_t)sg * R16_NPOS + q) * RB_B + lane * 4;
    uint4 v[YU + PU];
    unsigned bad;
    do {
#pragma unroll
      for (int u = 0; u < YU; ++u) v[u] = r16_ld4(ya + u * 128);
#pragma unroll
      for (int u = 0; u < PU; ++u) v[YU + u] = r16_ld4(pa + u * 128);
      bad = 0u;
#pragma unroll
      for (int u = 0; u < YU + PU; ++u) bad |= r16_bad(v[u]);
    } while (bad@@BRK@@);
#pragma unroll
    for (int u = 0; u < YU; ++u) r16_sent4(ya + u * 128);
#pragma unroll
    for (int u = 0; u < PU; ++u) r16_sent4(pa + u * 128);
    float s = 0.f;
#pragma unroll
    for (int u = 0; u < PU; ++u)
      s += (__uint_as_float(v[YU + u].x) + __uint_as_float(v[YU + u].y)) + (__uint_as_float(v[YU + u].z) + __uint_as_float(v[YU + u].w));
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    s += __shfl_xor_sync(0xffffffffu, s, 8);
    s += __shfl_xor_sync(0xffffffffu, s, 16);
#pragma unroll
    for (int u = 0; u < YU; ++u) *reinterpret_cast<uint4*>(stage + w * CW + u * 128 + lane * 4) = v[u];
    if (lane == 0) rs[w] = wt * rsqrtf(s * (1.f / HD) + 1e-8f);
  }
  CTA_SYNC();
  float a[CPT];
#pragma unroll
  for (int k = 0; k < CPT; ++k) a[k] = 0.f;
#pragma unroll
  for (int sl = 0; sl < 8; ++sl) {
    if (S.pos[tk * 8 + sl] >= 0) {  // (in slot order, as the base final)
      const float r = rs[sl];
      const float* const y = stage + sl * CW + tid * CPT;
      if (CPT == 4) {
        const float4 u = *reinterpret_cast<const float4*>(y);
        a[0] += r * u.x; a[1] += r * u.y; a[2 % CPT] += r * u.z; a[3 % CPT] += r * u.w;
      } else {
        const float2 u = *reinterpret_cast<const float2*>(y);
        a[0] += r * u.x; a[1 % CPT] += r * u.y;
      }
    }
  }
  __nv_bfloat16* const o = p.out + (size_t)tk * HD + sg * CW + tid * CPT;
  if (CPT == 4) {
    __nv_bfloat162 o0 = __floats2bfloat162_rn(a[0], a[1 % CPT]), o1 = __floats2bfloat162_rn(a[2 % CPT], a[3 % CPT]);
    uint2 ov;
    ov.x = *reinterpret_cast<uint32_t*>(&o0);
    ov.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(o) = ov;
  } else {
    __nv_bfloat162 o0 = __floats2bfloat162_rn(a[0], a[1 % CPT]);
    *reinterpret_cast<uint32_t*>(o) = *reinterpret_cast<uint32_t*>(&o0);
  }
}
@@PFIN@@"""

_R16D_PFIN = r"""
// r16d P: the same final as one CTA-uniform loop whose body holds ALL of the work (poll, partial sums, staging, the out sums):
// after the first (cold) pass every pass runs from a warm instruction cache, so the pass that sees the last data is warm. One
// bar.red.or per pass (named barrier 1, NT threads) both joins the staging and decides the exit; the stage is double-buffered by
// pass parity (a pass writes stage[k & 1] only after every thread finished reading it two passes ago). The out store and the
// sentinel restores follow the loop (one each per element).
__device__ __forceinline__ void r16_finp(const Params& p, Smem& S, int tid) {
  constexpr int CW = HD / R16_FS, YU = CW / 128, PU = RB_B / 128, CPT = CW / NT;
@@SKIP@@@@ITEM@@  const int lane = tid & 31, w = tid >> 5;
  unsigned char* const pv = r16_priv(p);
  float* const stage = reinterpret_cast<float*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);  // [2][8 slots][CW] y, then [2][8] r
  float* const rs = stage + 2 * 8 * CW;
  const int q = S.pos[tk * 8 + w];  // (warp-uniform)
  unsigned used = 0u;
#pragma unroll
  for (int s = 0; s < 8; ++s) used |= S.pos[tk * 8 + s] >= 0 ? (1u << s) : 0u;
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // (the ring was last written by TMA)
  const float wt = q >= 0 ? __ldg(p.topk_w + tk * 8 + w) : 0.f;
  const float* const ya = reinterpret_cast<const float*>(pv + R16_YOFF) + (size_t)(q >= 0 ? q : 0) * HD + sg * CW + lane * 4;
  const float* const pa = reinterpret_cast<const float*>(pv + R16_POFF) + ((size_t)sg * R16_NPOS + (q >= 0 ? q : 0)) * RB_B + lane * 4;
  uint4 v[YU + PU];
  float a[CPT];
  unsigned it = 0u, anyb;
  do {
    float* const st = stage + (it & 1u) * (8 * CW);
    float* const rr = rs + (it & 1u) * 8;
    unsigned bad = 0u;
    if (q >= 0) {
#pragma unroll
      for (int u = 0; u < YU; ++u) v[u] = r16_ld4(ya + u * 128);
#pragma unroll
      for (int u = 0; u < PU; ++u) v[YU + u] = r16_ld4(pa + u * 128);
#pragma unroll
      for (int u = 0; u < YU + PU; ++u) bad |= r16_bad(v[u]);
      float s = 0.f;
#pragma unroll
      for (int u = 0; u < PU; ++u)
        s += (__uint_as_float(v[YU + u].x) + __uint_as_float(v[YU + u].y)) + (__uint_as_float(v[YU + u].z) + __uint_as_float(v[YU + u].w));
      s += __shfl_xor_sync(0xffffffffu, s, 1);
      s += __shfl_xor_sync(0xffffffffu, s, 2);
      s += __shfl_xor_sync(0xffffffffu, s, 4);
      s += __shfl_xor_sync(0xffffffffu, s, 8);
      s += __shfl_xor_sync(0xffffffffu, s, 16);
#pragma unroll
      for (int u = 0; u < YU; ++u) *reinterpret_cast<uint4*>(st + w * CW + u * 128 + lane * 4) = v[u];
      if (lane == 0) rr[w] = wt * rsqrtf(s * (1.f / HD) + 1e-8f);
    }
    asm volatile("{\n .reg .pred q0, q1;\n setp.ne.u32 q0, %1, 0;\n bar.red.or.pred q1, 1, %2, q0;\n selp.u32 %0, 1, 0, q1;\n}\n"
                 : "=r"(anyb) : "r"(bad), "n"(NT) : "memory");
#pragma unroll
    for (int k = 0; k < CPT; ++k) a[k] = 0.f;
#pragma unroll
    for (int sl = 0; sl < 8; ++sl) {
      if (used & (1u << sl)) {  // (in slot order, as the base final)
        const float r = rr[sl];
        const float* const y = st + sl * CW + tid * CPT;
        if (CPT == 4) {
          const float4 u = *reinterpret_cast<const float4*>(y);
          a[0] += r * u.x; a[1] += r * u.y; a[2 % CPT] += r * u.z; a[3 % CPT] += r * u.w;
        } else {
          const float2 u = *reinterpret_cast<const float2*>(y);
          a[0] += r * u.x; a[1 % CPT] += r * u.y;
        }
      }
    }
    ++it;
  } while (anyb@@BRK@@);
  __nv_bfloat16* const o = p.out + (size_t)tk * HD + sg * CW + tid * CPT;
  if (CPT == 4) {
    __nv_bfloat162 o0 = __floats2bfloat162_rn(a[0], a[1 % CPT]), o1 = __floats2bfloat162_rn(a[2 % CPT], a[3 % CPT]);
    uint2 ov;
    ov.x = *reinterpret_cast<uint32_t*>(&o0);
    ov.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(o) = ov;
  } else {
    __nv_bfloat162 o0 = __floats2bfloat162_rn(a[0], a[1 % CPT]);
    *reinterpret_cast<uint32_t*>(o) = *reinterpret_cast<uint32_t*>(&o0);
  }
  if (q >= 0) {
#pragma unroll
    for (int u = 0; u < YU; ++u) r16_sent4(ya + u * 128);
#pragma unroll
    for (int u = 0; u < PU; ++u) r16_sent4(pa + u * 128);
  }
}
"""

_R16D_PSTART_NEW = r"""__device__ void ws_poller(const Params& p, Smem& S) {
  const int lane = TIDX & 31;
  // r16d: this TINY launch's generation (one relaxed RMW per CTA on one private word: arrivals in the low 16 bits, the generation
  // in the high 16; the last arriver turns (t, G) into (t + 1, 0); every RMW of this launch precedes that bump in the word's
  // coherence order, so every CTA reads the same t), its doneA copy t & 1; CTA 0 zeroes the next launch's copy (last used by the
  // previous TINY launch, which has completed)
  int* r16dA;
  {
    unsigned char* const pv = r16_priv(p);
    unsigned o = 0u;
    if (lane == 0) {
      unsigned* const tw = reinterpret_cast<unsigned*>(pv) + R16_TGW;
      o = atomicAdd(tw, 1u);
      if ((o & 0xFFFFu) == gridDim.x - 1u) atomicAdd(tw, 0x10000u - gridDim.x);
    }
    o = __shfl_sync(0xffffffffu, o, 0);
    const int tg = (int)(o >> 16);
    r16dA = reinterpret_cast<int*>(pv) + R16_DA0 + (tg & 1) * R16_DAN;
    if (blockIdx.x == 0) {
      int* const z = reinterpret_cast<int*>(pv) + R16_DA0 + ((tg + 1) & 1) * R16_DAN;
      z[lane] = 0;
      z[lane + 32] = 0;
    }
  }
  WS_JOIN();  // routing tables and this CTA's ranges are ready
"""

_R16D_WSBL = r"""  {  // r16d: the CTA's last phase-B units (WSBL: their epilogues ran on the compute warps, whose sums of squares go to the
     // sink): their per-unit partials, formed here from the same cross-warp sums (S.red is not written after the unit loop)
    for (int q = Nh; q < N; ++q) {
      const int v = hof_unit(S, 1, q - nA);
      const int j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B, bb = q % NRB;
      float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        const float4 t4 = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
        sm.x += t4.x; sm.y += t4.y; sm.z += t4.z; sm.w += t4.w;
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
        float sq = v0f * v0f + v1f * v1f;
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (tk < cnt && gg < R16_FS) r16_st(r16p + ((size_t)gg * R16_NPOS + st + tk) * RB_B + rb, sq);
      }
    }
  }
"""

_R16D_HOST = r"""  {  // r16d: TINY-private state at the end of scratch[2] (see R16_*): y rows, per-unit partials, ss sink, int words
    TORCH_CHECK(x.size(0) <= 4 && grid >= x.size(0) * okc::R16_FS, "r16d: the private rows hold 4 tokens");
    TORCH_CHECK(scratch[2].numel() * 2 >= (int64_t)okc::R16_BYTES + (int64_t)okc::EMAX * okc::ID * 2, "r16d: scratch[2] lacks the private tail");
    unsigned char* r16pv = (unsigned char*)scratch[2].data_ptr() + scratch[2].numel() * 2 - (int64_t)okc::R16_BYTES;
    p.ybuf = (float*)(r16pv + okc::R16_YOFF);
    p.ss = p.ss1 = (float*)(r16pv + okc::R16_KOFF);
    p.ctr = (int*)r16pv + okc::R16_CTR;
  }
"""

_R16D_TWIN = r"""  if (@@COND@@) {  // r16d same-codegen twin: the base grid barrier (TINY-private counter block, base generation protocol)
    int* const bc = p.ctr + 4 + (S.gen & 1);
    if (tid == 0) {
      __threadfence();
      const int old = atomicAdd(bc, 1);
      if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
      if ((int)blockIdx.x < p.M * R16_FS) while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
    }
    CTA_SYNC();
  }
"""

_R16D_ITEM0 = r"""  if ((int)blockIdx.x >= p.M * R16_FS) return;
  const int tk = (int)blockIdx.x / R16_FS, sg = (int)blockIdx.x - tk * R16_FS;
"""
_R16D_ITEME = r"""  int k16;
  {  // r16d E: final item k16 of this CTA: the CTAs with one phase-B unit fewer (they finish first) take items 0.., then the others
    const int G = (int)gridDim.x, c = (int)blockIdx.x, nb = S.njS * RB_B;
    const int qb = muldiv_g(nb, 1, G, __fdividef(1.f, (float)G));  // floor(nb / G): units of a short CTA
    const int lb = S.wsr[2] - c * qb;                                 // long CTAs (qb + 1 units) before c
    k16 = (S.wsr[3] - S.wsr[2] == qb) ? c - lb : (G - (nb - G * qb)) + lb;
  }
  if (k16 >= p.M * R16_FS) return;
  const int tk = k16 / R16_FS, sg = k16 - tk * R16_FS;
"""

_R16D_CWSBL = r"""  if (TIDX < OKC_WSBLK * 32) {  // r16d c: the WSBL units' per-unit partials, formed right away by the compute warp that ran the unit's
    // own epilogue (from the same cross-warp sums S.red it just read; the sums of squares of those epilogues went to the sink)
    const int lane = TIDX & 31, warp = TIDX >> 5, nA = S.hof[3], N = nA + S.hof[7], nB = N - nA;
    if (warp < nB) {
      float* const r16p = reinterpret_cast<float*>(r16_priv(p) + R16_POFF);
      const int gg = lane >> 2, tt = lane & 3, q = N - 1 - warp;
      const int v = hof_unit(S, 1, nB - 1 - warp);
      const int j = S.njF + v / RB_B, rb = v - (v / RB_B) * RB_B, bb = q % NRB;
      float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        const float4 t4 = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
        sm.x += t4.x; sm.y += t4.y; sm.z += t4.z; sm.w += t4.w;
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
        float sq = v0f * v0f + v1f * v1f;
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (tk < cnt && gg < R16_FS) r16_st(r16p + ((size_t)gg * R16_NPOS + st + tk) * RB_B + rb, sq);
      }
    }
  }
"""

_R16D_CTAIL = r"""#if OKC_WS && OKC_FINPF && OKC_FINCOL && !OKC_WGS && !OKC_SPLIT
@@CWSBL@@  WS_JOIN();  // r16d: this CTA's epilogue stores are issued; no grid barrier: the final CTAs poll the data they read
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
@@TWIN@@  r16_fin(p, S, tid);
#else
#error "r16d: TINY-style tail only"
#endif
  DBG(5);
}
"""


def _r16d_edit(s, kind):
    rp = _r16d_rp
    assert s.count("(void)jj; (void)pp;") == 1, "r16d: the base TINY source must carry NOPF"
    if kind == "plc":
        return s
    import re as _re
    m = _re.match(r"W(\d)", kind)
    assert m, kind
    FS = int(m.group(1))
    fl = kind[2:]
    twin = "x" if "x" in fl else "o" if "o" in fl else ""
    dev = _R16D_DEV.replace("@@PFIN@@", _R16D_PFIN if "P" in fl else "").replace("@@FS@@", str(FS))
    dev = dev.replace("@@ITEM@@", _R16D_ITEME if "E" in fl else _R16D_ITEM0)
    dev = dev.replace("@@BRK@@", " && p.zero != 0u" if "B" in fl else "")  # B: positive control (one poll round, no wait; never shipped)
    dev = dev.replace("@@SKIP@@", "  if (p.zero == 0u) return;  // r16d probe S: the final skipped (timing twin, never shipped)\n" if "S" in fl else "")
    s = rp(s, "__device__ void ws_helper(", dev + "__device__ void ws_helper(")

    def hlp(h):
        I = "        "
        h = rp(h, I + "if (tk < cnt) {\n" + I + "  st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n" + I
               + "  st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n" + I + "}\n",
               I + "if (tk < cnt) {  // r16d: strong relaxed stores (p.ybuf = TINY's private y rows)\n" + I
               + "  r16_st(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n" + I
               + "  r16_st(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n" + I + "}\n")
        h = rp(h, "\n" + I + "if (gg == 0 && tk < cnt) r10t_ssadd(&ss_cur()[st + tk], sq);\n",
               "\n" + I + "if (tk < cnt && gg < R16_FS) r16_st(r16p + ((size_t)gg * R16_NPOS + st + tk) * RB_B + rb, sq);  // r16d: the unit's partial, copy gg\n")
        h = rp(h, "  const int gg = lane >> 2, tt = lane & 3;\n",
               "  const int gg = lane >> 2, tt = lane & 3;\n  float* const r16p = reinterpret_cast<float*>(r16_priv(p) + R16_POFF);  // r16d: [copy][route][unit]\n")
        if "c" not in fl and "u" not in fl:
            h = rp(h, _R16D_HEND, _R16D_HEND[:-2] + _R16D_WSBL + "}\n")
        return h
    s = _r16d_slice(s, _R16D_HLP_A, _R16D_HLP_B, hlp)

    def pol(h):
        h = rp(h, _R16D_PSTART, _R16D_PSTART_NEW)
        h = rp(h, "          atomicAdd(p.doneA + jj, b - a);\n", "          atomicAdd(r16dA + jj, b - a);\n", 1)
        h = rp(h, "        atomicAdd(p.doneA + jj, b - a);\n", "        atomicAdd(r16dA + jj, b - a);\n", 1)
        h = rp(h, "        while (ld_acquire(p.doneA + jj) < RB_A) __nanosleep(32);\n", "        while (ld_acquire(r16dA + jj) < RB_A) __nanosleep(32);\n", 2)
        assert "p.doneA" not in h
        return h
    s = _r16d_slice(s, _R16D_POL_A, _R16D_POL_B, pol)
    a = s.index(_R16D_END_A)
    b = s.index(_R16D_END_B, a)
    tw = _R16D_TWIN.replace("@@COND@@", "p.zero != 0u" if twin == "x" else "p.zero == 0u") if twin else ""
    ct = _R16D_CTAIL.replace("@@TWIN@@", tw).replace("@@CWSBL@@", _R16D_CWSBL if "c" in fl else "")
    if "P" in fl:
        ct = _r16d_rp(ct, "  r16_fin(p, S, tid);\n", "  r16_finp(p, S, tid);\n")
    s = s[:a] + ct + s[b + len(_R16D_END_B):]
    s = rp(s, _R16D_DBG_OLD, _R16D_DBG_OLD + _R16D_HOST)
    if "r" in fl:
        s = rp(s, _R16D_STK_OLD, _R16D_STK_NEW)
    if "u" in fl:  # the WSBL epilogue (compute warps, after the unit loop) stores the unit's partial copies itself
        s = rp(s, "      if (gg == 0 && tk < cnt) r10t_ssadd(&ss_cur()[st + tk], sq2);\n",
               "      if (tk < cnt && gg < R16_FS) r16_st(reinterpret_cast<float*>(r16_priv(p) + R16_POFF) + ((size_t)gg * R16_NPOS + st + tk) * RB_B + rb, sq2);  // r16d u: the unit's partial, copy gg\n")
    return s


_r16d_d = TINY_DEFINES
_r16d_at = [d for d in _r16d_d if d.startswith("@")]
assert len(_r16d_at) == 1, _r16d_d
_r16d_k = _r16d_at[0][1:] + "_r16d" + _R16D_TAG
assert _r16d_k not in ALT_SRCS, _r16d_k
ALT_SRCS[_r16d_k] = _r16d_edit(ALT_SRCS[_r16d_at[0][1:]], _R16D_KIND) + "\n// r16d " + _R16D_TAG + " " + _R16D_KIND + "\n"
_R4X_PFX[ALT_SRCS[_r16d_k]] = "okc_moe_r16d" + _R16D_TAG + "_"
TINY_DEFINES = tuple("@" + _r16d_k if d == _r16d_at[0] else d for d in _r16d_d)
if _R16D_KIND != "plc":
    _r16d_parent_scratch = _make_scratch

    def _make_scratch(dev):
        # r16d: scratch[2] + the TINY-private tail (R16_BYTES): 2 KB of int words (zero), 32 y rows of HIDDEN fp32 and FS x 32 x 256
        # fp32 per-unit sum-of-squares partials (all 0xFFFFFFFF, the sentinel), the EMAX-float ss sink (zero). Every other build
        # indexes scratch[2] from its start and never reaches the tail.
        import re as _re
        fs = int(_re.match(r"W(\d)", _R16D_KIND).group(1))
        s = _r16d_parent_scratch(dev)
        iw = torch.zeros(1024, dtype=torch.bfloat16, device=dev)
        sen = torch.full((32 * HIDDEN + fs * 32 * 256,), -1, dtype=torch.int32, device=dev).view(torch.bfloat16)
        k = torch.zeros(2 * EMAX, dtype=torch.bfloat16, device=dev)
        s[2] = torch.cat([s[2], iw, sen, k])
        return s
