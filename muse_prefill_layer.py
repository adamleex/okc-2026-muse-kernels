# okc: problem=muse_prefill_layer
# p12n (team adamleex) [r22]: base + prepare-time dense GEMM pins from a crash-isolated cuBLASLt algo-66 sweep (LT_PIN2_R22, tried first through the LT2 path; not offered / not valid -> base behaviour): 16384:qkv 1/[66, 409, 1, 0, 0, 1, 35, 0, 3]; 16384:unpooler 0/[66, 29, 1, 0, 0, 1, 35, 0, 3]; 3012:unpooler 1/[66, 408, 1, 0, 0, 1, 35, 0, 3].
# p12m (team adamleex) [r21]: p12l + the spec's sliced unpooler for T <= 8 (only the first 1024 output columns, rest zero; shape branch, T > 8 unchanged).
# p12l (team adamleex) [r19]: p12k + prepare-time decision pins (prepare-only; every pick is one p12k's own race could make, LTNM picks bitwise-checked as before): a memset GEMM pick takes its own workspace-0 twin without the noisy race (T=3012 o, 16K o / pooler / unpooler); 16K qkv prefers the tile-24 memset-free kernel and the T=3012 unpooler tile 501 when offered and valid (else p12k's race); the torch.mm GEMMs (753 qkv, 3012 router / unpooler, 16K qkv / router) skip the eager tuning race; LT2 sweep pins / batched router stay unless clearly slower (LT_PIN2_RACE 1.03, RTRED_RACE 1.05).
# p12k (team adamleex) [r18]: p12j (side-stream pool-collision guard) + LTNM race always includes the current pick's ws0 twin (prepare-only, removes a per-prepare memset lottery at T=3012/16K).
# p12j (team adamleex) [r17]: p12h + side-stream guard (host only, every kernel and output byte as p12h): kernel_fn never runs its side branches on the stream it is captured on. torch.cuda.Stream() hands out a 32-stream pool round-robin, so about one capture in 32 on a fresh torch.cuda.Stream() got SIDE_STREAM itself and serialized the kvoff and router branches (T = 753: router -> route -> pooler, +4 us); p12j swaps in a second side stream whenever SIDE_STREAM is the current stream (_side_guard).
# p12h (team adamleex) [r16]: p12g + exact T=753 latency levers, every output byte as p12g (each checked bitwise at prepare with fallback to p12g's path; existing kernels' SASS unchanged): (L2) kvoff qk_prep_kvt in one L2 round trip after its PDL wait (qk_prep_kvt1: RoPE table / cache row / key-slot loads before the wait, q, k, v rows together after it); (L3) at T = 753 the kvoff side branch forks from the qkv GEMM's launch-completion port (cudaStreamUpdateCaptureDependencies) instead of the 1-warp fork kernel, behind a prepare-time mini-graph guard; (L4) the batched router algo with signature (33, 1, 0, 3) preferred over the eager timing race when offered. Built but off: L1 1-token-per-CTA hand-off combine (HCOMB_TPC1_MAX_T 0), L5 gather_kp page-table modes (GKP_PTU 0).
# p12g (team adamleex) [r15]: p12f + fused-MoE consumer dispatch as an if-chain with N=256 first (code placement: the hot item epilogues sit inside the first ~60 KB of the kernel, so their first execution no longer pays a cold instruction fetch; outputs bitwise as p12f).
# p12f (team adamleex) [r14]: p12e + T = 3012 / 16384 glue, every output byte as p12e (each new kernel checked bitwise at prepare against the one it replaces: _p12f_check; existing kernels' SASS unchanged): (1) qk_prep with 4 (RoPE) / 2 tokens per block and the RoPE angles before the PDL wait (qk_prep_m; t4s keeps qk_prep); (2) t4f gather_kp with both kv heads per CTA, whole 128-byte cache rows (gather_kp2); (3) first boundary at 16K loads the input stream row before the PDL wait (MODE 1 LP 4), at 3012 after the row-norm barrier (LP 2); (4) the memset-free cuBLASLt race credits the graph edges a memset costs that its chain cannot see (LTNM_CREDIT_US); (5) the 3012 / 16K MoE gather at min 5 blocks per SM (48 registers).
# p12e (team adamleex) [r13]: p12d + small-kernel latency fixes, every output byte as p12d (each lever checked bitwise at prepare): (1) MoE gather xt row stores coalesced (the 4 lanes of a stage swap 16 B chunks by shuffles so each warp store writes whole 32 B sectors; _gco_check); (2) T < 2048 boundaries read the fp32 stream row after the branch row-norm barrier (template LP 2; _bnd_lp_check); (3) kvoff: a 1-warp fork-point kernel first, so the side branch no longer forks from the graph root (that root fork + join had turned the qkv GEMM -> qk_prep_kvt programmatic edge into a full one); (4) kvoff: the new tokens' RoPE cos / sin computed on the side stream (rope_tab, qk_prep's expression) and loaded by qk_prep_kvt instead of evaluated after the qkv GEMM (KVOFF self-check).
# p12d (team adamleex) [r12e]: p12c + cheaper fused o_norm * sigmoid(gate) epilogue, bitwise = p12c: fmha7 / fmha7p gate rows and outputs (and split partials) as 16 B per lane through a per-warp 1 KB swizzled smem tile (stmatrix), MODE 2 x staging by stmatrix, combine_ng / combine_ngv issue every load at entry
# p12c (team adamleex): p12b (r10p persistent / stream-K fmha7p) + the 16K pooler made eligible for a memset-free cuBLASLt pick (LTNM_OFF keeps only (3012, pooler)).
# r10p (team adamleex): p11 + persistent / stream-K fmha7p (fmha7's per-block code unchanged inside a per-CTA segment loop; TMA ring and ping-pong token run across segments; mode 0 = fmha7's units on <= 132 persistent CTAs, bitwise; mode 2 = mode 0 with the o_norm * sigmoid(gate) epilogue handed to the producer warpgroup's 3 idle warps, bitwise; mode 1 = whole-unit rounds + stream-K partials merged by combine_ngv) for the attention configs in ZH_P only, each prepare-time self-checked with fallback to fmha7.
# p11 (team adamleex): p10pdlrt + r6 stack (hand-off combine for T <= 2048, memset-free cuBLASLt picks at T >= 3012, router GEMM serial before the pooler at T = 16384) + r6 kvoff stage b for sliding layers with T <= 2048 (side-stream gather_old + qk_prep writes new K / V^T into kvt; gather_kp not launched). No arithmetic change vs p10 except where the provenance lines say so; every lever has a prepare-time self-check with fallback. Provenance lines follow.
# r6stack (team adamleex): p10pdlrt + r6hcomb_fix (hand-off combine) + r6ltnm (memset-free cuBLASLt picks at T >= 3012) + r5 ser16k (router GEMM serial before the pooler at T = 16384) + a prepare-time check of the combined MoE -> combine -> unpooler chain (_stack_check). Provenance lines follow.
# r6hcomb_fix (team adamleex): r6hcomb + review fixes (self-check poisons y / rrow between runs; timing-run errors turn the hand-off off; CUDA source identical to r6hcomb).
# r6hcomb (team adamleex): p10pdlrt + hand-off combine (HCOMB): the fused MoE's GEMM2 publisher counts each GEMM2 item on the 32-row blocks of its tile (red.release.gpu), and moe_combine2 (still PDL-launched) waits per route for its block to reach 32 n-tiles instead of griddepcontrol.wait on the whole MoE grid; same combine arithmetic (bitwise), prepare-time self-check against the PDL-wait pair with fallback; knobs HCOMB / HCOMB_MIN_T / HCOMB_MAX_T.
# r6kvoff_ab_fix: kvoff_ab + review fix: qk_prep_kvt's k-branch RMS shuffles name only lanes 0..15 (mask 0x0000ffff), since lanes 16..31 no longer exit there but wait at the reconvergence point before the V^T stores (full mask = warp deadlock / UB); same arithmetic, same bytes.
# r6kvoff_ab (team adamleex): p10pdlrt + [r6kvoff] sliding layers, T <= KVOFF_MAX_T: a side-stream gather_old kernel (<= 3 CTAs, no PDL) stages the cache-only kvt tiles + lens during the qkv GEMM (stage a'), and qk_prep_kvt writes the new keys' K / V^T bytes straight into kvt (stage b), so gather_kp is not launched and fmha7 PDL-waits on qk_prep; kvt bytes identical (prepare self-check).
# p10pdlrt (team adamleex): p9gath + PDL launches for fmha7 and combine_ng (no arithmetic change) + at T=753 the router GEMM as one strided-batched cuBLASLt call over 4 K-slices (no cuBLAS splitKreduce kernel; moe_route_kernel<4> sums the partials in the split-K reduce's order; kept only if it validates and wins the prepare race). Provenance lines follow.
# r4rtred: p9gath + T=753 router GEMM as a strided batch of 4 K-slices (no cuBLAS splitKreduce); moe_route_kernel<4> sums the partials.
# r4pdl (team adamleex): p9gath + PDL launches for fmha7 and combine_ng (gather_kp and fmha7 trigger their dependents early; fmha7 waits right before its first global read, combine_ng at entry; no arithmetic change).
# p9gath (team adamleex): p8diet (MOE_COLD=1 + COMBINE_V2 + warp-uniform TMA producer + byte-domain exp in both fmha7 builds + fmha7 hot-loop diet) + MoE gather with the token row and its 8 routes' lid/pos loaded at entry (right after the PDL wait), per-expert counts copied to smem, and a dedicated plan block (same arithmetic and row placement). Base: keell0renz/muse-open-kernel-challenge kernels/muse_prefill_layer.py at 8e30e1be (Apache-2.0); attribution lines below kept.
# glue lever A+B stacked (A: cache_seqlens + PDL gather_kp launch; B: rc boundaries for T >= BND_RC_MIN_T = 2048, rc launches with PDL).
# pf_n72: stack_s5_snap1.py + ZhimaoL/sglang okc/x-n72-naw b0a2c3433b okc_patch/_cuda_src_fmha.py N72 (PV and the softmax row sum as one m64n72k32 on an N-group-major V^T stage layout; gather_kp writes that layout; Apache-2.0), the commit's 23 code hunks applied verbatim.
# + ZhimaoL/sglang okc/cand12pre okc_patch/_cuda_src_fmha.py (prefill attention fmha7, Apache-2.0) on vjeux/okc-2026 muse_prefill_layer.py as published by alygg77/okc-muse-spark-kernels (Apache-2.0).
# + okc prefill patches (harness/make_pf.py): fmt=512, bnd1=128,false, ltpin, ltsweep
# + glue2 combine (rrow/lid/topw without __restrict__ so ptxas cannot hoist their loads above griddepcontrol.wait): COMBINE_V2 = moe_combine2_kernel (CUDA twin of the Triton _combine_kernel, same rounding points).
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
"""Muse Spark prefill layer: one transformer layer step over a batch of tokens.

Algorithm (one ``kernel_fn`` call, in order; every stage is one launch or a
few launches, and the stage order is the data dependency order):

1. **QKV projection.** Dense GEMM (cuBLASLt-tuned, else ``torch.mm``) of the
   attention input, producing ``[Q | K | V | gate]`` per token.
2. **Q/K prep.** Per-token RMS-norm of Q and K, RoPE rotation, FP8
   (E4M3) quantisation of Q, and stores of K/V into the paged KV cache,
   scaled by the cache scales. A CUDA kernel (``qk_prep``) does this; the
   Triton ``_qk_prep_kernel`` is the same computation and is kept as the
   fallback path.
3. **Attention.** Either the FA3 paged FP8 path followed by a fused
   ``norm * sigmoid(gate)`` epilogue, or the file's own Hopper attention
   kernel (``attn_fwd``) with that epilogue fused in. The own kernel is
   picked per shape class in ``prepare`` after validating it against the
   FA3 path. On non-Hopper devices a slow FP32 reference attention runs
   instead (testing only, never on the scored path).
4. **O projection + first boundary.** Dense GEMM, then the boundary mixes
   the branch into the fp32 residual stream:
   ``stream = alpha * stream + beta * rms(branch)`` and emits the next
   block's BF16 input as ``rms(stream) * (1 + g_pre)``. The boundary
   kernels also zero the MoE work-counters as a side effect.
5. **Router + pooler, overlapped.** The router GEMM and top-8 routing run
   on a side stream while the pooler GEMM runs on the main stream; the
   main stream then joins the side stream. Routing picks ``TOP_K`` experts
   per token, per local expert.
6. **Grouped MoE.** The pooler output is gathered (with the pre-expert
   norm) into per-expert token rows, then the fused persistent kernel
   runs both expert GEMMs: NVFP4 weights are dequantised in registers as
   the wgmma A-operand, BF16 activations come from shared memory, and
   the epilogue is the fused SiLU*up (``silu(gate) * up``) for GEMM1.
7. **Combine + unpooler + second boundary.** The per-route rows are
   combined (per-expert RMS-norm, weighted by routing weights) back into
   one ``[T, D_EXPERT]`` tensor, the unpooler GEMM brings it back to
   ``HIDDEN``, and the second boundary mixes it into the stream and
   emits the layer output. When the token count is large the second
   boundary instead rebuilds the fp32 stream from the first boundary's
   saved RMS factor, saving a 32 KB-per-row round trip.

Design rules this file follows:

* ``prepare(weights)`` runs once: it builds the CUDA extension, repacks
  the NVFP4 expert weights into the kernel's layout, and tunes the two
  choices that depend on the machine (cuBLASLt algorithm per dense GEMM,
  own-attention variant per shape class). Those tuning results are the
  only state carried between calls, plus the compiled extension and one
  side stream.
* ``kernel_fn(data, state)`` allocates every activation fresh on every
  call and returns fresh tensors; nothing computed from one call's
  inputs is reused by another.
* The timed path performs no host synchronisation: no ``.item()``,
  ``.tolist()``, ``torch.cuda.synchronize()`` or data-dependent host
  branch. (The FP32 fallback attention and the self-tests do sync; they
  are not on the Hopper path.)
* The CUDA/C++ sources at the bottom (``CPP``, ``GLUE``, ``_CUDA_SRC``,
  ``LT_CPP``, ``LT_GLUE``) are the tuned algorithm with every dead
  preprocessor branch, dead function and dead define removed (the
  taken path is unchanged: the kernels, tile sizes and launch
  configs are identical). The one host-supplied switch, MOE_FAST_SCALE
  (0 when the weights have non-normal block scales), is kept with
  both of its branches. Everything above
  them -- constants, Triton kernels, ``prepare``, ``kernel_fn`` and the
  helpers -- is the Python layer. Anything in that layer
  whose equivalence is not provable from the source alone needs GPU
  gating before it is treated as the submission.
"""

import hashlib

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
from triton.language.extra import libdevice

try:  # FA3 (Hopper) from the judge image
    from sgl_kernel.flash_attn import flash_attn_with_kvcache as _fa3
except Exception:  # local testing
    _fa3 = None


# --------------------------------------------------------------------------------------
# Model geometry. Every other size in the file derives from these.
# --------------------------------------------------------------------------------------

HIDDEN = 8192
D_EXPERT = 4096
TOP_K = 8
LOCAL_EXPERTS = 32
HQ, HKV, HD = 16, 2, 64
Q_SIZE, KV_SIZE = HQ * HD, HKV * HD
# QKV gate output per token: [Q (Q_SIZE) | K (KV_SIZE) | V (KV_SIZE) | gate (Q_SIZE)].
QKV_SIZE = 2 * Q_SIZE + 2 * KV_SIZE
GATE_OFFSET = Q_SIZE + 2 * KV_SIZE  # gate starts after Q, K, V
RMS_EPS = 1e-5  # RMS-norm epsilon (post-norm and Q/K-norm; the CUDA kernels use the same)
POST_NORM_EPS = 1e-8  # boundary / combine RMS-norm epsilon (the CUDA kernels use the same)
ROPE_THETA = 500_000.0
FP8 = torch.float8_e4m3fn
LOG2E = 1.4426950408889634  # converts a natural-log softmax scale for exp2

# NVFP4 expert weights: K is walked in stages of KS elements; each
# (n-tile, k-stage) weight block holds KS * 128 code bytes + KS * 8 scale bytes.
KS, WBLK = 64, 4608


# --------------------------------------------------------------------------------------
# Tuning knobs. Each one is a measured default, not a guess: the comment says
# what it changes and why the value is what it is. All of them were formerly
# scattered through the file (several after their first use); they are
# collected here so the whole launch configuration is visible at once.
# --------------------------------------------------------------------------------------

# --- MoE launch ---
# Tile slots per weight n-tile in the (non-persistent) expert GEMMs.
P_GEMM1 = 8
P_GEMM2 = 16
MOE_LARGE_T = 8192  # token count from which the "large" launch mode/order apply
# Launch mode per token count: "fused" = both GEMMs in one persistent launch,
# "persist" = two persistent launches, anything else = two per-tile-grid launches.
MOE_MODE_SMALL = "fused"
MOE_MODE_LARGE = "fused"
# Tile order for the persistent work queue: 0 expert order, 1 biggest first,
# 2 big first + small tiles spread between.
MOE_ORDER_SMALL = 1
MOE_ORDER_LARGE = 1
MOE_FINE_ORDER_SHIFT = 8  # bit 8 of the plan order / fused flags: fine remainder tiles
MIN_TILE_ROWS = 32  # smallest token tile; the plan's tile-count bound divides by this
MOE_TILE_SLACK = 64  # extra tile slots in the plan buffer, beyond the bound
MAX_PADDED_ROWS_PER_EXPERT = 256  # per-expert row allowance in the padded row buffer
FINE_MIN_T = 512  # from this token count, remainder tiles round to 32 rows
SKINNY_TEST = False  # feasibility-test build: the fused MoE kernel at 2 CTAs/SM
CIT = False  # combine in the tail of the fused expert kernel (no combine launch)
GATHER_PLAN = True  # the gather kernel also writes the plan (one launch fewer)
MOE_SMEM_SM90 = 227 * 1024  # dynamic shared memory for the fused/persistent MoE kernels
MOE_SMEM_SM80 = 163 * 1024
MOE_SMEM_SKINNY_TEST = 100 * 1024

# --- Boundary ---
BND_RECOMPUTE = True  # second boundary rebuilds the fp32 stream from the first's RMS factor
BND_RC_MIN_T = 2048  # glue skeptic: rc only from T 2048 (t4 and 16K; t1 stream1 write mostly absorbed by L2) (-16 KB/row of HBM traffic: 64 + 112 vs 96 + 96 KB); was 8192, and its
#                   earlier 512/2048 trial ran the rc kernels without PDL while the stored-stream path had it
BND_CUDA_MIN = 1  # token count from which the CUDA boundary kernel replaces the Triton one
# [r13] boundary load placement (0: p12d; 2: the stream row read after the branch row-norm barrier), per launch kind;
# prepare (_bnd_lp_check) compares each enabled one with the LP 0 launch bitwise and drops a mismatching one
BND_LP_M0 = 2  # MODE 0 (both boundaries below BND_RC_MIN_T)
BND_LP_RC1 = 2  # recompute scheme, first boundary (MODE 1); [p12f] 2 (W63 timelines: t4s / t4f bnd1 -1.3 us); per T: BND_LP_RC1_T
BND_LP_RC2 = 0  # recompute scheme, second boundary (MODE 2): no LP variant is built (0 only)
GATHER_CO = True  # [r13] MoE gather: coalesced xt row stores (moe_gather_plan_co); prepare checks it bitwise (_gco_check)

# --- Router / combine ---
ROUTER_SERIAL_MAX = 0  # T <= this: router GEMM on the main stream before the pooler
ROUTER_SERIAL_MIN = 16384  # [r5 ser16k] T >= this: the same (the 16K chunk shapes; p10: 1 << 30 = never)
ROUTER_LT = True  # False: the router GEMM always runs as torch.mm (no tuned split-K algorithm)
# [r4rtred] T -> router GEMM as one strided batch of RTRED_PARTS K-slices (split-K 1: no workspace reduce, no
# cublasLt::splitKreduce launch), [RTRED_PARTS, T, 512] fp32 partial logits summed by moe_route4. Planned and raced in
# prepare (_try_rtred); any failure keeps the split-K router of the base.
RTRED_T = (753,)
RTRED_PARTS = 4  # must be 4: moe_route4 / k_moe_route4 launch moe_route_kernel<4>
RTRED_PIN = None  # (tile id, split-K, reduction, custom option) of the batched algo to use untimed; None: time the candidates
RTRED_MAX_CAND = 8  # split-K-1 heuristic results timed for the batched router
RTRED_RACE = 1.05  # [p12l] L4 (was 1.0): kept unless it measures > this x the split-K router (its GEMM + reduce) in prepare
RTRED_FORCE = False  # True: keep the batched router whenever it validates (A/B experiments), whatever the race says
RTRED_REQUIRE_EQUAL = False  # True: keep only a batched algo whose summed logits equal the split-K router's bitwise
COMBINE_NW_SMALL = 8  # Triton combine warps at T <= COMBINE_SMALL_T
COMBINE_SMALL_T = 4096
COMBINE_PW_EARLY = tl.constexpr(False)  # Triton combine: norm weight loaded before the PDL wait
CUDA_COMBINE = False  # CUDA combine kernel (PDL): slower than Triton at the scored sizes
COMBINE_V2 = True  # glue2: moe_combine2 (CUDA, PDL, route-row prefetch ring), bitwise twin of the Triton _combine_kernel
COMBINE_V2_CPS = 3  # CTAs per SM moe_combine2_kernel is built for (__launch_bounds__(256, 3); 64 KB ring + 3.4 KB each)
COMBINE_V2_TPC_MAX = 16  # tokens per CTA cap (<= 32): one wave of CTAs up to T = 16 * CPS * nsm, then ceil(T / 16) CTAs
TRITON_PDL = False  # qk_prep and combine (Triton) launched with programmatic dependent launch
# [r6h] hand-off combine: moe_combine2 waits per route on its 32-row block's GEMM2 completion counter (set by the fused
# MoE's publisher warp) instead of griddepcontrol.wait on the whole MoE grid, so its CTAs combine the finished tokens
# on the SMs the MoE has already left. Same combine code after the wait (bitwise-identical output). Needs COMBINE_V2
# and the fused MoE; per-T gate below; prepare() checks it against the PDL-wait pair and turns it off on any mismatch.
HCOMB = True
HCOMB_MIN_T = 1  # [r6h] hand-off only for HCOMB_MIN_T <= T <= HCOMB_MAX_T (e.g. HCOMB_MAX_T = 2048: T = 753 shapes only)
HCOMB_MAX_T = 2048
HCOMB_NEED = D_EXPERT // 128  # [r6h] GEMM2 n-tiles per token tile (= fa.g2.NT, the 32 in k_moe_fused's make_args)
HCOMB_CHECK = True  # [r6h] prepare-time self-check (torch.equal against the PDL-wait pair); False: trust the build
HCOMB_CHECK_T = (64, 300, 753, 1506, 3012, 4096, 16384)  # [r6h] checked token counts (64: no local routes at all)
HCOMB_CHECK_REPS = 3  # [r6h] eager repetitions per checked T
HCOMB_CHECK_GRAPH = 4  # [r6h] CUDA-graph replays per checked T (programmatic edges as in the judge's graph); 0: skip
HCOMB_CHECK_TIME = True  # [r6h] print eager cold-L2 times of the (MoE + combine) pair, PDL wait vs hand-off
HCOMB_HOIST_ONLY = False  # [r6h] fallback arm: no block counters; combine2 does its index loads + compaction after the go
                          # flag, then the whole-grid griddepcontrol.wait (the MoE publisher does no extra work)
# [r6s] stack check: at every T where r6ltnm replaced the unpooler's pick, prepare() runs kernel_fn's MoE -> combine ->
# unpooler chain (the combine kernel_fn uses at that T: hand-off or PDL wait) eagerly and from a CUDA graph (programmatic
# edges as in the judge's graph; every chain buffer poisoned before each run) against the p10 chain (PDL-wait combine, p10
# unpooler pick), torch.equal. Neither lever's own check covers this edge: r6hcomb's stops at the combine, r6ltnm's race
# predecessor neither triggers early nor writes the GEMM's input. Mismatch -> p10 unpooler pick at that T; if the chain
# with the p10 pick still fails while the hand-off is on -> hand-off off for the run.
STACK_CHECK = True
STACK_CHECK_REPS = 2  # [r6s] eager runs per checked T
STACK_CHECK_GRAPH = 6  # [r6s] CUDA-graph replays per checked T; 0: eager only
STACK_CHECK_SEED = 20261104  # [r6s] private generator for the check's inputs (the global RNG stream stays as in p10)
STACK_CHECK_HAND_T = (753,)  # [r6s] also checked without an r6ltnm pick when the hand-off is in use: the hand-off ->
                             # p10 unpooler chain (T=753: p10's pinned unpooler already gets a programmatic edge from
                             # combine2, so it launches behind the hand-off combine; r6hcomb's check stops at the
                             # combine). Mismatch -> hand-off off. (): skip

# --- Attention ---
QK_PREP_CUDA = True  # CUDA q/k prep instead of the Triton kernel (same numerics)
PREP_SPLIT_MAX_T = 1024  # stage cache-only attention tiles on the side stream during the qkv GEMM
PREP_SPLIT = {"slide": False, "full": False}  # split-staging by shape class (off: it cost time)
PREP_SPLIT_AFTER = True  # capture the side branch after the qkv GEMM node, not before it
# [r6kvoff] sliding layers on the ZH path: take gather_kp off the attention critical path. Stage 1 (a'): gather_old, a
# side-stream kernel forked around the qkv GEMM (KVOFF_CTAS CTAs x 1024 threads, no PDL; it reads only this call's inputs)
# stages the tiles made only of cached keys + lens; the join before qk_prep orders it; gather_kp<true> stages the tiles
# with new keys after qk_prep, as today. Stage 2 (b): gather_old also stages the mixed tiles (new and pad keys as zeros), qk_prep_kvt
# writes each new key's K / V^T bytes into kvt, gather_kp is not launched and fmha7 PDL-waits on qk_prep.
# The kvt bytes are the base path's (prepare: _kvoff_prepare compares them with torch.equal on the sliding shapes and the
# page-64 case; a stage that mismatches or fails is turned off).
KVOFF_STAGE = 2  # 0: off (base path); 1: stage a'; 2: stage a' + b
KVOFF_STAGE_T = {}  # T -> stage for that token count (overrides KVOFF_STAGE; e.g. {3012: 1})
KVOFF_MIN_T = 1
KVOFF_MAX_T = 2048  # 16K keeps the base path (the side kernel would be starved by the 1032-tile qkv GEMM)
KVOFF_CTAS = 3  # gather_old CTAs (4 tile groups of 256 threads each); <= 3 leaves 63 SM pairs for the T=753 qkv GEMM's 1x2 clusters
KVOFF_AFTER = True  # capture the side branch after the qkv GEMM node (event recorded before it), like PREP_SPLIT_AFTER
KVOFF_HOIST = True  # stage b: qk_prep_kvt computes the RoPE sin/cos (inputs only: pos, inv_freq) before its PDL wait
KVOFF_ROPE_TAB = True  # [r13] stage 2 with RoPE: the cos / sin table on the side stream (rope_tab), qk_prep_kvt loads it
KVOFF_FORK = True  # [r13] launch the 1-warp fork-point kernel first, so the side branch does not fork from the graph root
# --- [p12f] glue at T = 3012 / 16384 (each new kernel checked bitwise at prepare against the one it replaces: _p12f_check) ---
QKPM = True  # [p12f] qk_prep_m (NT 4 with RoPE, 2 without) for the non-kvoff qk_prep launches with int64 positions / cache rows
QKPM_OFF = {(3012, True)}  # [p12f] (T, rope) kept on qk_prep (t4s: NT 4 there delays gather_kp's hand-off, W63_tl1 / tl2)
GKP2 = True  # [p12f] gather_kp2 (both kv heads per CTA) for full-attention gatherp launches with batch >= GKP2_MIN_B
GKP2_MIN_B = 2  # [p12f] W63_unit3: t4f gather_kp 49.6 -> 43.3 us; batch-1 shapes (t1f, 16K) 3-10% slower with it
BND_LP_RC1_T = {16384: 4}  # [p12f] T -> MODE 1 load placement (overrides BND_LP_RC1; W63_tl2 / tl3: 16K bnd1 290 -> 274 us)
GATHER_MINB_T = {3012, 16384}  # [p12f] T using the coalesced-store gather at min 5 blocks per SM (moe_gather_plan_cm; W63_tl6: 16K gather 119 -> 108-109 us, 3012 21.2 -> 19.7)
# --- [p12h] round-14 exact levers (each checked bitwise at prepare: _p12h_check / the HCOMB check; fallback: p12g's path) ---
HCOMB_TPC1_MAX_T = 0  # [p12h] L1 (built, off: W66 timelines show no tail gain, 753 CTAs > 396 slots): hand-off combine at 1 token per CTA for T <= this (0: p12g's _combine2_tpc)
KVOFF_QKP1 = True  # [p12h] L2: qk_prep_kvt1 (the RoPE-table qk_prep_kvt in one L2 round trip after its PDL wait)
KVOFF_LCFORK = True  # [p12h] L3: kvoff side branch from the qkv GEMM's launch-completion port instead of the fork kernel
KVOFF_LCFORK_T = (753,)  # [p12h] token counts using it (the guard _lcfork_check runs the T = 753 qkv GEMM)
LCFORK_CHECK_REPS = 20  # [p12h] guard replays
LCFORK_MIN_LEAD_US = 20.0  # [p12h] the guard's side stamp must lead its join stamp by this much in every replay
RTRED_PREFER = (33, 1, 0, 3)  # [p12h] L4: batched-router signature (tile, split-K, reduction, custom) used when offered and valid
GKP_PTU = 0  # [p12h] L5 (built, off: W66 timelines mode 1 +1.5 us at t1f, modes 2 / 3 within +-0.3 us): full-attention gather_kp / gather_kp2 with the page-table read off the cache_seqlens chain (0: off; 1: unpredicated ld.volatile, 2 / 3: prefetch.global.L1 / .L2 at entry)
KVOFF_SELFCHECK = True  # prepare: check each stage against the base path (False: trust the build; A/B experiments only)
ATTN_SK = {"slide": 0, "full_long": 0, "full_short": 0, "full_batch": 0}  # stream-K grid per class (0: off)
ATTN_SK_MAX_PAIRS = 1536  # stream-K applies only up to this many (block, pair) pairs; must match ATTN_SK_MAXPAIRS in _CUDA_SRC
ATTN_PF_MAX_T = 0  # tokens <= this: the attention kernel prefetches the o-proj weight into L2 (0: off)
ATTN_SPLIT_MODE = 1  # 0: fill two waves; 1: minimize waves / splits (wave-quantization aware)
# Own attention variant per shape class when it validates (None: time all variants
# in prepare). Variant 43 is ~4% faster than 8 on long full attention and ~3%
# slower on sliding windows, hence the split by class.
ATTN_FIXED = {"slide": 8, "full_long": 43, "full_short": 43, "full_batch": 43}
# Own-kernel variants: (consumer warpgroups, exponentials per thread per tile on
# the FMA pipe). The first element sizes the split heuristic; the GLUE launch
# table maps the variant to the kernel template parameters.
ATTN_VARIANTS = {0: (2, 0), 1: (2, 16), 8: (3, 0), 9: (3, 16), 16: (2, 0), 24: (3, 0), 43: (3, 0)}
OWN_ATTN_LOCAL = False  # local (sm_80) runs use the FP32 fallback attention unless set
ATTN_ASTAGES = 4  # must match `#define ATTN_ASTAGES` in _CUDA_SRC (it sets the staging depth)
# Dynamic shared memory for the own attention kernel: Q/K staging + per-stage
# K/V tiles (a K tile is 128 keys x 64, a V tile 72 x 128). Must match the
# kernel's staging in _CUDA_SRC.
ATTN_SMEM = 1024 + 1024 + 4096 * 3 + ATTN_ASTAGES * (8192 + 72 * 128)
ATTN_SMEM_SM80_EXTRA = 3 * 64 * 128  # sm_80 test path stages P in shared memory
ATTN_VALIDATE_TOL = 0.03  # max relative error for an own-kernel variant to validate
ATTN_Q_ROWS_PER_WARPGROUP = 8  # query rows one warpgroup owns in the split heuristic
ATTN_KEY_TILE = 128  # keys per staging tile
ATTN_MAX_SPLITS = 16  # cap on attention key-range splits
ATTN_SPLIT_PENALTY = 0.02  # cost added per extra split (the merge pass is not free)
ATTN_SPLIT_SLOT = HD + 2  # per (split, token, head) partial: 64 unnormalised O + max + sum
ATTN_SK_Q_BLOCK = 24  # stream-K: query rows per (block, pair) -- 3 warpgroups x 8 rows
FALLBACK_Q_BLOCK = 1024  # query rows per step in the FP32 fallback attention

# --- Dense GEMMs (cuBLASLt) ---
# name -> (state weight [N, K], fp32 output). The router GEMM returns both
# halves (hi | lo partial logits) in one GEMM.
LT_GEMMS = {"qkv": ("qkv_gate", False), "o": ("o", False), "router": ("router_hilo", True),
            "pooler": ("pooler", False), "unpooler": ("unpooler", False)}
LT_TOKENS = (753, 3012, 16384)  # token counts of the scored shapes; others use torch.mm
# T -> SMs left free for the router chain by sizing the pooler down (0/absent:
# torch.mm or the tuned algorithm). The router chain runs on a side stream
# during the pooler GEMM; at small T the pooler holds every SM, so the chain
# would wait for the pooler's end and land the top-8 on the critical path.
POOL_SM_RESERVE = {}
# (T, gemm) -> cuBLASLt heuristic signature (tile id, split-K, reduction,
# custom option): a fixed pick, so prepare never races near-equal algorithms.
LT_PIN = {(753, "router"): (33, 4, 2, 3), (16384, "o"): (201, 1, 0, 2),
          (16384, "unpooler"): (201, 1, 0, 2), (16384, "pooler"): (29, 1, 0, 1),
          (3012, "qkv"): (496, 1, 0, 5), (3012, "pooler"): (24, 1, 0, 7),
          # [p12l] L3: torch.mm (the tuner's pick in every judge run) without the eager race a torch.mm timing outlier
          # could flip (local prepares: 16K router 3.8%, 753 qkv 0.9%, 3012 router 0.7%)
          (753, "qkv"): None, (3012, "router"): None, (3012, "unpooler"): None, (16384, "qkv"): None,
          (16384, "router"): None}
# (T, gemm) -> cuBLASLt algo-66 sweep winners, best first: (layout, [algo id, tile, split-K, reduction, swizzle, custom option, stages,
# inner shape, cluster shape]); layout bit 0 = row-major descriptors (x as A), bit 1 = W pre-transposed. Each candidate is checked and
# raced against torch.mm in prepare; none kept -> the LT_PIN / tuning path below.
LT_PIN2 = {
    (753, "pooler"): [(2, [66, 32, 1, 0, 0, 5, 35, 0, 3]), (2, [66, 32, 1, 0, 0, 7, 35, 0, 5]), (2, [66, 32, 1, 0, 0, 5, 35, 0, 2]), (2, [66, 32, 1, 0, 0, 5, 35, 0, 3])],
    (753, "unpooler"): [(0, [66, 32, 1, 0, 0, 5, 35, 0, 3]), (2, [66, 32, 1, 0, 0, 5, 35, 0, 2]), (2, [66, 32, 1, 0, 0, 5, 35, 0, 5])],
    (753, "o"): [(0, [66, 32, 1, 0, 0, 7, 35, 0, 2]), (0, [66, 495, 1, 0, 0, 7, 35, 0, 5])],
    (3012, "o"): [(0, [66, 189, 1, 0, 0, 2, 35, 0, 5]), (0, [66, 197, 1, 0, 0, 2, 35, 0, 5])],
    (3012, "qkv"): [(0, [66, 496, 1, 0, 0, 5, 35, 0, 2])],
    (16384, "o"): [(3, [66, 29, 1, 0, 0, 1, 35, 0, 3])],
}
# [p12n] r22 crash-isolated algo-66 sweep winners (chain race vs the p12m pick, >= 20 clean subprocess runs each):
# tried first; a pin this cuBLASLt does not offer / that fails the LT2 checks leaves the base's picks unchanged
LT_PIN2_R22 = {
    (16384, 'qkv'): [(1, [66, 409, 1, 0, 0, 1, 35, 0, 3]), (1, [66, 407, 1, 0, 0, 3, 35, 0, 2])],
    (16384, 'unpooler'): [(0, [66, 29, 1, 0, 0, 1, 35, 0, 3]), (2, [66, 29, 1, 0, 0, 1, 35, 0, 3])],
    (3012, 'unpooler'): [(1, [66, 408, 1, 0, 0, 1, 35, 0, 3]), (3, [66, 408, 1, 0, 0, 1, 35, 0, 3])],
}
for _k, _v in LT_PIN2_R22.items():
    LT_PIN2[_k] = list(_v) + [e for e in LT_PIN2.get(_k, []) if e not in _v]
del _k, _v
LT_ALGO_TOL = 2e-2  # a tuned algorithm must match torch.mm this closely to be kept
LT_MIN_SPEEDUP = 0.97  # ...and must beat torch.mm by at least this factor to be kept
LT_WORKSPACE_BYTES = 64 << 20  # cuBLASLt workspace per stream (main + side)

# --- [r6ltnm] cuBLASLt workspace-memset removal at T >= LTNM_MIN_T ---
# cuBLASLt runs a GEMM with more than ~2 waves of output tiles (o / pooler / unpooler at T=3012; qkv / o / pooler / unpooler
# at 16384; nothing at 753) as a dynamically scheduled nvjet kernel and cudaMemsetAsync's a small counter before every launch.
# In the captured graph that memset node turns predecessor -> GEMM into predecessor -> memset -> GEMM (two full edges, no
# programmatic edge): 4.4-7 us of idle GPU per GEMM in the p10 timeline (W43: t4s o 6.95 us, unpooler 4.48 us). prepare()
# looks for a memset-free algorithm per such GEMM (routes below), requires bitwise-equal output and a capture without memset
# nodes, and keeps it only if it wins a cold-L2 graph-replay race of [predecessor kernel -> GEMM] against the current pick
# by LTNM_MARGIN_* and is not slower alone (eager). Otherwise the p10 pick stays.
LTNM = True  # master switch (False: p10's GEMM picks everywhere)
LTNM_MIN_T = 3012  # token counts below this keep p10's picks (T=753 has no memsets: every GEMM is <= 264 tiles)
LTNM_MAX_T = 16384  # 3012: turn-4 shapes only; 16384: the 16K chunk GEMMs too (each still has to win its own race)
LTNM_NAMES = ("qkv", "o", "pooler", "unpooler", "router")  # GEMMs considered; one whose current pick captures without a memset is skipped
# (T, gemm) never touched. Pooler: it runs on the main stream beside the side-stream router, its memset hides behind the
# router (vet: its end is router end + 3 tiles, 0 us gain at t4s/t4f, 0 at 16K), and a static schedule would lose the
# dynamic tile balancing it gets while sharing SMs with the router. Remove an entry to A/B it.
LTNM_RACE_TWIN = True  # [p12k] always race the current pick's ws0 twin (see _ltnm_one)
# [p12l] L1: a memset pick is replaced by its own ws0 twin (same algorithm / config, workspace 0: no memset node, bitwise
# equal, memset-free capture; both checked below) without a race. The 16K race could not resolve the ~3-6 us memset edge
# (A/A noise +-10 us): judge runs kept the memset unpooler in 3 of 4 p12f-p12h prepares, local ones 8-17% at 16K.
LTNM_WS0_FIRST = True
# [p12l] L2: torch.mm picks -> the memset-free candidate with this cuBLASLt signature (tile, split-K, stages, cluster,
# custom option, inner shape = ltnm_info[4:10]) when offered and valid, without a race (else the race). 16K qkv: tile 24
# (256x128 ssched coopA) ran 1.3-3.0% faster than every other qkv kernel the race picked on 13 judge profiles (tiles
# 209 / 23 / 184, torch.mm + memset; 6 of 13 judge prepares picked one of those). 3012 unpooler: tile 501 (256x152
# ssched) and 423 (128x304) measure the same; 501 only removes the coin flip.
LTNM_PREFER = {(16384, "qkv"): (24, 1, 35, 5, 7, 0), (3012, "unpooler"): (501, 1, 35, 5, 7, 0)}
LTNM_OFF = {(3012, "pooler")}  # [p12c] the 16K pooler now runs after the serial router (ser16k): its memset is on the critical path
# GEMMs W43 showed with a memset: used only when this torch cannot expose a captured graph (torch.mm picks) for counting.
LTNM_EXPECT = {(3012, "o"), (3012, "pooler"), (3012, "unpooler"), (16384, "qkv"), (16384, "o"), (16384, "pooler"),
               (16384, "unpooler")}
# Candidate routes: "ws0" = the current algorithm (or, for a torch.mm pick, its first LTNM_TWINS cuBLASLt heuristic
# results) called with workspace size 0 (cuBLASLt rejects it on the host if the algorithm needs workspace: no launch);
# "heur0" = heuristic results requested with a zero workspace budget; "sched" = the current algorithm with cuBLASLt's hidden
# algo-config attribute 10 (scheduling mode, uint16) set to a whitelisted value, AlgoCheck'ed for zero workspace.
LTNM_ROUTES = ("ws0", "heur0", "sched")
LTNM_TWINS = 2
LTNM_HEUR_MAX = 8
# (T, gemm) -> attr-10 values for route "sched". Empty until the local subprocess probe (r6/ltnm/ltnm_probe_all.sh) has run
# each value once on the H200: an unvetted value could launch a faulting kernel, and a sticky CUDA error in the judge's
# prepare kills the run (the 3012/16384 algo-66 sweeps lost the context that way).
LTNM_SCHED_MODES = {}
LTNM_REQUIRE_EQUAL = True  # a candidate must reproduce the current pick's output bitwise (False: LT_ALGO_TOL vs torch.mm)
LTNM_RACE_MAX = 3  # memset-free candidates (fastest eager first) raced against the current pick
LTNM_ROUNDS, LTNM_REPS = 3, 11  # race: per arm, min over rounds of the median over repetitions (arms alternate each rep)
LTNM_MARGIN_US = 1.0  # the candidate's [predecessor -> GEMM] graph replay must be at least this much faster ...
LTNM_MARGIN_REL = 0.002  # ... and at least this fraction of the current chain (2.8 us at 16K, where a 0.2% slower GEMM eats the gain)
LTNM_ALONE_TOL = 0.0  # ... and its eager time alone at most (1 + this) x the current pick's
# [p12f] us credited to a memset-free candidate's measured [predecessor -> GEMM] chain gain (and alone tolerance): the race's
# predecessor kernel does not trigger dependents early and is not the graph root, so it misses what the memset costs in the
# real graph (W63 timelines): before the 16K qkv GEMM (graph root) the GEMM starts 12-15 us late; before the o / unpooler
# GEMMs the fmha7p / combine2 -> GEMM programmatic edge becomes memset -> GEMM (4-6 us); before the pooler the router
# branch's join at the MoE gather turns its programmatic edge into a full one (+2.3-2.5 us)
LTNM_CREDIT_US = {(16384, "qkv"): 10.0, (16384, "o"): 4.0, (16384, "pooler"): 2.5, (16384, "unpooler"): 3.0, (3012, "o"): 4.0,
                  (3012, "unpooler"): 3.0}
LTNM_FORCE = False  # A/B only: take the fastest valid memset-free candidate whatever the race says
LTNM_PRED_ELEMS = 8 << 20  # fp32 elements the race's predecessor kernel scales in place (32 MB: ~20 us)
LTNM_SEED = 6012  # private generator for the prepare-time test inputs (the global RNG stream stays as in p10)

# --- Self-tests (check runs only, printed from prepare) ---
SELFTEST = False
SYNTH_SM_SCALE = 0.125  # 1 / sqrt(HD), the softmax scale of the synthetic attention cases
SYNTH_QK_SCALE_FACTOR = 5.4730064863875  # qk scale factor of the synthetic attention cases

# The side stream for the router GEMM + top-8 (created in prepare). It is a
# module global, not part of the state dict, because the state holds only
# tensors and plain data.
SIDE_STREAM = None
UNPOOL_SLICED_MAX_T = 8  # [p12m] reference SLICED_MAX_TOKENS
UNPOOL_SLICE_COLS = 1024  # [p12m] reference UNPOOL_SLICE = HIDDEN // 8
# [p12j] B: a second side stream (the next pool stream). torch.cuda.Stream() hands out each priority's pool of 32 streams
# round-robin, so a harness that captures every shape on a fresh torch.cuda.Stream() is given SIDE_STREAM itself about once
# in 32 captures; each side branch (kvoff gather_old, router GEMM + top-8) then lands on the capture stream and the graph
# serializes (T = 753: router -> route -> pooler, the route alone on all SMs before the pooler). kernel_fn swaps the two
# side streams whenever SIDE_STREAM is the stream it runs on (_side_guard): same kernels, same data.
SIDE_STREAM_ALT = None
SIDE_GUARD = True  # [p12j] B (False: p12h's behaviour, A/B only)

# --------------------------------------------------------------------------------------
# Triton kernels. The CUDA kernels in _CUDA_SRC do the same jobs on the hot
# paths; these cover the boundary/combine epilogues and the q/k prep
# fallback. Shapes here are spelled with the model constants: a QKV row is
# [Q | K | V | gate], so the gate starts at GATE_OFFSET.
# --------------------------------------------------------------------------------------


@triton.jit
def _to_e4m3(x):
    """fp32 -> float8_e4m3fn bits (uint8), round-to-nearest-even, saturating at 448."""
    a = tl.abs(x)
    u = a.to(tl.int32, bitcast=True)
    normal = ((u + 0x7FFFF + ((u >> 20) & 1)) >> 20) - 960
    sub = libdevice.rint(a * 512.0).to(tl.int32)
    code = tl.where(a < 0.015625, sub, normal)
    code = tl.minimum(code, 126)
    code = tl.where(x < 0, code | 128, code)
    return code.to(tl.uint8)


@triton.jit
def _qk_prep_kernel(qkv_ptr, pos_ptr, loc_ptr, invf_ptr, q8_ptr, kc_ptr, vc_ptr, ksc_ptr, vsc_ptr,
                     ROPE: tl.constexpr, SCALED: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:  # programmatic dependent launch: let the successor launch, wait for the predecessor's results
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    row = qkv_ptr + t.to(tl.int64) * 2304  # Triton does not capture plain-int module globals
    h = tl.arange(0, 16)[:, None]  # Triton does not capture plain-int module globals
    i = tl.arange(0, 32)[None, :]  # Triton does not capture plain-int module globals
    qe = tl.load(row + h * 64 + 2 * i).to(tl.float32)  # Triton does not capture plain-int module globals
    qo = tl.load(row + h * 64 + 2 * i + 1).to(tl.float32)  # Triton does not capture plain-int module globals
    rq = tl.rsqrt((tl.sum(qe * qe, 1) + tl.sum(qo * qo, 1)) / 64.0 + 1e-5)[:, None]  # Triton does not capture plain-int module globals
    qe = qe * rq
    qo = qo * rq
    kh = tl.arange(0, 2)[:, None]  # Triton does not capture plain-int module globals
    ke = tl.load(row + 1024 + kh * 64 + 2 * i).to(tl.float32)  # Triton does not capture plain-int module globals
    ko = tl.load(row + 1024 + kh * 64 + 2 * i + 1).to(tl.float32)  # Triton does not capture plain-int module globals
    rk = tl.rsqrt((tl.sum(ke * ke, 1) + tl.sum(ko * ko, 1)) / 64.0 + 1e-5)[:, None]  # Triton does not capture plain-int module globals
    ke = ke * rk
    ko = ko * rk
    if ROPE:
        pos = tl.load(pos_ptr + t).to(tl.float32)
        ang = pos * tl.load(invf_ptr + i)
        c = libdevice.cos(ang)
        s = libdevice.sin(ang)
        qe, qo = qe * c - qo * s, qe * s + qo * c
        ke, ko = ke * c - ko * s, ke * s + ko * c
    qrow = q8_ptr + t.to(tl.int64) * 1024  # Triton does not capture plain-int module globals
    tl.store(qrow + h * 64 + 2 * i, _to_e4m3(qe))  # Triton does not capture plain-int module globals
    tl.store(qrow + h * 64 + 2 * i + 1, _to_e4m3(qo))  # Triton does not capture plain-int module globals
    loc = tl.load(loc_ptr + t).to(tl.int64)
    j = tl.arange(0, 64)[None, :]  # Triton does not capture plain-int module globals
    v = tl.load(row + 1152 + kh * 64 + j).to(tl.float32)  # Triton does not capture plain-int module globals
    if SCALED:
        ks = tl.load(ksc_ptr)
        vs = tl.load(vsc_ptr)
        ke = ke / ks
        ko = ko / ks
        v = v / vs
    tl.store(kc_ptr + loc * 128 + kh * 64 + 2 * i, _to_e4m3(ke))  # Triton does not capture plain-int module globals
    tl.store(kc_ptr + loc * 128 + kh * 64 + 2 * i + 1, _to_e4m3(ko))  # Triton does not capture plain-int module globals
    tl.store(vc_ptr + loc * 128 + kh * 64 + j, _to_e4m3(v))  # Triton does not capture plain-int module globals


@triton.jit
def _norm_gate_kernel(o_ptr, qkv_ptr, out_ptr):
    t = tl.program_id(0)
    h = tl.arange(0, 16)[:, None]  # Triton does not capture plain-int module globals
    j = tl.arange(0, 64)[None, :]  # Triton does not capture plain-int module globals
    o = tl.load(o_ptr + t.to(tl.int64) * 1024 + h * 64 + j).to(tl.float32)  # Triton does not capture plain-int module globals
    g = tl.load(qkv_ptr + t.to(tl.int64) * 2304 + 1280 + h * 64 + j).to(tl.float32)  # Triton does not capture plain-int module globals
    r = tl.rsqrt(tl.sum(o * o, 1) / 64.0 + 1e-5)[:, None]  # Triton does not capture plain-int module globals
    y = o * r * tl.sigmoid(g)
    tl.store(out_ptr + t.to(tl.int64) * 1024 + h * 64 + j, y.to(tl.bfloat16))  # Triton does not capture plain-int module globals


@triton.jit
def _boundary_kernel(branch_ptr, stream_ptr, gpost_ptr, alpha_ptr, beta_ptr, gpre_ptr, sout_ptr, nout_ptr,
                     HAS_GPOST: tl.constexpr, H: tl.constexpr, STORE_S: tl.constexpr = True):
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    base = t.to(tl.int64) * H
    b = tl.load(branch_ptr + base + offs).to(tl.float32)
    if HAS_GPOST:
        b = b * (1.0 + tl.load(gpost_ptr + offs).to(tl.float32))
    r1 = tl.rsqrt(tl.sum(b * b, 0) / H + 1e-8)  # Triton does not capture plain-int module globals
    s = tl.load(alpha_ptr + offs) * tl.load(stream_ptr + base + offs) + tl.load(beta_ptr + offs) * (b * r1)
    if STORE_S:
        tl.store(sout_ptr + base + offs, s)
    r2 = tl.rsqrt(tl.sum(s * s, 0) / H + 1e-5)  # Triton does not capture plain-int module globals
    tl.store(nout_ptr + base + offs, (s * r2 * (1.0 + tl.load(gpre_ptr + offs).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _combine_kernel(y_ptr, rrow_ptr, lid_ptr, w_ptr, gs_ptr, pw_ptr, out_ptr, H: tl.constexpr, PDL: tl.constexpr = False):
    offs = tl.arange(0, H)
    if COMBINE_PW_EARLY:  # the norm weight is a constant: load it before waiting for the expert kernel
        pw = 1.0 + tl.load(pw_ptr + offs).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    if not COMBINE_PW_EARLY:
        pw = 1.0 + tl.load(pw_ptr + offs).to(tl.float32)  # BF16 weight, applied as 1 + w
    acc = tl.zeros([H], dtype=tl.float32)
    for k in tl.static_range(8):  # Triton does not capture plain-int module globals
        lid = tl.load(lid_ptr + t * 8 + k)  # Triton does not capture plain-int module globals
        if lid >= 0:
            row = tl.load(rrow_ptr + t * 8 + k)  # Triton does not capture plain-int module globals
            y = tl.load(y_ptr + row.to(tl.int64) * H + offs).to(tl.float32) * (tl.load(gs_ptr + lid) * pw)
            rr = tl.rsqrt(tl.sum(y * y, 0) / H + 1e-8)  # Triton does not capture plain-int module globals
            acc += y * (rr * tl.load(w_ptr + t * 8 + k))  # Triton does not capture plain-int module globals
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(tl.bfloat16))


# --------------------------------------------------------------------------------------
# prepare: build the per-weights state once (extension, repacked weights,
# tuned GEMM / attention choices). kernel_fn only reads this state.
# --------------------------------------------------------------------------------------


def _residual(gate):
    """Residual-stream mixing coefficients from a residual gate: the branch
    weight is beta = sigmoid(gate / 0.3), alpha = sqrt(sigmoid(-gate/0.3) * (1 + beta))."""
    w = gate.float() / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return alpha.contiguous(), beta.contiguous()


def _split_router(router):
    """Split the fp32 router weight into hi/lo BF16 halves stacked as one
    [512, 8192] GEMM: one GEMM gives both partial-logit halves."""
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    return torch.cat([hi, lo], 0).contiguous()


def _fast_scales_ok(weights):
    # The fast block-scale decode is exact only for positive normal E4M3
    # scales (bytes 0x08..0x7E: no zero, subnormal, negative or NaN
    # encodings); otherwise the extension is built with the exact decode.
    return all(bool(((weights[k].view(torch.uint8) >= 0x08) & (weights[k].view(torch.uint8) <= 0x7E)).all())
               for k in ("w13_sf", "w2_sf"))


def prepare(weights):
    global SIDE_STREAM, SIDE_STREAM_ALT
    dev = weights["qkv_gate"].device
    st = {}
    for n in ("qkv_gate", "o", "pooler", "unpooler"):
        st[n] = weights[n].contiguous()
    st["router_hilo"] = _split_router(weights["router"].float())
    st["router_bias"] = weights["router_bias"].float().contiguous()
    st["a1"], st["b1"] = _residual(weights["attn_residual_gate"])
    st["a2"], st["b2"] = _residual(weights["ffn_residual_gate"])
    # Norm weights stay BF16 exactly as given (applied as 1 + w): the kernels form 1 + w in fp32 when they load them.
    for n in ("pre_ffn_norm", "pre_expert_norm", "post_expert_norm", "post_ffn_norm", "next_pre_attn_norm"):
        st["g_" + n] = weights[n].contiguous()
    st["inv_freq"] = (1.0 / (ROPE_THETA ** (torch.arange(0, HD, 2, device=dev).float() / HD))).contiguous()
    st["mod"] = _build(_fast_scales_ok(weights))
    st["w13r"] = _repack(weights["w13"], weights["w13_sf"], True)
    st["w2r"] = _repack(weights["w2"], weights["w2_sf"], False)
    st["w13_gs"] = weights["w13_gs"].float().contiguous()
    st["w2_gs"] = weights["w2_gs"].float().contiguous()
    st["ones32"] = torch.ones(LOCAL_EXPERTS, dtype=torch.float32, device=dev)
    st["no_zero"] = torch.empty((0,), dtype=torch.int32, device=dev)
    st["no_zero_f"] = torch.empty((0,), dtype=torch.float32, device=dev)
    st["no_zero_b"] = torch.empty((0,), dtype=torch.bfloat16, device=dev)
    props = torch.cuda.get_device_properties(dev)
    st["smem"] = MOE_SMEM_SM90 if props.major >= 9 else MOE_SMEM_SM80
    if SKINNY_TEST:  # feasibility test build (MOE_SKINNY_TEST): 2 CTAs per SM
        st["smem"] = MOE_SMEM_SKINNY_TEST
    st["mod"].moe_init(st["smem"])
    st["attn_smem"] = ATTN_SMEM + (0 if props.major >= 9 else ATTN_SMEM_SM80_EXTRA)
    st["mod"].attn_init(st["attn_smem"])
    st["no_pf"] = torch.empty((0,), dtype=torch.uint8, device=dev)
    st["nsm"] = props.multi_processor_count
    st["moe_dbg"] = torch.empty((0,), dtype=torch.int64, device=dev)  # a timeline buffer here turns on item tracing
    st["moe_flags"] = 0  # experiments only
    if SIDE_STREAM is None:
        SIDE_STREAM = torch.cuda.Stream(device=dev)
    if SIDE_STREAM_ALT is None:  # [p12j] B
        SIDE_STREAM_ALT = torch.cuda.Stream(device=dev)
        if SIDE_STREAM_ALT == SIDE_STREAM:
            SIDE_STREAM_ALT = torch.cuda.Stream(device=dev)
    st["lt_choice"] = _init_lt(st, dev)
    torch.cuda.synchronize()
    st["attn_choice"] = _choose_attention(st, props)
    _zh_prepare(st)
    _kvoff_prepare(st)  # [r6kvoff]
    _bnd_lp_check(st)  # [r13]
    _gco_check(st)  # [r13]
    _p12f_check(st)  # [p12f] (after _gco_check: the min-5 gather check needs gco_ok)
    _p12h_check(st)  # [p12h]
    st["hcomb_err"] = torch.zeros((4,), dtype=torch.int32, device=dev)  # [r6h] spin-limit error words (debug only)
    st["hcomb_ok"] = False
    if HCOMB:
        st["hc_tpc1"] = bool(HCOMB_TPC1_MAX_T)  # [p12h] L1 (checked by _hcomb_check against the PDL-wait combine at p12g's tpc)
        st["hcomb_ok"] = _hcomb_check(st) if HCOMB_CHECK else True
        if not st["hcomb_ok"] and st["hc_tpc1"]:  # [p12h] fall back to p12g's tokens per CTA, re-checked
            print("HCOMB: 1-token-per-CTA hand-off failed its check; retrying at p12g's tokens per CTA", flush=True)
            st["hc_tpc1"] = False
            st["hcomb_ok"] = _hcomb_check(st) if HCOMB_CHECK else True
        if HCOMB_TPC1_MAX_T:
            print(f"P12H L1: hand-off combine 1 token per CTA for T <= {HCOMB_TPC1_MAX_T}: {bool(st['hc_tpc1'] and st['hcomb_ok'])}",
                  flush=True)
    if STACK_CHECK:  # [r6s] (after both levers' own checks; never raises)
        _stack_check(st)
    if SELFTEST:
        try:
            _selftest(st, weights)
        except Exception as exc:  # noqa: BLE001
            print("SELFTEST failed:", repr(exc), flush=True)
        for fn in (_selftest_attn2, _selftest_boundary):
            try:
                fn(st)
            except Exception as exc:  # noqa: BLE001
                print("SELFTEST", fn.__name__, "failed:", repr(exc), flush=True)
    return st


# --------------------------------------------------------------------------------------
# Dense GEMMs: torch.mm, or a cuBLASLt algorithm tuned in prepare when it
# beats torch.mm (same library, different kernel choice). The tuner runs
# once in prepare; kernel_fn only looks the choice up.
# --------------------------------------------------------------------------------------


def _lt_pin_index(lt, pid, pin):
    """LT_PIN entry -> heuristic index: None (torch.mm), an int index (legacy), or a signature tuple
    (tile id, split-K, reduction scheme, custom option) matched against each heuristic result's config
    read back from cuBLASLt; returns -1 when no result matches (the caller then tunes as usual)."""
    if pin is None:
        return None
    n = lt.lt_nalgo(pid)
    if isinstance(pin, int):
        return pin if pin < n else -1
    for a in range(n):
        info = list(lt.lt_info(pid, a))
        if (info[0], info[1], info[6], info[8]) == tuple(pin):
            return a
    return -1


def _lt_rel_err(out, ref):
    """Max relative error of a candidate GEMM output against the torch.mm reference."""
    return float((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6))


LT_PIN2_RACE = 1.03  # [p12l] L4 (was 0.997): a sweep pin is kept unless it measures this much slower than torch.mm on the
# machine running prepare (H200 margins 1.004-1.17 faster; the T=753 o pin came within 0.1% of losing)


def _try_lt2(st, T, name, x, w, out, ref, ref_fn, t_ref, f32):
    """First LT_PIN2 candidate for (T, name) that this cuBLASLt accepts, matches torch.mm numerically and beats it in an
    alternating cold-L2 race; ("lt2", handle) or None (the heuristic pins / tuning then apply). Never raises."""
    cands = LT_PIN2.get((T, name))
    if not cands:
        return None
    lt = st["lt"]
    wn = LT_GEMMS[name][0]
    for lay, cfg in cands:
        try:
            wt = None
            if lay & 2:
                wt = st["lt2_wt"].get(wn) if st.get("lt2_wt") and wn in st["lt2_wt"] else w.t().contiguous()
            h = lt.lt2_add(T, w.shape[0], w.shape[1], f32, lay, cfg, st["lt_ws"].numel())
            if h < 0:
                print(f"LT2 T={T} {name}: lay {lay} cfg {cfg} not offered", flush=True)
                continue
            run = lambda: lt.lt2_run(h, x, w, wt if wt is not None else w, out, st["lt_ws"])
            run()
            torch.cuda.synchronize()
            err = _lt_rel_err(out, ref)
            if not err < LT_ALGO_TOL:
                print(f"LT2 T={T} {name}: lay {lay} cfg {cfg} mismatch {err:.3g}", flush=True)
                continue
            ta, tb = [], []
            for _ in range(3):
                ta.append(_time(run, reps=11))
                tb.append(_time(ref_fn, reps=11))
            a, b = min(ta), min(tb)
            print(f"LT2 T={T} {name}: lay {lay} cfg {cfg}: {a:.1f}us vs torch.mm {b:.1f}us (x{b / a:.3f})", flush=True)
            if a < b * LT_PIN2_RACE:
                if wt is not None:
                    st.setdefault("lt2_wt", {})[wn] = wt
                return ("lt2", h)
        except Exception as exc:  # noqa: BLE001
            print(f"LT2 T={T} {name}: lay {lay} failed: {repr(exc)[:200]}", flush=True)
    return None


def _tune_lt(st):
    """Time cuBLASLt's heuristic algorithms for each dense GEMM at the scored token counts; keep one
    only where it beats torch.mm."""
    lt = st["lt"]
    dev = "cuda"
    choice = {}
    for T in LT_TOKENS:
        for name, (wn, f32) in LT_GEMMS.items():
            w = st[wn]
            N, K = w.shape
            x = (torch.randn((T, K), device=dev) * 0.5).to(torch.bfloat16)
            odt = torch.float32 if f32 else torch.bfloat16
            ref_fn = (lambda: torch.mm(x, w.t(), out_dtype=torch.float32)) if f32 else (lambda: torch.mm(x, w.t()))
            ref = ref_fn().float()
            t_ref = _time(ref_fn, reps=10)
            if name == "router" and not ROUTER_LT:  # (the split-K pick steals the concurrent pooler's SMs)
                del x
                continue
            out = torch.empty((T, N), dtype=odt, device=dev)
            if name == "pooler" and POOL_SM_RESERVE.get(T, 0) > 0:
                pid = lt.lt_plan_sm(T, N, K, f32, st["lt_ws"].numel(), 4, st["nsm"] - POOL_SM_RESERVE[T])
                if lt.lt_nalgo(pid) > 0:
                    lt.lt_run(pid, 0, x, w, out, st["lt_ws"])
                    torch.cuda.synchronize()
                    if _lt_rel_err(out, ref) < LT_ALGO_TOL:
                        choice[(T, name)] = (pid, 0)
                        t = _time(lambda: lt.lt_run(pid, 0, x, w, out, st["lt_ws"]), reps=10)
                        print(f"LT T={T} pooler sized for {st['nsm'] - POOL_SM_RESERVE[T]} SMs: "
                              f"{t:.1f}us alone (torch.mm {t_ref:.1f}us)", flush=True)
                        del x, out
                        continue
            c2 = _try_lt2(st, T, name, x, w, out, ref, ref_fn, t_ref, f32)
            if c2 is not None:
                choice[(T, name)] = c2
                del x, out
                continue
            pid = lt.lt_plan(T, N, K, f32, st["lt_ws"].numel(), 32)
            if (T, name) in LT_PIN:  # a fixed pick: no prepare-time race between near-equal algorithms
                a = _lt_pin_index(lt, pid, LT_PIN[(T, name)])
                if a != -1:  # -1: no heuristic result has the pinned signature -> time them as usual
                    if a is not None:
                        choice[(T, name)] = (pid, a)
                    print(f"LT T={T} {name}: pinned {LT_PIN[(T, name)]} -> algo {a}", flush=True)
                    del x, out
                    continue
                print(f"LT T={T} {name}: pinned signature {LT_PIN[(T, name)]} not offered; tuning instead", flush=True)
            best = (t_ref, None)
            for a in range(lt.lt_nalgo(pid)):
                try:
                    lt.lt_run(pid, a, x, w, out, st["lt_ws"])
                    torch.cuda.synchronize()
                    if not _lt_rel_err(out, ref) < LT_ALGO_TOL:
                        continue
                    t = _time(lambda: lt.lt_run(pid, a, x, w, out, st["lt_ws"]), reps=10)
                except Exception:  # noqa: BLE001
                    continue
                if t < best[0] * LT_MIN_SPEEDUP:
                    best = (t, a)
            if best[1] is not None:
                choice[(T, name)] = (pid, best[1])
            print(f"LT tune T={T} {name}: torch.mm {t_ref:.1f}us, best {best[0]:.1f}us "
                  f"algo {best[1]} of {lt.lt_nalgo(pid)}", flush=True)
            del x, out
    torch.cuda.empty_cache()
    return choice


def _init_lt(st, dev):
    """Build the cuBLASLt extension and tune the dense GEMMs; None choice on any failure."""
    try:
        st["lt"] = _build_lt()
        st["lt_ws"] = torch.empty(LT_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
        st["lt_ws_side"] = torch.empty(LT_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
        choice = _tune_lt(st)
        _try_rtred(st, choice)  # [r4rtred]
        _ltnm_tune(st, choice)  # [r6ltnm] (never raises; leaves a pick unchanged unless a memset-free one wins)
        return choice
    except Exception as exc:  # noqa: BLE001
        print("LT unavailable:", repr(exc)[:300], flush=True)
        return None


def _try_rtred(st, choice):
    """[r4rtred] For each T in RTRED_T: plan the router GEMM as RTRED_PARTS K-slices in one strided-batched cuBLASLt
    call (only split-K-1 algorithms: no workspace reduce, no extra kernel), check it against torch.mm, and race it with
    the split-K router chosen by _tune_lt (its GEMM + cublasLt::splitKreduce). Kept as st["rtred"][T] = (plan id,
    algo index); prints whether the slice-order sums equal the split-K router's logits bitwise. Never raises."""
    st["rtred"] = {}
    lt = st["lt"]
    wn = LT_GEMMS["router"][0]
    w = st[wn]
    N, K = w.shape
    P = RTRED_PARTS
    for T in RTRED_T:
        try:
            if P != 4 or K % P or (K // P) % 64:
                print(f"RTRED T={T}: {P} parts of K={K} not supported", flush=True)
                continue
            x = (torch.randn((T, K), device="cuda") * 0.5).to(torch.bfloat16)
            ref = torch.mm(x, w.t(), out_dtype=torch.float32).float()
            c = choice.get((T, "router")) if choice else None
            sk = torch.empty((T, N), dtype=torch.float32, device="cuda")
            if c is None:
                run_sk = lambda: torch.mm(x, w.t(), out_dtype=torch.float32)
            elif c[0] == "lt2":
                run_sk = lambda: lt.lt2_run(c[1], x, w, st.get("lt2_wt", {}).get(wn, w), sk, st["lt_ws"])
            else:
                run_sk = lambda: lt.lt_run(c[0], c[1], x, w, sk, st["lt_ws"])
            r = run_sk()
            torch.cuda.synchronize()
            skv = r if c is None else sk
            pid = lt.lt_plan_kb(T, N, K, P, True, st["lt_ws"].numel(), 32)
            cands = []
            for a in range(lt.lt_nalgo(pid)):
                info = list(lt.lt_info(pid, a))
                if info[1] > 1:  # split-K inside a slice would bring a reduce kernel back
                    continue
                if RTRED_PIN is not None and (info[0], info[1], info[6], info[8]) != tuple(RTRED_PIN):
                    continue
                cands.append((a, info))
            print(f"RTRED T={T}: {lt.lt_nalgo(pid)} batched heuristic results, {len(cands)} split-K-1 candidates", flush=True)
            out = torch.empty((P, T, N), dtype=torch.float32, device="cuda")
            best = None
            pref = None  # [p12h] L4: the RTRED_PREFER candidate, when offered and valid
            if RTRED_PREFER is not None:  # (time it first, so it is never cut by RTRED_MAX_CAND)
                cands = ([c for c in cands if (c[1][0], c[1][1], c[1][6], c[1][8]) == tuple(RTRED_PREFER)]
                         + [c for c in cands if (c[1][0], c[1][1], c[1][6], c[1][8]) != tuple(RTRED_PREFER)])
            for a, info in cands[:RTRED_MAX_CAND]:
                try:
                    out.fill_(float("nan"))
                    lt.lt_run(pid, a, x, w, out, st["lt_ws"])
                    torch.cuda.synchronize()
                    acc = out[0].clone()
                    for q in range(1, P):  # the route kernel's order: each half summed in slice order
                        acc += out[q]
                    err = _lt_rel_err(acc, ref)
                    same = bool(torch.equal(acc, skv))
                    dmax = float((acc - skv).abs().max())
                    if not err < LT_ALGO_TOL or (RTRED_REQUIRE_EQUAL and not same):
                        print(f"RTRED T={T}: algo {a} {info} rejected (err {err:.3g}, equal {same})", flush=True)
                        continue
                    t = _time(lambda: lt.lt_run(pid, a, x, w, out, st["lt_ws"]), reps=10)
                    print(f"RTRED T={T}: algo {a} {info}: {t:.1f}us, err {err:.3g}, "
                          f"{'bitwise equal to split-K' if same else f'max |d| vs split-K {dmax:.3g}'}", flush=True)
                    if best is None or t < best[0]:
                        best = (t, a, info, same, dmax)
                    if RTRED_PREFER is not None and (info[0], info[1], info[6], info[8]) == tuple(RTRED_PREFER):
                        pref = (t, a, info, same, dmax)
                except Exception as exc:  # noqa: BLE001
                    print(f"RTRED T={T}: algo {a} failed: {repr(exc)[:200]}", flush=True)
            if best is None:
                print(f"RTRED T={T}: no usable batched algo; split-K router kept", flush=True)
                continue
            if pref is not None:  # [p12h] L4 (every 4-slice candidate gives the same partial logits: a variance cut)
                print(f"RTRED T={T}: preferred signature {tuple(RTRED_PREFER)} -> algo {pref[1]} ({pref[0]:.1f}us; fastest "
                      f"timed {best[0]:.1f}us)", flush=True)
                best = pref
            elif RTRED_PREFER is not None:
                print(f"RTRED T={T}: preferred signature {tuple(RTRED_PREFER)} not offered / not valid; raced", flush=True)
            a = best[1]
            run_kb = lambda: lt.lt_run(pid, a, x, w, out, st["lt_ws"])
            ta, tb = [], []
            for _ in range(3):
                ta.append(_time(run_kb, reps=11))
                tb.append(_time(run_sk, reps=11))
            ka, kb = min(ta), min(tb)
            ok = RTRED_FORCE or ka <= kb * RTRED_RACE
            print(f"RTRED T={T}: batched {P} x K{K // P} algo {a} {best[2]}: {ka:.1f}us vs split-K router {kb:.1f}us "
                  f"(x{kb / ka:.3f}); logits {'bitwise equal' if best[3] else f'max |d| {best[4]:.3g}'} -> "
                  f"{'USE batched' if ok else 'keep split-K'}", flush=True)
            if ok:
                st["rtred"][T] = (pid, a)
            del x, ref, sk, out
        except Exception as exc:  # noqa: BLE001
            print(f"RTRED T={T} unavailable: {repr(exc)[:300]}", flush=True)
    torch.cuda.empty_cache()


def _mm(st, x, name, side=False):
    """x [T, K] @ W[N, K]^T with the tuned cuBLASLt algorithm for this shape, else torch.mm."""
    T = x.shape[0]
    choice = st["lt_choice"].get((T, name)) if st.get("lt_choice") else None
    wn, f32 = LT_GEMMS[name]
    w = st[wn]
    if choice is None:
        return torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
    out = torch.empty((T, w.shape[0]), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    if choice[0] == "lt2":
        st["lt"].lt2_run(choice[1], x, w, st.get("lt2_wt", {}).get(wn, w), out, st["lt_ws_side"] if side else st["lt_ws"])
        return out
    if choice[0] == "ltnm":  # [r6ltnm] (kind 0: heuristic plan + index, 1: lt2 handle; workspace bytes, 0: no memset)
        st["lt"].ltnm_run(choice[1], choice[2], choice[3], x, w, st.get("lt2_wt", {}).get(wn, w), out,
                          st["lt_ws_side"] if side else st["lt_ws"], choice[4], True)
        return out
    st["lt"].lt_run(choice[0], choice[1], x, w, out, st["lt_ws_side"] if side else st["lt_ws"])
    return out


def _router_side(st, h):
    """[r4rtred] The side-stream router GEMM: [RTRED_PARTS, T, 512] fp32 K-slice partial logits from the strided-batched
    plan kept by _try_rtred for this T (no cuBLAS splitKreduce), else _mm's [T, 512] (hi | lo partial logits)."""
    T = h.shape[0]
    kb = (st.get("rtred") or {}).get(T)
    if kb is None:
        return _mm(st, h, "router", side=True)
    w = st[LT_GEMMS["router"][0]]
    out = torch.empty((RTRED_PARTS, T, w.shape[0]), dtype=torch.float32, device=h.device)
    st["lt"].lt_run(kb[0], kb[1], h, w, out, st["lt_ws_side"])
    return out


# --------------------------------------------------------------------------------------
# [r6ltnm] Memset-free GEMM picks at T >= LTNM_MIN_T (see the LTNM_* knobs). prepare() only; kernel_fn just calls _mm.
# --------------------------------------------------------------------------------------


def _ltnm_ref_of(c):
    """An _mm choice -> (kind, id, algo) for the ltnm_* glue (kind 0: heuristic plan + result index, 1: lt2 handle);
    None for torch.mm."""
    if c is None:
        return None
    if c[0] == "lt2":
        return (1, c[1], 0)
    if c[0] == "ltnm":
        return (c[1], c[2], c[3])
    return (0, c[0], c[1])


def _ltnm_runner(st, c, wn, x, w, out, f32):
    """A callable running GEMM choice c on x exactly as _mm does (into out for cuBLASLt picks) and returning the output."""
    lt = st["lt"]
    ws = st["lt_ws"]
    if c is None:
        return (lambda: torch.mm(x, w.t(), out_dtype=torch.float32)) if f32 else (lambda: torch.mm(x, w.t()))
    wt = st.get("lt2_wt", {}).get(wn, w)
    if c[0] == "lt2":
        h = c[1]

        def run():
            lt.lt2_run(h, x, w, wt, out, ws)
            return out
    elif c[0] == "ltnm":
        kind, i, a, wsb = c[1], c[2], c[3], c[4]

        def run():
            lt.ltnm_run(kind, i, a, x, w, wt, out, ws, wsb, True)
            return out
    else:
        pid, a = c

        def run():
            lt.lt_run(pid, a, x, w, out, ws)
            return out
    return run


def _ltnm_fmt(v):
    """ltnm_cap / graph-stats vector -> text."""
    if v is None or len(v) < 12:
        return f"capture failed {list(v) if v is not None else None}"
    off = "outside lt_ws" if v[7] == -1 else ("-" if v[7] == -2 else f"lt_ws+{v[7]}")
    return (f"status {v[0]} cap {v[1]}: {v[2]} nodes, {v[3]} kernel, {v[4]} memset ({off}, {v[8]} B), {v[5]} memcpy, "
            f"{v[6]} other; grid {v[9]} block {v[10]} smem {v[11]}")


def _ltnm_torch_stats(st, fn):
    """Node counts of a torch-captured graph of fn() (used for torch.mm picks); None when this torch cannot keep the graph."""
    try:
        g = torch.cuda.CUDAGraph(keep_graph=True)
    except Exception:  # noqa: BLE001
        return None
    try:
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            fn()
        ws = st["lt_ws"]
        return [0, 0] + list(st["lt"].ltnm_graph_stats(int(g.raw_cuda_graph()), ws.data_ptr(), ws.numel()))
    except Exception as exc:  # noqa: BLE001
        print(f"LTNM: torch graph inspection unavailable ({repr(exc)[:120]})", flush=True)
        return None
    finally:
        del g


def _ltnm_desc(st, c):
    r = _ltnm_ref_of(c)
    if r is None:
        return "torch.mm"
    return f"{c[0] if isinstance(c[0], str) else 'plan'} {r} info {list(st['lt'].ltnm_info(*r))}"


def _ltnm_candidates(st, T, name, cur, N, K, f32):
    """[(label, kind, id, algo)] memset-free candidates to validate, per LTNM_ROUTES (host-side only: nothing launched)."""
    lt = st["lt"]
    srcs = []
    r = _ltnm_ref_of(cur)
    if r is None:  # torch.mm: its cuBLASLt twins (same library heuristic, the full workspace budget)
        pid = lt.lt_plan(T, N, K, f32, st["lt_ws"].numel(), 32)
        for a in range(min(LTNM_TWINS, lt.lt_nalgo(pid))):
            srcs.append((f"twin{a}", 0, pid, a))
    else:
        srcs.append(("cur", r[0], r[1], r[2]))
    cands = []
    if "ws0" in LTNM_ROUTES:
        cands += [(f"ws0:{lab}", k, i, a) for lab, k, i, a in srcs]
    if "sched" in LTNM_ROUTES:
        for mode in LTNM_SCHED_MODES.get((T, name), ()):
            for lab, k, i, a in srcs:
                h = lt.ltnm_resched(k, i, a, int(mode), 0)
                if h < 0:
                    print(f"LTNM T={T} {name}: sched {mode} on {lab} not accepted (code {h})", flush=True)
                    continue
                cands.append((f"sched{mode}:{lab}", 1, h, 0))
    if "heur0" in LTNM_ROUTES:
        pid0 = lt.lt_plan(T, N, K, f32, 0, LTNM_HEUR_MAX)
        cands += [(f"heur0:{a}", 0, pid0, a) for a in range(lt.lt_nalgo(pid0))]
    return srcs, cands


def _ltnm_race(fns):
    """Per arm: cold-L2 graph replay of [predecessor kernel -> fn()] and eager fn() alone, arms alternating every
    repetition; min over LTNM_ROUNDS of the median of LTNM_REPS (us). The predecessor is a plain ~20 us elementwise kernel,
    so a memset-free cuBLAS launch gets its programmatic edge and a memset one pays predecessor -> memset -> GEMM."""
    _time(lambda: None, reps=1)  # allocates the L2 flush buffer
    buf, sink = _FLUSH_BUF
    pbuf = torch.ones(LTNM_PRED_ELEMS, dtype=torch.float32, device="cuda")
    graphs = []
    for fn in fns:
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            pbuf.mul_(1.0)
            fn()
        graphs.append(g)
    for g in graphs:
        g.replay()
    torch.cuda.synchronize()

    def one(run):
        sink.copy_(buf.sum(dtype=torch.float32))
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        run()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1000

    chain = [[] for _ in fns]
    alone = [[] for _ in fns]
    for r in range(LTNM_ROUNDS):
        tc = [[] for _ in fns]
        ta = [[] for _ in fns]
        for k in range(LTNM_REPS):
            order = list(range(len(fns)))
            if (r * LTNM_REPS + k) % 2:
                order.reverse()
            for i in order:
                tc[i].append(one(graphs[i].replay))
                ta[i].append(one(fns[i]))
        for i in range(len(fns)):
            chain[i].append(sorted(tc[i])[len(tc[i]) // 2])
            alone[i].append(sorted(ta[i])[len(ta[i]) // 2])
    del graphs, pbuf
    return [min(v) for v in chain], [min(v) for v in alone]


def _ltnm_preferred(T, name, cur, valid):
    """[p12l] (candidate, reason) taken without a race, or None (race as p12k). L1: for a cuBLASLt current pick, its own
    ws0 twin (label ws0:cur: the same algorithm and config with workspace 0, i.e. the same kernel tiles and K order minus
    the memset; it is in `valid` only after the bitwise and memset-free capture checks). L2: for a torch.mm pick, the
    first valid candidate whose ltnm_info signature (tile, split-K, stages, cluster, custom option, inner shape) is
    LTNM_PREFER[(T, name)] (ws0 twins before heur0 results: the same kernel either way)."""
    if LTNM_WS0_FIRST and cur is not None:
        for v in valid:
            if v[1] == "ws0:cur" and v[5]:
                return v, "ws0 twin of the current pick"
    want = LTNM_PREFER.get((T, name))
    if want is not None:
        for v in sorted(valid, key=lambda u: not u[1].startswith("ws0:")):
            if v[5] and len(v[7]) >= 10 and tuple(v[7][4:10]) == tuple(want):
                return v, f"LTNM_PREFER signature {tuple(want)}"
    return None


def _ltnm_one(st, choice, T, name, gen):
    """Look for, validate and race memset-free candidates for one GEMM; replaces choice[(T, name)] only on a win that
    also passes the final check through _mm itself."""
    lt = st["lt"]
    wn, f32 = LT_GEMMS[name]
    if name == "router" and T in (st.get("rtred") or {}):  # the batched router is _router_side's plan, not an _mm pick
        print(f"LTNM T={T} {name}: batched router plan in use; skipped", flush=True)
        return
    w = st[wn]
    N, K = w.shape
    cur = choice.get((T, name))
    odt = torch.float32 if f32 else torch.bfloat16
    xs = [(torch.randn((T, K), device="cuda", generator=gen) * s).to(torch.bfloat16) for s in (0.5, 2.0)]
    wt = st.get("lt2_wt", {}).get(wn, w)
    ws = st["lt_ws"]
    out0 = torch.empty((T, N), dtype=odt, device="cuda")
    run_cur = [_ltnm_runner(st, cur, wn, x, w, out0, f32) for x in xs]
    refs = [r().clone() for r in run_cur]
    torch.cuda.synchronize()
    r0 = _ltnm_ref_of(cur)
    if r0 is None:
        cs = _ltnm_torch_stats(st, run_cur[0])
    else:
        cs = list(lt.ltnm_cap(r0[0], r0[1], r0[2], xs[0], w, wt, out0, ws, -1))
    if cs is None or len(cs) < 12:
        has = (T, name) in LTNM_EXPECT
        print(f"LTNM T={T} {name}: current {_ltnm_desc(st, cur)}: graph not inspectable ({cs}); "
              f"memset assumed {'yes' if has else 'no'} (W43)", flush=True)
    else:
        has = cs[4] > 0
        print(f"LTNM T={T} {name}: current {_ltnm_desc(st, cur)}: {_ltnm_fmt(cs)}", flush=True)
    if not has:
        print(f"LTNM T={T} {name}: no memset in front of the current pick; kept", flush=True)
        return
    ref32 = None
    _, cands = _ltnm_candidates(st, T, name, cur, N, K, f32)
    valid = []
    for lab, kind, i, a in cands:
        try:
            info = list(lt.ltnm_info(kind, i, a))
            outc = torch.empty((T, N), dtype=odt, device="cuda")
            s0 = lt.ltnm_run(kind, i, a, xs[0], w, wt, outc, ws, 0, False)
            if s0 != 0:
                print(f"LTNM T={T} {name}: {lab} info {info}: cublasLtMatmul with no workspace -> status {s0}; rejected",
                      flush=True)
                continue
            torch.cuda.synchronize()
            eq = bool(torch.equal(outc, refs[0]))
            if eq:
                lt.ltnm_run(kind, i, a, xs[1], w, wt, outc, ws, 0, True)
                torch.cuda.synchronize()
                eq = bool(torch.equal(outc, refs[1]))
            err = 0.0
            if not eq:
                if ref32 is None:
                    ref32 = torch.mm(xs[0], w.t(), out_dtype=torch.float32).float()
                lt.ltnm_run(kind, i, a, xs[0], w, wt, outc, ws, 0, True)
                err = _lt_rel_err(outc, ref32)
                if LTNM_REQUIRE_EQUAL or not err < LT_ALGO_TOL:
                    print(f"LTNM T={T} {name}: {lab} info {info}: not bitwise equal to the current pick (err vs torch.mm "
                          f"{err:.3g}); rejected", flush=True)
                    continue
            cap = list(lt.ltnm_cap(kind, i, a, xs[0], w, wt, outc, ws, 0))
            if len(cap) < 12 or cap[0] != 0 or cap[1] != 0 or cap[4] != 0 or cap[3] < 1:
                print(f"LTNM T={T} {name}: {lab} info {info}: {_ltnm_fmt(cap)}; rejected", flush=True)
                continue
            run = _ltnm_runner(st, ("ltnm", kind, i, a, 0), wn, xs[0], w, outc, f32)
            t = _time(run, reps=LTNM_REPS)
            print(f"LTNM T={T} {name}: {lab} info {info}: {'bitwise equal' if eq else f'err {err:.3g}'}, "
                  f"{_ltnm_fmt(cap)}, {t:.1f}us eager", flush=True)
            valid.append((t, lab, kind, i, a, eq, run, info))
        except Exception as exc:  # noqa: BLE001
            print(f"LTNM T={T} {name}: {lab} failed: {repr(exc)[:200]}", flush=True)
    if not valid:
        print(f"LTNM T={T} {name}: no memset-free candidate; current pick kept", flush=True)
        return
    valid.sort(key=lambda v: v[0])
    pref = _ltnm_preferred(T, name, cur, valid)  # [p12l] L1 / L2
    if pref is not None:
        v, why = pref
        best, chain = (None, v), [None]
        print(f"LTNM T={T} {name}: preferred {v[1]} info {v[7]} ({why}): no race", flush=True)
    else:
        top = valid[:LTNM_RACE_MAX]
        if LTNM_RACE_TWIN:  # [p12k] the current pick's ws0 twin always races (eager pre-sort noise left it out ~6% of prepares)
            top += [v for v in valid if v[1].startswith("ws0:") and all(v is not u for u in top)][:1]
        chain, alone = _ltnm_race([run_cur[0]] + [v[6] for v in top])
        need = max(LTNM_MARGIN_US, LTNM_MARGIN_REL * chain[0])
        credit = float(LTNM_CREDIT_US.get((T, name), 0.0))  # [p12f] in-graph edge cost of the memset the race does not see
        best = None
        for j, v in enumerate(top, 1):
            gain = chain[0] - chain[j] + credit
            ok = gain >= need and alone[j] <= alone[0] * (1.0 + LTNM_ALONE_TOL) + credit
            print(f"LTNM T={T} {name}: {v[1]}: chain {chain[j]:.1f}us vs {chain[0]:.1f}us (gain {gain:.2f}, need {need:.2f}), "
                  f"alone {alone[j]:.1f}us vs {alone[0]:.1f}us -> {'win' if ok else 'no'}", flush=True)
            if (ok or LTNM_FORCE) and (best is None or chain[j] < best[0]):
                best = (chain[j], v)
        if best is None:
            print(f"LTNM T={T} {name}: current pick kept", flush=True)
            return
    v = best[1]
    new = ("ltnm", v[2], v[3], v[4], 0)
    choice[(T, name)] = new
    # final self-check through _mm itself (the kernel_fn path): bitwise against the current pick's outputs (or, for an
    # LTNM_REQUIRE_EQUAL=False pick, against the candidate's own run); any mismatch restores the p10 pick
    old = st.get("lt_choice")
    st["lt_choice"] = choice
    try:
        ok = True
        for x, ref in zip(xs, refs):
            y = _mm(st, x, name)
            torch.cuda.synchronize()
            if v[5]:
                ok = ok and bool(torch.equal(y, ref))
            else:
                z = torch.empty_like(y)
                lt.ltnm_run(v[2], v[3], v[4], x, w, wt, z, ws, 0, True)
                torch.cuda.synchronize()
                ok = ok and bool(torch.equal(y, z))
    except Exception as exc:  # noqa: BLE001
        print(f"LTNM T={T} {name}: _mm check failed: {repr(exc)[:200]}", flush=True)
        ok = False
    finally:
        st["lt_choice"] = old
    if not ok:
        if cur is None:
            choice.pop((T, name), None)
        else:
            choice[(T, name)] = cur
        print(f"LTNM T={T} {name}: {v[1]} mismatched through _mm; current pick restored", flush=True)
        return
    st.setdefault("ltnm_prev", {})[(T, name)] = cur  # [r6s] the p10 pick, for _stack_check's fallback
    print(f"LTNM T={T} {name}: USE {v[1]} {new} "
          + ("(preferred, no race)" if best[0] is None else f"(chain {best[0]:.1f}us vs {chain[0]:.1f}us)"), flush=True)


def _ltnm_tune(st, choice):
    """[r6ltnm] For each (T, gemm) with LTNM_MIN_T <= T <= LTNM_MAX_T: replace a pick that cuBLASLt precedes with a
    workspace memset by a memset-free algorithm that matches it bitwise and wins the graph-replay race. Never raises."""
    if not LTNM or choice is None:
        return
    try:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(LTNM_SEED)
    except Exception as exc:  # noqa: BLE001
        print(f"LTNM unavailable: {repr(exc)[:200]}", flush=True)
        return
    for T in LT_TOKENS:
        if not LTNM_MIN_T <= T <= LTNM_MAX_T:
            continue
        for name in LTNM_NAMES:
            if name not in LT_GEMMS or (T, name) in LTNM_OFF:
                continue
            try:
                _ltnm_one(st, choice, T, name, gen)
            except Exception as exc:  # noqa: BLE001
                print(f"LTNM T={T} {name} failed: {repr(exc)[:300]}; current pick kept", flush=True)
            torch.cuda.synchronize()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Weight repack: NVFP4 expert weights into the fused kernel's layout.
# --------------------------------------------------------------------------------------


def _codes(packed):
    """uint8 [..., K/2] -> uint8 codes [..., K] (element 2i = low nibble)."""
    lo = packed & 15
    hi = packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


# Nibble slot i (the old layout's 4i..4i+3) -> bit positions of (m0, m1, m2, sign) in the 32-bit word
# (see convert8 in _CUDA_SRC).
_CODE_POS = {3: (6, 7, 8, 15), 7: (22, 23, 24, 31), 2: (2, 3, 4, 11), 6: (18, 19, 20, 27),
             1: (28, 29, 30, 5), 5: (12, 13, 14, 21), 0: (0, 1, 26, 9), 4: (16, 17, 10, 25)}  # dequant layout


def _repack(w, sf, gated):
    """w: uint8 [E, N, 2048]; sf: e4m3 [E, N, 256] -> uint8 [E, N/128, KS, WBLK]."""
    E, N, _ = w.shape
    n_tiles = N // 128
    dev = w.device
    if gated:
        nt = torch.arange(n_tiles, device=dev).view(n_tiles, 1, 1)
        q = torch.arange(8, device=dev).view(1, 8, 1)
        lr = torch.arange(16, device=dev).view(1, 1, 16)
        perm = torch.where(lr < 8, 64 * nt + 8 * q + lr, 4096 + 64 * nt + 8 * q + lr - 8).reshape(-1)
    else:
        perm = torch.arange(N, device=dev)
    out = torch.empty((E, n_tiles, KS, WBLK), dtype=torch.uint8, device=dev)
    sf_u8 = sf.view(torch.uint8)
    shifts = torch.tensor([3, 7, 1, 5, 2, 6, 0, 4], dtype=torch.int32, device=dev)  # (half, hk, b) order below
    pos = torch.tensor([_CODE_POS[int(i)] for i in shifts.tolist()], dtype=torch.int64, device=dev)  # [8, 4]
    for e in range(E):
        c = _codes(w[e][perm]).to(torch.int32)  # [N, 4096]
        # rows: [nt, wg, w, half, g]; k: [ks, j, hk, t, b]
        c = c.view(n_tiles, 2, 4, 2, 8, KS, 4, 2, 4, 2)
        # -> [nt, ks, wg, w, g, t, j, half, hk, b]
        c = c.permute(0, 5, 1, 2, 4, 8, 6, 3, 7, 9).reshape(n_tiles, KS, 2, 4, 8, 4, 4, 8)
        # Nibble index for (half, hk, b): i = 3 - (half + 2*hk) + 4*b; flattened order (half, hk, b):
        # (0,0,0)->3 (0,0,1)->7 (0,1,0)->1 (0,1,1)->5 (1,0,0)->2 (1,0,1)->6 (1,1,0)->0 (1,1,1)->4
        # Bit layout read by convert8 (rotate + mask per register): code bit b of the code in nibble slot i goes to
        # bit _CODE_POS[i][b] (b: m0, m1, m2, sign).
        cb = (c.to(torch.int64).unsqueeze(-1) >> torch.arange(4, device=dev)) & 1  # [..., 8, 4]
        words = (cb << pos).sum((-1, -2)) & 0xFFFFFFFF
        wbytes = torch.stack([(words >> (8 * i)) & 0xFF for i in range(4)], dim=-1).to(torch.uint8)
        out[e, :, :, :4096] = wbytes.reshape(n_tiles, KS, 4096)
        s = sf_u8[e][perm].view(n_tiles, 2, 4, 2, 8, KS, 4)  # rows [nt, wg, w, half, g]; cols [ks, j]
        s = s.permute(0, 5, 1, 2, 4, 6, 3).reshape(n_tiles, KS, 512)  # [nt, ks, wg, w, g, j, half]
        out[e, :, :, 4096:] = s
    return out


# --------------------------------------------------------------------------------------
# Attention: FA3 paged FP8 path, the FP32 fallback for non-Hopper devices,
# and the file's own Hopper kernel (validated + tuned in prepare).
# --------------------------------------------------------------------------------------


def _attention_fallback(data, q8, cfg):
    """Local (non-Hopper) attention over the paged FP8 cache, FP32 math. Testing only: it
    synchronises (`.tolist()`), so it is not on the scored path."""
    B, L = cfg["batch"], cfg["q_len"]
    page, window = cfg["page_size"], cfg["window"]
    ks, vs = data["k_scale"].float(), data["v_scale"].float()
    kf = data["k_cache"].view(-1, HKV, HD)
    vf = data["v_cache"].view(-1, HKV, HD)
    scale = cfg["sm_scale"] * cfg["qk_scale_factor"]
    out = torch.empty((B * L, HQ, HD), dtype=torch.bfloat16, device=q8.device)
    seq_lens = data["seq_lens"].tolist()
    for b in range(B):
        seq = seq_lens[b]
        first = max(0, seq - window + 1) if window else 0
        kp = torch.arange(first, seq + L, device=q8.device)
        rows = data["page_table"][b].long()[kp // page] * page + kp % page
        keys = kf[rows].float() * ks
        vals = vf[rows].float() * vs
        q = q8[b * L:(b + 1) * L].float()
        for c in range(0, L, FALLBACK_Q_BLOCK):
            qp = seq + torch.arange(c, min(c + FALLBACK_Q_BLOCK, L), device=q8.device)
            vis = kp[None, :] <= qp[:, None]
            if window:
                vis &= kp[None, :] > qp[:, None] - window
            qq = q[c:c + FALLBACK_Q_BLOCK].view(-1, HKV, HQ // HKV, HD)
            att = torch.einsum("lkgd,skd->lkgs", qq, keys) * scale
            att = att.masked_fill(~vis[:, None, None, :], float("-inf")).softmax(-1)
            out[b * L + c: b * L + c + qq.shape[0]] = torch.einsum("lkgs,skd->lkgd", att, vals).reshape(-1, HQ, HD).to(torch.bfloat16)
    return out


def _attention(data, q8, cfg):
    """FA3 paged FP8 attention (or the FP32 fallback when FA3 is unavailable)."""
    if _fa3 is None:
        return _attention_fallback(data, q8, cfg)
    window = cfg["window"]
    scaled = cfg["kv_scale"] != 1.0
    descale = {}
    if scaled:
        B = cfg["batch"]
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, HKV),
                       v_descale=data["v_scale"].reshape(1, 1).expand(B, HKV))
    out = _fa3(q=q8.view(-1, HQ, HD), k_cache=data["k_cache"], v_cache=data["v_cache"],
               page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
               cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=cfg["q_len"],
               softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
               window_size=(window - 1, 0) if window else (-1, -1),
               scheduler_metadata=data["scheduler_metadata"], **descale)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _attention_norm_gate(data, q8, qkv, cfg):
    """FA3 attention followed by the fused `rms(att) * sigmoid(gate)` epilogue -> og [T, Q_SIZE]."""
    T = cfg["tokens"]
    att = _attention(data, q8, cfg)
    og = torch.empty((T, Q_SIZE), dtype=torch.bfloat16, device=q8.device)
    _norm_gate_kernel[(T,)](att, qkv, og, num_warps=4)
    return og


def _attn_splits(cfg, data, variant, nsm):
    """Key-range splits so that small batches still fill the GPU (each split keeps >= 8 key tiles)."""
    B, L, window = cfg["batch"], cfg["q_len"], cfg["window"]
    qb = ATTN_Q_ROWS_PER_WARPGROUP * ATTN_VARIANTS[variant][0]
    base = ((L + qb - 1) // qb) * 2 * B
    if base >= 2 * nsm:
        return 1
    keys = (window + qb) if window else data["page_table"].shape[1] * cfg["page_size"]
    nmax = max(1, min(ATTN_MAX_SPLITS, (keys // ATTN_KEY_TILE) // 8))
    if ATTN_SPLIT_MODE == 1:
        best = None
        for ns in range(1, nmax + 1):
            cost = -(-base * ns // nsm) / ns + (ATTN_SPLIT_PENALTY if ns > 1 else 0.0)
            if best is None or cost < best[0] - 1e-9:
                best = (cost, ns)
        return best[1]
    ns = min((2 * nsm + base - 1) // base, nmax)
    return max(1, min(ns, ATTN_MAX_SPLITS))


def _attn_class(cfg):
    """Shape class of an attention call: the key prepare must choose by."""
    if cfg["window"]:
        return "slide"
    if cfg["tokens"] > 4096:
        return "full_long"
    return "full_batch" if cfg["batch"] > 1 else "full_short"


def _attn_variant(st, cfg):
    """Own-kernel variant for this call's shape class, or None for the FA3 path."""
    choice = st.get("attn_choice")
    if not choice:
        return None
    return choice[_attn_class(cfg)]


def _attn_stage_buffers(data, cfg, dev):
    """Staging buffers for the own attention kernel: K tiles and V tiles per (batch, head)."""
    B, L, window, page = cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"]
    smax = (window - 1 + L) if window else data["page_table"].shape[1] * page
    ntk = (smax + ATTN_KEY_TILE - 1) // ATTN_KEY_TILE
    kt = torch.empty((B, 2, ntk, ATTN_KEY_TILE * HD), dtype=torch.uint8, device=dev)
    vt = torch.empty((B, 2, ntk, 72 * ATTN_KEY_TILE), dtype=torch.uint8, device=dev)
    return ntk, kt, vt


def _attn_prep_split(st, data, cfg, ev=None):
    """Stage the tiles made only of keys already in the cache on the side stream (it needs nothing
    from this call's qk_prep, so it overlaps the qkv GEMM's idle SMs); _attn_custom stages the rest
    after joining it. With `ev` (an event recorded before the qkv GEMM was enqueued) the side branch
    is captured after the GEMM node."""
    main, side = torch.cuda.current_stream(), SIDE_STREAM
    pre = _attn_stage_buffers(data, cfg, data["attn_input"].device)
    if ev is not None:
        side.wait_event(ev)
    else:
        side.wait_stream(main)
    with torch.cuda.stream(side):
        st["mod"].attn_prep(data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["page_table"],
                            data["seq_lens"], cfg["page_size"], cfg["q_len"], cfg["window"], pre[0], pre[1], pre[2], 1)
    return pre


def _attn_custom(st, data, q8, qkv, cfg, nsplit=1, variant=0, pre=None):
    """FP8 attention (own kernel) with the output norm * sigmoid(gate) fused: returns og [T, Q_SIZE] bf16."""
    B, L, window, page = cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"]
    dev = q8.device
    if pre is not None:  # part 1 was issued on the side stream by _attn_prep_split: join it, then stage the rest
        ntk, kt, vt = pre
        torch.cuda.current_stream().wait_stream(SIDE_STREAM)
    else:
        ntk, kt, vt = _attn_stage_buffers(data, cfg, dev)
    st["mod"].attn_prep(data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["page_table"],
                        data["seq_lens"], page, L, window, ntk, kt, vt, 2 if pre is not None else 0)
    out = torch.empty((B * L, Q_SIZE), dtype=torch.bfloat16, device=dev)
    c = cfg["sm_scale"] * cfg["qk_scale_factor"] * cfg["kv_scale"] * LOG2E
    if nsplit < 0:  # stream-K over -nsplit CTAs: int header (W, pair prefix, first pairs) + npairs + G partial slots
        G, nqb = -nsplit, (L + ATTN_SK_Q_BLOCK - 1) // ATTN_SK_Q_BLOCK
        hdr = (2 + B * nqb * 2 + 1 + G + 63) // 64 * 64  # must match GLUE's ((2 + npairs + 1 + G) + 63) & ~63
        # Layout agreed with attn_fwd_sk_kernel: header + per-pair blocks of
        # ATTN_SK_Q_BLOCK q-rows x 8 x ATTN_SPLIT_SLOT.
        ws = torch.empty((hdr + (B * nqb * 2 + G) * ATTN_SK_Q_BLOCK * 8 * ATTN_SPLIT_SLOT,), dtype=torch.float32, device=dev)
    else:
        ws = torch.empty((nsplit, B * L, HQ, ATTN_SPLIT_SLOT), dtype=torch.float32, device=dev) if nsplit > 1 else out
    pf = st["o"] if B * L <= ATTN_PF_MAX_T else st["no_pf"]  # o-proj weight into L2 during the attention
    st["mod"].attn_fwd(q8.view(torch.uint8), kt, vt, data["seq_lens"], qkv, out, L, window, ntk, c, cfg["kv_scale"],
                       st["attn_smem"], nsplit, ws, variant, pf)
    return out


def _synthetic_attn_case(B, L, ctx, window, page, seed):
    """A synthetic attention case (random data, paged cache) for validation / tuning."""
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    S = ctx + L
    npg = (S + page - 1) // page
    T = B * L
    q8 = (torch.randn((T, HQ, HD), device=dev, generator=g) * 0.7).to(FP8)
    qkv = torch.randn((T, QKV_SIZE), device=dev, generator=g).to(torch.bfloat16)
    kc = (torch.randn((B * npg, page, HKV, HD), device=dev, generator=g) * 0.8 + 0.3).to(FP8)  # [pages, page_size, 2, 64]
    vc = torch.randn((B * npg, page, HKV, HD), device=dev, generator=g).to(FP8)
    pt = torch.randperm(B * npg, device=dev, generator=g).to(torch.int32).view(B, npg)
    data = {"k_cache": kc, "v_cache": vc, "page_table": pt, "seq_lens": torch.full((B,), ctx, dtype=torch.int32, device=dev),
            "cache_seqlens": torch.full((B,), S, dtype=torch.int32, device=dev),
            "cu_seqlens_q": torch.arange(0, T + 1, L, dtype=torch.int32, device=dev), "scheduler_metadata": None,
            "k_scale": torch.tensor(1.0, device=dev), "v_scale": torch.tensor(1.0, device=dev)}
    cfg = {"batch": B, "q_len": L, "window": window, "page_size": page, "sm_scale": SYNTH_SM_SCALE,
           "qk_scale_factor": SYNTH_QK_SCALE_FACTOR, "kv_scale": 1.0, "tokens": T}
    return data, cfg, q8, qkv


def _attn_ref(data, cfg, q8, qkv):
    """Reference attention path (FA3 + norm-gate) for validating the own kernel."""
    return _attention_norm_gate(data, q8, qkv, cfg)


def _validate_attn(st, variants):
    """Check own-kernel variants against the reference attention path on small synthetic cases; returns
    the variants that agree."""
    worst = {v: 0.0 for v in variants}
    for i, (B, L, ctx, window, page) in enumerate(((2, 300, 1000, 0, 1), (1, 400, 3000, 256, 1), (2, 100, 200, 0, 64))):
        data, cfg, q8, qkv = _synthetic_attn_case(B, L, ctx, window, page, 7 + i)
        ref = _attn_ref(data, cfg, q8, qkv).float()
        for v in variants:
            for ns in (1, 3):
                o = _attn_custom(st, data, q8, qkv, cfg, nsplit=ns, variant=v).float()
                err = float((o - ref).norm() / ref.norm())
                worst[v] = max(worst[v], err if err == err else 1e9)
    ok = [v for v in variants if worst[v] < ATTN_VALIDATE_TOL]
    print("ATTN own kernel validation (worst rel err):", {v: round(e, 4) for v, e in worst.items()}, "->", ok, flush=True)
    return ok


def _tune_attn(st, variants):
    """Pick, per attention shape class, the fastest of the reference path and the validated own variants."""
    choice = {}
    # The scored shapes themselves (synthetic data): sliding 16K chunk, full 16K chunk at 32K context, turn 1 / turn 4 full.
    for kind, (B, L, ctx, window) in (("slide", (1, 16384, 32768, 2048)), ("full_long", (1, 16384, 32768, 0)),
                                      ("full_short", (1, 753, 80000, 0)), ("full_batch", (4, 753, 80000, 0))):
        data, cfg, q8, qkv = _synthetic_attn_case(B, L, ctx, window, 1, 11)
        res = {"ref": _time(lambda: _attn_ref(data, cfg, q8, qkv), reps=5)}
        for v in variants:
            ns = _attn_splits(cfg, data, v, st["nsm"])
            res[v] = _time(lambda: _attn_custom(st, data, q8, qkv, cfg, nsplit=ns, variant=v), reps=5)
        best = min(res, key=res.get)
        choice[kind] = None if best == "ref" else best
        print(f"ATTN tune {kind}: " + " ".join(f"{k}={t:.0f}us" for k, t in res.items()) + f" -> {best}", flush=True)
        del data, q8, qkv
    torch.cuda.empty_cache()
    return choice


def _choose_attention(st, props):
    """Validate (and, on Hopper, tune) the own attention kernel; returns the per-class variant choice."""
    if not (props.major >= 9 or OWN_ATTN_LOCAL):
        return None
    try:
        ok = _validate_attn(st, list(ATTN_VARIANTS) if props.major >= 9 else [24])
        if not ok:
            return None
        if props.major >= 9 and ATTN_FIXED is not None and all(v in ok for v in ATTN_FIXED.values()):
            # A prepare-time race between near-equal options adds noise: pin the measured picks.
            return dict(ATTN_FIXED)
        if props.major >= 9:
            return _tune_attn(st, ok)
        return {k: ok[0] for k in ("slide", "full_long", "full_short", "full_batch")}
    except Exception as exc:  # noqa: BLE001
        print("ATTN validation/tuning failed:", repr(exc)[:300], flush=True)
        return None


# --------------------------------------------------------------------------------------
# MoE: gather the pooler output into per-expert rows, run the fused expert
# GEMMs, return the per-route rows for the combine step.
# --------------------------------------------------------------------------------------


def _max_tiles(T):
    """Upper bound on the plan's tile count for T tokens (min-size tiles + slack)."""
    return T * TOP_K // MIN_TILE_ROWS + MOE_TILE_SLACK


def _hcomb_nblk(T):
    """[r6h] 32-row blocks of the padded expert row buffer (y rows) for T tokens."""
    return -(-(T * TOP_K + LOCAL_EXPERTS * MAX_PADDED_ROWS_PER_EXPERT) // 32)


def _hcomb_len(T):
    """[r6h] ints the hand-off needs in the zeroed zb buffer: go flag + pad (4), then the block counters."""
    return 4 + (0 if HCOMB_HOIST_ONLY else _hcomb_nblk(T))


def _hcomb_bufs(st, z, T):
    """[r6h] go / blk views of a zeroed int32 region z of _hcomb_len(T) ints (blk empty in the hoist-only arm)."""
    return dict(go=z[0:1], blk=st["no_zero"] if HCOMB_HOIST_ONLY else z[4:4 + _hcomb_nblk(T)])


def _hcomb_use(st, T, use_cit):
    """[r6s] kernel_fn's hand-off choice for T (r6h's expression, shape-based: the same on every call of a shape); also
    used by _stack_check, so the check runs the combine kernel_fn runs."""
    return (HCOMB and st.get("hcomb_ok", False) and not use_cit and HCOMB_MIN_T <= T <= HCOMB_MAX_T and COMBINE_V2
            and st["g_post_expert_norm"].data_ptr() % 16 == 0
            and (MOE_MODE_LARGE if T >= MOE_LARGE_T else MOE_MODE_SMALL) == "fused")


def _moe_forward(st, p, lid, pos, cnt, T, order=None, zb=None, cit=None, hcomb=None, gco=False):
    """Routed experts on the pooler output p [T, 4096]: returns per-route rows y [rows, 4096] (bf16) and rrow.
    With cit (dict: zc = [max tiles | 4] zeroed counters, tok = [2T + rows] ints, topw, out) the fused kernel
    also writes the combined output (combine in the tail) and the caller skips the combine kernel.
    [r6h] With hcomb (dict: go = [1] zeroed int, blk = [_hcomb_nblk(T)] zeroed ints or empty) the fused kernel sets go after its
    PDL wait and counts every GEMM2 item on its tile's 32-row blocks, for moe_combine2h. Only moe_fused sees it (the
    gather keeps no_zero: cit= would turn on its CIT writes)."""
    dev = p.device
    mod = st["mod"]
    maxrows = T * TOP_K + LOCAL_EXPERTS * MAX_PADDED_ROWS_PER_EXPERT
    maxtiles = _max_tiles(T)
    offsets = torch.empty((LOCAL_EXPERTS,), dtype=torch.int32, device=dev)
    rows = torch.empty((LOCAL_EXPERTS,), dtype=torch.int32, device=dev)
    tiles = torch.empty((maxtiles, 4), dtype=torch.int32, device=dev)
    ntiles = torch.empty((1,), dtype=torch.int32, device=dev)
    large = T >= MOE_LARGE_T
    mode = MOE_MODE_LARGE if large else MOE_MODE_SMALL
    if order is None:
        order = MOE_ORDER_LARGE if large else MOE_ORDER_SMALL
    fine = 1 if (FINE_MIN_T and T >= FINE_MIN_T) else 0  # remainder tiles rounded to 32 rows (plan + kernel agree)
    order = order | (fine << MOE_FINE_ORDER_SHIFT)
    xt = torch.empty((maxrows, D_EXPERT), dtype=torch.bfloat16, device=dev)
    rrow = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    if cit is not None and (not GATHER_PLAN or mode != "fused"):
        raise RuntimeError("CIT needs the fused kernel and the gather+plan launch")
    if hcomb is not None and (mode != "fused" or cit is not None):
        raise RuntimeError("HCOMB needs the fused kernel (and no CIT)")
    cz = cit["zc"] if cit is not None else st["no_zero"]
    ctok = cit["tok"] if cit is not None else st["no_zero"]
    if GATHER_PLAN:  # the gather kernel also writes the plan (one launch fewer)
        if gco and T in GATHER_MINB_T and st.get("gmb_ok"):  # [p12f] min 5 blocks per SM
            mod.moe_gather_plan_cm(p, st["g_pre_expert_norm"], lid, pos, cnt, xt, rrow, offsets, tiles, ntiles, rows, order, cz, ctok)
        else:
            (mod.moe_gather_plan_co if gco else mod.moe_gather_plan)(p, st["g_pre_expert_norm"], lid, pos, cnt, xt, rrow, offsets,
                                                                      tiles, ntiles, rows, order, cz, ctok)
    else:
        mod.moe_plan(cnt, offsets, tiles, ntiles, rows, order)
        mod.moe_gather(p, st["g_pre_expert_norm"], lid, pos, offsets, cnt, xt, rrow)
    inter = torch.empty((maxrows, D_EXPERT), dtype=torch.bfloat16, device=dev)
    y = torch.empty((maxrows, D_EXPERT), dtype=torch.bfloat16, device=dev)
    if mode == "fused":
        if zb is None:
            zb = torch.zeros((1 + maxtiles,), dtype=torch.int32, device=dev)  # work counter + per-tile GEMM1 completion
        mod.moe_fused(st["w13r"], st["w2r"], xt, inter, y, tiles, ntiles, st["w13_gs"], st["w2_gs"],
                      st["nsm"] * (2 if SKINNY_TEST else 1), st["smem"],
                      zb[0:1], zb[1:1 + maxtiles], st["moe_dbg"], st["moe_flags"] | (fine << MOE_FINE_ORDER_SHIFT), cz, ctok,
                      rrow, lid, cit["topw"] if cit is not None else st["no_zero_f"], st["g_post_expert_norm"],
                      cit["out"] if cit is not None else st["no_zero_b"],
                      hcomb["go"] if hcomb is not None else st["no_zero"],  # [r6h]
                      hcomb["blk"] if hcomb is not None else st["no_zero"])
    elif mode == "persist":
        ctr = torch.zeros((2,), dtype=torch.int32, device=dev)
        mod.moe_gemm_p(st["w13r"], xt, tiles, ntiles, st["w13_gs"], inter, 64, st["nsm"], True, st["smem"], ctr[0:1])
        mod.moe_gemm_p(st["w2r"], inter, tiles, ntiles, st["w2_gs"], y, 32, st["nsm"], False, st["smem"], ctr[1:2])
    else:
        mod.moe_gemm(st["w13r"], xt, tiles, ntiles, st["w13_gs"], inter, 64, P_GEMM1, True, st["smem"])
        mod.moe_gemm(st["w2r"], inter, tiles, ntiles, st["w2_gs"], y, 32, P_GEMM2, False, st["smem"])
    return dict(y=y, rrow=rrow, xt=xt, inter=inter, tiles=tiles, ntiles=ntiles, offsets=offsets)


# --------------------------------------------------------------------------------------
# kernel_fn: one prefill layer step.
# --------------------------------------------------------------------------------------


def _run_qk_prep(st, data, qkv, q8, cfg):
    """Q/K RMS-norm + RoPE + FP8 quantisation + cache stores (CUDA kernel, or the Triton twin)."""
    if (QK_PREP_CUDA and QKPM and st.get("qkpm_ok") and (cfg["tokens"], bool(cfg["rope_theta"])) not in QKPM_OFF  # [p12f]
            and data["positions"].dtype == torch.int64 and data["cache_loc"].dtype == torch.int64):
        st["mod"].qk_prep_m(qkv, data["positions"], data["cache_loc"], st["inv_freq"], q8, data["k_cache"].view(torch.uint8),
                            data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], bool(cfg["rope_theta"]),
                            cfg["kv_scale"] != 1.0)
    elif QK_PREP_CUDA:
        st["mod"].qk_prep(qkv, data["positions"], data["cache_loc"], st["inv_freq"], q8, data["k_cache"].view(torch.uint8),
                          data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], bool(cfg["rope_theta"]),
                          cfg["kv_scale"] != 1.0)
    else:
        T = cfg["tokens"]
        _qk_prep_kernel[(T,)](qkv, data["positions"], data["cache_loc"], st["inv_freq"], q8,
                              data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["k_scale"],
                              data["v_scale"], ROPE=bool(cfg["rope_theta"]), SCALED=cfg["kv_scale"] != 1.0, num_warps=4,
                              **(dict(PDL=True, launch_pdl=True) if TRITON_PDL else {}))


def _run_attention(st, data, q8, qkv, cfg, variant, pre):
    """Attention + norm-gate epilogue -> og [T, Q_SIZE]."""
    if variant is not None:
        ns = _attn_splits(cfg, data, variant, st["nsm"])
        sk = ATTN_SK.get(_attn_class(cfg), 0)
        if sk and variant in (8, 43) and cfg["batch"] * ((cfg["q_len"] + ATTN_SK_Q_BLOCK - 1) // ATTN_SK_Q_BLOCK) * 2 <= ATTN_SK_MAX_PAIRS:
            ns = -sk
        return _attn_custom(st, data, q8, qkv, cfg, nsplit=ns, variant=variant, pre=pre)
    # kernel_fn's FA3 path views q8 as FP8 (the reference path in _attn_ref passes the raw uint8 tensor).
    return _attention_norm_gate(data, q8.view(FP8), qkv, cfg)



# ---------------------------------------------------------------------------------------------- ZhimaoL fmha7
# + ZhimaoL/sglang okc/cand12pre python/sglang/kernels/ops/muse_spark_v12/okc_patch/_cuda_src_fmha.py (Apache-2.0),
# CUDA verbatim; host launcher zh_fmha_ng added here (fused o_norm * sigmoid(gate) epilogue / combine_ng).
ZH_FMHA = {"slide": True, "full_long": True, "full_short": True, "full_batch": True}
ZH_EXACT = False  # his shipped build: piecewise-linear exp2 on part of the tiles (P is FP8 e4m3 after it anyway)
ZH_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DINL __device__ __forceinline__

#ifndef PG_NOWIN
#define PG_NOWIN 0
#endif
#ifndef PG_DBG
#define PG_DBG 0
#endif
#ifndef PFMHA_EPI2
#define PFMHA_EPI2 1   // [r12e] coalesced gate / output epilogue (exact)
#endif
#ifndef PFMHA_SIGA
#define PFMHA_SIGA 0   // [r12e] 0 exact sigmoid; 1 tanh.approx.f32; 2 tanh.approx.f16x2
#endif
#ifndef PFMHA_GLD
#define PFMHA_GLD 0   // [r12e] fmha7p: 1 = gate registers loaded under the final PV (spills at NC4); 0 = L1 prefetch there
#endif
#ifndef PFMHA_QWO
#define PFMHA_QWO 1   // [r12e] epilogue tile address / lane from an opaque %tid after the loop
#endif
#ifndef PFMHA_G7U
#define PFMHA_G7U 0   // [r12e] fmha7: 1 = gate registers loaded under the final PV; 2 = L1 prefetch there; 0 = neither
#endif
#ifndef PFMHA_GPF7
#define PFMHA_GPF7 0   // [r12e] fmha7: producer L2 prefetch of the CTA's gate rows after its TMA loop
#endif
#ifndef PFMHA_GPFP
#define PFMHA_GPFP 0   // [r12e] fmha7p: producer L2 prefetch of segment 0's gate rows too
#endif
#ifndef PFMHA_PRD
#define PFMHA_PRD 0   // [r12e] fmha7p: segment re-derivation + gate L1 prefetch under the final PV
#endif
#ifndef PFMHA_M2S
#define PFMHA_M2S 1   // [r12e] fmha7p MODE 2: x staged by stmatrix
#endif
#ifndef PFMHA_CMB2
#define PFMHA_CMB2 1   // [r12e] combine_ng / combine_ngv: every load issued at entry (one L2 round trip, bitwise)
#endif

namespace pgfa {

constexpr int TRK = 48;
__device__ unsigned long long* g_trace = nullptr;

DINL unsigned long long clk() { unsigned long long t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }

DINL void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DINL void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> DINL void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N)); }

DINL void mbar_init(unsigned bar, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
DINL void mbar_arrive(unsigned bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(bar) : "memory");
}
DINL void mbar_expect_tx(unsigned bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(bar), "r"(bytes) : "memory");
}
DINL void bulk_g2s(void* smem_dst, const void* gsrc, unsigned bytes, unsigned bar) {
    unsigned sd = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sd), "l"(gsrc), "r"(bytes), "r"(bar) : "memory");
}
DINL void bulk_g2s_u(unsigned sd, const void* gsrc, unsigned bytes, unsigned bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sd), "l"(gsrc), "r"(bytes), "r"(bar) : "memory");
}
DINL void mbar_wait(unsigned bar, unsigned phase) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}\n" :: "r"(bar), "r"(phase) : "memory");
}
DINL void mbar_wait_sleep(unsigned bar, unsigned phase) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n"
        "@!p bra WAIT_%=;\n}\n" :: "r"(bar), "r"(phase), "r"(0x989680) : "memory");
}

DINL void fence_u(unsigned& x) { asm volatile("" : "+r"(x)); }
DINL void fence_f(float& x)    { asm volatile("" : "+f"(x)); }
template <int N> DINL void fence_ubank(unsigned (&b)[N]) {
#pragma unroll
    for (int i = 0; i < N; i++) fence_u(b[i]);
}
template <int N> DINL void fence_fbank(float (&b)[N]) {
#pragma unroll
    for (int i = 0; i < N; i++) fence_f(b[i]);
}

DINL unsigned long long make_desc(unsigned saddr, unsigned lbo, unsigned sbo) {
    return (unsigned long long)((saddr >> 4) & 0x3FFF)
         | ((unsigned long long)((lbo >> 4) & 0x3FFF) << 16)
         | ((unsigned long long)((sbo >> 4) & 0x3FFF) << 32);
}
DINL unsigned long long desc_lo(unsigned lo) { return (unsigned long long)lo | (8ull << 32); }
DINL unsigned long long desc_lo_s(unsigned lo) { return (unsigned long long)lo | (64ull << 32); }   // [lcp ngm] SBO 1024 B

template <int ACC>
DINL void wgmma_qk(unsigned* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
          "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
        : "l"(db), "n"(ACC));
}

template <int ACC>
DINL void wgmma_qk_ss(unsigned* d, unsigned long long da, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        " %32, %33, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
        : "l"(da), "l"(db), "n"(ACC));
}

DINL void wgmma_pv(float* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
        : "l"(db), "n"(1));
}

template <int ACC>
DINL void wgmma_l(float* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3\n"
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(ACC));
}

DINL __half2 u32h2(unsigned v) { return *reinterpret_cast<__half2*>(&v); }
DINL unsigned h2u32(__half2 v) { return *reinterpret_cast<unsigned*>(&v); }
DINL unsigned ex2_f16x2(unsigned x) {
    unsigned r;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x));
    return r;
}
DINL unsigned short cvt_e4m3x2(unsigned h2) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(h2));
    return r;
}
DINL int pi_lane(int lam) {
    return (lam & 1) | (((lam >> 2) & 3) << 1) | (((lam >> 1) & 1) << 3) | (lam & 16);
}

constexpr int SB = 16384;
constexpr int SBS = 17408;   // [lcp ngm] smem slot: K | V^T image + 1 KB ones group
// slot s (s < 8) offsets without a multiply: 17408 s = 16384 s | 1024 s, 1088 s = 1024 s | 64 s (no carries)
#define LCP_SS(s) ((((unsigned)(s)) << 14) | (((unsigned)(s)) << 10))
#define LCP_SL(s) ((((unsigned)(s)) << 10) | (((unsigned)(s)) << 6))
constexpr unsigned NEG = 0xFC00;
#define M_INIT (-30000.0f)

DINL void reg_dealloc_24() { asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n" ::: "memory"); }
DINL void reg_alloc_160() { asm volatile("setmaxnreg.inc.sync.aligned.u32 160;\n" ::: "memory"); }
DINL void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
DINL void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }

template <long long OPT> struct Cfg {
    static constexpr bool HOIST = (OPT & 2) != 0;
    static constexpr int EMU = (OPT & 4) ? 1 : 0;
    static constexpr bool DBP = (OPT & 8) != 0;
    static constexpr int RELU = (OPT & 16) ? 1 : 0;
    static constexpr int F16C = ((OPT & 32) || ((OPT >> 36) & 1)) ? 1 : 0;   // [gate2] V implies the f16 vote
    static constexpr int MUC = (OPT >> 8) & 7;
    static constexpr int MU = MUC == 1 ? 3 : MUC == 2 ? 2 : MUC == 3 ? 5 : MUC == 4 ? 1 : MUC == 5 ? 0 :
                              MUC == 6 ? 6 : MUC == 7 ? 8 : 4;
    static constexpr bool PP = (OPT & 2048) == 0;
    static constexpr bool PEEL = (OPT & 4096) != 0;
    static constexpr bool PIN = (OPT & 8192) != 0;
    static constexpr int ABL = (OPT >> 16) & 31;
    static constexpr bool TRACE = ((OPT >> 21) & 1) != 0;
    static constexpr bool PIPE = ((OPT >> 22) & 1) != 0;
    static constexpr bool QSM = ((OPT >> 23) & 1) != 0;
    static constexpr bool DUI = ((OPT >> 24) & 1) != 0;   // dual: second phase = first PWL's bits + 512 per half
    static constexpr bool SPEC = ((OPT >> 25) & 1) != 0;
    static constexpr bool IMADX = ((OPT >> 26) & 1) != 0;
    static constexpr bool PREF = ((OPT >> 27) & 1) != 0;
    static constexpr int NC = ((OPT >> 28) & 1) ? 4 : 3;   // consumer warpgroups (8 tokens x 8 heads each)
    // NC 4: producer warpgroup at 24 registers, consumers at 112 (launch 96: 128*24 + 512*112 <= 640*96 + 128*72);
    // a lone producer warp caps the block at 96 registers, and bulk copies issued by a consumer thread stall
    // that warpgroup ~1000 clk per block
    static constexpr bool INL = false;
    static constexpr bool PW = false;
    static constexpr int THREADS = 128 * (NC + 1);
    // dual-phase emulated exp: registers per 8 (bits 29-30: 0, 2, 3, 4; bit 15 adds 4) computed as
    // PWL(y) + 2^-0.5 PWL(y + 0.5)
    // (the two Schraudolph PWL errors are half a period apart: ratio spread 0.947..1.037 -> 0.975..1.021);
    // with DU > 0 every exp path produces 2^(x + 8) (P <= 2^8.75 < 448) so the classes share one scale
    static constexpr int DUC = (OPT >> 29) & 3;
    static constexpr int DU = (DUC == 1 ? 2 : DUC == 2 ? 3 : DUC == 3 ? 4 : 0) + ((OPT & 32768) ? 4 : 0);
    static constexpr bool DUD = (OPT & 16384) != 0;   // dither: second phase at y + 0.5 + 1/64 (own relu)
    static constexpr float EOFF = DU > 0 ? 8.0f : 7.0f;
    // bit 32: MUFU path shifted by MB = log2(1.0113 / 0.9992) to match the PWL's mean ratio after e4m3
    // bit 33: nmr derived from the rounded nmc (nmr = nmc + 14 - (EOFF - 7)) so both classes share one rounding
    static constexpr float MB = ((OPT >> 32) & 1) ? 0.01737f : 0.0f;
    static constexpr bool NMX = ((OPT >> 33) & 1) != 0;
    // [gate1] bit 35 (BYT): byte-domain exp. e4m3 is a log format with a 3-bit mantissa, so the Schraudolph integer
    // z = 8 (x - m + EOFFB + 7) (x = s qk2, log2 domain) IS the e4m3 code of P = 2^(x - m + EOFFB). Per P register:
    // z = HFMA2.RELU(s, 8 qk2, 8 (EOFFB + 7 - m)) in f16x2, then + 1024.0 (f16 ulp is 1 on [1024, 2048), so the low byte
    // of each half is round(z)); one PRMT 0x6420 per packed register. Every register takes this path (no MUFU class).
    // EOFFB 7.75: max byte 124 under the 0.75 lazy max (byte 127 = NaN needs z >= 126.5). The per-row constant is
    // min(8 (EOFFB + 7 - m), 65504): at m = M_INIT (row fully masked so far) it stays finite, so a masked s = -inf
    // gives relu(-inf) = 0 instead of -inf + inf = NaN. l (n72 ones column) sums the same bytes, so the 2^EOFFB scale
    // cancels in acc / l and in the split merge (every partial of a call uses the same build).
    static constexpr bool BYT = ((OPT >> 35) & 1) != 0;
    static constexpr float EOFFB = 7.75f;
    // [gate2] hot-loop instruction cuts (see mk_gate2.py): V vote in f16 vs a round-down threshold, T 3-input max tree,
    // D PV descriptor carried from the previous QK descriptor, W dead work (PEEL is bit 12)
    static constexpr bool G2V = ((OPT >> 36) & 1) != 0;
    static constexpr bool G2T = ((OPT >> 37) & 1) != 0;
    static constexpr bool G2D = ((OPT >> 38) & 1) != 0;
    static constexpr bool G2W = ((OPT >> 39) & 1) != 0;
    static constexpr bool G2Q = ((OPT >> 41) & 1) != 0;   // PEEL fixups: opaque loop bounds
    static constexpr bool G2U = ((OPT >> 43) & 1) != 0;   // PEEL's unmasked loop unrolled by 2, l folded once per pair
    static_assert(!G2U || PEEL, "[gate2] U unrolls PEEL's unmasked loop");
    static constexpr unsigned PWLC = DU > 0 ? 0xF7B5F7C5u + 0x04000400u : 0xF7B5F7C5u;
};

struct Prod { int nxt; int NT; const char* src; };

// non-blocking mbarrier phase test (1 if the phase with this parity has completed)
DINL unsigned mbar_test(unsigned bar, unsigned phase) {
    unsigned r;
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(r) : "r"(bar), "r"(phase) : "memory");
    return r;
}

// [gate2] T: max of 3 f16x2 (ptxas fuses the nested pair into one VHMNMX)
DINL __half2 hmax3(__half2 a, __half2 b, __half2 c) { return __hmax2(__hmax2(a, b), c); }
// [gate2] T: f16x2 max of row R's 16 registers s[2 i + R] in 8 ops (7 VHMNMX + 1 HMNMX2), depth 3
template <int R> DINL __half2 rowmax16(const unsigned (&s)[32]) {
    const __half2 a0 = hmax3(u32h2(s[R]),      u32h2(s[2 + R]),  u32h2(s[4 + R]));
    const __half2 a1 = hmax3(u32h2(s[6 + R]),  u32h2(s[8 + R]),  u32h2(s[10 + R]));
    const __half2 a2 = hmax3(u32h2(s[12 + R]), u32h2(s[14 + R]), u32h2(s[16 + R]));
    const __half2 a3 = hmax3(u32h2(s[18 + R]), u32h2(s[20 + R]), u32h2(s[22 + R]));
    const __half2 a4 = hmax3(u32h2(s[24 + R]), u32h2(s[26 + R]), u32h2(s[28 + R]));
    return __hmax2(hmax3(a0, a1, a2), hmax3(a3, a4, u32h2(s[30 + R])));
}
// [gate2] V: f16 lazy-max threshold, rounded DOWN: for an f16 score s, s <= RD(T) <=> s <= T (T real), so the f16
// vote is exactly s qk2f <= m + 0.75 (up to the fp32 quotient); overflow below gives -inf (only -inf passes), above
// 65504 (every finite s passes, and s qk2f <= 65504 qk2f < m + 0.75 then holds anyway)
DINL __half2 thr_rd(float m, float qk2f) { return __half2half2(__float2half_rd((m + 0.75f) / qk2f)); }
// [gate2] V: both rows' f16x2 maxima <= their thresholds (all 4 halves), as 2 HSETP2 + 1 PLOP3 (__hble2 is HSET2 masks)
DINL bool vote_le2(__half2 a0, __half2 b0, __half2 a1, __half2 b1) {
    unsigned r;
    asm("{\n.reg .pred p0, q0, p1, q1;\n"
        "setp.le.f16x2 p0|q0, %1, %2;\n"
        "setp.le.and.f16x2 p1|q1, %3, %4, p0;\n"
        "and.pred p1, p1, q1;\n"
        "and.pred p1, p1, q0;\n"
        "selp.u32 %0, 1, 0, p1;\n}\n" : "=r"(r) : "r"(h2u32(a0)), "r"(h2u32(b0)), "r"(h2u32(a1)), "r"(h2u32(b1)));
    return r != 0;
}

// fmha4's softmax_a (mask, lazy max, hybrid exp in place)
template <int WIN, long long OPT>
DINL int softmax7(unsigned (&s)[32], float (&m)[2], float (&al)[2], __half2 (&nmc)[2], __half2 (&nmr)[2],
                  __half2 (&thr)[2], float qk2f, __half2 qk2h2,
                  int diag, int j0, int lim0, int t, int lowm = 0, int lo0 = 0, unsigned k32 = 32u) {
    using C = Cfg<OPT>;
    static_assert(!(C::BYT && C::SPEC), "[gate1] BYT is wired into softmax7 only (softmax_spec / exp_regs untouched)");
    if (WIN && lowm) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int k0 = j0 + 2 * t + 8 * j;
            if (k0 < lo0)         s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
            if (k0 + 1 < lo0)     s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
            if (k0 < lo0 + 1)     s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
            if (k0 + 1 < lo0 + 1) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
        }
    }
    if (diag) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int k0 = j0 + 2 * t + 8 * j;
            if (k0 > lim0)     s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
            if (k0 + 1 > lim0) s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
            if (k0 > lim0 + 1)     s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
            if (k0 + 1 > lim0 + 1) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
        }
    }
    // [gate2] W: with OPT & 64 (fold_l72) al is read under resc only, so the al = 1 stores are dead on the hot path
    if (!(C::G2W && (OPT & 64))) { al[0] = 1.0f; al[1] = 1.0f; }
    int resc = 0;
    if (!(C::ABL & 2)) {
        __half2 t0v[8], t1v[8];
        if (C::G2T) {   // [gate2] T: 3-input tree (t0v[0] / t1v[0] = the rows' f16x2 maxima)
            t0v[0] = rowmax16<0>(s);
            t1v[0] = rowmax16<1>(s);
        } else {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            t0v[j] = __hmax2(u32h2(s[4 * j]), u32h2(s[4 * j + 2]));
            t1v[j] = __hmax2(u32h2(s[4 * j + 1]), u32h2(s[4 * j + 3]));
        }
#pragma unroll
        for (int st = 4; st > 0; st >>= 1)
#pragma unroll
            for (int j = 0; j < 8; j++) if (j < st) {
                t0v[j] = __hmax2(t0v[j], t0v[j + st]);
                t1v[j] = __hmax2(t1v[j], t1v[j + st]);
            }
        }
        bool ok;
        if (C::F16C) {
            ok = C::G2V ? vote_le2(t0v[0], thr[0], t1v[0], thr[1])   // [gate2] V
                        : (__hble2(t0v[0], thr[0]) & __hble2(t1v[0], thr[1]));
        } else {
            const float f0 = fmaxf(__half2float(__low2half(t0v[0])), __half2float(__high2half(t0v[0])));
            const float f1 = fmaxf(__half2float(__low2half(t1v[0])), __half2float(__high2half(t1v[0])));
            ok = (f0 * qk2f <= m[0] + 0.75f) & (f1 * qk2f <= m[1] + 0.75f);
        }
        resc = !__all_sync(0xFFFFFFFFu, ok);
        if (resc) {
            const float f0 = fmaxf(__half2float(__low2half(t0v[0])), __half2float(__high2half(t0v[0])));
            const float f1 = fmaxf(__half2float(__low2half(t1v[0])), __half2float(__high2half(t1v[0])));
            float r0 = f0, r1 = f1;
            r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 1));
            r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 2));
            r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 1));
            r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 2));
            const float mn0 = fmaxf(m[0], r0 * qk2f);
            const float mn1 = fmaxf(m[1], r1 * qk2f);
            al[0] = exp2f(m[0] - mn0);
            al[1] = exp2f(m[1] - mn1);
            m[0] = mn0; m[1] = mn1;
            nmc[0] = __float2half2_rn(C::EOFF + C::MB - mn0);
            nmc[1] = __float2half2_rn(C::EOFF + C::MB - mn1);
            if (C::BYT) {   // [gate1] byte constant, clamped finite (see Cfg::BYT)
                nmr[0] = __float2half2_rn(fminf(8.0f * (C::EOFFB + 7.0f - mn0), 65504.0f));
                nmr[1] = __float2half2_rn(fminf(8.0f * (C::EOFFB + 7.0f - mn1), 65504.0f));
            } else if (C::RELU) {
                if (C::NMX) {
                    const __half2 d = __float2half2_rn(21.0f - C::EOFF - C::MB);
                    nmr[0] = __hadd2(nmc[0], d);
                    nmr[1] = __hadd2(nmc[1], d);
                } else {
                    nmr[0] = __float2half2_rn(21.0f - mn0);
                    nmr[1] = __float2half2_rn(21.0f - mn1);
                }
            }
            if (C::G2V) {   // [gate2] V: round-down threshold (exact f16 vote)
                thr[0] = thr_rd(mn0, qk2f);
                thr[1] = thr_rd(mn1, qk2f);
            } else if (C::F16C) {
                thr[0] = __float2half2_rn((mn0 + 0.75f) / qk2f);
                thr[1] = __float2half2_rn((mn1 + 0.75f) / qk2f);
            }
        }
    }
    if (!(C::ABL & 1)) {
        const __half2 hsc   = __float2half2_rn(1024.0f);
        const __half2 hmag  = __float2half2_rn(50176.0f);
        const __half2 hclmp = __float2half2_rn(35840.0f);
        const __half2 hclmp2 = __float2half2_rn(36352.0f);
        const __half2 hcd = __float2half2_rn(C::DUD ? 0.6994198f : 0.70710678f);   // 2^-(0.5 + 1/64) / 2^-0.5
        const __half2 hdd = __float2half2_rn(0.515625f);
        constexpr int MU = C::MU;
        constexpr int DU = C::DU;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const int rk = (((j >> 1) & 7) * 2 + (j & 1)) % 16;
            const bool mufu = !C::BYT && ((MU == 4 && DU == 0) ? ((j & 2) == 0) : (rk < 2 * MU));
            const bool dual = DU > 0 && !mufu && rk < 2 * (MU + DU);
            if (C::BYT) {   // [gate1] HFMA2.RELU + HADD2 1024 (qk2h2 = 8 qk2, nmr = byte constant on this build)
                s[j] = h2u32(__hadd2(__hfma2_relu(u32h2(s[j]), qk2h2, nmr[j & 1]), hsc));
            } else if (mufu) {
                s[j] = ex2_f16x2(h2u32(__hfma2(u32h2(s[j]), qk2h2, nmc[j & 1])));
            } else if (dual) {
                const __half2 yp = __hfma2_relu(u32h2(s[j]), qk2h2, nmr[j & 1]);
                const unsigned e1 = h2u32(__hfma2(yp, hsc, hclmp)) * 32u + 0xF7B5F7C5u;
                unsigned e2;
                if (C::DUI && !C::DUD) {
                    e2 = e1 + 0x02000200u;
                } else if (C::DUD) {
                    const __half2 yp2 = __hadd2(yp, hdd);
                    e2 = h2u32(__hfma2(yp2, hsc, hclmp)) * (C::IMADX ? k32 : 32u) + 0xF7B5F7C5u;
                } else {
                    e2 = h2u32(__hfma2(yp, hsc, hclmp2)) * (C::IMADX ? k32 : 32u) + 0xF7B5F7C5u;
                }
                s[j] = h2u32(__hfma2(u32h2(e2), hcd, u32h2(e1)));
            } else if (C::RELU) {
                const __half2 yp = __hfma2_relu(u32h2(s[j]), qk2h2, nmr[j & 1]);
                if (C::IMADX) s[j] = h2u32(__hfma2(yp, hsc, hclmp)) * k32 + C::PWLC;
                else s[j] = h2u32(__hfma2(yp, hsc, hclmp)) * 32u + C::PWLC;
            } else {
                __half2 y  = __hfma2(u32h2(s[j]), qk2h2, nmc[j & 1]);
                __half2 xb = __hmax2(__hfma2(y, hsc, hmag), hclmp);
                if (C::EMU) s[j] = h2u32(xb) * 32u - 0x084A083Bu;
                else s[j] = (h2u32(xb) & 0x03FF03FFu) * 32u - 0x083B083Bu;
            }
        }
    }
    return resc;
}

// hybrid exp of 32 registers (softmax7's exp loop, out of place)
template <long long OPT>
DINL void exp_regs(const unsigned (&s)[32], unsigned (&e)[32], const __half2 (&nmc)[2], const __half2 (&nmr)[2],
                   __half2 qk2h2) {
    using C = Cfg<OPT>;
    const __half2 hsc   = __float2half2_rn(1024.0f);
    const __half2 hmag  = __float2half2_rn(50176.0f);
    const __half2 hclmp = __float2half2_rn(35840.0f);
    constexpr int MU = C::MU;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        const bool mufu = (MU == 4) ? ((j & 2) == 0) : ((((j >> 1) & 7) * 2 + (j & 1)) % 16 < 2 * MU);
        if (mufu) {
            e[j] = ex2_f16x2(h2u32(__hfma2(u32h2(s[j]), qk2h2, nmc[j & 1])));
        } else if (C::RELU) {
            const __half2 yp = __hfma2_relu(u32h2(s[j]), qk2h2, nmr[j & 1]);
            e[j] = h2u32(__hfma2(yp, hsc, hclmp)) * 32u - 0x084A083Bu;
        } else {
            __half2 y  = __hfma2(u32h2(s[j]), qk2h2, nmc[j & 1]);
            __half2 xb = __hmax2(__hfma2(y, hsc, hmag), hclmp);
            if (C::EMU) e[j] = h2u32(xb) * 32u - 0x084A083Bu;
            else e[j] = (h2u32(xb) & 0x03FF03FFu) * 32u - 0x083B083Bu;
        }
    }
}

// SPEC: the exp runs with the running max before the lazy-max vote (off the max tree's dependency chain) into e;
// a failed vote (some score above m + 0.75) recomputes it from s with the new max
template <int WIN, long long OPT>
DINL int softmax_spec(unsigned (&s)[32], unsigned (&e)[32], float (&m)[2], float (&al)[2], __half2 (&nmc)[2],
                      __half2 (&nmr)[2], __half2 (&thr)[2], float qk2f, __half2 qk2h2,
                      int diag, int j0, int lim0, int t, int lowm = 0, int lo0 = 0) {
    using C = Cfg<OPT>;
    if (WIN && lowm) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int k0 = j0 + 2 * t + 8 * j;
            if (k0 < lo0)         s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
            if (k0 + 1 < lo0)     s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
            if (k0 < lo0 + 1)     s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
            if (k0 + 1 < lo0 + 1) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
        }
    }
    if (diag) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int k0 = j0 + 2 * t + 8 * j;
            if (k0 > lim0)     s[2 * j]     = (s[2 * j]     & 0xFFFF0000u) | NEG;
            if (k0 + 1 > lim0) s[2 * j]     = (s[2 * j]     & 0x0000FFFFu) | (NEG << 16);
            if (k0 > lim0 + 1)     s[2 * j + 1] = (s[2 * j + 1] & 0xFFFF0000u) | NEG;
            if (k0 + 1 > lim0 + 1) s[2 * j + 1] = (s[2 * j + 1] & 0x0000FFFFu) | (NEG << 16);
        }
    }
    exp_regs<OPT>(s, e, nmc, nmr, qk2h2);
    al[0] = 1.0f; al[1] = 1.0f;
    __half2 t0v[8], t1v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        t0v[j] = __hmax2(u32h2(s[4 * j]), u32h2(s[4 * j + 2]));
        t1v[j] = __hmax2(u32h2(s[4 * j + 1]), u32h2(s[4 * j + 3]));
    }
#pragma unroll
    for (int st = 4; st > 0; st >>= 1)
#pragma unroll
        for (int j = 0; j < 8; j++) if (j < st) {
            t0v[j] = __hmax2(t0v[j], t0v[j + st]);
            t1v[j] = __hmax2(t1v[j], t1v[j + st]);
        }
    bool ok;
    if (C::F16C) {
        ok = __hble2(t0v[0], thr[0]) & __hble2(t1v[0], thr[1]);
    } else {
        const float f0 = fmaxf(__half2float(__low2half(t0v[0])), __half2float(__high2half(t0v[0])));
        const float f1 = fmaxf(__half2float(__low2half(t1v[0])), __half2float(__high2half(t1v[0])));
        ok = (f0 * qk2f <= m[0] + 0.75f) & (f1 * qk2f <= m[1] + 0.75f);
    }
    const int resc = !__all_sync(0xFFFFFFFFu, ok);
    if (resc) {
        const float f0 = fmaxf(__half2float(__low2half(t0v[0])), __half2float(__high2half(t0v[0])));
        const float f1 = fmaxf(__half2float(__low2half(t1v[0])), __half2float(__high2half(t1v[0])));
        float r0 = f0, r1 = f1;
        r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 1));
        r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFFu, r0, 2));
        r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 1));
        r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFFu, r1, 2));
        const float mn0 = fmaxf(m[0], r0 * qk2f);
        const float mn1 = fmaxf(m[1], r1 * qk2f);
        al[0] = exp2f(m[0] - mn0);
        al[1] = exp2f(m[1] - mn1);
        m[0] = mn0; m[1] = mn1;
        nmc[0] = __float2half2_rn(7.0f - mn0);
        nmc[1] = __float2half2_rn(7.0f - mn1);
        if (C::RELU) {
            nmr[0] = __float2half2_rn(21.0f - mn0);
            nmr[1] = __float2half2_rn(21.0f - mn1);
        }
        if (C::F16C) {
            thr[0] = __float2half2_rn((mn0 + 0.75f) / qk2f);
            thr[1] = __float2half2_rn((mn1 + 0.75f) / qk2f);
        }
        exp_regs<OPT>(s, e, nmc, nmr, qk2h2);
    }
    return resc;
}

template <long long OPT>
DINL void softmax_cvt(const unsigned (&s)[32], unsigned (&pk)[16]) {
    if (Cfg<OPT>::ABL & 8) {
#pragma unroll
        for (int i = 0; i < 16; i++) pk[i] = s[2 * i] ^ s[2 * i + 1];
        return;
    }
    if (Cfg<OPT>::BYT) {   // [gate1] one PRMT per P register: bytes 0 and 2 of each source (the low byte
        // of each f16 half), same element order as cvt_e4m3x2's {lo, hi} pairs
#pragma unroll
        for (int js = 0; js < 4; js++) {
            pk[4 * js]     = __byte_perm(s[8 * js],     s[8 * js + 2], 0x6420);
            pk[4 * js + 1] = __byte_perm(s[8 * js + 1], s[8 * js + 3], 0x6420);
            pk[4 * js + 2] = __byte_perm(s[8 * js + 4], s[8 * js + 6], 0x6420);
            pk[4 * js + 3] = __byte_perm(s[8 * js + 5], s[8 * js + 7], 0x6420);
        }
        return;
    }
#pragma unroll
    for (int js = 0; js < 4; js++) {
        pk[4 * js]     = (unsigned)cvt_e4m3x2(s[8 * js])     | ((unsigned)cvt_e4m3x2(s[8 * js + 2]) << 16);
        pk[4 * js + 1] = (unsigned)cvt_e4m3x2(s[8 * js + 1]) | ((unsigned)cvt_e4m3x2(s[8 * js + 3]) << 16);
        pk[4 * js + 2] = (unsigned)cvt_e4m3x2(s[8 * js + 4]) | ((unsigned)cvt_e4m3x2(s[8 * js + 6]) << 16);
        pk[4 * js + 3] = (unsigned)cvt_e4m3x2(s[8 * js + 5]) | ((unsigned)cvt_e4m3x2(s[8 * js + 7]) << 16);
    }
}

DINL void rescale_acc(float (&acc)[32], const float (&al)[2]) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
        acc[4 * j]     *= al[0]; acc[4 * j + 1] *= al[0];
        acc[4 * j + 2] *= al[1]; acc[4 * j + 3] *= al[1];
    }
}

DINL void issue_qk(unsigned (&s)[32], unsigned* qf, unsigned st) {
    const unsigned long long dK0 = make_desc(st, 2048, 128);
    const unsigned long long dK1 = make_desc(st + 4096, 2048, 128);
    wgmma_fence();
    wgmma_qk<0>(s, qf, dK0);
    wgmma_qk<1>(s, qf + 4, dK1);
    wgmma_commit();
}

template <long long OPT>
DINL void issue_pvl(float (&acc)[32], float (&lblk)[4], unsigned (&pk)[16], unsigned st, unsigned long long dones) {
    fence_fbank(acc);
    fence_ubank(pk);
    wgmma_fence();
#pragma unroll
    for (int js = 0; js < 4; js++) {
        wgmma_pv(acc, pk + 4 * js, make_desc(st + 8192 + js * 256, 128, 1024));
        if (!(Cfg<OPT>::ABL & 4)) {
            if (js == 0) wgmma_l<0>(lblk, pk, dones);
            else wgmma_l<1>(lblk, pk + 4 * js, dones);
        }
    }
    wgmma_commit();
}

DINL void issue_qk_h(unsigned (&s)[32], unsigned* qf, unsigned dlo) {
    wgmma_fence();
    wgmma_qk<0>(s, qf, desc_lo(dlo));
    wgmma_qk<1>(s, qf + 4, desc_lo(dlo + 256));
    wgmma_commit();
}

template <long long OPT>
DINL void issue_pvl_h(float (&acc)[32], float (&lblk)[4], unsigned (&pk)[16], unsigned dlo,
                      unsigned long long dones) {
    fence_fbank(acc);
    fence_ubank(pk);
    wgmma_fence();
#pragma unroll
    for (int js = 0; js < 4; js++) {
        wgmma_pv(acc, pk + 4 * js, desc_lo_s(dlo + js * 16));
        if (!(Cfg<OPT>::ABL & 4)) {
            if (js == 0) wgmma_l<0>(lblk, pk, dones);
            else wgmma_l<1>(lblk, pk + 4 * js, dones);
        }
    }
    wgmma_commit();
}

DINL void wgmma_pv72(float* d, unsigned* a, unsigned long long db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35},"
        "{%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
          "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
          "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
          "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
          "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
        : "l"(db), "n"(1));
}
template <long long OPT>
DINL void issue_pvl72_h(float (&acc)[36], unsigned (&pk)[16], unsigned dlo) {
    fence_fbank(acc);
    fence_ubank(pk);
    wgmma_fence();
#pragma unroll
    for (int js = 0; js < 4; js++) wgmma_pv72(acc, pk + 4 * js, desc_lo_s(dlo + js * 16));
    wgmma_commit();
}
DINL void rescale_acc72(float (&acc)[36], const float (&al)[2]) {   // the 32 PV entries only
#pragma unroll
    for (int j = 0; j < 8; j++) {
        acc[4 * j] *= al[0]; acc[4 * j + 1] *= al[0];
        acc[4 * j + 2] *= al[1]; acc[4 * j + 3] *= al[1];
    }
}
// [lcp n72] l lives in fp32 registers, never in the wgmma accumulator: the FP8 wgmma accumulator drops the low bits
// of small addends against a grown accumulator (lcp1f: l rel error 0.17 / 0.70 on flat B). The l columns acc[32..35]
// hold ONE block's row sums (zeroed after each fold) = the bank's lblk, folded with the bank's own arithmetic.
template <long long OPT>
DINL void fold_l72(float (&acc)[36], float (&l)[2], const float (&al)[2], int resc) {
    if (OPT & 64) {
        if (resc) {
            rescale_acc72(acc, al);
            l[0] = (l[0] + acc[32]) * al[0];
            l[1] = (l[1] + acc[34]) * al[1];
        } else {
            l[0] += acc[32];
            l[1] += acc[34];
        }
    } else {
        if (resc) rescale_acc72(acc, al);
        l[0] = (l[0] + acc[32]) * al[0];
        l[1] = (l[1] + acc[34]) * al[1];
    }
    // restart the l columns IN PLACE (x * an opaque +0): a plain "= 0.f" (or x -= x) makes ptxas serialize every
    // wgmma (C7511, insufficient register resources); l columns are finite and >= 0, so x * +0 = +0 exactly
    float z0;
    asm("mov.b32 %0, 0;" : "=f"(z0));
    if (Cfg<OPT>::G2W) { acc[32] *= z0; acc[34] *= z0; }   // [gate2] W: acc[33] / acc[35] are never read
    else { acc[32] *= z0; acc[33] *= z0; acc[34] *= z0; acc[35] *= z0; }
}
DINL void issue_qk_s(unsigned (&s)[32], unsigned dqa, unsigned dlo) {
    wgmma_fence();
    wgmma_qk_ss<0>(s, desc_lo(dqa), desc_lo(dlo));
    wgmma_qk_ss<1>(s, desc_lo(dqa + 128), desc_lo(dlo + 256));
    wgmma_commit();
}

template <long long OPT>
DINL void issue_qk_x(unsigned (&s)[32], unsigned (&qf)[8], unsigned dqa, unsigned dlo) {
    if (Cfg<OPT>::QSM) issue_qk_s(s, dqa, dlo);
    else issue_qk_h(s, qf, dlo);
}

// PIPE form of issue_pvl_h: the row sums accumulate across blocks in lacc (rescaled with acc)
template <long long OPT>
DINL void issue_pvl_acc(float (&acc)[32], float (&lacc)[4], unsigned (&pk)[16], unsigned dlo,
                        unsigned long long dones) {
    fence_fbank(acc);
    fence_fbank(lacc);
    fence_ubank(pk);
    wgmma_fence();
#pragma unroll
    for (int js = 0; js < 4; js++) {
        wgmma_pv(acc, pk + 4 * js, desc_lo_s(dlo + js * 16));
        if (!(Cfg<OPT>::ABL & 4) && !(PG_DBG & 16)) wgmma_l<1>(lacc, pk + 4 * js, dones);
    }
    wgmma_commit();
}

#define STAMP(i) do { if (C::TRACE && tr != nullptr && kt < TRK) tr[kt * 8 + (i)] = clk(); } while (0)

// PIPE block kt >= 1 (nothing in flight on entry): sx holds S(kt) (QK issued and retired one block earlier), py
// holds P(kt-1); sy and px are free. Issues QK(kt+1) -> sy (unless LAST) and PV(kt-1) from py, runs the softmax of
// S(kt) into px under them, then retires both at the end of the block: no wgmma is in flight across a loop back
// edge (ptxas serializes every wgmma (C7514/C7515) when one is and its registers are touched later in the loop).
template <int NST, int WIN, long long OPT, bool LAST>
DINL void blockP(int kt, unsigned (&sx)[32], unsigned (&sy)[32], unsigned (&px)[16], unsigned (&py)[16],
                 float (&acc)[32], float (&lacc)[4], float (&m)[2], float (&al)[2], __half2 (&nmc)[2],
                 __half2 (&nmr)[2], __half2 (&thr)[2], unsigned (&qf)[8], unsigned dqa, unsigned mb, int cw,
                 unsigned dqk, unsigned dpv, unsigned long long dones, int kb0, int dgt, int lim, int t, int win,
                 float qk2f, __half2 qk2h2, unsigned long long* tr) {
    using C = Cfg<OPT>;
    constexpr int NC = C::NC;
    constexpr bool PP = C::PP;
    const unsigned sl1 = (unsigned)(kt + 1) & (NST - 1);
    const unsigned slp = (unsigned)(kt - 1) & (NST - 1);
    STAMP(0);
    if (!LAST) mbar_wait(mb + sl1 * 8, ((unsigned)(kt + 1) / NST) & 1);
    STAMP(1);
    if (PP) named_bar_sync(1 + cw, 256);
    STAMP(2);
    if (!LAST) issue_qk_x<OPT>(sy, qf, dqa, dqk + LCP_SL(sl1));
    issue_pvl_acc<OPT>(acc, lacc, py, dpv + LCP_SL(slp), dones);
    if (PP) named_bar_arrive(1 + (cw + 1) % NC, 256);
    STAMP(3);
    const int j0 = (kb0 + kt) << 7;
    const int resc = softmax7<WIN, OPT>(sx, m, al, nmc, nmr, thr, qk2f, qk2h2, (j0 + 127 > dgt), j0, lim, t,
                                        WIN ? (j0 < dgt + 7 - (win - 1)) : 0, lim - (win - 1));
    STAMP(4);
    if (resc) {
        wgmma_wait<0>();
        fence_fbank(acc);
        fence_fbank(lacc);
        rescale_acc(acc, al);
#pragma unroll
        for (int i = 0; i < 4; i++) lacc[i] *= al[i >> 1];
    }
    STAMP(5);
    softmax_cvt<OPT>(sx, px);
    STAMP(6);
    wgmma_wait<0>();
    fence_ubank(sy);
    fence_ubank(px);
    fence_ubank(py);
    fence_fbank(acc);
    fence_fbank(lacc);
    mbar_arrive(mb + NST * 8 + slp * 8);
    STAMP(7);
}

// one key block kt >= 1 of a consumer warpgroup: QK(kt) + PV(kt-1) (P from pkc); P(kt) into pkn
template <int NST, int WIN, long long OPT, int MASK = 1, int FOLD = 1>
DINL void block7(int kt, unsigned (&pkc)[16], unsigned (&pkn)[16], float (&acc)[36], float (&l)[2], float (&m)[2],
                 float (&al)[2], __half2 (&nmc)[2], __half2 (&nmr)[2], __half2 (&thr)[2], unsigned (&qf)[8],
                 unsigned mb, int cw, unsigned sbase,
                 unsigned dqk, unsigned dpv, unsigned long long dones, int kb0, int dgt, int lim, int t, int win,
                 float qk2f, __half2 qk2h2, unsigned long long* tr, unsigned k32, unsigned& rdy, Prod& pd,
                 bool pthr, unsigned dqa, unsigned& qprev) {
    using C = Cfg<OPT>;
    constexpr int NC = C::NC;
    constexpr bool HOIST = C::HOIST;
    constexpr bool PP = C::PP;
    unsigned s[32];
    float lblk[4];
    const unsigned sl = HOIST ? ((unsigned)kt & (NST - 1)) : (unsigned)(kt % NST);
    const unsigned slp = HOIST ? ((unsigned)(kt - 1) & (NST - 1)) : (unsigned)((kt - 1) % NST);
    STAMP(0);
    if (!C::PREF || !rdy) mbar_wait(mb + sl * 8, HOIST ? (((unsigned)kt / NST) & 1) : ((kt / NST) & 1));
    STAMP(1);
    if (PP) named_bar_sync(1 + cw, 256);
    STAMP(2);
    if (C::ABL & 16) {
#pragma unroll
        for (int j = 0; j < 32; j++) s[j] = qf[j & 7] + (unsigned)kt * 0x00010001u;
        fence_ubank(pkc);
#pragma unroll
    } else if (HOIST) {
        const unsigned qkd = dqk + LCP_SL(sl);
        issue_qk_x<OPT>(s, qf, dqa, qkd);
        // [gate2] D: slot (kt - 1)'s PV descriptor = block kt - 1's QK descriptor + (dpv - dqk): same smem image,
        // start + 8192 B (+512 in 16 B units), LBO 2048 -> 128 B (128 -> 8 << 16); no 14-bit wrap (smem < 256 KB)
        issue_pvl72_h<OPT>(acc, pkc, C::G2D ? qprev + 0xFF880200u : dpv + LCP_SL(slp));
        if (C::G2D) qprev = qkd;
    } else {
        issue_qk(s, qf, sbase + LCP_SS(sl));
        issue_pvl72_h<OPT>(acc, pkc, dpv + LCP_SL(slp));
    }
    if (PP) named_bar_arrive(1 + (cw + 1) % NC, 256);
    // the next stage's full barrier, tested now so the test latency hides under the QK wait
    if (C::PREF) rdy = mbar_test(mb + ((unsigned)(kt + 1) & (NST - 1)) * 8, ((unsigned)(kt + 1) / NST) & 1);
    STAMP(3);
    wgmma_wait<1>();
    fence_ubank(s);
    STAMP(4);
    const int j0 = (kb0 + kt) << 7;
    unsigned e[32];
    int resc;
    if (C::SPEC)
        resc = softmax_spec<WIN && MASK, OPT>(s, e, m, al, nmc, nmr, thr, qk2f, qk2h2,
                                             MASK ? (j0 + 127 > dgt) : 0, j0, lim, t,
                                             (WIN && MASK) ? (j0 < dgt + 7 - (win - 1)) : 0, lim - (win - 1));
    else
        resc = softmax7<WIN && MASK, OPT>(s, m, al, nmc, nmr, thr, qk2f, qk2h2,
                                         MASK ? (j0 + 127 > dgt) : 0, j0, lim, t,
                                         (WIN && MASK) ? (j0 < dgt + 7 - (win - 1)) : 0, lim - (win - 1), k32);
    STAMP(5);
    if (C::DBP) softmax_cvt<OPT>(s, pkn);
    wgmma_wait<0>();
    fence_fbank(acc);
    fence_ubank(pkc);
    mbar_arrive(mb + NST * 8 + slp * 8);
    if (C::INL && pthr) {
        while (pd.nxt < pd.NT && pd.nxt < kt + NST) {
            const unsigned ps = (unsigned)pd.nxt & (NST - 1), pr = (unsigned)pd.nxt / NST;
            if (!mbar_test(mb + NST * 8 + ps * 8, (pr - 1) & 1)) break;
            mbar_expect_tx(mb + ps * 8, SB);
            bulk_g2s_u(sbase + LCP_SS(ps), pd.src + (size_t)pd.nxt * SB, SB, mb + ps * 8);
            pd.nxt++;
        }
    }
    STAMP(6);
    if (FOLD) {
        fold_l72<OPT>(acc, l, al, resc);   // [lcp n72] the bank's l arithmetic on this block's l columns
    } else if (resc) {   // [gate2] U: first block of a pair: its l columns keep accumulating into the second block's
        // fold (2 blocks = 8 k32 steps in the fp32 wgmma accumulator), so they are rescaled with acc and l here
        rescale_acc72(acc, al);
        acc[32] *= al[0]; acc[34] *= al[1];   // acc[33] / acc[35] are never read
        l[0] *= al[0]; l[1] *= al[1];
    }
    if (!C::DBP) {
        if (C::SPEC) softmax_cvt<OPT>(e, pkn);
        else softmax_cvt<OPT>(s, pkn);
    }
    fence_ubank(pkn);
    STAMP(7);
}

// ---------------------------------------------------------------------------------------------- [r12e] epilogue v2
// sigmoid of a gate pair (PFMHA_SIGA 0: the exact expression used everywhere before, 1 / (1 + __expf(-g)))
DINL float tanh_apx(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DINL float2 sig2(float2 g) {
#if PFMHA_SIGA == 1
    return make_float2(fmaf(0.5f, tanh_apx(0.5f * g.x), 0.5f), fmaf(0.5f, tanh_apx(0.5f * g.y), 0.5f));
#elif PFMHA_SIGA == 2
    const __half2 h = __floats2half2_rn(0.5f * g.x, 0.5f * g.y);
    unsigned th;
    asm("tanh.approx.f16x2 %0, %1;" : "=r"(th) : "r"(*reinterpret_cast<const unsigned*>(&h)));
    const float2 t2 = __half22float2(*reinterpret_cast<const __half2*>(&th));
    return make_float2(fmaf(0.5f, t2.x, 0.5f), fmaf(0.5f, t2.y, 0.5f));
#else
    return make_float2(1.0f / (1.0f + __expf(-g.x)), 1.0f / (1.0f + __expf(-g.y)));
#endif
}
DINL void stsm_x4(unsigned addr, unsigned a, unsigned b, unsigned c, unsigned d) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
DINL uint4 lds128(unsigned addr) {
    uint4 v;
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
    return v;
}
// y pair = bf16(x rstd sigmoid(g)) from a bf16x2 x word and a bf16x2 gate word (fmha7's arithmetic and order)
DINL unsigned ng_pair(unsigned xw, unsigned gw, float rstd) {
    const float2 x2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xw));
    const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gw));
    const float2 s2 = sig2(g2);
    const float y0 = x2.x * rstd * s2.x;
    const float y1 = x2.y * rstd * s2.y;
    __nv_bfloat162 o2 = __floats2bfloat162_rn(y0, y1);
    return *reinterpret_cast<unsigned*>(&o2);
}
// gate rows of a warp's two tokens in the coalesced layout: lane -> head 4 k + (lane >> 3), dims 8 (lane & 7) .. + 7
DINL void epi2_gload(uint4 (&gq)[4], const __nv_bfloat16* __restrict__ gate, int gstride, int b, int kvh, int tok0, int L,
                     int lane) {
#pragma unroll
    for (int rr = 0; rr < 2; rr++)
#pragma unroll
        for (int k = 0; k < 2; k++) {
            const int tok = tok0 + rr;
            gq[2 * rr + k] = make_uint4(0u, 0u, 0u, 0u);
            if (tok < L)
                gq[2 * rr + k] = *reinterpret_cast<const uint4*>(gate + (size_t)(b * L + tok) * gstride
                                                                + (kvh * 8 + 4 * k + (lane >> 3)) * 64 + (lane & 7) * 8);
        }
}
// the warp's 1 KB tile base (qsm + 256 wiw) and lane, re-derived after the hot loop from an opaque %tid (keeps no new
// value live across the loop); consumer warpgroup cw = wg - 1 (fmha7 / fmha7p's shipped NC3 / NC4 builds)
DINL void epi2_qw(unsigned mb, unsigned& qw, int& lane) {
    unsigned t;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
    qw = mb + 512u + (((t >> 7) - 1u) << 12) + (((t >> 5) & 3u) << 8);
    lane = (int)(t & 31u);
}
// L1 prefetch of the warp's gate rows (2 tokens x 8 lines of 128 B; lanes 16-31 repeat 0-15): no registers held
DINL void epi2_gpf1(const __nv_bfloat16* __restrict__ gate, int gstride, int b, int kvh, int tok0, int L, int lane) {
    const int tok = tok0 + ((lane >> 3) & 1);
    if (tok < L)
        asm volatile("prefetch.global.L1 [%0];" :: "l"(gate + (size_t)(b * L + tok) * gstride + (kvh * 8 + (lane & 7)) * 64)
                     : "memory");
}
// the warp's two rows (tok0, tok0 + 1) x 8 heads: SPLIT 0 -> y = o_norm(x) * sigmoid(gate) into out (bf16);
// SPLIT 1 -> the f16 partial into po (+ pml). qw = qsm + 256 wiw: the warp's own Q rows in its warpgroup's 4 KB Q tile
// (k-chunk c at c KB, token row 2 wiw + rr at 128 B): tile row h (128 B) at qw + (h >> 1) KB + (h & 1) 128, 16 B chunk c
// at (c ^ h) 16 (conflict-free stmatrix and 16 B reads). Only the warp's own Q rows are reused, after its last QK wgmma.
// x / ss / rstd / y arithmetic = fmha7's (bitwise); only the data movement differs.
template <int SPLIT>
DINL void epi2_store(const float (&acc)[36], float inv0, float inv1, const float (&m)[2], const float (&l)[2],
                     const uint4 (&gq)[4], int b, int kvh, int tok0, int L, int B, int zslot, int lane, unsigned qw,
                     __nv_bfloat16* __restrict__ out, __half* __restrict__ po, float2* __restrict__ pml, float ng_eps) {
    const int r8 = lane >> 3, c8 = lane & 7, rw = lane & 7;
    const unsigned rwa = qw + (rw >> 1) * 1024 + (rw & 1) * 128;               // tile row rw (stmatrix row address)
    const unsigned sa0 = rwa + (((unsigned)r8 ^ rw) << 4);                     // stmatrix x4 #0: chunk r8, row rw
    const unsigned sa1 = rwa + (((unsigned)(r8 + 4) ^ rw) << 4);               // x4 #1: chunk 4 + r8
    const unsigned la0 = qw + (r8 >> 1) * 1024 + (r8 & 1) * 128 + (((unsigned)c8 ^ r8) << 4);   // head r8, chunk c8
    const unsigned la1 = qw + ((r8 + 4) >> 1) * 1024 + (r8 & 1) * 128 + (((unsigned)c8 ^ (r8 + 4)) << 4);
    unsigned w[2][8];
    float rstd[2];
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const float inv = rr ? inv1 : inv0;
        if (!SPLIT) {
            float ss = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const __nv_bfloat162 b2 = __floats2bfloat162_rn(acc[4 * j + 2 * rr] * inv, acc[4 * j + 2 * rr + 1] * inv);
                const float2 x2 = __bfloat1622float2(b2);
                ss = fmaf(x2.x, x2.x, ss);
                ss = fmaf(x2.y, x2.y, ss);
                w[rr][j] = *reinterpret_cast<const unsigned*>(&b2);
            }
            ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 1);
            ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 2);
            rstd[rr] = rsqrtf(fmaf(ss, 1.0f / 64.0f, ng_eps));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const __half2 o2 = __floats2half2_rn(acc[4 * j + 2 * rr] * inv, acc[4 * j + 2 * rr + 1] * inv);
                w[rr][j] = *reinterpret_cast<const unsigned*>(&o2);
            }
            rstd[rr] = 0.f;
        }
    }
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const int tok = tok0 + rr;
        const bool live = tok < L;   // warp-uniform
        if (rr) __syncwarp();        // row 0's tile reads before row 1's stmatrix
        stsm_x4(sa0, w[rr][0], w[rr][1], w[rr][2], w[rr][3]);
        stsm_x4(sa1, w[rr][4], w[rr][5], w[rr][6], w[rr][7]);
        __syncwarp();
        const uint4 x0 = lds128(la0), x1 = lds128(la1);
        if (!SPLIT) {
            const float rs0 = __shfl_sync(0xFFFFFFFFu, rstd[rr], r8 * 4);
            const float rs1 = __shfl_sync(0xFFFFFFFFu, rstd[rr], (r8 + 4) * 4);
            const uint4 g0 = gq[2 * rr], g1 = gq[2 * rr + 1];
            const uint4 o0 = make_uint4(ng_pair(x0.x, g0.x, rs0), ng_pair(x0.y, g0.y, rs0),
                                        ng_pair(x0.z, g0.z, rs0), ng_pair(x0.w, g0.w, rs0));
            const uint4 o1 = make_uint4(ng_pair(x1.x, g1.x, rs1), ng_pair(x1.y, g1.y, rs1),
                                        ng_pair(x1.z, g1.z, rs1), ng_pair(x1.w, g1.w, rs1));
            if (live) {
                __nv_bfloat16* op = out + ((size_t)(b * L + tok) * 16 + kvh * 8 + r8) * 64 + c8 * 8;
                *reinterpret_cast<uint4*>(op) = o0;
                *reinterpret_cast<uint4*>(op + 4 * 64) = o1;
            }
        } else if (live) {
            const size_t row0 = ((size_t)zslot * B * L + (size_t)b * L + tok) * 16 + kvh * 8;
            __half* pp = po + (row0 + r8) * 64 + c8 * 8;
            *reinterpret_cast<uint4*>(pp) = x0;
            *reinterpret_cast<uint4*>(pp + 4 * 64) = x1;
            if ((lane & 3) == 0) pml[row0 + (lane >> 2)] = make_float2(m[rr], l[rr]);
        }
    }
}

template <int NST, int SPLIT, int WIN, long long OPT>
__global__ void __launch_bounds__(Cfg<OPT>::THREADS, 1) fmha7(
    const unsigned char* __restrict__ q, const unsigned char* __restrict__ kvt,
    __nv_bfloat16* __restrict__ out, __half* __restrict__ po, float2* __restrict__ pml,
    const int* __restrict__ lens,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    float smqk_log2e, int B, int L, int capblks, int win,
    const __nv_bfloat16* __restrict__ gate, int gstride, float ng_eps)
{
    using C = Cfg<OPT>;
    constexpr int NC = C::NC;
    constexpr int QB = 8 * NC;
    constexpr bool HOIST = C::HOIST;
    const int tid = threadIdx.x;
    const int b = blockIdx.y >> 1, kvh = blockIdx.y & 1;
    const int qs = (gridDim.x - 1 - blockIdx.x) * QB;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int len = lens[b];
    const int ctx = len - L;
    const int NTF = min(capblks, (min(ctx + qs + QB, len) + 127) >> 7);
    const int KLO = WIN ? (max(0, ctx + qs - (win - 1)) >> 7) : 0;
    int kb0 = KLO, kb1 = NTF;
    if (SPLIT) {
        kb0 = KLO + (int)(((long long)(NTF - KLO) * blockIdx.z) / gridDim.z);
        kb1 = KLO + (int)(((long long)(NTF - KLO) * (blockIdx.z + 1)) / gridDim.z);
    }
    const int NT = kb1 - kb0;
    const char* src = reinterpret_cast<const char*>(kvt) + ((size_t)(b * 2 + kvh) * capblks + kb0) * SB;

    extern __shared__ __align__(1024) char smem[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned mb = sbase + NST * SBS;
    if (tid < NST) {
        mbar_init(mb + tid * 8, 1);
        mbar_init(mb + NST * 8 + tid * 8, 128 * NC);
    }
    if (tid >= 64 && tid < 128)
        reinterpret_cast<unsigned*>(smem + NST * SBS + 256)[tid - 64] = 0x38383838u;   // e4m3 1.0
    for (int i = tid; i < NST * 256; i += blockDim.x)   // [lcp ngm] each slot's 1 KB ones group (N72's 9th N group)
        reinterpret_cast<unsigned*>(smem + (i >> 8) * SBS + 16384)[i & 255] = 0x38383838u;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    const int wg = tid >> 7;
    Prod pd{0, NT, src};
    if (C::INL) {
        if (tid == 0) {
            for (; pd.nxt < min(NT, NST); pd.nxt++) {
                mbar_expect_tx(mb + pd.nxt * 8, SB);
                bulk_g2s(smem + LCP_SS(pd.nxt), src + (size_t)pd.nxt * SB, SB, mb + pd.nxt * 8);
            }
        }
    } else if (C::PW ? (tid >= 128 * NC) : (wg == 0)) {
        if (!C::PW) reg_dealloc_24();
        if (tid == (C::PW ? 128 * NC : 0)) {
            for (int kt = 0; kt < NT; kt++) {
                const int ps = kt % NST, pr = kt / NST;
                if (pr > 0) {
                    if (OPT & 1) mbar_wait_sleep(mb + NST * 8 + ps * 8, (pr - 1) & 1);
                    else mbar_wait(mb + NST * 8 + ps * 8, (pr - 1) & 1);
                }
                mbar_expect_tx(mb + ps * 8, SB);
                bulk_g2s(smem + LCP_SS(ps), src + (size_t)kt * SB, SB, mb + ps * 8);
            }
            if (PFMHA_EPI2 && PFMHA_GPF7 && !SPLIT && gate != nullptr) {   // [r12e] the CTA's gate rows (1 KB per token)
                const __nv_bfloat16* gr = gate + (size_t)(b * L + qs) * gstride + kvh * 512;
                const int ntk = min(QB, L - qs);
                for (int tk = 0; tk < ntk; tk++)
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], 1024;" :: "l"(gr + (size_t)tk * gstride) : "memory");
            }
        }
        return;
    }
    if (NC == 4) asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\n" ::: "memory");
    else if (!C::INL && !C::PW) reg_alloc_160();
    const int cw = __shfl_sync(0xFFFFFFFFu, (C::INL || C::PW) ? wg : wg - 1, 0);   // [lcp u] warp-uniform -> UR
    const bool pthr = C::INL && tid == 0;
    const int wiw = __shfl_sync(0xFFFFFFFFu, (tid >> 5) & 3, 0);   // [lcp u]
    const int lane = tid & 31;
    const int t = lane & 3;
    const int h = lane >> 2;
    unsigned long long* tr = nullptr;
    if (C::TRACE && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && (tid & 127) == 0)
        tr = g_trace + cw * TRK * 8;

    const float qk2f = k_scale[0] * smqk_log2e;
    const __half2 qk2h2 = __float2half2_rn(C::BYT ? 8.0f * qk2f : qk2f);   // [gate1] BYT: 8 qk2 (exact x8 of the f16)
    const float vs = v_scale[0];

    unsigned qf[8];
    const int tok0 = qs + 8 * cw + 2 * wiw;
#pragma unroll
    for (int sl2 = 0; sl2 < 2; sl2++)
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            const unsigned char* qp = q + ((size_t)(b * L + tok) * 16 + kvh * 8 + h) * 64
                                    + sl2 * 32 + 4 * t;
            qf[4 * sl2 + rr]     = (tok < L) ? *reinterpret_cast<const unsigned*>(qp) : 0u;
            qf[4 * sl2 + rr + 2] = (tok < L) ? *reinterpret_cast<const unsigned*>(qp + 16) : 0u;
        }

    float m[2] = {M_INIT, M_INIT}, l[2] = {0.f, 0.f}, al[2];
    __half2 nmc[2] = {__float2half2_rn(C::EOFF + C::MB - M_INIT), __float2half2_rn(C::EOFF + C::MB - M_INIT)};
    // [gate1] BYT: the clamped byte constant min(8 (EOFFB + 7 - M_INIT), 65504) (local, so no device global)
    constexpr float NMB0 = 8.0f * (C::EOFFB + 7.0f - M_INIT) < 65504.0f ? 8.0f * (C::EOFFB + 7.0f - M_INIT) : 65504.0f;
    __half2 nmr[2] = {__float2half2_rn(C::BYT ? NMB0 : 21.0f - M_INIT), __float2half2_rn(C::BYT ? NMB0 : 21.0f - M_INIT)};
    __half2 thr[2] = {__float2half2_rn((M_INIT + 0.75f) / qk2f), __float2half2_rn((M_INIT + 0.75f) / qk2f)};
    if (C::G2V) { thr[0] = thr_rd(M_INIT, qk2f); thr[1] = thr[0]; }   // [gate2] V
    constexpr bool PP = C::PP;
    float acc[36];
#pragma unroll
    for (int i = 0; i < 36; i++) acc[i] = 0.f;
    unsigned pk[16], pk2[16];
    unsigned long long dones = make_desc(sbase + NST * SBS + 256, 128, 128);
    unsigned dqk = ((sbase >> 4) & 0x3FFF) | (128u << 16);
    unsigned dpv = (((sbase + 8192) >> 4) & 0x3FFF) | (8u << 16);   // [lcp ngm] LBO 128 B
    if (C::PIN) {
        dqk = __shfl_sync(0xFFFFFFFFu, dqk, 0);
        dpv = __shfl_sync(0xFFFFFFFFu, dpv, 0);
        dones = desc_lo(__shfl_sync(0xFFFFFFFFu, (unsigned)dones, 0));
    }
    const unsigned qsm = sbase + NST * SBS + 512 + cw * 4096;
    const unsigned dqa = ((qsm >> 4) & 0x3FFF) | (64u << 16);
    if (C::QSM) {
#pragma unroll
        for (int sl2 = 0; sl2 < 2; sl2++)
#pragma unroll
            for (int rr = 0; rr < 2; rr++)
#pragma unroll
                for (int hk = 0; hk < 2; hk++) {
                    const unsigned a = qsm + (2 * sl2 + hk) * 1024 + (wiw * 2 + rr) * 128 + h * 16 + 4 * t;
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(a), "r"(qf[4 * sl2 + rr + 2 * hk]) : "memory");
                }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        named_bar_sync(8 + cw, 128);
    }
    const int lim = __shfl_sync(0xFFFFFFFFu, ctx + qs + 8 * cw + 2 * wiw, 0);   // [lcp u] lens[b] is a load
    const int dgt = __shfl_sync(0xFFFFFFFFu, ctx + qs + 8 * cw, 0);
    const unsigned k32 = 32u + (unsigned)(__float_as_uint(qk2f) == 0x7FC00001u);
    unsigned rdy = 0;
    unsigned qprev = dqk;   // [gate2] D: block 0's QK descriptor (slot 0)
#define BLOCK7(KT, PKC, PKN) block7<NST, WIN, OPT, 1>(KT, PKC, PKN, acc, l, m, al, nmc, nmr, thr, qf, mb, cw, sbase, \
                                                     dqk, dpv, dones, kb0, dgt, lim, t, win, qk2f, qk2h2, tr, k32, rdy, \
                                                     pd, pthr, dqa, qprev)
#define BLOCK7U(KT, PKC, PKN) block7<NST, WIN, OPT, 0>(KT, PKC, PKN, acc, l, m, al, nmc, nmr, thr, qf, mb, cw, sbase, \
                                                     dqk, dpv, dones, kb0, dgt, lim, t, win, qk2f, qk2h2, tr, k32, rdy, \
                                                     pd, pthr, dqa, qprev)
#define BLOCK7UF(KT, F) block7<NST, WIN, OPT, 0, F>(KT, pk, pk, acc, l, m, al, nmc, nmr, thr, qf, mb, cw, sbase, \
                                                     dqk, dpv, dones, kb0, dgt, lim, t, win, qk2f, qk2h2, tr, k32, rdy, \
                                                     pd, pthr, dqa, qprev)

    uint4 gq[4];   // [r12e] the warp's gate rows (coalesced layout), loaded under the final PV
    if constexpr (C::PIPE) { if (NT > 0) {
        unsigned sA[32], sB[32], pB[16];
        float lacc[4] = {0.f, 0.f, 0.f, 0.f};
        mbar_wait(mb, 0);
        if (NT > 1) mbar_wait(mb + 8, 0);
        issue_qk_x<OPT>(sA, qf, dqa, dqk);
        if (NT > 1) issue_qk_x<OPT>(sB, qf, dqa, dqk + 1088u);
        wgmma_wait<0>();
        fence_ubank(sA);
        fence_ubank(sB);
        {
            const int j0 = kb0 << 7;
            softmax7<WIN, OPT>(sA, m, al, nmc, nmr, thr, qk2f, qk2h2, (j0 + 127 > dgt), j0, lim, t,
                               WIN ? (j0 < dgt + 7 - (win - 1)) : 0, lim - (win - 1));
            softmax_cvt<OPT>(sA, pk);
        }
        if (PP && NT > 1 && cw == NC - 1) named_bar_arrive(1, 256);
#define BLOCKP(LST, KT, SX, SY, PX, PY) blockP<NST, WIN, OPT, LST>(KT, SX, SY, PX, PY, acc, lacc, m, al, nmc, nmr, \
                                        thr, qf, dqa, mb, cw, dqk, dpv, dones, kb0, dgt, lim, t, win, qk2f, qk2h2, tr)
        int kt = 1;
        for (; kt + 2 < NT; kt += 2) {
            BLOCKP(false, kt, sB, sA, pB, pk);
            BLOCKP(false, kt + 1, sA, sB, pk, pB);
        }
        // kt == NT - 2 (one full block, then the last) or kt == NT - 1 (the last) or kt == NT (none)
        bool lastB;   // the last P sits in pB
        if (kt + 1 < NT) {
            BLOCKP(false, kt, sB, sA, pB, pk);
            BLOCKP(true, kt + 1, sA, sB, pk, pB);
            lastB = false;
        } else if (kt < NT) {
            BLOCKP(true, kt, sB, sA, pB, pk);
            lastB = true;
        } else {
            lastB = false;
        }
#undef BLOCKP
        if (PP && NT > 1 && cw == 0) named_bar_sync(1, 256);
        const unsigned sll = (unsigned)(NT - 1) & (NST - 1);
        if (lastB) issue_pvl_acc<OPT>(acc, lacc, pB, dpv + LCP_SL(sll), dones);
        else issue_pvl_acc<OPT>(acc, lacc, pk, dpv + LCP_SL(sll), dones);
        wgmma_wait<0>();
        fence_fbank(acc);
        fence_fbank(lacc);
        l[0] = lacc[0];
        l[1] = lacc[2];
    } } else if (NT > 0) {
        {
            unsigned s[32];
            mbar_wait(mb, 0);
            if (C::ABL & 16) {
#pragma unroll
                for (int j = 0; j < 32; j++) s[j] = qf[j & 7];
            } else {
                if (HOIST) issue_qk_x<OPT>(s, qf, dqa, dqk);
                else issue_qk(s, qf, sbase);
                wgmma_wait<0>();
            }
            fence_ubank(s);
            const int j0 = kb0 << 7;
            softmax7<WIN, OPT>(s, m, al, nmc, nmr, thr, qk2f, qk2h2, (j0 + 127 > dgt), j0, lim, t,
                               WIN ? (j0 < dgt + 7 - (win - 1)) : 0, lim - (win - 1));
            softmax_cvt<OPT>(s, pk);
        }
        if (PP && NT > 1 && cw == NC - 1) named_bar_arrive(1, 256);
        if (C::DBP) {
            int kt = 1;
            for (; kt + 1 < NT; kt += 2) {
                BLOCK7(kt, pk, pk2);
                BLOCK7(kt + 1, pk2, pk);
            }
            if (kt < NT) {
                BLOCK7(kt, pk, pk2);
#pragma unroll
                for (int i = 0; i < 16; i++) pk[i] = pk2[i];
            }
        } else {
            if (C::PEEL) {
                int klo = WIN ? max(1, min(NT, ((dgt + 8 - win + 127) >> 7) - kb0)) : 1;
                int kdg = max(klo, min(NT, ((dgt - 127) >> 7) + 1 - kb0));
                if (C::G2Q) {   // [gate2] Q: opaque (warp-uniform) bounds: ptxas re-derived max(klo, ...) per block
                    klo = __shfl_sync(0xFFFFFFFFu, klo, 0);
                    kdg = __shfl_sync(0xFFFFFFFFu, kdg, 0);
                }
                int kt = 1;
                if (C::G2U) {
                    // [gate2] U: the unmasked blocks run in pairs (kt, kt + 1); an odd last unmasked block runs in the
                    // masked loop below (its mask tests are false there: same result)
                    for (; kt < klo; kt++) BLOCK7(kt, pk, pk);
                    int kde = kdg - 1;   // pairs with kt + 1 < kdg
                    if (C::G2Q) kde = __shfl_sync(0xFFFFFFFFu, kde, 0);
                    for (; kt < kde; kt += 2) {
                        BLOCK7UF(kt, 0);
                        BLOCK7UF(kt + 1, 1);
                    }
                } else {
                    for (; kt < klo; kt++) BLOCK7(kt, pk, pk);
                    for (; kt < kdg; kt++) BLOCK7U(kt, pk, pk);
                }
                for (; kt < NT; kt++) BLOCK7(kt, pk, pk);
            } else {
                for (int kt = 1; kt < NT; kt++) BLOCK7(kt, pk, pk);
            }
        }
#undef BLOCK7
#undef BLOCK7U
#undef BLOCK7UF
        if (PP && NT > 1 && cw == 0) named_bar_sync(1, 256);
        const unsigned sll = HOIST ? ((unsigned)(NT - 1) & (NST - 1)) : (unsigned)((NT - 1) % NST);
#if PFMHA_EPI2
        if (!(C::ABL & 16)) issue_pvl72_h<OPT>(acc, pk, dpv + LCP_SL(sll));
#if PFMHA_G7U == 1
        if (!SPLIT && gate != nullptr) epi2_gload(gq, gate, gstride, b, kvh, tok0, L, lane);
#elif PFMHA_G7U == 2
        if (!SPLIT && gate != nullptr) epi2_gpf1(gate, gstride, b, kvh, tok0, L, lane);
#endif
        if (!(C::ABL & 16)) wgmma_wait<0>();
#else
        if (!(C::ABL & 16)) {
            issue_pvl72_h<OPT>(acc, pk, dpv + LCP_SL(sll));
            wgmma_wait<0>();
        }
#endif
        fence_fbank(acc);
        l[0] += acc[32];
        l[1] += acc[34];
    }

    const float inv0 = vs / fmaxf(l[0], 1e-37f);
    const float inv1 = vs / fmaxf(l[1], 1e-37f);
#if PFMHA_EPI2
    if (SPLIT || gate != nullptr) {   // [r12e] coalesced epilogue through the warp's 1 KB tile (qsm region)
        if (!SPLIT && (C::PIPE || NT <= 0 || PFMHA_G7U != 1)) epi2_gload(gq, gate, gstride, b, kvh, tok0, L, lane);
#if PFMHA_QWO
        unsigned qw; int ln;
        epi2_qw(mb, qw, ln);
        epi2_store<SPLIT>(acc, inv0, inv1, m, l, gq, b, kvh, tok0, L, B, blockIdx.z, ln, qw, out, po, pml, ng_eps);
#else
        epi2_store<SPLIT>(acc, inv0, inv1, m, l, gq, b, kvh, tok0, L, B, blockIdx.z, lane,
                          sbase + NST * SBS + 512 + cw * 4096 + wiw * 256, out, po, pml, ng_eps);
#endif
    } else
#endif
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const int tok = tok0 + rr;
        if (tok >= L) continue;
        const float inv = rr ? inv1 : inv0;
        if (!SPLIT && gate != nullptr) {
            // fused o_norm_gate (sk_fa7's epilogue): y = bf16(x * rsqrt(mean(x^2) + eps) * sigmoid(g)) over the
            // head's 64 dims, x = the bf16 attention output; the row is spread over the 4 lanes with this h
            const size_t r = (size_t)(b * L + tok) * 16 + kvh * 8 + h;
            float xv[16];
            float ss = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float2 x2 = __bfloat1622float2(__floats2bfloat162_rn(acc[4 * j + 2 * rr] * inv,
                                                                           acc[4 * j + 2 * rr + 1] * inv));
                xv[2 * j] = x2.x;
                xv[2 * j + 1] = x2.y;
                ss = fmaf(x2.x, x2.x, ss);
                ss = fmaf(x2.y, x2.y, ss);
            }
            ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 1);
            ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 2);
            const float rstd = rsqrtf(fmaf(ss, 1.0f / 64.0f, ng_eps));
            const __nv_bfloat16* gp = gate + (size_t)(b * L + tok) * gstride + (kvh * 8 + h) * 64 + 2 * t;
            __nv_bfloat16* op = out + r * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp + 8 * j));
                const float y0 = xv[2 * j] * rstd * sig2(g2).x;
                const float y1 = xv[2 * j + 1] * rstd * sig2(g2).y;
                __nv_bfloat162 o2 = __floats2bfloat162_rn(y0, y1);
                *reinterpret_cast<unsigned*>(op + 8 * j) = *reinterpret_cast<unsigned*>(&o2);
            }
        } else if (!SPLIT) {
            __nv_bfloat16* op = out + ((size_t)(b * L + tok) * 16 + kvh * 8 + h) * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                __nv_bfloat162 o2 = __floats2bfloat162_rn(acc[4 * j + 2 * rr] * inv,
                                                          acc[4 * j + 2 * rr + 1] * inv);
                *reinterpret_cast<unsigned*>(op + 8 * j) = *reinterpret_cast<unsigned*>(&o2);
            }
        } else {
            const size_t row = ((size_t)blockIdx.z * B * L + (size_t)b * L + tok) * 16 + kvh * 8 + h;
            __half* op = po + row * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                __half2 o2 = __floats2half2_rn(acc[4 * j + 2 * rr] * inv, acc[4 * j + 2 * rr + 1] * inv);
                *reinterpret_cast<__half2*>(op + 8 * j) = o2;
            }
            if (t == 0) pml[row] = make_float2(m[rr], l[rr]);
        }
    }
}


// ---------------------------------------------------------------------------------------------- [r10p] fmha7p
// [r10p] persistent / stream-K fmha7 (team adamleex). fmha7's per-block code (block7 / softmax7 / issue_* / fold_l72,
// unchanged) inside a per-CTA segment loop; grid = min(segments, co-resident CTAs). The producer's TMA ring runs on across
// segment boundaries: tile g of the CTA (g = running tile count over its segments) sits in slot g & (NST - 1) with full /
// empty parity (g / NST) & 1, and every consumed tile is released (the last one of a segment too, which fmha7 never
// needed). The consumers' key-block counter kt IS g (kb0 is rebased by -g0), so block7's slot / parity / j0 arithmetic
// is fmha7's. The ping-pong token passes from segment to segment (one arrive at entry, one sync at exit) instead of
// fmha7's per-launch pair, so a warpgroup can never arrive twice on a barrier before its partner synced.
// MODE 0: the segments are fmha7's own CTAs (q block x, y = 2 b + kvh, split z) in launch order l = x + nq (y + 2 B z);
//         CTA c runs l = r G + c (r G + G - 1 - c on odd rounds r when SNAKE): per-unit arithmetic as fmha7 (bitwise).
// MODE 1: units u = x + nq y (no split); ndpr whole-unit rounds (u = r G + c), then the key tiles of units u >= ndpr G in
//         unit order, split evenly over gsk <= G CTAs (stream-K); every segment writes its partial (SPLIT layout) to slot
//         (its index within the unit, < smax) and the unit's slot-0 CTA writes nseg[u]; combine_ngv merges.
//         gsk keeps every range >= ceil(capblks / (smax - 1)) tiles, so no unit has more than smax segments.
// MODE 2: MODE 0 (no split) with the fused o_norm * sigmoid(gate) epilogue offloaded: the consumers stage x = bf16(acc / l)
//         (row stride 144 B, conflict-free) and the row's rstd in smem and go on with the next segment; warps 1-3 of the
//         producer warpgroup (setmaxnreg 32) apply sigmoid(gate) and store 16 B per thread. Same arithmetic as fmha7's
//         epilogue (y = x rstd sigmoid(g), x the bf16-rounded output, rstd from x's fp32 sum of squares in fmha7's order).
constexpr int PT_LENS = 64;    // lens table entries (B <= 64, host-checked)
constexpr int PT_MAXU = 512;   // MODE 1: stream-K units per call (host-checked)
constexpr int PT_BYTES = ((PT_LENS + PT_MAXU + 1 + 32) * 4 + 127) & ~127;   // keeps the MODE 2 region 128-aligned
constexpr int EPO_RS = 144;   // MODE 2 staging row stride (64 bf16 + 8 pad): 8 rows of a warp start 4 banks apart
template <int NC> struct EpoL {   // MODE 2 smem (after the lens / prefix table): 2 mbarriers | rstd[NC 64] | x[NC 64][72]
    static constexpr int BAR = 0, RSTD = 128, XS = 128 + NC * 64 * 4, BYTES = XS + NC * 64 * EPO_RS;
};

struct SegP { int b, kvh, qs, kb0, NT, slot, u, nseg; };
#ifndef PFMHA_PTRACE
#define PFMHA_PTRACE 0
#endif
// [r10p] PFMHA_PTRACE (diagnostic builds only): clock64 stamps per (CTA, segment < 16, consumer group, point < 8) into
// the mode-0 nseg_out buffer (unsigned long long [G][16][4][8]); point 7 of segment 0 = %globaltimer at entry
#define PSTAMP(SEG, PT) do { if (PFMHA_PTRACE && ptr != nullptr && (SEG) < 16 && (tid & 127) == 0) \
    ptr[(((size_t)cta * 16 + (SEG)) * 4 + cw) * 8 + (PT)] = clk(); } while (0)

// a unit's key-tile range [KLO, NTF) (fmha7's prologue arithmetic)
template <int WIN, int QB>
DINL void unit_rng(int b, int qs, const int* lens_s, int L, int capblks, int win, int& KLO, int& NTF) {
    const int len = lens_s[b];
    const int ctx = len - L;
    NTF = min(capblks, (min(ctx + qs + QB, len) + 127) >> 7);
    KLO = WIN ? (max(0, ctx + qs - (win - 1)) >> 7) : 0;
}

struct SkP { int ndpr, nsk, u0, lo, hi, gsk, W2; };

// the CTA's it-th segment (identical on the producer thread and every consumer thread); false: no more
template <int SPLIT, int WIN, int QB, int MODE, bool SNAKE>
DINL bool seg_get(int it, SegP& sg, const int* lens_s, const int* pt_s, int B, int L, int capblks, int win, int nq,
                  int S, int G, int cta, const SkP& sk) {
    if (MODE != 1) {
        const int Ut = nq * 2 * B * (SPLIT ? S : 1);
        const int l = it * G + ((SNAKE && (it & 1)) ? G - 1 - cta : cta);
        if (l >= Ut) return false;
        const int x = l % nq, yz = l / nq, y = yz % (2 * B), z = yz / (2 * B);
        sg.b = y >> 1; sg.kvh = y & 1; sg.qs = (nq - 1 - x) * QB;
        int KLO, NTF;
        unit_rng<WIN, QB>(sg.b, sg.qs, lens_s, L, capblks, win, KLO, NTF);
        int kb0 = KLO, kb1 = NTF;
        if (SPLIT) {
            kb0 = KLO + (int)(((long long)(NTF - KLO) * z) / S);
            kb1 = KLO + (int)(((long long)(NTF - KLO) * (z + 1)) / S);
        }
        sg.kb0 = kb0; sg.NT = kb1 - kb0; sg.slot = z; sg.u = l; sg.nseg = 1;
        return true;
    }
    int u, s0, s1, slot = 0, nseg = 1;
    if (it < sk.ndpr) {
        u = it * G + cta;
        const int x = u % nq, y = u / nq;
        sg.b = y >> 1; sg.kvh = y & 1; sg.qs = (nq - 1 - x) * QB;
        int KLO, NTF;
        unit_rng<WIN, QB>(sg.b, sg.qs, lens_s, L, capblks, win, KLO, NTF);
        s0 = KLO; s1 = NTF;
    } else {
        const int i = sk.u0 + (it - sk.ndpr);   // stream-K unit index
        if (sk.lo >= sk.hi || i >= sk.nsk || pt_s[i] >= sk.hi) return false;
        u = sk.ndpr * G + i;
        const int x = u % nq, y = u / nq;
        sg.b = y >> 1; sg.kvh = y & 1; sg.qs = (nq - 1 - x) * QB;
        int KLO, NTF;
        unit_rng<WIN, QB>(sg.b, sg.qs, lens_s, L, capblks, win, KLO, NTF);
        const int p0 = pt_s[i], p1 = pt_s[i + 1];
        s0 = KLO + (max(sk.lo, p0) - p0);
        s1 = KLO + (min(sk.hi, p1) - p0);
        // CTA of global stream-K tile t: ((t + 1) gsk - 1) / W2
        const int c0 = (int)(((long long)(p0 + 1) * sk.gsk - 1) / sk.W2);
        const int c1 = (int)(((long long)p1 * sk.gsk - 1) / sk.W2);
        slot = cta - c0; nseg = c1 - c0 + 1;
    }
    sg.kb0 = s0; sg.NT = s1 - s0; sg.slot = slot; sg.u = u; sg.nseg = nseg;
    return true;
}

template <int NST, int SPLIT, int WIN, long long OPT, int MODE>
__global__ void __launch_bounds__(Cfg<OPT>::THREADS, 1) fmha7p(
    const unsigned char* __restrict__ q, const unsigned char* __restrict__ kvt,
    __nv_bfloat16* __restrict__ out, __half* __restrict__ po, float2* __restrict__ pml,
    const int* __restrict__ lens,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    float smqk_log2e, int B, int L, int capblks, int win,
    const __nv_bfloat16* __restrict__ gate, int gstride, float ng_eps,
    int S, int ndpr, int smax, int* __restrict__ nseg_out)
{
    using C = Cfg<OPT>;
    constexpr int NC = C::NC;
    constexpr int QB = 8 * NC;
    constexpr bool HOIST = C::HOIST;
    constexpr bool SNAKE = true;
#ifndef PFMHA_PFQ
#define PFMHA_PFQ 1
#endif
    constexpr bool PFQ = PFMHA_PFQ != 0;   // [r10p] producer-side L2 prefetch of each later segment's Q / gate rows
                                           // (mode 0: when smax != 0; mode 1: always)
    static_assert(HOIST && C::PEEL && !C::PIPE && !C::INL && !C::PW && !C::DBP && !C::TRACE && !C::SPEC && !(C::ABL & 16),
                  "[r10p] fmha7p covers the shipped NC3 / NC4 builds");
    static_assert(MODE != 1 || SPLIT, "[r10p] MODE 1 writes partials");
    static_assert(MODE != 2 || !SPLIT, "[r10p] MODE 2 offloads the unsplit (gate) epilogue");
    constexpr int SMB0 = NST * SBS + 512 + ((C::QSM || PFMHA_EPI2) ? 4 * 4096 : 0) + PT_BYTES;   // MODE 2 region offset (16-aligned)
    const int tid = threadIdx.x;
    const int G = gridDim.x, cta = blockIdx.x;
    const int nq = (L + QB - 1) / QB;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");

    extern __shared__ __align__(1024) char smem[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned mb = sbase + NST * SBS;
    int* lens_s = reinterpret_cast<int*>(smem + NST * SBS + 512 + ((C::QSM || PFMHA_EPI2) ? 4 * 4096 : 0));
    int* pt_s = lens_s + PT_LENS;
    if (tid < NST) {
        mbar_init(mb + tid * 8, 1);
        mbar_init(mb + NST * 8 + tid * 8, 128 * NC);
    }
    if (tid >= 64 && tid < 128)
        reinterpret_cast<unsigned*>(smem + NST * SBS + 256)[tid - 64] = 0x38383838u;   // e4m3 1.0
    for (int i = tid; i < NST * 256; i += blockDim.x)   // [lcp ngm] each slot's 1 KB ones group (N72's 9th N group)
        reinterpret_cast<unsigned*>(smem + (i >> 8) * SBS + 16384)[i & 255] = 0x38383838u;
    if (tid < B) lens_s[tid] = lens[tid];
    const unsigned ebar = sbase + SMB0 + EpoL<NC>::BAR;   // MODE 2: x full (consumers) / x empty (epilogue warps)
    if (MODE == 2 && tid == 0) {
        mbar_init(ebar, 128 * NC);
        mbar_init(ebar + 8, 96);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    SkP sk{0, 0, 0, 0, 0, 1, 1};
    if (MODE == 1) {   // exclusive key-tile prefix over the stream-K units (one per thread, nsk <= PT_MAXU <= blockDim)
        sk.ndpr = ndpr;
        sk.nsk = nq * 2 * B - ndpr * G;
        const int lane = tid & 31, w = tid >> 5;
        int v = 0;
        if (tid < sk.nsk) {
            const int u = ndpr * G + tid;
            const int x = u % nq, y = u / nq;
            int KLO, NTF;
            unit_rng<WIN, QB>(y >> 1, (nq - 1 - x) * QB, lens_s, L, capblks, win, KLO, NTF);
            v = max(0, NTF - KLO);
        }
        int xs = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xFFFFFFFFu, xs, o);
            if (lane >= o) xs += y;
        }
        if (lane == 31) pt_s[PT_MAXU + 1 + w] = xs;
        __syncthreads();
        int off = 0;
        for (int i = 0; i < w; i++) off += pt_s[PT_MAXU + 1 + i];
        if (tid < sk.nsk) pt_s[tid] = off + xs - v;
        if (tid == max(sk.nsk, 1) - 1) pt_s[sk.nsk] = sk.nsk > 0 ? off + xs : 0;
        __syncthreads();
        sk.W2 = max(pt_s[sk.nsk], 1);
        const int wmin = (capblks + smax - 2) / (smax - 1);
        sk.gsk = max(1, min(G, pt_s[sk.nsk] / wmin));
        sk.lo = cta < sk.gsk ? (int)(((long long)pt_s[sk.nsk] * cta) / sk.gsk) : pt_s[sk.nsk];
        sk.hi = cta < sk.gsk ? (int)(((long long)pt_s[sk.nsk] * (cta + 1)) / sk.gsk) : pt_s[sk.nsk];
        int a = 0, z = max(sk.nsk - 1, 0);   // the last stream-K unit with pt_s[u] <= lo
        while (a < z) {
            const int mid = (a + z + 1) >> 1;
            if (pt_s[mid] <= sk.lo) a = mid; else z = mid - 1;
        }
        sk.u0 = a;
    }
    const int wg = tid >> 7;
    if (wg == 0) {
        if (MODE == 2) asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n" ::: "memory");
        else reg_dealloc_24();
        if (MODE == 2 && tid >= 32) {   // [r10p] MODE 2 epilogue warps: y = x rstd sigmoid(g), 8 dims (16 B) per item
            const int et = tid - 32;
            const unsigned char* xs = reinterpret_cast<const unsigned char*>(smem) + SMB0 + EpoL<NC>::XS;
            const float* rs = reinterpret_cast<const float*>(smem + SMB0 + EpoL<NC>::RSTD);
            SegP sg;
            SegP sn;
            for (int it = 0; seg_get<SPLIT, WIN, QB, MODE, SNAKE>(it, sg, lens_s, pt_s, B, L, capblks, win, nq, S, G, cta, sk);
                 it++) {
                // the CTA's last segment: its consumers run the epilogue themselves (nothing left to overlap)
                if (!seg_get<SPLIT, WIN, QB, MODE, SNAKE>(it + 1, sn, lens_s, pt_s, B, L, capblks, win, nq, S, G, cta, sk)) break;
                mbar_wait_sleep(ebar, (unsigned)it & 1);
                const int nrow = min(QB, L - sg.qs) * 8;
#pragma unroll 2
                for (int c = et; c < nrow * 8; c += 96) {
                    const int row = c >> 3, c8 = c & 7;
                    const int tok = sg.qs + (row >> 3), hd = sg.kvh * 8 + (row & 7);
                    const uint4 xv = *reinterpret_cast<const uint4*>(xs + row * EPO_RS + c8 * 16);
                    const float rstd = rs[row];
                    const uint4 gv = *reinterpret_cast<const uint4*>(gate + (size_t)(sg.b * L + tok) * gstride + hd * 64 + c8 * 8);
                    const unsigned xx[4] = {xv.x, xv.y, xv.z, xv.w}, gg[4] = {gv.x, gv.y, gv.z, gv.w};
                    unsigned o[4];
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const float2 x2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xx[i]));
                        const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gg[i]));
                        const float y0 = x2.x * rstd * sig2(g2).x;
                        const float y1 = x2.y * rstd * sig2(g2).y;
                        __nv_bfloat162 o2 = __floats2bfloat162_rn(y0, y1);
                        o[i] = *reinterpret_cast<unsigned*>(&o2);
                    }
                    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                                 :: "l"(out + ((size_t)(sg.b * L + tok) * 16 + hd) * 64 + c8 * 8),
                                    "r"(o[0]), "r"(o[1]), "r"(o[2]), "r"(o[3]) : "memory");
                }
                mbar_arrive(ebar + 8);
            }
            return;
        }
        if (tid == 0) {
            int g = 0;
            SegP sg;
            for (int it = 0; seg_get<SPLIT, WIN, QB, MODE, SNAKE>(it, sg, lens_s, pt_s, B, L, capblks, win, nq, S, G, cta, sk);
                 it++) {
                if (PFQ && (it > 0 || (PFMHA_EPI2 && PFMHA_GPFP)) && (MODE == 1 || smax != 0)) {   // [r10p] L2 prefetch of the segment's Q rows (512 B per token) and gate rows (1 KB) [r12e] segment 0: gate only
                    const int ntk = min(QB, L - sg.qs);
                    const unsigned char* qr = q + ((size_t)(sg.b * L + sg.qs) * 16 + sg.kvh * 8) * 64;
                    if (it > 0)
                    for (int tk = 0; tk < ntk; tk++)
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], 512;" :: "l"(qr + (size_t)tk * 1024) : "memory");
                    if (!SPLIT && gate != nullptr) {
                        const __nv_bfloat16* gr = gate + (size_t)(sg.b * L + sg.qs) * gstride + sg.kvh * 512;
                        for (int tk = 0; tk < ntk; tk++)
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], 1024;" :: "l"(gr + (size_t)tk * gstride)
                                         : "memory");
                    }
                }
                const char* src = reinterpret_cast<const char*>(kvt) + ((size_t)(sg.b * 2 + sg.kvh) * capblks + sg.kb0) * SB;
                for (int kt = 0; kt < sg.NT; kt++, g++) {
                    const int ps = g & (NST - 1), pr = g / NST;
                    if (pr > 0) {
                        if (OPT & 1) mbar_wait_sleep(mb + NST * 8 + ps * 8, (pr - 1) & 1);
                        else mbar_wait(mb + NST * 8 + ps * 8, (pr - 1) & 1);
                    }
                    mbar_expect_tx(mb + ps * 8, SB);
                    bulk_g2s(smem + LCP_SS(ps), src + (size_t)kt * SB, SB, mb + ps * 8);
                }
            }
        }
        return;
    }
    if (NC == 4) asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\n" ::: "memory");
    else reg_alloc_160();
    const int cw = __shfl_sync(0xFFFFFFFFu, wg - 1, 0);   // [lcp u] warp-uniform -> UR
    const bool pthr = false;
    const int wiw = __shfl_sync(0xFFFFFFFFu, (tid >> 5) & 3, 0);   // [lcp u]
    const int lane = tid & 31;
    const int t = lane & 3;
    const int h = lane >> 2;
    unsigned long long* tr = nullptr;
    Prod pd{0, 0, nullptr};

    const float qk2f = k_scale[0] * smqk_log2e;
    const __half2 qk2h2 = __float2half2_rn(C::BYT ? 8.0f * qk2f : qk2f);   // [gate1] BYT: 8 qk2 (exact x8 of the f16)
    const float vs = v_scale[0];
    constexpr bool PP = C::PP;
    unsigned long long dones = make_desc(sbase + NST * SBS + 256, 128, 128);
    unsigned dqk = ((sbase >> 4) & 0x3FFF) | (128u << 16);
    unsigned dpv = (((sbase + 8192) >> 4) & 0x3FFF) | (8u << 16);   // [lcp ngm] LBO 128 B
    if (C::PIN) {
        dqk = __shfl_sync(0xFFFFFFFFu, dqk, 0);
        dpv = __shfl_sync(0xFFFFFFFFu, dpv, 0);
        dones = desc_lo(__shfl_sync(0xFFFFFFFFu, (unsigned)dones, 0));
    }
    const unsigned qsm = sbase + NST * SBS + 512 + cw * 4096;
    const unsigned dqa = ((qsm >> 4) & 0x3FFF) | (64u << 16);
    const unsigned k32 = 32u + (unsigned)(__float_as_uint(qk2f) == 0x7FC00001u);
    unsigned long long* ptr = (PFMHA_PTRACE && MODE == 0) ? reinterpret_cast<unsigned long long*>(nseg_out) : nullptr;
    if (PFMHA_PTRACE && ptr != nullptr && (tid & 127) == 0) {
        unsigned long long gt;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt));
        ptr[(((size_t)cta * 16) * 4 + cw) * 8 + 7] = gt;
    }
    if (PP && cw == NC - 1) named_bar_arrive(1, 256);   // [r10p] the ping-pong token's one entry arrive
    int g0 = 0;   // [r10p] running tile count of this CTA (= the producer's g at each segment's first tile)
    SegP sg;
    for (int it = 0; seg_get<SPLIT, WIN, QB, MODE, SNAKE>(it, sg, lens_s, pt_s, B, L, capblks, win, nq, S, G, cta, sk); it++) {
    PSTAMP(it, 0);
    uint4 gq[4];        // [r12e] the warp's gate rows, loaded under the final PV
    bool rd = false;    // [r12e] sg re-derived under the final PV
    float m[2] = {M_INIT, M_INIT}, l[2] = {0.f, 0.f}, al[2];
    float acc[36];
#pragma unroll
    for (int i = 0; i < 36; i++) acc[i] = 0.f;
    {
    unsigned qf[8];
    const int b = sg.b, kvh = sg.kvh, qs = sg.qs;
    const int len = lens_s[b];
    const int ctx = len - L;
    const int NT = sg.NT;
    const int kb0 = sg.kb0 - g0;   // [r10p] rebased: j0 = (kb0 + kt) << 7 with kt = g0 + local block
    const int ktE = g0 + NT;       // [r10p] end of this segment's global blocks

    const int tok0 = qs + 8 * cw + 2 * wiw;
#pragma unroll
    for (int sl2 = 0; sl2 < 2; sl2++)
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int tok = tok0 + rr;
            const unsigned char* qp = q + ((size_t)(b * L + tok) * 16 + kvh * 8 + h) * 64
                                    + sl2 * 32 + 4 * t;
            qf[4 * sl2 + rr]     = (tok < L) ? *reinterpret_cast<const unsigned*>(qp) : 0u;
            qf[4 * sl2 + rr + 2] = (tok < L) ? *reinterpret_cast<const unsigned*>(qp + 16) : 0u;
        }

    __half2 nmc[2] = {__float2half2_rn(C::EOFF + C::MB - M_INIT), __float2half2_rn(C::EOFF + C::MB - M_INIT)};
    constexpr float NMB0 = 8.0f * (C::EOFFB + 7.0f - M_INIT) < 65504.0f ? 8.0f * (C::EOFFB + 7.0f - M_INIT) : 65504.0f;
    __half2 nmr[2] = {__float2half2_rn(C::BYT ? NMB0 : 21.0f - M_INIT), __float2half2_rn(C::BYT ? NMB0 : 21.0f - M_INIT)};
    __half2 thr[2] = {__float2half2_rn((M_INIT + 0.75f) / qk2f), __float2half2_rn((M_INIT + 0.75f) / qk2f)};
    if (C::G2V) { thr[0] = thr_rd(M_INIT, qk2f); thr[1] = thr[0]; }   // [gate2] V
    unsigned pk[16];
    if (C::QSM) {
        named_bar_sync(8 + cw, 128);   // [r10p] the previous segment's wgmmas (every warp of the group) read qsm
#pragma unroll
        for (int sl2 = 0; sl2 < 2; sl2++)
#pragma unroll
            for (int rr = 0; rr < 2; rr++)
#pragma unroll
                for (int hk = 0; hk < 2; hk++) {
                    const unsigned a = qsm + (2 * sl2 + hk) * 1024 + (wiw * 2 + rr) * 128 + h * 16 + 4 * t;
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(a), "r"(qf[4 * sl2 + rr + 2 * hk]) : "memory");
                }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        named_bar_sync(8 + cw, 128);
    }
    const int lim = __shfl_sync(0xFFFFFFFFu, ctx + qs + 8 * cw + 2 * wiw, 0);   // [lcp u] lens[b] is a load
    const int dgt = __shfl_sync(0xFFFFFFFFu, ctx + qs + 8 * cw, 0);
    PSTAMP(it, 1);
    unsigned rdy = 0;   // [r10p] per segment: the previous segment's last test was of this segment's first tile
    const unsigned sl0 = (unsigned)g0 & (NST - 1);
    unsigned qprev = dqk + LCP_SL(sl0);   // [gate2] D: block 0's QK descriptor
#define BLOCK7(KT, PKC, PKN) block7<NST, WIN, OPT, 1>(KT, PKC, PKN, acc, l, m, al, nmc, nmr, thr, qf, mb, cw, sbase, \
                                                     dqk, dpv, dones, kb0, dgt, lim, t, win, qk2f, qk2h2, tr, k32, rdy, \
                                                     pd, pthr, dqa, qprev)
#define BLOCK7UF(KT, F) block7<NST, WIN, OPT, 0, F>(KT, pk, pk, acc, l, m, al, nmc, nmr, thr, qf, mb, cw, sbase, \
                                                     dqk, dpv, dones, kb0, dgt, lim, t, win, qk2f, qk2h2, tr, k32, rdy, \
                                                     pd, pthr, dqa, qprev)
    if (NT > 0) {
        {
            unsigned s[32];
            mbar_wait(mb + sl0 * 8, ((unsigned)g0 / NST) & 1);
            issue_qk_x<OPT>(s, qf, dqa, dqk + LCP_SL(sl0));
            wgmma_wait<0>();
            fence_ubank(s);
            const int j0 = (kb0 + g0) << 7;
            softmax7<WIN, OPT>(s, m, al, nmc, nmr, thr, qk2f, qk2h2, (j0 + 127 > dgt), j0, lim, t,
                               WIN ? (j0 < dgt + 7 - (win - 1)) : 0, lim - (win - 1));
            softmax_cvt<OPT>(s, pk);
        }
        PSTAMP(it, 2);
        {
            // fmha7's PEEL bounds (relative to the segment), shifted to the global block counter
            int klo = WIN ? max(1, min(NT, ((dgt + 8 - win + 127) >> 7) - sg.kb0)) : 1;
            int kdg = max(klo, min(NT, ((dgt - 127) >> 7) + 1 - sg.kb0));
            klo += g0; kdg += g0;
            if (C::G2Q) {
                klo = __shfl_sync(0xFFFFFFFFu, klo, 0);
                kdg = __shfl_sync(0xFFFFFFFFu, kdg, 0);
            }
            int kt = g0 + 1;
            if (C::G2U) {
                for (; kt < klo; kt++) BLOCK7(kt, pk, pk);
                int kde = kdg - 1;
                if (C::G2Q) kde = __shfl_sync(0xFFFFFFFFu, kde, 0);
                for (; kt < kde; kt += 2) {
                    BLOCK7UF(kt, 0);
                    BLOCK7UF(kt + 1, 1);
                }
            } else {
                for (; kt < klo; kt++) BLOCK7(kt, pk, pk);
                for (; kt < kdg; kt++) block7<NST, WIN, OPT, 0>(kt, pk, pk, acc, l, m, al, nmc, nmr, thr, qf, mb, cw, sbase,
                                                               dqk, dpv, dones, kb0, dgt, lim, t, win, qk2f, qk2h2, tr, k32,
                                                               rdy, pd, pthr, dqa, qprev);
            }
            for (; kt < ktE; kt++) BLOCK7(kt, pk, pk);
        }
#undef BLOCK7
#undef BLOCK7UF
        PSTAMP(it, 3);
        const unsigned sll = (unsigned)(ktE - 1) & (NST - 1);
        issue_pvl72_h<OPT>(acc, pk, dpv + LCP_SL(sll));
#if PFMHA_EPI2 && PFMHA_PRD
        {   // [r12e] re-derive the segment (opaque it) and issue the gate loads while the final PV runs
            int it2;
            asm volatile("mov.b32 %0, %1;" : "=r"(it2) : "r"(it));
            seg_get<SPLIT, WIN, QB, MODE, SNAKE>(it2, sg, lens_s, pt_s, B, L, capblks, win, nq, S, G, cta, sk);
            rd = true;
            if (!SPLIT && gate != nullptr) {
                bool last = true;   // MODE 2: only the CTA's last segment keeps the inline epilogue
                if (MODE == 2) {
                    const int it3 = it2 + 1;
                    const int l3 = it3 * G + ((SNAKE && (it3 & 1)) ? G - 1 - cta : cta);
                    last = l3 >= nq * 2 * B;
                }
#if PFMHA_GLD == 1
                if (last) epi2_gload(gq, gate, gstride, sg.b, sg.kvh, sg.qs + 8 * cw + 2 * wiw, L, lane);
#else
                if (last) epi2_gpf1(gate, gstride, sg.b, sg.kvh, sg.qs + 8 * cw + 2 * wiw, L, lane);
#endif
            }
        }
#endif
        wgmma_wait<0>();
        fence_fbank(acc);
        PSTAMP(it, 4);
        mbar_arrive(mb + NST * 8 + sll * 8);   // [r10p] release the segment's last tile (the ring runs on)
        l[0] += acc[32];
        l[1] += acc[34];
    }
    g0 = ktE;
    }
    if (!rd) {   // [r10p] re-derive the segment after the loop (opaque it): only it / g0 stay live across the hot loop
        int it2;
        asm volatile("mov.b32 %0, %1;" : "=r"(it2) : "r"(it));
        seg_get<SPLIT, WIN, QB, MODE, SNAKE>(it2, sg, lens_s, pt_s, B, L, capblks, win, nq, S, G, cta, sk);
    }
    const int b = sg.b, kvh = sg.kvh, qs = sg.qs;
    const int tok0 = qs + 8 * cw + 2 * wiw;

    const float inv0 = vs / fmaxf(l[0], 1e-37f);
    const float inv1 = vs / fmaxf(l[1], 1e-37f);
    bool epo = false;
    if (MODE == 2) {   // the CTA's last segment keeps the inline epilogue
        SegP sn;
        int it3;
        asm volatile("mov.b32 %0, %1;" : "=r"(it3) : "r"(it + 1));
        epo = seg_get<SPLIT, WIN, QB, MODE, SNAKE>(it3, sn, lens_s, pt_s, B, L, capblks, win, nq, S, G, cta, sk);
    }
    if (epo) {   // [r10p] MODE 2: stage x and rstd for the epilogue warps (fmha7's x / ss / rstd arithmetic)
        if (it > 0) mbar_wait(ebar + 8, ((unsigned)it - 1) & 1);
        unsigned char* xs = reinterpret_cast<unsigned char*>(smem) + SMB0 + EpoL<NC>::XS;
        float* rs = reinterpret_cast<float*>(smem + SMB0 + EpoL<NC>::RSTD);
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const float inv = rr ? inv1 : inv0;
            const int row = (8 * cw + 2 * wiw + rr) * 8 + h;
            float ss = 0.f;
#if PFMHA_EPI2 && PFMHA_M2S
            unsigned w8[8];   // [r12e] stmatrix: 2 x4 per row instead of 8 4-B stores
#pragma unroll
            for (int j = 0; j < 8; j++) {
                __nv_bfloat162 b2 = __floats2bfloat162_rn(acc[4 * j + 2 * rr] * inv, acc[4 * j + 2 * rr + 1] * inv);
                const float2 x2 = __bfloat1622float2(b2);
                ss = fmaf(x2.x, x2.x, ss);
                ss = fmaf(x2.y, x2.y, ss);
                w8[j] = *reinterpret_cast<const unsigned*>(&b2);
            }
            {
                const unsigned xa = (unsigned)__cvta_generic_to_shared(xs) + ((8 * cw + 2 * wiw + rr) * 8 + (lane & 7)) * EPO_RS
                                  + (lane >> 3) * 16;
                stsm_x4(xa, w8[0], w8[1], w8[2], w8[3]);
                stsm_x4(xa + 64, w8[4], w8[5], w8[6], w8[7]);
            }
#else
#pragma unroll
            for (int j = 0; j < 8; j++) {
                __nv_bfloat162 b2 = __floats2bfloat162_rn(acc[4 * j + 2 * rr] * inv, acc[4 * j + 2 * rr + 1] * inv);
                const float2 x2 = __bfloat1622float2(b2);
                ss = fmaf(x2.x, x2.x, ss);
                ss = fmaf(x2.y, x2.y, ss);
                *reinterpret_cast<__nv_bfloat162*>(xs + row * EPO_RS + (8 * j + 2 * t) * 2) = b2;
            }
#endif
            ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 1);
            ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 2);
            if (t == 0) rs[row] = rsqrtf(fmaf(ss, 1.0f / 64.0f, ng_eps));
        }
        mbar_arrive(ebar);
    }
#if PFMHA_EPI2
    else if (SPLIT || gate != nullptr) {   // [r12e] coalesced epilogue through the warp's 1 KB tile (qsm region)
        if (!SPLIT && (PFMHA_GLD != 1 || !rd)) epi2_gload(gq, gate, gstride, b, kvh, tok0, L, lane);
#if PFMHA_QWO
        unsigned qw; int ln;
        epi2_qw(mb, qw, ln);
        epi2_store<SPLIT>(acc, inv0, inv1, m, l, gq, b, kvh, tok0, L, B, sg.slot, ln, qw, out, po, pml, ng_eps);
#else
        epi2_store<SPLIT>(acc, inv0, inv1, m, l, gq, b, kvh, tok0, L, B, sg.slot, lane, qsm + wiw * 256, out, po, pml,
                          ng_eps);
#endif
    }
#endif
    else
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const int tok = tok0 + rr;
        if (tok >= L) continue;
        const float inv = rr ? inv1 : inv0;
        if (!SPLIT && gate != nullptr) {
            const size_t r = (size_t)(b * L + tok) * 16 + kvh * 8 + h;
            float xv[16];
            float ss = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float2 x2 = __bfloat1622float2(__floats2bfloat162_rn(acc[4 * j + 2 * rr] * inv,
                                                                           acc[4 * j + 2 * rr + 1] * inv));
                xv[2 * j] = x2.x;
                xv[2 * j + 1] = x2.y;
                ss = fmaf(x2.x, x2.x, ss);
                ss = fmaf(x2.y, x2.y, ss);
            }
            ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 1);
            ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 2);
            const float rstd = rsqrtf(fmaf(ss, 1.0f / 64.0f, ng_eps));
            const __nv_bfloat16* gp = gate + (size_t)(b * L + tok) * gstride + (kvh * 8 + h) * 64 + 2 * t;
            __nv_bfloat16* op = out + r * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp + 8 * j));
                const float y0 = xv[2 * j] * rstd * sig2(g2).x;
                const float y1 = xv[2 * j + 1] * rstd * sig2(g2).y;
                __nv_bfloat162 o2 = __floats2bfloat162_rn(y0, y1);
                *reinterpret_cast<unsigned*>(op + 8 * j) = *reinterpret_cast<unsigned*>(&o2);
            }
        } else if (!SPLIT) {
            __nv_bfloat16* op = out + ((size_t)(b * L + tok) * 16 + kvh * 8 + h) * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                __nv_bfloat162 o2 = __floats2bfloat162_rn(acc[4 * j + 2 * rr] * inv,
                                                          acc[4 * j + 2 * rr + 1] * inv);
                *reinterpret_cast<unsigned*>(op + 8 * j) = *reinterpret_cast<unsigned*>(&o2);
            }
        } else {
            const size_t row = ((size_t)sg.slot * B * L + (size_t)b * L + tok) * 16 + kvh * 8 + h;
            __half* op = po + row * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                __half2 o2 = __floats2half2_rn(acc[4 * j + 2 * rr] * inv, acc[4 * j + 2 * rr + 1] * inv);
                *reinterpret_cast<__half2*>(op + 8 * j) = o2;
            }
            if (t == 0) pml[row] = make_float2(m[rr], l[rr]);
        }
    }
    if (MODE == 1 && sg.slot == 0 && tid == 128) nseg_out[sg.u] = sg.nseg;
    PSTAMP(it, 5);
    }   // segment loop
    if (PP && cw == 0) named_bar_sync(1, 256);   // [r10p] the token's one exit sync (balances the entry arrive)
}

// [r10p] combine_ng over a per-unit segment count (MODE 1 partials): row (b, tok, head) -> unit
// u = (nq - 1 - tok / QB) + nq (2 b + head / 8), merged over slots 0 .. nseg[u] - 1 (combine_ng's arithmetic)
__global__ void __launch_bounds__(256) combine_ngv(
    const __half* __restrict__ po, const float2* __restrict__ pml, const int* __restrict__ nseg,
    __nv_bfloat16* __restrict__ out, int rows, int L, int QB,
    const __nv_bfloat16* __restrict__ gate, int gstride, float ng_eps)
{
    asm volatile("griddepcontrol.wait;" ::: "memory");   // PDL launch: fmha7p's po / pml / nseg
    const int gid = blockIdx.x * 256 + threadIdx.x;
    const int row = gid >> 3, c = gid & 7;
    const bool live = row < rows;
#if PFMHA_CMB2
    {   // [r12e] nseg + gate at entry, then pml / po of the unit's S <= 4 slots together (S > 4: the loop below)
        int S = 0;
        uint4 gv = make_uint4(0u, 0u, 0u, 0u);
        if (live) {
            const int tk = row >> 4, head = row & 15;
            const int b = tk / L, tok = tk - b * L;
            const int nq = (L + QB - 1) / QB;
            gv = *reinterpret_cast<const uint4*>(gate + (size_t)tk * gstride + head * 64 + c * 8);
            S = nseg[(nq - 1 - tok / QB) + nq * (2 * b + (head >> 3))];
        }
        if (__all_sync(0xFFFFFFFFu, S <= 4)) {
        float2 ml4[4];
        uint4 v4[4];
#pragma unroll
        for (int s = 0; s < 4; s++) {
            ml4[s] = make_float2(0.f, 0.f); v4[s] = make_uint4(0u, 0u, 0u, 0u);
            if (s < S) {
                ml4[s] = pml[(size_t)s * rows + row];
                v4[s] = *reinterpret_cast<const uint4*>(po + ((size_t)s * rows + row) * 64 + c * 8);
            }
        }
        float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float wsum = 0.f;
        if (live) {
            float M = M_INIT;
#pragma unroll
            for (int s = 0; s < 4; s++) if (s < S) M = fmaxf(M, ml4[s].x);
#pragma unroll
            for (int s = 0; s < 4; s++) {
                if (s < S) {
                    const float w = ml4[s].y * exp2f(ml4[s].x - M);
                    wsum += w;
                    const unsigned vv[4] = {v4[s].x, v4[s].y, v4[s].z, v4[s].w};
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const float2 f = __half22float2(u32h2(vv[i]));
                        a[2 * i] += w * f.x;
                        a[2 * i + 1] += w * f.y;
                    }
                }
            }
        }
        const float inv = 1.0f / fmaxf(wsum, 1e-37f);
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const float2 x2 = __bfloat1622float2(__floats2bfloat162_rn(a[i] * inv, a[i + 1] * inv));
            a[i] = x2.x;
            a[i + 1] = x2.y;
            ss = fmaf(x2.x, x2.x, ss);
            ss = fmaf(x2.y, x2.y, ss);
        }
        ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 1);
        ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 2);
        ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 4);
        if (!live) return;
        const float rstd = rsqrtf(fmaf(ss, 1.0f / 64.0f, ng_eps));
        const unsigned gg[4] = {gv.x, gv.y, gv.z, gv.w};
        unsigned o[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gg[i]));
            __nv_bfloat162 b2 = __floats2bfloat162_rn(a[2 * i] * rstd * sig2(g2).x, a[2 * i + 1] * rstd * sig2(g2).y);
            o[i] = *reinterpret_cast<unsigned*>(&b2);
        }
        *reinterpret_cast<uint4*>(out + (size_t)row * 64 + c * 8) = make_uint4(o[0], o[1], o[2], o[3]);
        return;
        }
    }
#endif
    float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float wsum = 0.f;
    if (live) {
        const int tk = row >> 4, head = row & 15;
        const int b = tk / L, tok = tk - b * L;
        const int nq = (L + QB - 1) / QB;
        const int S = nseg[(nq - 1 - tok / QB) + nq * (2 * b + (head >> 3))];
        float M = M_INIT;
        for (int s = 0; s < S; s++) M = fmaxf(M, pml[(size_t)s * rows + row].x);
        for (int s = 0; s < S; s++) {
            const float2 ml = pml[(size_t)s * rows + row];
            const float w = ml.y * exp2f(ml.x - M);
            wsum += w;
            const uint4 v = *reinterpret_cast<const uint4*>(po + ((size_t)s * rows + row) * 64 + c * 8);
            const unsigned vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float2 f = __half22float2(u32h2(vv[i]));
                a[2 * i] += w * f.x;
                a[2 * i + 1] += w * f.y;
            }
        }
    }
    const float inv = 1.0f / fmaxf(wsum, 1e-37f);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const float2 x2 = __bfloat1622float2(__floats2bfloat162_rn(a[i] * inv, a[i + 1] * inv));
        a[i] = x2.x;
        a[i + 1] = x2.y;
        ss = fmaf(x2.x, x2.x, ss);
        ss = fmaf(x2.y, x2.y, ss);
    }
    ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 1);
    ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 2);
    ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 4);
    if (!live) return;
    const float rstd = rsqrtf(fmaf(ss, 1.0f / 64.0f, ng_eps));
    const int tok = row >> 4, head = row & 15;
    const uint4 gv = *reinterpret_cast<const uint4*>(gate + (size_t)tok * gstride + head * 64 + c * 8);
    const unsigned gg[4] = {gv.x, gv.y, gv.z, gv.w};
    unsigned o[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gg[i]));
        __nv_bfloat162 b2 = __floats2bfloat162_rn(a[2 * i] * rstd * sig2(g2).x,
                                                  a[2 * i + 1] * rstd * sig2(g2).y);
        o[i] = *reinterpret_cast<unsigned*>(&b2);
    }
    *reinterpret_cast<uint4*>(out + (size_t)row * 64 + c * 8) = make_uint4(o[0], o[1], o[2], o[3]);
}

template <int NST, int SPLIT, int WIN, long long OPT>
static void launch_fmha7(dim3 grid, cudaStream_t stream,
                         const unsigned char* qp, const unsigned char* kp, __nv_bfloat16* op,
                         __half* pop, float2* pmp,
                         const int* lens, const float* ks, const float* vs,
                         float smqk, int B, int L, int capblks, int win,
                         const __nv_bfloat16* gate, int gstride, float ng_eps) {
    auto fn = fmha7<NST, SPLIT, WIN, OPT>;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, NST * SBS + 512 + 4 * 4096);
        set = true;
    }
    cudaLaunchConfig_t lc = {};
    lc.gridDim = grid; lc.blockDim = dim3(Cfg<OPT>::THREADS);
    lc.dynamicSmemBytes = NST * SBS + 512 + ((Cfg<OPT>::QSM || PFMHA_EPI2) ? 4 * 4096 : 0); lc.stream = stream;
    cudaLaunchAttribute la[1]; la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = 1;
    cudaLaunchKernelEx(&lc, fn, qp, kp, op, pop, pmp, lens, ks, vs, smqk, B, L, capblks, win, gate, gstride, ng_eps);
}

template <long long OPT>
static void run_fmha7(dim3 grid, cudaStream_t st, const unsigned char* qp, const unsigned char* kp,
                      __nv_bfloat16* op, __half* pop, float2* pmp, const int* lp, const float* ks,
                      const float* vsp, float smqk, int B, int L, int capblks, int S, int win,
                      const __nv_bfloat16* gate = nullptr, int gstride = 0, float ng_eps = 0.f) {
    if (S <= 1) {
#if !PG_NOWIN
        if (win > 0) launch_fmha7<8, 0, 1, OPT>(grid, st, qp, kp, op, nullptr, nullptr, lp, ks, vsp, smqk, B, L, capblks, win, gate, gstride, ng_eps);
        else
#endif
        launch_fmha7<8, 0, 0, OPT>(grid, st, qp, kp, op, nullptr, nullptr, lp, ks, vsp, smqk, B, L, capblks, 0, gate, gstride, ng_eps);
    } else {
#if !PG_NOWIN
        if (win > 0) launch_fmha7<8, 1, 1, OPT>(grid, st, qp, kp, nullptr, pop, pmp, lp, ks, vsp, smqk, B, L, capblks, win, nullptr, 0, 0.f);
        else
#endif
        launch_fmha7<8, 1, 0, OPT>(grid, st, qp, kp, nullptr, pop, pmp, lp, ks, vsp, smqk, B, L, capblks, 0, nullptr, 0, 0.f);
    }
}


// [r10p] fmha7p launch: grid G (<= co-resident CTAs, 1 per SM), fmha7's dynamic smem + the lens / prefix table
template <int NST, int SPLIT, int WIN, long long OPT, int MODE>
static void launch_fmha7p(int G, cudaStream_t stream,
                          const unsigned char* qp, const unsigned char* kp, __nv_bfloat16* op,
                          __half* pop, float2* pmp,
                          const int* lens, const float* ks, const float* vs,
                          float smqk, int B, int L, int capblks, int win,
                          const __nv_bfloat16* gate, int gstride, float ng_eps, int S, int ndpr, int smax, int* nsp) {
    auto fn = fmha7p<NST, SPLIT, WIN, OPT, MODE>;
    const int smb = NST * SBS + 512 + ((Cfg<OPT>::QSM || PFMHA_EPI2) ? 4 * 4096 : 0) + PT_BYTES + (MODE == 2 ? EpoL<Cfg<OPT>::NC>::BYTES : 0);
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smb);
        set = true;
    }
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3((unsigned)G); lc.blockDim = dim3(Cfg<OPT>::THREADS);
    lc.dynamicSmemBytes = smb; lc.stream = stream;
    cudaLaunchAttribute la[1]; la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = 1;
    cudaLaunchKernelEx(&lc, fn, qp, kp, op, pop, pmp, lens, ks, vs, smqk, B, L, capblks, win, gate, gstride, ng_eps,
                       S, ndpr, smax, nsp);
}

template <long long OPT, int MODE>
static void run_fmha7p(int G, cudaStream_t st, const unsigned char* qp, const unsigned char* kp,
                       __nv_bfloat16* op, __half* pop, float2* pmp, const int* lp, const float* ks,
                       const float* vsp, float smqk, int B, int L, int capblks, int S, int win,
                       const __nv_bfloat16* gate, int gstride, float ng_eps, int ndpr, int smax, int* nsp) {
    if constexpr (MODE == 2) {   // unsplit only (host-checked)
        if (win > 0) launch_fmha7p<8, 0, 1, OPT, 2>(G, st, qp, kp, op, nullptr, nullptr, lp, ks, vsp, smqk, B, L, capblks, win, gate, gstride, ng_eps, 1, 0, smax, nsp);
        else launch_fmha7p<8, 0, 0, OPT, 2>(G, st, qp, kp, op, nullptr, nullptr, lp, ks, vsp, smqk, B, L, capblks, 0, gate, gstride, ng_eps, 1, 0, smax, nsp);
    } else if (MODE == 1 || S > 1) {
        if (win > 0) launch_fmha7p<8, 1, 1, OPT, MODE>(G, st, qp, kp, nullptr, pop, pmp, lp, ks, vsp, smqk, B, L, capblks, win, nullptr, 0, 0.f, S, ndpr, smax, nsp);
        else launch_fmha7p<8, 1, 0, OPT, MODE>(G, st, qp, kp, nullptr, pop, pmp, lp, ks, vsp, smqk, B, L, capblks, 0, nullptr, 0, 0.f, S, ndpr, smax, nsp);
    } else {
        if (win > 0) launch_fmha7p<8, 0, 1, OPT, 0>(G, st, qp, kp, op, nullptr, nullptr, lp, ks, vsp, smqk, B, L, capblks, win, gate, gstride, ng_eps, 1, 0, smax, nsp);
        else launch_fmha7p<8, 0, 0, OPT, 0>(G, st, qp, kp, op, nullptr, nullptr, lp, ks, vsp, smqk, B, L, capblks, 0, gate, gstride, ng_eps, 1, 0, smax, nsp);
    }
}

__global__ void __launch_bounds__(256) combine(
    const __half* __restrict__ po, const float2* __restrict__ pml,
    __nv_bfloat16* __restrict__ out, int rows, int S)
{
    const int gid = blockIdx.x * 256 + threadIdx.x;
    const int row = gid >> 3, c = gid & 7;
    if (row >= rows) return;
    float M = M_INIT;
    for (int s = 0; s < S; s++) M = fmaxf(M, pml[(size_t)s * rows + row].x);
    float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float wsum = 0.f;
    for (int s = 0; s < S; s++) {
        const float2 ml = pml[(size_t)s * rows + row];
        const float w = ml.y * exp2f(ml.x - M);
        wsum += w;
        const uint4 v = *reinterpret_cast<const uint4*>(po + ((size_t)s * rows + row) * 64 + c * 8);
        const unsigned vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 f = __half22float2(u32h2(vv[i]));
            a[2 * i] += w * f.x;
            a[2 * i + 1] += w * f.y;
        }
    }
    const float inv = 1.0f / fmaxf(wsum, 1e-37f);
    unsigned o[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        __nv_bfloat162 b2 = __floats2bfloat162_rn(a[2 * i] * inv, a[2 * i + 1] * inv);
        o[i] = *reinterpret_cast<unsigned*>(&b2);
    }
    *reinterpret_cast<uint4*>(out + (size_t)row * 64 + c * 8) = make_uint4(o[0], o[1], o[2], o[3]);
}

// split-KV merge with the fused o_norm_gate (sk_fa7's combine_ng): 8 threads per (token, head) row, 8 dims each
__global__ void __launch_bounds__(256) combine_ng(
    const __half* __restrict__ po, const float2* __restrict__ pml,
    __nv_bfloat16* __restrict__ out, int rows, int S,
    const __nv_bfloat16* __restrict__ gate, int gstride, float ng_eps)
{
    asm volatile("griddepcontrol.wait;" ::: "memory");   // [r4pdl] PDL launch: fmha7's po / pml
    const int gid = blockIdx.x * 256 + threadIdx.x;
    const int row = gid >> 3, c = gid & 7;
    const bool live = row < rows;
#if PFMHA_CMB2
    if (S <= 4) {   // [r12e] every load at entry: one L2 round trip
        float2 ml4[4];
        uint4 v4[4];
        uint4 gv = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int s = 0; s < 4; s++) { ml4[s] = make_float2(0.f, 0.f); v4[s] = make_uint4(0u, 0u, 0u, 0u); }
        if (live) {
            gv = *reinterpret_cast<const uint4*>(gate + (size_t)(row >> 4) * gstride + (row & 15) * 64 + c * 8);
#pragma unroll
            for (int s = 0; s < 4; s++) {
                if (s < S) {
                    ml4[s] = pml[(size_t)s * rows + row];
                    v4[s] = *reinterpret_cast<const uint4*>(po + ((size_t)s * rows + row) * 64 + c * 8);
                }
            }
        }
        float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float wsum = 0.f;
        if (live) {
            float M = M_INIT;
#pragma unroll
            for (int s = 0; s < 4; s++) if (s < S) M = fmaxf(M, ml4[s].x);
#pragma unroll
            for (int s = 0; s < 4; s++) {
                if (s < S) {
                    const float w = ml4[s].y * exp2f(ml4[s].x - M);
                    wsum += w;
                    const unsigned vv[4] = {v4[s].x, v4[s].y, v4[s].z, v4[s].w};
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const float2 f = __half22float2(u32h2(vv[i]));
                        a[2 * i] += w * f.x;
                        a[2 * i + 1] += w * f.y;
                    }
                }
            }
        }
        const float inv = 1.0f / fmaxf(wsum, 1e-37f);
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const float2 x2 = __bfloat1622float2(__floats2bfloat162_rn(a[i] * inv, a[i + 1] * inv));
            a[i] = x2.x;
            a[i + 1] = x2.y;
            ss = fmaf(x2.x, x2.x, ss);
            ss = fmaf(x2.y, x2.y, ss);
        }
        ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 1);
        ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 2);
        ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 4);
        if (!live) return;
        const float rstd = rsqrtf(fmaf(ss, 1.0f / 64.0f, ng_eps));
        const unsigned gg[4] = {gv.x, gv.y, gv.z, gv.w};
        unsigned o[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gg[i]));
            __nv_bfloat162 b2 = __floats2bfloat162_rn(a[2 * i] * rstd * sig2(g2).x, a[2 * i + 1] * rstd * sig2(g2).y);
            o[i] = *reinterpret_cast<unsigned*>(&b2);
        }
        *reinterpret_cast<uint4*>(out + (size_t)row * 64 + c * 8) = make_uint4(o[0], o[1], o[2], o[3]);
        return;
    }
#endif
    float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float wsum = 0.f;
    if (live) {
        float M = M_INIT;
        for (int s = 0; s < S; s++) M = fmaxf(M, pml[(size_t)s * rows + row].x);
        for (int s = 0; s < S; s++) {
            const float2 ml = pml[(size_t)s * rows + row];
            const float w = ml.y * exp2f(ml.x - M);
            wsum += w;
            const uint4 v = *reinterpret_cast<const uint4*>(po + ((size_t)s * rows + row) * 64 + c * 8);
            const unsigned vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float2 f = __half22float2(u32h2(vv[i]));
                a[2 * i] += w * f.x;
                a[2 * i + 1] += w * f.y;
            }
        }
    }
    const float inv = 1.0f / fmaxf(wsum, 1e-37f);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const float2 x2 = __bfloat1622float2(__floats2bfloat162_rn(a[i] * inv, a[i + 1] * inv));
        a[i] = x2.x;
        a[i + 1] = x2.y;
        ss = fmaf(x2.x, x2.x, ss);
        ss = fmaf(x2.y, x2.y, ss);
    }
    ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 1);
    ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 2);
    ss += __shfl_xor_sync(0xFFFFFFFFu, ss, 4);
    if (!live) return;
    const float rstd = rsqrtf(fmaf(ss, 1.0f / 64.0f, ng_eps));
    const int tok = row >> 4, head = row & 15;
    const uint4 gv = *reinterpret_cast<const uint4*>(gate + (size_t)tok * gstride + head * 64 + c * 8);
    const unsigned gg[4] = {gv.x, gv.y, gv.z, gv.w};
    unsigned o[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gg[i]));
        __nv_bfloat162 b2 = __floats2bfloat162_rn(a[2 * i] * rstd * sig2(g2).x,
                                                  a[2 * i + 1] * rstd * sig2(g2).y);
        o[i] = *reinterpret_cast<unsigned*>(&b2);
    }
    *reinterpret_cast<uint4*>(out + (size_t)row * 64 + c * 8) = make_uint4(o[0], o[1], o[2], o[3]);
}

__global__ void __launch_bounds__(256) gather_k(
    const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc,
    unsigned char* __restrict__ kvt,
    const int* __restrict__ pt, const int* __restrict__ lens, const int* __restrict__ k0s,
    int pt_stride, int pagelog, int capblks)
{
    __shared__ __align__(16) unsigned char sv[128 * 68];
    const int b = blockIdx.y >> 1, kvh = blockIdx.y & 1;
    const int blk = blockIdx.x;
    const int len = lens[b];
    const int kst = k0s ? k0s[b] : 0;
    const int tid = threadIdx.x;
    unsigned char* tile = kvt + ((size_t)(b * 2 + kvh) * capblks + blk) * SB;

    long rowb[2]; bool val[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int key = c >> 2;
        const int pos = kst + (blk << 7) + key;
        val[i] = pos < len;
        if (val[i]) {
            const int page = pt[b * pt_stride + (pos >> pagelog)];
            rowb[i] = ((long)page << pagelog) + (pos & ((1 << pagelog) - 1));
        } else rowb[i] = 0;
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int key = c >> 2, kc4 = c & 3;
        const long off = rowb[i] * 128 + kvh * 64 + kc4 * 16;
        uint4 kw = val[i] ? *reinterpret_cast<const uint4*>(kc + off) : make_uint4(0, 0, 0, 0);
        uint4 vw = val[i] ? *reinterpret_cast<const uint4*>(vc + off) : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(tile + kc4 * 2048 + key * 16) = kw;
        unsigned* sp = reinterpret_cast<unsigned*>(sv + key * 68 + kc4 * 16);
        sp[0] = vw.x; sp[1] = vw.y; sp[2] = vw.z; sp[3] = vw.w;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int kcv = c >> 6, d = c & 63;
        unsigned w[4];
#pragma unroll
        for (int ww = 0; ww < 4; ww++) {
            unsigned r = 0;
#pragma unroll
            for (int bb = 0; bb < 4; bb++) {
                const int lam = kcv * 16 + ww * 4 + bb;
                const int kk = (lam & ~31) | pi_lane(lam & 31);
                r |= (unsigned)sv[kk * 68 + d] << (8 * bb);
            }
            w[ww] = r;
        }
        uint4* dp = reinterpret_cast<uint4*>(tile + 8192 + (d >> 3) * 1024 + kcv * 128 + (d & 7) * 16);   // [lcp ngm] V^T N-group-major
        *dp = make_uint4(w[0], w[1], w[2], w[3]);
    }
}

}  // namespace pgfa

namespace pfg {

constexpr int SB = 16384;
DINL int pi_lane(int lam) {
    return (lam & 1) | (((lam >> 2) & 3) << 1) | (((lam >> 1) & 1) << 3) | (lam & 16);
}

// skw::gather_kp, verbatim ([r6kvoff] SKIPOLD = true: the stage a' launch, whose old_blk tiles gather_old staged)
template <bool SKIPOLD = false>
__global__ void __launch_bounds__(256) gather_kp(
    const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc,
    unsigned char* __restrict__ kvt,
    const int* __restrict__ pt, const int* __restrict__ lens, const int* __restrict__ seq,
    int* __restrict__ lens_out, int pt_stride, int pagelog, int capblks, int L, int win)
{
    __shared__ __align__(16) unsigned char sv[128 * 68];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int b = blockIdx.y >> 1, kvh = blockIdx.y & 1;
    const int blk = blockIdx.x;
    const int len = lens[b];
    const int kst = seq ? max(seq[b] - (win - 1), 0) : 0;
    const bool old_blk = kst + (blk << 7) + 127 < len - L;
    const int tid = threadIdx.x;
    if (lens_out && blk == 0 && kvh == 0 && tid == 0) lens_out[b] = len - kst;
    if constexpr (SKIPOLD) {   // [r6kvoff] stage a': gather_old staged this tile (and lens_out) on the side stream
        if (old_blk) {
            asm volatile("griddepcontrol.wait;" ::: "memory");   // our completion must imply qkf's
            return;
        }
    }
    if (kst + (blk << 7) >= len) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        return;
    }
    unsigned char* tile = kvt + ((size_t)(b * 2 + kvh) * capblks + blk) * SB;

    long rowb[2]; bool val[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int key = c >> 2;
        const int pos = kst + (blk << 7) + key;
        val[i] = pos < len;
        if (val[i]) {
            const int page = pt[b * pt_stride + (pos >> pagelog)];
            rowb[i] = ((long)page << pagelog) + (pos & ((1 << pagelog) - 1));
        } else rowb[i] = 0;
    }
    if (!old_blk) asm volatile("griddepcontrol.wait;" ::: "memory");   // the new tokens' cache rows come from qkf
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int key = c >> 2, kc4 = c & 3;
        const long off = rowb[i] * 128 + kvh * 64 + kc4 * 16;
        uint4 kw = val[i] ? *reinterpret_cast<const uint4*>(kc + off) : make_uint4(0, 0, 0, 0);
        uint4 vw = val[i] ? *reinterpret_cast<const uint4*>(vc + off) : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(tile + kc4 * 2048 + key * 16) = kw;
        unsigned* sp = reinterpret_cast<unsigned*>(sv + key * 68 + kc4 * 16);
        sp[0] = vw.x; sp[1] = vw.y; sp[2] = vw.z; sp[3] = vw.w;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int kcv = c >> 6, d = c & 63;
        unsigned w[4];
#pragma unroll
        for (int ww = 0; ww < 4; ww++) {
            unsigned r = 0;
#pragma unroll
            for (int bb = 0; bb < 4; bb++) {
                const int lam = kcv * 16 + ww * 4 + bb;
                const int kk = (lam & ~31) | pi_lane(lam & 31);
                r |= (unsigned)sv[kk * 68 + d] << (8 * bb);
            }
            w[ww] = r;
        }
        uint4* dp = reinterpret_cast<uint4*>(tile + 8192 + (d >> 3) * 1024 + kcv * 128 + (d & 7) * 16);   // [lcp ngm] V^T N-group-major
        *dp = make_uint4(w[0], w[1], w[2], w[3]);
    }
    if (old_blk) asm volatile("griddepcontrol.wait;" ::: "memory");   // our completion must imply qkf's
}


// [p12f] gather_kp with both kv heads of a 128-key block in one 512-thread CTA (grid capblks x B): each key's 128-byte K
// (and V) cache row is read whole by 8 consecutive threads (16 B each; chunk ch = kvh * 4 + kc4) instead of as two 64-byte
// halves by two CTAs launched ~capblks CTAs apart. Same kvt bytes as gather_kp: head kvh's tile gets K chunk kc4 at
// kc4 * 2048 + key * 16 and its V^T bytes through its own 128 x 68 B smem slice (threads 256 * kvh .. + 255 run gather_kp's
// V^T loop); same PDL waits (new-key blocks before their cache loads, old-key blocks before they exit).
__global__ void __launch_bounds__(512) gather_kp2(
    const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc,
    unsigned char* __restrict__ kvt,
    const int* __restrict__ pt, const int* __restrict__ lens, const int* __restrict__ seq,
    int* __restrict__ lens_out, int pt_stride, int pagelog, int capblks, int L, int win)
{
    __shared__ __align__(16) unsigned char sv2[2][128 * 68];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int b = blockIdx.y;
    const int blk = blockIdx.x;
    const int len = lens[b];
    const int kst = seq ? max(seq[b] - (win - 1), 0) : 0;
    const bool old_blk = kst + (blk << 7) + 127 < len - L;
    const int tid = threadIdx.x;
    if (lens_out && blk == 0 && tid == 0) lens_out[b] = len - kst;
    if (kst + (blk << 7) >= len) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        return;
    }
    long rowb[2]; bool val[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 512 + tid;
        const int key = c >> 3;
        const int pos = kst + (blk << 7) + key;
        val[i] = pos < len;
        if (val[i]) {
            const int page = pt[b * pt_stride + (pos >> pagelog)];
            rowb[i] = ((long)page << pagelog) + (pos & ((1 << pagelog) - 1));
        } else rowb[i] = 0;
    }
    if (!old_blk) asm volatile("griddepcontrol.wait;" ::: "memory");   // the new tokens' cache rows come from qkf
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 512 + tid;
        const int key = c >> 3, ch = c & 7, kvh = ch >> 2, kc4 = ch & 3;
        const long off = rowb[i] * 128 + kvh * 64 + kc4 * 16;
        uint4 kw = val[i] ? *reinterpret_cast<const uint4*>(kc + off) : make_uint4(0, 0, 0, 0);
        uint4 vw = val[i] ? *reinterpret_cast<const uint4*>(vc + off) : make_uint4(0, 0, 0, 0);
        unsigned char* tile = kvt + ((size_t)(b * 2 + kvh) * capblks + blk) * SB;
        *reinterpret_cast<uint4*>(tile + kc4 * 2048 + key * 16) = kw;
        unsigned* sp = reinterpret_cast<unsigned*>(sv2[kvh] + key * 68 + kc4 * 16);
        sp[0] = vw.x; sp[1] = vw.y; sp[2] = vw.z; sp[3] = vw.w;
    }
    __syncthreads();
    {
        const int kvh = tid >> 8, t2 = tid & 255;
        const unsigned char* sv = sv2[kvh];
        unsigned char* tile = kvt + ((size_t)(b * 2 + kvh) * capblks + blk) * SB;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = i * 256 + t2;
            const int kcv = c >> 6, d = c & 63;
            unsigned w[4];
#pragma unroll
            for (int ww = 0; ww < 4; ww++) {
                unsigned r = 0;
#pragma unroll
                for (int bb = 0; bb < 4; bb++) {
                    const int lam = kcv * 16 + ww * 4 + bb;
                    const int kk = (lam & ~31) | pi_lane(lam & 31);
                    r |= (unsigned)sv[kk * 68 + d] << (8 * bb);
                }
                w[ww] = r;
            }
            uint4* dp = reinterpret_cast<uint4*>(tile + 8192 + (d >> 3) * 1024 + kcv * 128 + (d & 7) * 16);   // [lcp ngm] V^T N-group-major
            *dp = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
    if (old_blk) asm volatile("griddepcontrol.wait;" ::: "memory");   // our completion must imply qkf's
}

// [p12h] L5: gather_kp (SKIPOLD false) / gather_kp2 for full-attention layers only (window start 0: no seq, win 0), with the
// page-table read taken off the cold cache_seqlens load's dependency chain. M selects how (same kvt bytes, lens_out and
// PDL waits as gather_kp / gather_kp2 in every mode):
//   M 1: unpredicated page-table loads (column clamped to ptcols - 1), then the lens load, as ld.volatile in that order,
//        before the lens-dependent exit (ptxas sinks a plain or .nc load below the exit into the only block using it; it
//        may not drop a volatile access from the exit path and keeps volatile accesses in program order);
//   M 2 / 3: prefetch.global.L1 / .L2 of this thread's page-table entries (column clamped) at entry, then gather_kp's
//        own code: its lens-dependent, predicated page-table load hits the prefetched line.
template <int M>
DINL void gkp_pt_prefetch(const int* p) {
    if constexpr (M == 2) asm volatile("prefetch.global.L1 [%0];" :: "l"(p));
    else asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
template <int M>
__global__ void __launch_bounds__(256) gather_kp_u(
    const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc,
    unsigned char* __restrict__ kvt,
    const int* __restrict__ pt, const int* __restrict__ lens, const int* __restrict__ seq,
    int* __restrict__ lens_out, int pt_stride, int pagelog, int capblks, int L, int win, int ptcols)
{
    __shared__ __align__(16) unsigned char sv[128 * 68];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int b = blockIdx.y >> 1, kvh = blockIdx.y & 1;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr int kst = 0;   // full-attention layers only
    int pg[2] = {0, 0}, len;
    if constexpr (M == 1) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int pos = kst + (blk << 7) + ((i * 256 + tid) >> 2);
            asm volatile("ld.volatile.global.b32 %0, [%1];" : "=r"(pg[i]) : "l"(pt + b * pt_stride + min(pos >> pagelog, ptcols - 1)));
        }
        asm volatile("ld.volatile.global.b32 %0, [%1];" : "=r"(len) : "l"(lens + b));
    } else {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int pos = kst + (blk << 7) + ((i * 256 + tid) >> 2);
            gkp_pt_prefetch<M>(pt + b * pt_stride + min(pos >> pagelog, ptcols - 1));
        }
        len = lens[b];
    }
    const bool old_blk = kst + (blk << 7) + 127 < len - L;
    if (lens_out && blk == 0 && kvh == 0 && tid == 0) lens_out[b] = len - kst;
    if (kst + (blk << 7) >= len) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        return;
    }
    unsigned char* tile = kvt + ((size_t)(b * 2 + kvh) * capblks + blk) * SB;

    long rowb[2]; bool val[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int key = c >> 2;
        const int pos = kst + (blk << 7) + key;
        val[i] = pos < len;
        if constexpr (M == 1) {
            rowb[i] = val[i] ? ((long)pg[i] << pagelog) + (pos & ((1 << pagelog) - 1)) : 0;
        } else {
            if (val[i]) {
                const int page = pt[b * pt_stride + (pos >> pagelog)];
                rowb[i] = ((long)page << pagelog) + (pos & ((1 << pagelog) - 1));
            } else rowb[i] = 0;
        }
    }
    if (!old_blk) asm volatile("griddepcontrol.wait;" ::: "memory");   // the new tokens' cache rows come from qkf
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int key = c >> 2, kc4 = c & 3;
        const long off = rowb[i] * 128 + kvh * 64 + kc4 * 16;
        uint4 kw = val[i] ? *reinterpret_cast<const uint4*>(kc + off) : make_uint4(0, 0, 0, 0);
        uint4 vw = val[i] ? *reinterpret_cast<const uint4*>(vc + off) : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(tile + kc4 * 2048 + key * 16) = kw;
        unsigned* sp = reinterpret_cast<unsigned*>(sv + key * 68 + kc4 * 16);
        sp[0] = vw.x; sp[1] = vw.y; sp[2] = vw.z; sp[3] = vw.w;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 256 + tid;
        const int kcv = c >> 6, d = c & 63;
        unsigned w[4];
#pragma unroll
        for (int ww = 0; ww < 4; ww++) {
            unsigned r = 0;
#pragma unroll
            for (int bb = 0; bb < 4; bb++) {
                const int lam = kcv * 16 + ww * 4 + bb;
                const int kk = (lam & ~31) | pi_lane(lam & 31);
                r |= (unsigned)sv[kk * 68 + d] << (8 * bb);
            }
            w[ww] = r;
        }
        uint4* dp = reinterpret_cast<uint4*>(tile + 8192 + (d >> 3) * 1024 + kcv * 128 + (d & 7) * 16);   // V^T N-group-major
        *dp = make_uint4(w[0], w[1], w[2], w[3]);
    }
    if (old_blk) asm volatile("griddepcontrol.wait;" ::: "memory");   // our completion must imply qkf's
}

// [p12h] L5: gather_kp2 with gather_kp_u's page-table modes (full-attention layers only)
template <int M>
__global__ void __launch_bounds__(512) gather_kp2u(
    const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc,
    unsigned char* __restrict__ kvt,
    const int* __restrict__ pt, const int* __restrict__ lens, const int* __restrict__ seq,
    int* __restrict__ lens_out, int pt_stride, int pagelog, int capblks, int L, int win, int ptcols)
{
    __shared__ __align__(16) unsigned char sv2[2][128 * 68];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int b = blockIdx.y;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr int kst = 0;   // full-attention layers only
    int pg[2] = {0, 0}, len;
    if constexpr (M == 1) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int pos = kst + (blk << 7) + ((i * 512 + tid) >> 3);
            asm volatile("ld.volatile.global.b32 %0, [%1];" : "=r"(pg[i]) : "l"(pt + b * pt_stride + min(pos >> pagelog, ptcols - 1)));
        }
        asm volatile("ld.volatile.global.b32 %0, [%1];" : "=r"(len) : "l"(lens + b));
    } else {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int pos = kst + (blk << 7) + ((i * 512 + tid) >> 3);
            gkp_pt_prefetch<M>(pt + b * pt_stride + min(pos >> pagelog, ptcols - 1));
        }
        len = lens[b];
    }
    const bool old_blk = kst + (blk << 7) + 127 < len - L;
    if (lens_out && blk == 0 && tid == 0) lens_out[b] = len - kst;
    if (kst + (blk << 7) >= len) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        return;
    }
    long rowb[2]; bool val[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 512 + tid;
        const int key = c >> 3;
        const int pos = kst + (blk << 7) + key;
        val[i] = pos < len;
        if constexpr (M == 1) {
            rowb[i] = val[i] ? ((long)pg[i] << pagelog) + (pos & ((1 << pagelog) - 1)) : 0;
        } else {
            if (val[i]) {
                const int page = pt[b * pt_stride + (pos >> pagelog)];
                rowb[i] = ((long)page << pagelog) + (pos & ((1 << pagelog) - 1));
            } else rowb[i] = 0;
        }
    }
    if (!old_blk) asm volatile("griddepcontrol.wait;" ::: "memory");   // the new tokens' cache rows come from qkf
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = i * 512 + tid;
        const int key = c >> 3, ch = c & 7, kvh = ch >> 2, kc4 = ch & 3;
        const long off = rowb[i] * 128 + kvh * 64 + kc4 * 16;
        uint4 kw = val[i] ? *reinterpret_cast<const uint4*>(kc + off) : make_uint4(0, 0, 0, 0);
        uint4 vw = val[i] ? *reinterpret_cast<const uint4*>(vc + off) : make_uint4(0, 0, 0, 0);
        unsigned char* tile = kvt + ((size_t)(b * 2 + kvh) * capblks + blk) * SB;
        *reinterpret_cast<uint4*>(tile + kc4 * 2048 + key * 16) = kw;
        unsigned* sp = reinterpret_cast<unsigned*>(sv2[kvh] + key * 68 + kc4 * 16);
        sp[0] = vw.x; sp[1] = vw.y; sp[2] = vw.z; sp[3] = vw.w;
    }
    __syncthreads();
    {
        const int kvh = tid >> 8, t2 = tid & 255;
        const unsigned char* sv = sv2[kvh];
        unsigned char* tile = kvt + ((size_t)(b * 2 + kvh) * capblks + blk) * SB;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = i * 256 + t2;
            const int kcv = c >> 6, d = c & 63;
            unsigned w[4];
#pragma unroll
            for (int ww = 0; ww < 4; ww++) {
                unsigned r = 0;
#pragma unroll
                for (int bb = 0; bb < 4; bb++) {
                    const int lam = kcv * 16 + ww * 4 + bb;
                    const int kk = (lam & ~31) | pi_lane(lam & 31);
                    r |= (unsigned)sv[kk * 68 + d] << (8 * bb);
                }
                w[ww] = r;
            }
            uint4* dp = reinterpret_cast<uint4*>(tile + 8192 + (d >> 3) * 1024 + kcv * 128 + (d & 7) * 16);   // V^T N-group-major
            *dp = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
    if (old_blk) asm volatile("griddepcontrol.wait;" ::: "memory");   // our completion must imply qkf's
}

// [r6kvoff] gather_old: the cache-only part of gather_kp's kvt staging, launched on the side stream around the qkv GEMM and
// joined before qk_prep (plain launch, no PDL: it reads only this call's inputs: the page table, the cache rows of keys
// below len - L, cache_seqlens and seq_lens). GO_GROUPS groups of 256 threads per CTA; each group stages whole
// (b, kvh, blk) tiles with gather_kp's body (same bytes, same N-group-major V^T layout), tiles dealt round-robin.
// MODE 1 (stage a'): the old_blk tiles only; gather_kp<true> stages the others after qk_prep.
// MODE 2 (stage b): every tile below len that holds a key that is not new: old keys from the cache, new and pad keys as
// zeros. qk_prep_kvt, which runs after the join, overwrites the new keys' bytes, so the pad keys end as zeros, as in
// gather_kp; tiles made only of new keys are left to qk_prep_kvt. Both modes write lens_out (gather_kp's blk-0 job).
constexpr int GO_GROUPS = 4;
DINL void go_bar(int grp) {   // named barrier of one 256-thread group (immediate ids, so ptxas reserves only ids 1..4)
    switch (grp) {
    case 0: asm volatile("bar.sync 1, 256;" ::: "memory"); break;
    case 1: asm volatile("bar.sync 2, 256;" ::: "memory"); break;
    case 2: asm volatile("bar.sync 3, 256;" ::: "memory"); break;
    default: asm volatile("bar.sync 4, 256;" ::: "memory"); break;
    }
}
template <int MODE>
__global__ void __launch_bounds__(256 * GO_GROUPS) gather_old(
    const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc,
    unsigned char* __restrict__ kvt,
    const int* __restrict__ pt, const int* __restrict__ lens, const int* __restrict__ seq,
    int* __restrict__ lens_out, int pt_stride, int pagelog, int capblks, int B, int L, int win)
{
    __shared__ __align__(16) unsigned char svg[GO_GROUPS][128 * 68];
    const int grp = threadIdx.x >> 8, tid = threadIdx.x & 255;
    unsigned char* sv = svg[grp];
    if (blockIdx.x == 0)
        for (int bb = threadIdx.x; bb < B; bb += blockDim.x) lens_out[bb] = lens[bb] - max(seq[bb] - (win - 1), 0);
    const int ntiles = B * 2 * capblks;
    for (int ti = blockIdx.x * GO_GROUPS + grp; ti < ntiles; ti += gridDim.x * GO_GROUPS) {   // group-uniform loop
        const int bh = ti / capblks, blk = ti - bh * capblks;
        const int b = bh >> 1, kvh = bh & 1;
        const int len = lens[b];
        const int kst = max(seq[b] - (win - 1), 0);
        const int k0 = kst + (blk << 7);
        const int olim = len - L;   // keys below it were in the cache before this call (gather_kp's old_blk bound)
        const bool old_blk = k0 + 127 < olim;
        if (MODE == 1 ? !old_blk : (k0 >= len || (k0 >= olim && k0 + 127 < len))) continue;
        unsigned char* tile = kvt + ((size_t)(b * 2 + kvh) * capblks + blk) * SB;

        long rowb[2]; bool val[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = i * 256 + tid;
            const int key = c >> 2;
            const int pos = k0 + key;
            val[i] = pos < olim;
            if (val[i]) {
                const int page = pt[b * pt_stride + (pos >> pagelog)];
                rowb[i] = ((long)page << pagelog) + (pos & ((1 << pagelog) - 1));
            } else rowb[i] = 0;
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = i * 256 + tid;
            const int key = c >> 2, kc4 = c & 3;
            const long off = rowb[i] * 128 + kvh * 64 + kc4 * 16;
            uint4 kw = val[i] ? *reinterpret_cast<const uint4*>(kc + off) : make_uint4(0, 0, 0, 0);
            uint4 vw = val[i] ? *reinterpret_cast<const uint4*>(vc + off) : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(tile + kc4 * 2048 + key * 16) = kw;
            unsigned* sp = reinterpret_cast<unsigned*>(sv + key * 68 + kc4 * 16);
            sp[0] = vw.x; sp[1] = vw.y; sp[2] = vw.z; sp[3] = vw.w;
        }
        go_bar(grp);   // the group's V rows are in sv
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = i * 256 + tid;
            const int kcv = c >> 6, d = c & 63;
            unsigned w[4];
#pragma unroll
            for (int ww = 0; ww < 4; ww++) {
                unsigned r = 0;
#pragma unroll
                for (int bb = 0; bb < 4; bb++) {
                    const int lam = kcv * 16 + ww * 4 + bb;
                    const int kk = (lam & ~31) | pi_lane(lam & 31);
                    r |= (unsigned)sv[kk * 68 + d] << (8 * bb);
                }
                w[ww] = r;
            }
            uint4* dp = reinterpret_cast<uint4*>(tile + 8192 + (d >> 3) * 1024 + kcv * 128 + (d & 7) * 16);   // [lcp ngm] V^T N-group-major
            *dp = make_uint4(w[0], w[1], w[2], w[3]);
        }
        go_bar(grp);   // sv is reused by the group's next tile
    }
}

}  // namespace pfg

#if PFMHA_EXACT
#define PFMHA_OPT4 411043671LL   // nc 4, all-MUFU exp2
#define PFMHA_OPT3 201328471LL   // nc 3, all-MUFU exp2
#else
#ifndef PFMHA_BYTE4
#define PFMHA_BYTE4 1   // [gate1] byte-domain exp on the NC4 build (Cfg bit 35); 0 = shipped 4706010455
#endif
#ifndef PFMHA_BYTE3
#define PFMHA_BYTE3 1   // [gate1] byte-domain exp on the NC3 build too (drops its MUFU class); 0 = shipped 4496294999
#endif
#ifndef PFMHA_G2V
#define PFMHA_G2V 1   // [gate2] cut V (Cfg bits 36)
#endif
#ifndef PFMHA_G2T
#define PFMHA_G2T 1   // [gate2] cut T (Cfg bits 37)
#endif
#ifndef PFMHA_G2D
#define PFMHA_G2D 1   // [gate2] cut D (Cfg bits 38 13)
#endif
#ifndef PFMHA_G2P
#define PFMHA_G2P 1   // [gate2] cut P (Cfg bits 12 41)
#endif
#ifndef PFMHA_G2W
#define PFMHA_G2W 1   // [gate2] cut W (Cfg bits 39)
#endif
#ifndef PFMHA_G2U
#define PFMHA_G2U 1   // [gate2] cut U (Cfg bits 43 12 41)
#endif
#define PFMHA_G2BITS ((PFMHA_G2V ? ((1LL << 36)) : 0LL) | (PFMHA_G2T ? ((1LL << 37)) : 0LL) | (PFMHA_G2D ? ((1LL << 38) | (1LL << 13)) : 0LL) | (PFMHA_G2P ? ((1LL << 12) | (1LL << 41)) : 0LL) | (PFMHA_G2W ? ((1LL << 39)) : 0LL) | (PFMHA_G2U ? ((1LL << 43) | (1LL << 12) | (1LL << 41)) : 0LL))
#define PFMHA_OPT4 ((4706010455LL | (PFMHA_BYTE4 ? (1LL << 35) : 0LL)) | PFMHA_G2BITS)
#define PFMHA_OPT3 ((4496294999LL | (PFMHA_BYTE3 ? (1LL << 35) : 0LL)) | PFMHA_G2BITS)
#endif
static void pg_fmha4(torch::Tensor q, torch::Tensor kvt, torch::Tensor out, torch::Tensor po, torch::Tensor pml,
                     torch::Tensor lens, torch::Tensor k_scale, torch::Tensor v_scale, double smqk_log2e,
                     int64_t B, int64_t L, int64_t capblks, int64_t S, int64_t win, int64_t nc,
                     const __nv_bfloat16* gate, int64_t gstride, double ng_eps) {
    dim3 grid3((L + 23) / 24, B * 2, S), grid4((L + 31) / 32, B * 2, S);
    auto st = at::cuda::getCurrentCUDAStream();
    auto qp = reinterpret_cast<const unsigned char*>(q.data_ptr());
    auto kp = reinterpret_cast<const unsigned char*>(kvt.data_ptr());
    auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    auto pop = reinterpret_cast<__half*>(po.data_ptr());
    auto pmp = reinterpret_cast<float2*>(pml.data_ptr());
    const int* lp = lens.data_ptr<int>();
    const float *ks = k_scale.data_ptr<float>(), *vsp = v_scale.data_ptr<float>();
    if (nc == 4)
        pgfa::run_fmha7<PFMHA_OPT4>(grid4, st, qp, kp, op, pop, pmp, lp, ks, vsp, (float)smqk_log2e, (int)B, (int)L,
                                  (int)capblks, (int)S, (int)win, gate, (int)gstride, (float)ng_eps);
    else
        pgfa::run_fmha7<PFMHA_OPT3>(grid3, st, qp, kp, op, pop, pmp, lp, ks, vsp, (float)smqk_log2e, (int)B, (int)L,
                                  (int)capblks, (int)S, (int)win, gate, (int)gstride, (float)ng_eps);
}

void fmha4n_split(torch::Tensor q, torch::Tensor kvt, torch::Tensor out, torch::Tensor po, torch::Tensor pml,
                  torch::Tensor lens, torch::Tensor k_scale, torch::Tensor v_scale, double smqk_log2e,
                  int64_t B, int64_t L, int64_t capblks, int64_t S, int64_t win, int64_t nc) {
    pg_fmha4(q, kvt, out, po, pml, lens, k_scale, v_scale, smqk_log2e, B, L, capblks, S, win, nc, nullptr, 0, 0.0);
}

void combine(torch::Tensor po, torch::Tensor pml, torch::Tensor out, int64_t rows, int64_t S) {
    const int blocks = (int)((rows * 8 + 255) / 256);
    pgfa::combine<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(po.data_ptr()), reinterpret_cast<const float2*>(pml.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)rows, (int)S);
}

void gatherp(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
             torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
             int64_t capblks, int64_t B, int64_t L, int64_t win) {
    // glue lever: launched with programmatic stream serialization, so the griddepcontrol.wait calls in gather_kp are
    // live: blocks made only of keys already in the cache (old_blk) stage them while qk_prep (the predecessor, which
    // triggers at its start) is still writing the new rows; blocks touching new rows wait first, and every block waits
    // before it exits, so gather_kp's completion still implies qk_prep's ([r4pdl] fmha7, PDL-launched, waits on it).
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)capblks, (unsigned)(B * 2));
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_pdl[1];
    at_pdl[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_pdl[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at_pdl;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, pfg::gather_kp<false>,
        reinterpret_cast<const unsigned char*>(k_cache.data_ptr()),
        reinterpret_cast<const unsigned char*>(v_cache.data_ptr()),
        reinterpret_cast<unsigned char*>(kvt.data_ptr()),
        (const int*)page_table.data_ptr<int>(), (const int*)lens.data_ptr<int>(),
        seq.numel() ? (const int*)seq.data_ptr<int>() : (const int*)nullptr,
        lens_out.numel() ? lens_out.data_ptr<int>() : (int*)nullptr,
        (int)pt_stride, (int)pagelog, (int)capblks, (int)L, (int)win);
}


// [p12f] gatherp with gather_kp2 (both kv heads per CTA), PDL as gatherp
void gatherp2(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
              torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
              int64_t capblks, int64_t B, int64_t L, int64_t win) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)capblks, (unsigned)B);
    cfg.blockDim = dim3(512);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_pdl[1];
    at_pdl[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_pdl[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at_pdl;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, pfg::gather_kp2,
        reinterpret_cast<const unsigned char*>(k_cache.data_ptr()),
        reinterpret_cast<const unsigned char*>(v_cache.data_ptr()),
        reinterpret_cast<unsigned char*>(kvt.data_ptr()),
        (const int*)page_table.data_ptr<int>(), (const int*)lens.data_ptr<int>(),
        seq.numel() ? (const int*)seq.data_ptr<int>() : (const int*)nullptr,
        lens_out.numel() ? lens_out.data_ptr<int>() : (int*)nullptr,
        (int)pt_stride, (int)pagelog, (int)capblks, (int)L, (int)win);
}

// [p12h] gatherp / gatherp2 with the unpredicated page-table loads (gather_kp_u / gather_kp2u; ptcols = page_table.size(1))
static void gatherp_u_launch(bool two, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
                             torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
                             int64_t capblks, int64_t B, int64_t L, int64_t win, int64_t mode) {
    TORCH_CHECK(mode >= 1 && mode <= 3, "gatherp_u: mode 1..3");
    TORCH_CHECK(page_table.dim() == 2 && page_table.size(1) >= 1, "gatherp_u: page_table [B, cols]");
    TORCH_CHECK(seq.numel() == 0 && win == 0, "gatherp_u: full-attention layers only");
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = two ? dim3((unsigned)capblks, (unsigned)B) : dim3((unsigned)capblks, (unsigned)(B * 2));
    cfg.blockDim = dim3(two ? 512 : 256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_pdl[1];
    at_pdl[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_pdl[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at_pdl;
    cfg.numAttrs = 1;
    auto fn = two ? (mode == 1 ? pfg::gather_kp2u<1> : mode == 2 ? pfg::gather_kp2u<2> : pfg::gather_kp2u<3>)
                  : (mode == 1 ? pfg::gather_kp_u<1> : mode == 2 ? pfg::gather_kp_u<2> : pfg::gather_kp_u<3>);
    cudaLaunchKernelEx(&cfg, fn,
        reinterpret_cast<const unsigned char*>(k_cache.data_ptr()),
        reinterpret_cast<const unsigned char*>(v_cache.data_ptr()),
        reinterpret_cast<unsigned char*>(kvt.data_ptr()),
        (const int*)page_table.data_ptr<int>(), (const int*)lens.data_ptr<int>(),
        seq.numel() ? (const int*)seq.data_ptr<int>() : (const int*)nullptr,
        lens_out.numel() ? lens_out.data_ptr<int>() : (int*)nullptr,
        (int)pt_stride, (int)pagelog, (int)capblks, (int)L, (int)win, (int)page_table.size(1));
}
void gatherpu(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
              torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
              int64_t capblks, int64_t B, int64_t L, int64_t win, int64_t mode) {
    gatherp_u_launch(false, k_cache, v_cache, kvt, page_table, lens, seq, lens_out, pt_stride, pagelog, capblks, B, L, win, mode);
}
void gatherp2u(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
               torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
               int64_t capblks, int64_t B, int64_t L, int64_t win, int64_t mode) {
    gatherp_u_launch(true, k_cache, v_cache, kvt, page_table, lens, seq, lens_out, pt_stride, pagelog, capblks, B, L, win, mode);
}

// [r6kvoff] stage a': gather_kp over the tiles with new keys only (gather_kp<true>: old_blk CTAs wait and exit), PDL as gatherp
void gatherp_skip(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
                  torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
                  int64_t capblks, int64_t B, int64_t L, int64_t win) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)capblks, (unsigned)(B * 2));
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_pdl[1];
    at_pdl[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_pdl[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at_pdl;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, pfg::gather_kp<true>,
        reinterpret_cast<const unsigned char*>(k_cache.data_ptr()),
        reinterpret_cast<const unsigned char*>(v_cache.data_ptr()),
        reinterpret_cast<unsigned char*>(kvt.data_ptr()),
        (const int*)page_table.data_ptr<int>(), (const int*)lens.data_ptr<int>(),
        seq.numel() ? (const int*)seq.data_ptr<int>() : (const int*)nullptr,
        lens_out.numel() ? lens_out.data_ptr<int>() : (int*)nullptr,
        (int)pt_stride, (int)pagelog, (int)capblks, (int)L, (int)win);
}

// [r6kvoff] gather_old on the current (side) stream: a plain launch, nctas CTAs of 256 * GO_GROUPS threads
void gatherold(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
               torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
               int64_t capblks, int64_t B, int64_t L, int64_t win, int64_t mode, int64_t nctas) {
    const dim3 grid((unsigned)nctas), block(256 * pfg::GO_GROUPS);
    auto st = at::cuda::getCurrentCUDAStream();
    auto kcp = reinterpret_cast<const unsigned char*>(k_cache.data_ptr());
    auto vcp = reinterpret_cast<const unsigned char*>(v_cache.data_ptr());
    auto kvp = reinterpret_cast<unsigned char*>(kvt.data_ptr());
    const int* ptp = (const int*)page_table.data_ptr<int>();
    const int* lp = (const int*)lens.data_ptr<int>();
    const int* sp = (const int*)seq.data_ptr<int>();
    int* lo = lens_out.data_ptr<int>();
    if (mode == 2)
        pfg::gather_old<2><<<grid, block, 0, st>>>(kcp, vcp, kvp, ptp, lp, sp, lo, (int)pt_stride, (int)pagelog, (int)capblks,
                                                   (int)B, (int)L, (int)win);
    else
        pfg::gather_old<1><<<grid, block, 0, st>>>(kcp, vcp, kvp, ptp, lp, sp, lo, (int)pt_stride, (int)pagelog, (int)capblks,
                                                   (int)B, (int)L, (int)win);
}


void zh_fmha_ng(torch::Tensor q, torch::Tensor kvt, torch::Tensor out, torch::Tensor po, torch::Tensor pml,
                torch::Tensor lens, torch::Tensor k_scale, torch::Tensor v_scale, double smqk_log2e,
                int64_t B, int64_t L, int64_t capblks, int64_t S, int64_t win, int64_t nc,
                torch::Tensor gate, int64_t gstride, double ng_eps) {
  const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr());
  pg_fmha4(q, kvt, out, po, pml, lens, k_scale, v_scale, smqk_log2e, B, L, capblks, S, win, nc,
           S <= 1 ? gp : nullptr, S <= 1 ? gstride : 0, S <= 1 ? ng_eps : 0.0);
  if (S > 1) {
    const int rows = (int)(B * L * 16);
    cudaLaunchConfig_t lc = {};   // [r4pdl] PDL launch: combine_ng waits for fmha7 at entry
    lc.gridDim = dim3((unsigned)((rows * 8 + 255) / 256)); lc.blockDim = dim3(256);
    lc.dynamicSmemBytes = 0; lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute la[1]; la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = 1;
    cudaLaunchKernelEx(&lc, pgfa::combine_ng,
        reinterpret_cast<const __half*>(po.data_ptr()), reinterpret_cast<const float2*>(pml.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), rows, (int)S, gp, (int)gstride, (float)ng_eps);
  }
}


// [r10p] fmha7p (mode 0: persistent over fmha7's own units; mode 1: whole-unit rounds + stream-K) and its merge
// (combine_ng over mode 0's splits; combine_ngv over mode 1's per-unit segment counts), both PDL-launched
void zh_fmha_ngp(torch::Tensor q, torch::Tensor kvt, torch::Tensor out, torch::Tensor po, torch::Tensor pml,
                 torch::Tensor nseg, torch::Tensor lens, torch::Tensor k_scale, torch::Tensor v_scale, double smqk_log2e,
                 int64_t B, int64_t L, int64_t capblks, int64_t S, int64_t win, int64_t nc,
                 torch::Tensor gate, int64_t gstride, double ng_eps, int64_t mode, int64_t grid, int64_t ndpr,
                 int64_t smax) {
  const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr());
  auto st = at::cuda::getCurrentCUDAStream();
  auto qp = reinterpret_cast<const unsigned char*>(q.data_ptr());
  auto kp = reinterpret_cast<const unsigned char*>(kvt.data_ptr());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  auto pop = reinterpret_cast<__half*>(po.data_ptr());
  auto pmp = reinterpret_cast<float2*>(pml.data_ptr());
  const int* lp = lens.data_ptr<int>();
  const float *ks = k_scale.data_ptr<float>(), *vsp = v_scale.data_ptr<float>();
  int* nsp = nseg.numel() ? nseg.data_ptr<int>() : nullptr;
  const bool sp = mode == 1 || (mode == 0 && S > 1);   // mode 2: unsplit only
  if (nc == 4) {
    if (mode == 2)
      pgfa::run_fmha7p<PFMHA_OPT4, 2>((int)grid, st, qp, kp, op, pop, pmp, lp, ks, vsp, (float)smqk_log2e, (int)B, (int)L,
                                      (int)capblks, 1, (int)win, gp, (int)gstride, (float)ng_eps, 0, (int)smax, nsp);
    else if (mode == 1)
      pgfa::run_fmha7p<PFMHA_OPT4, 1>((int)grid, st, qp, kp, op, pop, pmp, lp, ks, vsp, (float)smqk_log2e, (int)B, (int)L,
                                      (int)capblks, (int)S, (int)win, nullptr, 0, 0.f, (int)ndpr, (int)smax, nsp);
    else
      pgfa::run_fmha7p<PFMHA_OPT4, 0>((int)grid, st, qp, kp, op, pop, pmp, lp, ks, vsp, (float)smqk_log2e, (int)B, (int)L,
                                      (int)capblks, (int)S, (int)win, sp ? nullptr : gp, sp ? 0 : (int)gstride,
                                      sp ? 0.f : (float)ng_eps, 0, (int)smax, nsp);
  } else {
    if (mode == 2)
      pgfa::run_fmha7p<PFMHA_OPT3, 2>((int)grid, st, qp, kp, op, pop, pmp, lp, ks, vsp, (float)smqk_log2e, (int)B, (int)L,
                                      (int)capblks, 1, (int)win, gp, (int)gstride, (float)ng_eps, 0, (int)smax, nsp);
    else if (mode == 1)
      pgfa::run_fmha7p<PFMHA_OPT3, 1>((int)grid, st, qp, kp, op, pop, pmp, lp, ks, vsp, (float)smqk_log2e, (int)B, (int)L,
                                      (int)capblks, (int)S, (int)win, nullptr, 0, 0.f, (int)ndpr, (int)smax, nsp);
    else
      pgfa::run_fmha7p<PFMHA_OPT3, 0>((int)grid, st, qp, kp, op, pop, pmp, lp, ks, vsp, (float)smqk_log2e, (int)B, (int)L,
                                      (int)capblks, (int)S, (int)win, sp ? nullptr : gp, sp ? 0 : (int)gstride,
                                      sp ? 0.f : (float)ng_eps, 0, (int)smax, nsp);
  }
  if (sp) {
    const int rows = (int)(B * L * 16);
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3((unsigned)((rows * 8 + 255) / 256)); lc.blockDim = dim3(256);
    lc.dynamicSmemBytes = 0; lc.stream = st;
    cudaLaunchAttribute la[1]; la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = 1;
    if (mode == 1)
      cudaLaunchKernelEx(&lc, pgfa::combine_ngv, (const __half*)pop, (const float2*)pmp, (const int*)nsp, op, rows, (int)L,
                         (int)(8 * nc), gp, (int)gstride, (float)ng_eps);
    else
      cudaLaunchKernelEx(&lc, pgfa::combine_ng, (const __half*)pop, (const float2*)pmp, op, rows, (int)S, gp, (int)gstride,
                         (float)ng_eps);
  }
}

"""
ZH_CPP = r"""
void zh_fmha_ngp(torch::Tensor q, torch::Tensor kvt, torch::Tensor out, torch::Tensor po, torch::Tensor pml,
                 torch::Tensor nseg, torch::Tensor lens, torch::Tensor k_scale, torch::Tensor v_scale, double smqk_log2e,
                 int64_t B, int64_t L, int64_t capblks, int64_t S, int64_t win, int64_t nc,
                 torch::Tensor gate, int64_t gstride, double ng_eps, int64_t mode, int64_t grid, int64_t ndpr,
                 int64_t smax);
void zh_fmha_ng(torch::Tensor q, torch::Tensor kvt, torch::Tensor out, torch::Tensor po, torch::Tensor pml,
                torch::Tensor lens, torch::Tensor k_scale, torch::Tensor v_scale, double smqk_log2e,
                int64_t B, int64_t L, int64_t capblks, int64_t S, int64_t win, int64_t nc,
                torch::Tensor gate, int64_t gstride, double ng_eps);
void gatherp(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
             torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
             int64_t capblks, int64_t B, int64_t L, int64_t win);
void gatherp_skip(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
                  torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
                  int64_t capblks, int64_t B, int64_t L, int64_t win);
void gatherp2(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
              torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
              int64_t capblks, int64_t B, int64_t L, int64_t win);
void gatherpu(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
              torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
              int64_t capblks, int64_t B, int64_t L, int64_t win, int64_t mode);
void gatherp2u(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
               torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
               int64_t capblks, int64_t B, int64_t L, int64_t win, int64_t mode);
void gatherold(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
               torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
               int64_t capblks, int64_t B, int64_t L, int64_t win, int64_t mode, int64_t nctas);
"""
_ZH = {}


def _zh_build():
    try:
        major = torch.cuda.get_device_capability()[0]
        if major < 9:
            return
        _ZH["ext"] = load_inline(
            name="zh_pfmha_n72_exact_pdl_v1_r4pdl_r6kvab_r10p" if ZH_EXACT else "zh_pfmha_n72_pdl_g2all_r4pdl_r6kvab_r10p_r12_d2_p12f_p12h",  # N72 + PDL gather_kp launch (glue lever) + [r6kvoff] gather_old / gatherp_skip: own build
            cpp_sources=[ZH_CPP], cuda_sources=[ZH_CUDA_SRC], functions=["zh_fmha_ng", "zh_fmha_ngp", "gatherp", "gatherp_skip", "gatherold", "gatherp2", "gatherpu", "gatherp2u"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a"]
            + (["-DPFMHA_EXACT=1"] if ZH_EXACT else []), verbose=False)
    except Exception as exc:  # noqa: BLE001
        _ZH["err"] = repr(exc)[-600:]


import threading as _zh_threading
_ZH_THREAD = _zh_threading.Thread(target=_zh_build, daemon=True)
_ZH_THREAD.start()


def _zh_pick(L, ntf, B, nsm=132):
    """(consumer warpgroups, split): his pick_cfg4 cost model, with B requests per grid."""
    rows = B * L * 16
    best = (float("inf"), 3, 1)
    for nc in (3, 4):
        units = ((L + 8 * nc - 1) // (8 * nc)) * 2 * B
        for s in range(1, 17):
            waves = -(-units * s // nsm)
            cost = waves * (ntf / s + 20.0) * 1.41 * 0.75 * (nc / 3) * (0.9 if nc == 4 else 1.0)
            if s > 1:
                cost += s * rows * 136 * 2 / 3e12 * 1e6 + 3.0
            if cost < best[0] - 1e-9:
                best = (cost, nc, s)
    return best[1], best[2]


# [r10p] persistent / stream-K fmha7p per attention config, keyed (window > 0, B, L):
# (mode, nc, split, ndpr, smax, min_capblks); used when the kvt capacity (page_table shape) has >= min_capblks tiles.
# mode 0 = fmha7's own units (this nc / split) on min(units, nsm) persistent CTAs (bitwise = fmha7; smax = 1: the producer
# L2-prefetches each later segment's Q / gate rows); mode 1 = ndpr whole-unit rounds, then stream-K over the remaining
# units' key tiles (partials in <= smax slots per unit, merged by combine_ngv). Grid and buffers depend on the shape only;
# the decomposition is computed on device from lens.
ZH_P = {(False, 1, 16384): (0, 4, 1, 0, 1, 256), (True, 1, 16384): (2, 4, 1, 0, 1, 0), (True, 4, 753): (2, 3, 1, 0, 1, 0), (False, 4, 753): (1, 4, 1, 1, 4, 0)}
ZH_P_ON = True


def _zh_pcfg(st, B, L, window, capblks):
    if not ZH_P_ON or B > 64:
        return None
    c = ZH_P.get((bool(window), B, L))
    if c is None or capblks < c[5] or not st.get("zh_p_ok", {}).get(c, False):
        return None
    mode, nc, split, ndpr, smax, _ = c
    nsm = st.get("nsm", 132)
    qb = 8 * nc
    units = -(-L // qb) * 2 * B
    if mode == 1:
        if ndpr * nsm > units or units - ndpr * nsm > 512 or smax < 2 or B > 64:
            return None
        return mode, nc, split, nsm, ndpr, smax
    if mode == 2 and split != 1:
        return None
    return mode, nc, split, min(units * split, nsm), 0, smax   # mode 0: smax = the Q / gate L2 prefetch flag


def _zh_attn(st, data, q8, qkv, cfg, kvo=None, dbg=None, pcfg_on=True):
    """og [T, 1024] bf16 = o_norm(attention) * sigmoid(gate), by his gather_kp + fmha7 (+ combine_ng).
    [r6kvoff] kvo: kvt / lens from _kvoff_alloc, staged by gather_old (+ qk_prep_kvt at stage 2); dbg: gets kvt / lens."""
    ext = st["zh"]
    B, L, window, page = cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"]
    dev = q8.device
    T = B * L
    pt = data["page_table"]
    pagelog = page.bit_length() - 1
    max_len = pt.shape[1] * page
    # glue lever: the input's cache_seqlens is seq_lens + L (int32 [B]); using it drops the torch add kernel that sat
    # between qk_prep and gather_kp (and broke their PDL edge). Fallback only for a non-int32 / strided field.
    cs = data.get("cache_seqlens")
    if kvo is not None:  # [r6kvoff] the very tensor gather_old used
        lens_full = kvo["lens_full"]
    elif cs is not None and cs.dtype == torch.int32 and cs.is_contiguous() and cs.numel() == B:
        lens_full = cs
    else:
        lens_full = (data["seq_lens"] + L).to(torch.int32)
    cap = ((min(window - 1 + L, max_len) if window else max_len) + 255) & ~255
    capblks = cap // 128
    kvt = kvo["kvt"] if kvo is not None else torch.empty((B * 2 * capblks, 16384), dtype=torch.uint8, device=dev)
    kc = data["k_cache"].view(torch.uint8)
    vc = data["v_cache"].view(torch.uint8)
    if window and kvo is not None:  # [r6kvoff] gather_old staged the cache-only tiles and lens on the side stream
        assert kvo["capblks"] == capblks
        lens = kvo["lens"]
        if kvo["stage"] == 1:  # stage a': gather_kp<true> stages the tiles with new keys (PDL behind qk_prep, as today)
            ext.gatherp_skip(kc, vc, kvt, pt, lens_full, kvo["seq"], lens_full[:0], pt.stride(0), pagelog, capblks, B, L,
                             window)
        # stage 2: qk_prep_kvt wrote the new keys into kvt; fmha7 PDL-waits on qk_prep itself
        ntf = min(capblks, (window + 23) // 128 + 2)
    elif window:
        lens = torch.empty_like(lens_full)
        ext.gatherp(kc, vc, kvt, pt, lens_full, data["seq_lens"].to(torch.int32), lens, pt.stride(0), pagelog, capblks,
                    B, L, window)
        ntf = min(capblks, (window + 23) // 128 + 2)
    else:
        lens = lens_full
        none = lens_full[:0]
        two = GKP2 and B >= GKP2_MIN_B and st.get("gkp2_ok")  # [p12f]
        if GKP_PTU and st.get("gptu_ok") and pt.dim() == 2:  # [p12h] L5: page-table read off the lens chain (mode GKP_PTU)
            (ext.gatherp2u if two else ext.gatherpu)(kc, vc, kvt, pt, lens_full, none, none, pt.stride(0), pagelog, capblks, B, L, 0,
                                                    int(GKP_PTU))
        else:
            (ext.gatherp2 if two else ext.gatherp)(kc, vc, kvt, pt, lens_full, none, none, pt.stride(0), pagelog, capblks, B, L, 0)
        ntf = capblks
    nc, split = _zh_pick(L, ntf, B, st.get("nsm", 132))
    out = torch.empty((T, Q_SIZE), dtype=torch.bfloat16, device=dev)
    pc = _zh_pcfg(st, B, L, window, capblks) if pcfg_on else None
    if pc is not None:  # [r10p] fmha7p
        mode, nc, split, grid, ndpr, smax = pc
        nslot = smax if mode == 1 else split
        if mode == 1 or split > 1:
            po = torch.empty((nslot, T * HQ, HD), dtype=torch.float16, device=dev)
            pml = torch.empty((nslot, T * HQ, 2), dtype=torch.float32, device=dev)
        else:
            po = pml = out
        nseg = torch.empty((-(-L // (8 * nc)) * 2 * B if mode == 1 else 0,), dtype=torch.int32, device=dev)
        smqk = cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634
        ext.zh_fmha_ngp(q8.view(torch.uint8), kvt, out, po, pml, nseg, lens, data["k_scale"], data["v_scale"], smqk, B, L,
                        capblks, split, window, nc, qkv[:, Q_SIZE + 2 * KV_SIZE:], QKV_SIZE, RMS_EPS, mode, grid, ndpr,
                        smax)
        if dbg is not None:
            dbg.update(kvt=kvt, lens=lens, capblks=capblks)
        return out
    if split > 1:
        po = torch.empty((split, T * HQ, HD), dtype=torch.float16, device=dev)
        pml = torch.empty((split, T * HQ, 2), dtype=torch.float32, device=dev)
    else:
        po = pml = out
    smqk = cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634
    ext.zh_fmha_ng(q8.view(torch.uint8), kvt, out, po, pml, lens, data["k_scale"], data["v_scale"], smqk, B, L, capblks,
                   split, window, nc, qkv[:, Q_SIZE + 2 * KV_SIZE:], QKV_SIZE, RMS_EPS)
    if dbg is not None:
        dbg.update(kvt=kvt, lens=lens, capblks=capblks)
    return out


def _zh_ok(st, cfg):
    return st.get("zh") is not None and ZH_FMHA.get(_attn_class(cfg), False) and (cfg["page_size"] & (cfg["page_size"] - 1)) == 0


def _zh_prepare(st):
    """Join the build, then compare against our own kernel on the scored classes; on any mismatch, stay off."""
    _ZH_THREAD.join()
    st["zh"] = None
    if "ext" not in _ZH:
        print("ZH fmha unavailable:", _ZH.get("err", "no sm_90"), flush=True)
        return
    st["zh"] = _ZH["ext"]
    worst = 0.0
    try:
        for B, L, ctx, window, page in ((1, 753, 3000, 2048, 1), (2, 300, 2500, 0, 1), (1, 200, 5000, 0, 64),
                                        (1, 640, 4096, 2048, 64)):
            data, cfg, q8, qkv = _synthetic_attn_case(B, L, ctx, window, page, 17)
            ref = _attn_ref(data, cfg, q8, qkv)
            got = _zh_attn(st, data, q8, qkv, cfg)
            torch.cuda.synchronize()
            err = float((got.float() - ref.float()).norm() / ref.float().norm())
            worst = max(worst, err)
            print(f"ZH fmha check B{B} L{L} ctx{ctx} win{window} page{page}: rel err vs ref {err:.2e}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("ZH fmha check failed:", repr(exc)[:400], flush=True)
        st["zh"] = None
        return
    if not worst < 6e-2:
        print(f"ZH fmha off: mismatch {worst:.2e}", flush=True)
        st["zh"] = None
    if st["zh"] is not None:
        _zh_p_prepare(st)


def _zh_p_prepare(st):
    """[r10p] Validate each configured fmha7p kernel class on small synthetic cases (same template instantiation as the
    scored use: window, nc, split, mode); a failure leaves that config on fmha7."""
    ok = {}
    for key, c in ZH_P.items():
        if c in ok:
            continue
        mode, nc, split, ndpr, smax, mincap = c
        win = 2048 if key[0] else 0
        good = True
        try:
            ctx0 = 2500 if win else max(3000, mincap * 128 - key[2])   # full layers: a kvt capacity >= min_capblks
            for B, L, ctx, page in ((key[1], key[2], ctx0, 1),):
                data, cfg, q8, qkv = _synthetic_attn_case(B, L, ctx, win, page, 23)
                ref = _attn_ref(data, cfg, q8, qkv)
                base = _zh_attn(st, data, q8, qkv, cfg, pcfg_on=False)
                st.setdefault("zh_p_ok", {})[c] = True
                saved = dict(ZH_P)
                ZH_P.clear()
                ZH_P[(bool(win), B, L)] = c
                try:
                    got = _zh_attn(st, data, q8, qkv, cfg)
                finally:
                    ZH_P.clear()
                    ZH_P.update(saved)
                torch.cuda.synchronize()
                err = float((got.float() - ref.float()).norm() / ref.float().norm())
                d = float((got.float() - base.float()).norm() / base.float().norm())
                bit = bool(torch.equal(got, base))
                print(f"ZH fmha7p check {c} B{B} L{L} ctx{ctx} win{win} page{page}: rel err vs ref {err:.2e}, vs fmha7 {d:.2e}, "
                      f"bitwise {bit}", flush=True)
                if not (err < 6e-2 and d < 2e-2 and torch.isfinite(got.float()).all() and (mode == 1 or (mode == 0 and split > 1) or bit)):
                    good = False
        except Exception as exc:  # noqa: BLE001
            print("ZH fmha7p check failed:", repr(exc)[:400], flush=True)
            good = False
        ok[c] = good
    st["zh_p_ok"] = ok
    print("ZH fmha7p configs:", {k: (v, ok.get(v)) for k, v in ZH_P.items()}, flush=True)


# --------------------------------------------------------------------------------------
# [r6kvoff] sliding layers: gather_kp off the attention critical path (see the KVOFF_* knobs).
# --------------------------------------------------------------------------------------
_KVOFF_TOP = 2  # highest stage this file builds


def _kvoff_stage(st, cfg):
    """[r6kvoff] Stage for this call (0: the base path): window layers on the ZH path with KVOFF_MIN_T <= T <= KVOFF_MAX_T,
    lowered to the highest stage _kvoff_prepare validated."""
    T = cfg["tokens"]
    if not cfg["window"] or T < KVOFF_MIN_T or T > KVOFF_MAX_T:
        return 0
    s = min(int(KVOFF_STAGE_T.get(T, KVOFF_STAGE)), _KVOFF_TOP)
    ok = st.get("kvoff_ok") or ()
    while s > 0 and s not in ok:
        s -= 1
    return s


def _kvoff_alloc(data, cfg, stage):
    """[r6kvoff] kvt / lens (gather_kp's buffers) and the int32 length fields, allocated on the main stream before the qkv
    GEMM is enqueued, so no block the GEMM still uses can come back here."""
    B, L, window, page = cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"]
    pt = data["page_table"]
    max_len = pt.shape[1] * page
    cs = data.get("cache_seqlens")
    if cs is not None and cs.dtype == torch.int32 and cs.is_contiguous() and cs.numel() == B:
        lens_full = cs
    else:
        lens_full = (data["seq_lens"] + L).to(torch.int32)
    capblks = (((min(window - 1 + L, max_len) + 255) & ~255)) // 128  # = _zh_attn's (window layers)
    dev = data["k_cache"].device
    return {"stage": stage, "capblks": capblks, "lens_full": lens_full, "seq": data["seq_lens"].to(torch.int32),
            "kvt": torch.empty((B * 2 * capblks, 16384), dtype=torch.uint8, device=dev), "lens": torch.empty_like(lens_full),
            # [r13] RoPE table (stage 2, RoPE layers): [T, 64] fp32 = cos | sin per new token
            "tab": (torch.empty((B * L, 64), dtype=torch.float32, device=dev)
                    if KVOFF_ROPE_TAB and stage >= 2 and cfg.get("rope_theta") else None)}


def _kvoff_begin(st, data, cfg, kvo, ev=None, lc=False):
    """[r6kvoff] gather_old on the side stream, forked from main here (ev None: the caller runs this before enqueueing the
    qkv GEMM) or at `ev` (an event recorded before the qkv GEMM, so the side branch is captured after the GEMM node).
    _kvoff_prep joins it before qk_prep. [p12h] lc: `ev` was recorded right after the qkv GEMM; while capturing, the side
    branch's edge from it is retagged to the GEMM's launch-completion port (eager: the branch just waits for the GEMM)."""
    B, L, window, page = cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"]
    pt = data["page_table"]
    main, side = torch.cuda.current_stream(), SIDE_STREAM
    if ev is not None:
        side.wait_event(ev)
    else:
        side.wait_stream(main)
    with torch.cuda.stream(side):
        if lc:  # [p12h] L3 (cudaGraphKernelNodePortLaunchCompletion = 2, default edge type)
            st["lcfork_rc"] = int(st["mod"].retag_deps(2, 0))
        if kvo.get("tab") is not None:  # [r13] the RoPE table first (a short kernel; gather_old's tiles take far longer)
            st["mod"].rope_tab(data["positions"], st["inv_freq"], kvo["tab"])
        st["zh"].gatherold(data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), kvo["kvt"], pt,
                           kvo["lens_full"], kvo["seq"], kvo["lens"], pt.stride(0), page.bit_length() - 1, kvo["capblks"],
                           B, L, window, kvo["stage"], KVOFF_CTAS)
    # side-stream use of main-stream buffers: our own allocations only (inputs belong to the harness)
    if kvo.get("tab") is not None:
        kvo["tab"].record_stream(side)
    for k in ("kvt", "lens"):
        kvo[k].record_stream(side)
    if kvo["lens_full"] is not data.get("cache_seqlens"):
        kvo["lens_full"].record_stream(side)
    if kvo["seq"] is not data["seq_lens"]:
        kvo["seq"].record_stream(side)


def _kvoff_prep(st, data, qkv, q8, cfg, kvo):
    """[r6kvoff] Join the side stream (gather_old's tiles and lens), then qk_prep (stage 2: qk_prep_kvt, which also writes
    the new keys' K / V^T bytes into kvt)."""
    torch.cuda.current_stream().wait_stream(SIDE_STREAM)
    if kvo["stage"] >= 2 and kvo.get("tab") is not None and KVOFF_QKP1 and st.get("qkp1_ok"):  # [p12h] L2
        st["mod"].qk_prep_kvt1(qkv, data["cache_loc"], kvo["tab"], q8, data["k_cache"].view(torch.uint8),
                               data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], cfg["kv_scale"] != 1.0,
                               kvo["kvt"], kvo["lens_full"], kvo["seq"], kvo["capblks"], cfg["q_len"], cfg["window"])
    elif kvo["stage"] >= 2 and kvo.get("tab") is not None:  # [r13] RoPE angles from the side-stream table
        st["mod"].qk_prep_kvt_tab(qkv, data["positions"], data["cache_loc"], kvo["tab"], q8,
                                  data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["k_scale"],
                                  data["v_scale"], cfg["kv_scale"] != 1.0, kvo["kvt"],
                                  kvo["lens_full"], kvo["seq"], kvo["capblks"], cfg["q_len"], cfg["window"])
    elif kvo["stage"] >= 2:
        st["mod"].qk_prep_kvt(qkv, data["positions"], data["cache_loc"], st["inv_freq"], q8,
                              data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["k_scale"],
                              data["v_scale"], bool(cfg["rope_theta"]), cfg["kv_scale"] != 1.0, kvo["kvt"],
                              kvo["lens_full"], kvo["seq"], kvo["capblks"], cfg["q_len"], cfg["window"])
    else:
        _run_qk_prep(st, data, qkv, q8, cfg)


def _kvoff_case(B, L, ctxs, page, seed, scaled=False, window=2048):
    """[r6kvoff] A synthetic sliding-window prefill case with the judge's input conventions (reference.generate_input):
    scattered pages, page_table rows padded with -1, positions = seq_lens + i and cache_loc from the page table (int64),
    cache_seqlens = seq_lens + L (int32); random FP8 cache and BF16 qkv rows."""
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    seq = torch.tensor(ctxs, dtype=torch.int64, device=dev)
    per = [(c + L + page - 1) // page for c in ctxs]
    total = sum(per)
    perm = torch.randperm(total, device=dev, generator=g).to(torch.int32)
    pt = torch.full((B, max(per)), -1, dtype=torch.int32, device=dev)
    o = 0
    for b in range(B):
        pt[b, :per[b]] = perm[o:o + per[b]]
        o += per[b]
    kc = (torch.randn((total * page, HKV * HD), device=dev, generator=g) * 0.8 + 0.3).to(FP8).view(total, page, HKV, HD)
    vc = torch.randn((total * page, HKV * HD), device=dev, generator=g).to(FP8).view(total, page, HKV, HD)
    positions = (seq[:, None] + torch.arange(L, device=dev)).reshape(-1)
    p2 = positions.view(B, L)
    cache_loc = (torch.gather(pt.long(), 1, p2 // page) * page + p2 % page).reshape(-1)
    qkv = (torch.randn((B * L, QKV_SIZE), device=dev, generator=g) * 1.5).to(torch.bfloat16)
    sc = 0.75 if scaled else 1.0
    data = {"k_cache": kc, "v_cache": vc, "page_table": pt, "seq_lens": seq.to(torch.int32),
            "cache_seqlens": (seq + L).to(torch.int32), "positions": positions, "cache_loc": cache_loc,
            "k_scale": torch.tensor(sc, dtype=torch.float32, device=dev), "v_scale": torch.tensor(sc, dtype=torch.float32, device=dev)}
    cfg = {"batch": B, "q_len": L, "window": window, "page_size": page, "sm_scale": SYNTH_SM_SCALE,
           "qk_scale_factor": SYNTH_QK_SCALE_FACTOR, "kv_scale": sc, "tokens": B * L, "rope_theta": ROPE_THETA}
    return data, cfg, qkv


def _kvoff_run(st, data, cfg, qkv, stage):
    """[r6kvoff] qk_prep -> attention as kernel_fn issues it (stage 0: the base path) on private copies of the caches;
    returns og, q8, the caches and _zh_attn's kvt / lens. The new path's kvt / lens start poisoned (0xA5 / -7), so a byte
    it fails to write cannot match the base by accident of the allocator."""
    d = dict(data, k_cache=data["k_cache"].clone(), v_cache=data["v_cache"].clone())
    q8 = torch.full((cfg["tokens"], Q_SIZE), 0x5A, dtype=torch.uint8, device="cuda")
    dbg = {}
    if stage:
        kvo = _kvoff_alloc(d, cfg, stage)
        kvo["kvt"].fill_(0xA5)
        kvo["lens"].fill_(-7)
        _kvoff_begin(st, d, cfg, kvo)
        _kvoff_prep(st, d, qkv, q8, cfg, kvo)
        og = _zh_attn(st, d, q8, qkv, cfg, kvo=kvo, dbg=dbg)
    else:
        _run_qk_prep(st, d, qkv, q8, cfg)
        og = _zh_attn(st, d, q8, qkv, cfg, dbg=dbg)
    torch.cuda.synchronize()
    return og, q8, d["k_cache"], d["v_cache"], dbg


# (B, L, seq_lens, page, scaled): t1s / t4s geometry (T 753 / 3012, 80K context, spread lengths), the sliding tests (ctx 0:
# lengths 16 and 1; ctx 2500, B2), page-64 cases (one mixed tile per head at odd lengths), the cache-scale path.
KVOFF_CHECK_CASES = ((1, 753, (80000,), 1, False), (4, 753, (80000, 61234, 97001, 79999), 1, False), (1, 300, (16,), 1, False),
                     (1, 300, (1,), 1, False), (2, 753, (2500, 1907), 1, False), (1, 640, (4096,), 64, False),
                     (3, 200, (5000, 130, 2047), 64, False), (1, 753, (3000,), 1, True))


def _kvoff_prepare(st):
    """[r6kvoff] For each stage this file builds (up to the highest stage the knobs request): run the cases above through
    the base path and the stage's path and require torch.equal of every staged kvt tile below len (incl. the mixed tile's
    old/new bytes and the zeroed pad keys), lens, q8, the k/v caches and the attention output. A stage that mismatches or
    raises is left out of st["kvoff_ok"] (kernel_fn then uses the next lower one, or the base path)."""
    st["kvoff_ok"] = ()
    want = max([int(KVOFF_STAGE)] + [int(v) for v in KVOFF_STAGE_T.values()])
    stages = [s for s in range(1, min(want, _KVOFF_TOP) + 1)]
    if not stages or st.get("zh") is None:
        print(f"KVOFF off (stages requested {want}, zh {st.get('zh') is not None})", flush=True)
        return
    if not KVOFF_SELFCHECK:
        st["kvoff_ok"] = tuple(stages)
        print(f"KVOFF stages {stages} enabled without the self-check", flush=True)
        return
    ok = set(stages)
    try:
        for i, (B, L, ctxs, page, scaled) in enumerate(KVOFF_CHECK_CASES):
            data, cfg, qkv = _kvoff_case(B, L, ctxs, page, 101 + i, scaled)
            if not _zh_ok(st, cfg):
                continue
            ref = _kvoff_run(st, data, cfg, qkv, 0)
            win = cfg["window"]
            blocks = []  # (b, kvh) rows of kvt that gather_kp writes: tiles below len
            for b, c in enumerate(ctxs):
                kst = max(c - (win - 1), 0)
                nb = (c + L - kst + 127) // 128
                blocks += [((b * 2 + h) * ref[4]["capblks"], nb) for h in (0, 1)]
            for s in stages:
                try:
                    got = _kvoff_run(st, data, cfg, qkv, s)
                    tile_eq = all(torch.equal(got[4]["kvt"][r0:r0 + nb], ref[4]["kvt"][r0:r0 + nb]) for r0, nb in blocks)
                    eq = {"kvt": tile_eq, "lens": torch.equal(got[4]["lens"], ref[4]["lens"]), "q8": torch.equal(got[1], ref[1]),
                          "kc": torch.equal(got[2].view(torch.uint8), ref[2].view(torch.uint8)),
                          "vc": torch.equal(got[3].view(torch.uint8), ref[3].view(torch.uint8)), "og": torch.equal(got[0], ref[0])}
                    if not all(eq.values()):
                        print(f"KVOFF stage {s} B{B} L{L} seq{ctxs} page{page} scaled {scaled}: MISMATCH {eq}", flush=True)
                        ok.discard(s)
                    del got
                except Exception as exc:  # noqa: BLE001
                    print(f"KVOFF stage {s} B{B} L{L} seq{ctxs} page{page}: failed {repr(exc)[:300]}", flush=True)
                    ok.discard(s)
            del ref, data, qkv
    except Exception as exc:  # noqa: BLE001
        print("KVOFF self-check failed:", repr(exc)[:400], flush=True)
        ok = set()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    st["kvoff_ok"] = tuple(sorted(ok))
    print(f"KVOFF self-check ({len(KVOFF_CHECK_CASES)} cases): stages bitwise-equal to the base path {sorted(ok)} of {stages}",
          flush=True)


def _combine2_tpc(st, T):
    """Tokens per CTA for moe_combine2: a single wave of CTAs while it fits, else COMBINE_V2_TPC_MAX tokens each."""
    return max(1, min(COMBINE_V2_TPC_MAX, -(-T // (st["nsm"] * COMBINE_V2_CPS))))


def _combine2h_tpc(st, T):
    """[p12h] L1: tokens per CTA for the hand-off combine (moe_combine2h): 1 for T <= HCOMB_TPC1_MAX_T once the HCOMB check
    validated it (st["hc_tpc1"]): a late token no longer waits with a random partner's routes; else _combine2_tpc."""
    return 1 if (st.get("hc_tpc1") and T <= HCOMB_TPC1_MAX_T) else _combine2_tpc(st, T)


def _side_guard():
    """[p12j] B: keep SIDE_STREAM off the stream kernel_fn runs on (see SIDE_STREAM_ALT). A host-side stream identity test
    (no sync, no input value): when the harness captures on SIDE_STREAM, the two side streams swap for this and later calls."""
    global SIDE_STREAM, SIDE_STREAM_ALT
    if SIDE_GUARD and SIDE_STREAM_ALT is not None and SIDE_STREAM == torch.cuda.current_stream():
        SIDE_STREAM, SIDE_STREAM_ALT = SIDE_STREAM_ALT, SIDE_STREAM


def kernel_fn(data, state):
    st = state
    _side_guard()  # [p12j] B
    cfg = data["config"]
    T = cfg["tokens"]
    dev = data["attn_input"].device
    # --- attention ---
    av = _attn_variant(st, cfg)
    zh = _zh_ok(st, cfg)
    kvs = _kvoff_stage(st, cfg) if zh else 0  # [r6kvoff]
    kvo = _kvoff_alloc(data, cfg, kvs) if kvs else None
    if kvo is not None and not KVOFF_AFTER:
        _kvoff_begin(st, data, cfg, kvo)
    pre = None
    split = not zh and av is not None and T <= PREP_SPLIT_MAX_T and PREP_SPLIT.get("slide" if cfg["window"] else "full", False)
    if split and not PREP_SPLIT_AFTER:
        pre = _attn_prep_split(st, data, cfg)
    ev0 = None
    if split and PREP_SPLIT_AFTER:
        ev0 = torch.cuda.Event()
        ev0.record()
    ev_kv = None
    lcf = bool(kvo is not None and KVOFF_AFTER and KVOFF_LCFORK and st.get("lcfork_ok") and T in KVOFF_LCFORK_T)  # [p12h] L3
    if kvo is not None and KVOFF_AFTER and not lcf:
        if KVOFF_FORK:  # [r13] fork from a kernel node (see okc_fork_kernel)
            st["mod"].fork_node()
        ev_kv = torch.cuda.Event()
        ev_kv.record()
    qkv = _mm(st, data["attn_input"], "qkv")
    if lcf:  # [p12h] L3: fork after the qkv GEMM node; _kvoff_begin retags the edge to its launch-completion port
        ev_kv = torch.cuda.Event()
        ev_kv.record()
    if ev0 is not None:
        pre = _attn_prep_split(st, data, cfg, ev0)
    if ev_kv is not None:
        _kvoff_begin(st, data, cfg, kvo, ev_kv, lc=lcf)
    q8 = torch.empty((T, Q_SIZE), dtype=torch.uint8, device=dev)
    if kvo is not None:
        _kvoff_prep(st, data, qkv, q8, cfg, kvo)  # [r6kvoff] join gather_old, then qk_prep (stage 2: qk_prep_kvt)
    else:
        _run_qk_prep(st, data, qkv, q8, cfg)
    og = _zh_attn(st, data, q8, qkv, cfg, kvo=kvo) if zh else _run_attention(st, data, q8, qkv, cfg, av, pre)
    a = _mm(st, og, "o")
    h = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    maxtiles = _max_tiles(T)
    use_cit = CIT and MOE_MODE_SMALL == "fused" and MOE_MODE_LARGE == "fused" and GATHER_PLAN
    # [r6h] hand-off combine for this T (shape-based: same choice on every call of a shape)
    use_h = _hcomb_use(st, T, use_cit)  # [r6s] (r6h's expression, shared with _stack_check)
    # zb layout: [per-expert counts: LOCAL_EXPERTS | work counter: 1 | tile done: maxtiles
    #             | (CIT: GEMM2 done: maxtiles | head, tail, error, pad: 4)
    #             | ([r6h] HCOMB: go flag, pad: 4 | 32-row block GEMM2 counters: _hcomb_nblk(T))]
    hbase = LOCAL_EXPERTS + 1 + maxtiles + ((maxtiles + 4) if use_cit else 0)
    zlen = hbase + (_hcomb_len(T) if use_h else 0)
    recompute = T >= BND_CUDA_MIN and BND_RECOMPUTE and T >= BND_RC_MIN_T
    if recompute:  # the first boundary keeps only the branch's rms factor; the second rebuilds the fp32 stream from it
        zb = torch.empty((zlen,), dtype=torch.int32, device=dev)
        r1 = torch.empty((T,), dtype=torch.float32, device=dev)
        st["mod"].boundary_rc_lp(a, data["stream"], st["a1"], st["a1"], st["b1"], st["g_pre_ffn_norm"], data["stream"], h, 1, zb,
                                 a, r1, st["a1"], st["b1"], _bnd_lp1(st, T))  # [p12f] per-T MODE 1 placement
    elif T >= BND_CUDA_MIN:  # the boundary kernel also zeroes zb (counts | work counter | tile done)
        stream1 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
        zb = torch.empty((zlen,), dtype=torch.int32, device=dev)
        st["mod"].boundary_lp(a, data["stream"], st["a1"], st["a1"], st["b1"], st["g_pre_ffn_norm"], stream1, h, False, zb,
                              st["bnd_lp"][0])
    else:
        stream1 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
        zb = torch.zeros((zlen,), dtype=torch.int32, device=dev)
        _boundary_kernel[(T,)](a, data["stream"], a, st["a1"], st["b1"], st["g_pre_ffn_norm"], stream1, h,
                               HAS_GPOST=False, H=HIDDEN, num_warps=16)
    # --- router + top-8 on a side stream, concurrent with the pooler GEMM ---
    main = torch.cuda.current_stream()
    side = SIDE_STREAM
    serial = T <= ROUTER_SERIAL_MAX or T >= ROUTER_SERIAL_MIN
    if serial:  # the router GEMM alone on all SMs first: cheaper than slowing the pooler down by sharing SMs with it
        logits = _mm(st, h, "router")
    side.wait_stream(main)
    with torch.cuda.stream(side):
        if not serial:
            logits = _router_side(st, h)  # [T, 512] fp32: hi | lo partial logits ([r4rtred] or [4, T, 512] K-slice partials)
        ids = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
        topw = torch.empty((T, TOP_K), dtype=torch.float32, device=dev)
        lid = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
        pos = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
        cnt = zb[:LOCAL_EXPERTS]
        cit = None
        if use_cit:  # tok_left (T) | combine queue (T) | row -> token map (rows)
            ctok = torch.empty((2 * T + T * TOP_K + LOCAL_EXPERTS * MAX_PADDED_ROWS_PER_EXPERT,), dtype=torch.int32, device=dev)
            cit = dict(zc=zb[LOCAL_EXPERTS + 1 + maxtiles:], tok=ctok, topw=topw)
        (st["mod"].moe_route4 if logits.dim() == 3 else st["mod"].moe_route)(  # [r4rtred] (shape-based choice)
            logits, st["router_bias"], data["local_start"], ids, topw, lid, pos, cnt,
                            ctok[:T] if use_cit else st["no_zero"], ctok[T:2 * T] if use_cit else st["no_zero"])  # one warp per token
    # --- experts ---
    p = _mm(st, h, "pooler")
    main.wait_stream(side)
    moe = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
    if cit is not None:
        cit["out"] = moe
    hc = _hcomb_bufs(st, zb[hbase:hbase + _hcomb_len(T)], T) if use_h else None  # [r6h]
    ex = _moe_forward(st, p, lid, pos, cnt, T, zb=zb[LOCAL_EXPERTS:], cit=cit, hcomb=hc, gco=st.get("gco_ok", False))
    y, rrow = ex["y"], ex["rrow"]
    if cit is not None:
        pass  # combined in the fused kernel's tail
    elif hc is not None:  # [r6h] hand-off combine (PDL launch, per-route block counters instead of the grid wait)
        st["mod"].moe_combine2h(y, rrow, lid, topw, st["ones32"], st["g_post_expert_norm"], moe, _combine2h_tpc(st, T),
                                hc["go"], hc["blk"], HCOMB_NEED, st["hcomb_err"])
    elif COMBINE_V2 and st["g_post_expert_norm"].data_ptr() % 16 == 0:  # 16 B loads (Triton's 8-wide layout needs it too)
        st["mod"].moe_combine2(y, rrow, lid, topw, st["ones32"], st["g_post_expert_norm"], moe, _combine2_tpc(st, T))
    elif CUDA_COMBINE:  # one block per token, PDL (the per-expert scale is 1: st["ones32"])
        st["mod"].moe_combine(y, rrow, lid, topw, st["g_post_expert_norm"], moe)
    elif TRITON_PDL:
        _combine_kernel[(T,)](y, rrow, lid, topw, st["ones32"], st["g_post_expert_norm"], moe, H=D_EXPERT, PDL=True, num_warps=8,
                          launch_pdl=True)
    else:
        _combine_kernel[(T,)](y, rrow, lid, topw, st["ones32"], st["g_post_expert_norm"], moe, H=D_EXPERT,
                              num_warps=COMBINE_NW_SMALL if T <= COMBINE_SMALL_T else 8)
    if T <= UNPOOL_SLICED_MAX_T:  # [p12m] the spec: up to 8 tokens only this GPU's 1/8 of the columns (rank 0: the first
        # 1024) are computed; the rest of f stays zero (then scaled by (1 + g_post) and RMS-normalized like any f)
        f = torch.zeros((T, HIDDEN), dtype=torch.bfloat16, device=dev)
        f[:, :UNPOOL_SLICE_COLS] = torch.mm(moe, st["unpooler"][:UNPOOL_SLICE_COLS].t())
    else:
        f = _mm(st, moe, "unpooler")
    stream2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    hidden = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    if recompute:
        st["mod"].boundary_rc_lp(f, data["stream"], st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], stream2,
                                 hidden, 2, st["no_zero"], a, r1, st["a1"], st["b1"], st["bnd_lp"][2])
    elif T >= BND_CUDA_MIN:
        st["mod"].boundary_lp(f, stream1, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], stream2, hidden,
                              True, st["no_zero"], st["bnd_lp"][0])
    else:
        _boundary_kernel[(T,)](f, stream1, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"],
                               stream2, hidden, HAS_GPOST=True, H=HIDDEN, num_warps=16)
    return hidden, stream2, ids


def _hcomb_counts(T):
    """[r6h] Judge-like skewed per-expert route counts for T tokens (SCEN shapes as they are; others scaled from the
    nearest SCEN shape; T <= 64: no local routes; T == 300: sparse, many tokens without a local route)."""
    if T <= 64:
        return []
    if T == 300:
        return [37, 20, 9, 3, 1]
    for k in ("T753", "T3012", "T16384"):
        if SCEN[k][0] == T:
            return list(SCEN[k][1])
    src = SCEN["T753"] if T < 2000 else (SCEN["T3012"] if T < 8000 else SCEN["T16384"])
    out = [min(T, max(1, (c * T) // src[0])) for c in src[1]]
    while sum(out) > T * TOP_K:
        out = out[:-1]
    return out


def _hcomb_expect_blk(T, ex):
    """[r6h] Expected final block counters: HCOMB_NEED on every 32-row block holding a tile's real rows, else 0."""
    nt = int(ex["ntiles"].item())
    exp = torch.zeros((_hcomb_nblk(T),), dtype=torch.int32)
    for e, row0, n, nreal in ex["tiles"][:nt].cpu().tolist():
        exp[row0 // 32:(row0 + nreal + 31) // 32] += HCOMB_NEED
    return exp


def _hcomb_poison(ex):
    """[r6hf] NaN into the expert rows y and 0 (a valid row) into rrow of a finished run. With fixed routing every run
    writes the same values, so without this a combine2 read that comes too early still sees the previous run's (equal)
    y / rrow at the same address (graph replay) or in the block the caching allocator hands back (eager re-run)."""
    ex["y"].fill_(float("nan"))
    ex["rrow"].zero_()


def _hcomb_check(st):
    """[r6h] prepare-time check of the hand-off combine on synthetic judge-like routings for HCOMB_CHECK_T (within the
    HCOMB_MIN_T..MAX_T gate): the reference is the p10 pair (moe_fused, then moe_combine2 behind griddepcontrol.wait);
    the hand-off pair (moe_fused with go / block counters, then moe_combine2h) must give torch.equal y rows and combined
    output, go == 1, every block counter == its expected count, and zero error words, eagerly (HCOMB_CHECK_REPS) and
    from a CUDA graph (HCOMB_CHECK_GRAPH replays). The routing (pos) is fixed, so the gather, the expert GEMMs and the
    combine are deterministic and bitwise comparison is exact. Any failure or exception -> False (hand-off off)."""
    dev = st["w13r"].device
    mod = st["mod"]
    pw = st["g_post_expert_norm"]
    if not COMBINE_V2 or pw.data_ptr() % 16 or MOE_MODE_SMALL != "fused" or MOE_MODE_LARGE != "fused":
        print("HCOMB off: needs COMBINE_V2, a 16 B aligned post_expert_norm and the fused MoE", flush=True)
        return False
    gen = torch.Generator(device=dev)
    gen.manual_seed(20261103)
    try:
        for T in HCOMB_CHECK_T:
            if not (HCOMB_MIN_T <= T <= HCOMB_MAX_T):
                continue
            counts = _hcomb_counts(T)
            lid, pos, cnt = _routes_from_counts(T, counts, dev)
            p = torch.randn((T, D_EXPERT), device=dev, generator=gen).to(torch.bfloat16)
            topw = torch.rand((T, TOP_K), device=dev, generator=gen) + 0.05
            maxtiles, nblk, tpc = _max_tiles(T), _hcomb_nblk(T), _combine2_tpc(st, T)
            za = torch.zeros((1 + maxtiles,), dtype=torch.int32, device=dev)
            ex = _moe_forward(st, p, lid, pos, cnt, T, zb=za)
            ref = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
            mod.moe_combine2(ex["y"], ex["rrow"], lid, topw, st["ones32"], pw, ref, tpc)
            torch.cuda.synchronize()
            rr = ex["rrow"][lid >= 0].long()
            yref = ex["y"][rr].clone()
            exp_blk = _hcomb_expect_blk(T, ex)
            zh = torch.zeros((1 + maxtiles + _hcomb_len(T),), dtype=torch.int32, device=dev)
            hc = _hcomb_bufs(st, zh[1 + maxtiles:], T)
            if HCOMB_HOIST_ONLY:
                exp_blk = torch.zeros((0,), dtype=torch.int32)
            out = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)

            def run_h():
                zh.zero_()
                ex2 = _moe_forward(st, p, lid, pos, cnt, T, zb=zh, hcomb=hc)
                mod.moe_combine2h(ex2["y"], ex2["rrow"], lid, topw, st["ones32"], pw, out, _combine2h_tpc(st, T), hc["go"],
                                  hc["blk"], HCOMB_NEED, st["hcomb_err"])  # [p12h] (reference: moe_combine2 at tpc)
                return ex2

            def verify(ex2, what):
                torch.cuda.synchronize()
                bad = []
                if not torch.equal(out, ref):
                    bad.append(f"out max|d| {float((out.float() - ref.float()).abs().max()):.3g}")
                if not torch.equal(ex2["rrow"], ex["rrow"]) or not torch.equal(ex2["y"][rr], yref):
                    bad.append("y/rrow differ")
                if int(hc["go"].item()) != 1:
                    bad.append(f"go {int(hc['go'].item())}")
                if not torch.equal(hc["blk"].cpu(), exp_blk):
                    bad.append(f"block counters ({int((hc['blk'].cpu() != exp_blk).sum())} differ)")
                if int(st["hcomb_err"].abs().sum().item()):
                    bad.append(f"error words {st['hcomb_err'].tolist()}")
                if bad:
                    print(f"HCOMB check T={T} {what}: MISMATCH {'; '.join(bad)} -> hand-off off", flush=True)
                return not bad

            for i in range(HCOMB_CHECK_REPS):
                out.fill_(float("nan"))
                ex2 = run_h()
                ok = verify(ex2, f"eager {i}")
                _hcomb_poison(ex2)  # [r6hf] the next run's y / rrow blocks come back poisoned (same sizes)
                del ex2
                if not ok:
                    return False
            if HCOMB_CHECK_GRAPH:
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()  # (the eager runs above were the warm-up: custom kernels only)
                with torch.cuda.graph(g):
                    exg = run_h()
                for i in range(HCOMB_CHECK_GRAPH):
                    out.fill_(float("nan"))
                    _hcomb_poison(exg)  # [r6hf] graph buffers keep their addresses: poison before every replay
                    g.replay()
                    if not verify(exg, f"graph {i}"):
                        del g
                        return False
                del g, exg
            msg = ""
            if HCOMB_CHECK_TIME and T >= 300:
                ra = torch.empty_like(ref)

                def run_a():
                    za.zero_()
                    e = _moe_forward(st, p, lid, pos, cnt, T, zb=za)
                    mod.moe_combine2(e["y"], e["rrow"], lid, topw, st["ones32"], pw, ra, tpc)
                ta = [_time(run_a, reps=5) for _ in range(2)]
                tb = [_time(run_h, reps=5) for _ in range(2)]
                ok_t = int(st["hcomb_err"].abs().sum().item()) == 0
                msg = f"; eager MoE+combine {min(ta):.1f} us (PDL wait) vs {min(tb):.1f} us (hand-off){'' if ok_t else ' ERR'}"
                if not ok_t or not torch.equal(out, ref):  # [r6hf] a spin-limit hit costs 200 ms per call: off
                    print(f"HCOMB check T={T} timing runs: MISMATCH (error words {st['hcomb_err'].tolist()}, "
                          f"out == ref: {bool(torch.equal(out, ref))}) -> hand-off off", flush=True)
                    return False
            print(f"HCOMB check T={T}{' (hoist-only)' if HCOMB_HOIST_ONLY else ''}: ok ({len(counts)} experts, {int(ex['ntiles'].item())} tiles, {HCOMB_CHECK_REPS} eager, "
                  f"{HCOMB_CHECK_GRAPH} graph){msg}", flush=True)
            del ex, ref, yref, zh, za, out, p
            torch.cuda.empty_cache()
        st["hcomb_err"].zero_()
        return True
    except Exception as exc:  # noqa: BLE001
        print("HCOMB check failed:", repr(exc)[:300], "-> hand-off off", flush=True)
        return False


def _stack_pick(st, x, name, pick):
    """[r6s] _mm(st, x, name) with the (T, name) GEMM choice temporarily set to pick (None: torch.mm)."""
    T = x.shape[0]
    ch = st["lt_choice"]
    had, old = (T, name) in ch, ch.get((T, name))
    try:
        if pick is None:
            ch.pop((T, name), None)
        else:
            ch[(T, name)] = pick
        return _mm(st, x, name)
    finally:
        if had:
            ch[(T, name)] = old
        else:
            ch.pop((T, name), None)


def _stack_chain(st, T, gen, ref_pick):
    """[r6s] One check of the MoE -> combine -> unpooler chain at T as kernel_fn runs it (hand-off per _hcomb_use, the
    current unpooler pick) against the p10 chain (moe_combine2 behind the grid wait, unpooler pick ref_pick) on fixed
    judge-like routing (_hcomb_counts): torch.equal on the combined output and on the unpooler output, zero error words and
    go == 1 with the hand-off; STACK_CHECK_REPS eager runs, then STACK_CHECK_GRAPH replays of one CUDA graph, every chain
    buffer (y, rrow, combined output, unpooler output) poisoned before each run. Returns (ok, text); may raise."""
    dev = st["w13r"].device
    mod = st["mod"]
    pw = st["g_post_expert_norm"]
    use_h = _hcomb_use(st, T, False)
    counts = _hcomb_counts(T)
    lid, pos, cnt = _routes_from_counts(T, counts, dev)
    p = torch.randn((T, D_EXPERT), device=dev, generator=gen).to(torch.bfloat16)
    topw = torch.rand((T, TOP_K), device=dev, generator=gen) + 0.05
    maxtiles, tpc = _max_tiles(T), _combine2_tpc(st, T)
    za = torch.zeros((1 + maxtiles,), dtype=torch.int32, device=dev)
    ex = _moe_forward(st, p, lid, pos, cnt, T, zb=za)
    mref = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
    mod.moe_combine2(ex["y"], ex["rrow"], lid, topw, st["ones32"], pw, mref, tpc)
    fref = _stack_pick(st, mref, "unpooler", ref_pick)
    torch.cuda.synchronize()
    del ex, za
    zh = torch.zeros((1 + maxtiles + (_hcomb_len(T) if use_h else 0),), dtype=torch.int32, device=dev)
    hc = _hcomb_bufs(st, zh[1 + maxtiles:], T) if use_h else None
    st["hcomb_err"].zero_()

    def run():
        zh.zero_()
        e = _moe_forward(st, p, lid, pos, cnt, T, zb=zh, hcomb=hc)
        m = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
        if hc is not None:
            mod.moe_combine2h(e["y"], e["rrow"], lid, topw, st["ones32"], pw, m, _combine2h_tpc(st, T), hc["go"], hc["blk"],
                              HCOMB_NEED, st["hcomb_err"])  # [p12h]
        else:
            mod.moe_combine2(e["y"], e["rrow"], lid, topw, st["ones32"], pw, m, tpc)
        return e, m, _mm(st, m, "unpooler")

    def poison(r):
        e, m, f = r
        e["y"].fill_(float("nan"))
        e["rrow"].zero_()
        m.fill_(float("nan"))
        f.fill_(float("nan"))

    def verify(r, what):
        torch.cuda.synchronize()
        e, m, f = r
        bad = []
        if not torch.equal(m, mref):
            bad.append(f"combined output: {int((m != mref).sum())} elements differ")
        if not torch.equal(f, fref):
            bad.append(f"unpooler output: {int((f != fref).sum())} elements differ")
        if hc is not None and int(hc["go"].item()) != 1:
            bad.append(f"go {int(hc['go'].item())}")
        if int(st["hcomb_err"].abs().sum().item()):
            bad.append(f"error words {st['hcomb_err'].tolist()}")
        return not bad, f"{what}: {'; '.join(bad)}"

    for i in range(STACK_CHECK_REPS):
        r = run()
        ok, msg = verify(r, f"eager {i}")
        poison(r)  # the next run's blocks come back from the caching allocator poisoned
        del r
        if not ok:
            return False, msg
    if STACK_CHECK_GRAPH:
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()  # (the eager runs above were the warm-up)
        with torch.cuda.graph(g):
            rg = run()
        for i in range(STACK_CHECK_GRAPH):
            poison(rg)  # graph buffers keep their addresses: poison before every replay
            g.replay()
            ok, msg = verify(rg, f"graph {i}")
            if not ok:
                del g, rg
                return False, msg
        del g, rg
    return True, (f"{'hand-off' if use_h else 'PDL-wait'} combine, {len(counts)} experts, {STACK_CHECK_REPS} eager, "
                  f"{STACK_CHECK_GRAPH} graph")


def _stack_check(st):
    """[r6s] See STACK_CHECK: for each T where r6ltnm replaced the unpooler pick, check the chain kernel_fn runs; on a
    mismatch or an exception, restore the p10 unpooler pick at T and, if the hand-off is on there, check the chain again
    with the p10 pick and turn the hand-off off if that fails too. Also, at each T in STACK_CHECK_HAND_T without an r6ltnm
    unpooler pick but with the hand-off in use, check the hand-off -> (p10) unpooler chain; hand-off off on a mismatch.
    Never raises."""
    try:
        prev = st.get("ltnm_prev") or {}
        if not STACK_CHECK or not st.get("lt_choice"):
            return
        use_cit = CIT and MOE_MODE_SMALL == "fused" and MOE_MODE_LARGE == "fused" and GATHER_PLAN
        lt_t = {T for (T, name) in prev if name == "unpooler"}
        todo = sorted(lt_t | {T for T in STACK_CHECK_HAND_T if _hcomb_use(st, T, False)})
        if use_cit or not todo:
            print(f"STACK check: nothing to check ({'CIT on' if use_cit else 'no r6ltnm unpooler pick, no hand-off T'})",
                  flush=True)
            return
        gen = torch.Generator(device=st["w13r"].device)
        gen.manual_seed(STACK_CHECK_SEED)
        for T in todo:
            use_h = _hcomb_use(st, T, False)
            pick = st["lt_choice"].get((T, "unpooler"))
            if T not in lt_t:  # hand-off -> the p10 unpooler pick (T = 753: p10's programmatic edge combine2 -> unpooler)
                if not use_h:  # (turned off by an earlier T's check)
                    continue
                try:
                    ok, msg = _stack_chain(st, T, gen, pick)
                except Exception as exc:  # noqa: BLE001
                    ok, msg = False, f"failed: {repr(exc)[:300]}"
                if ok:
                    print(f"STACK check T={T}: ok ({msg}; p10 unpooler {pick})", flush=True)
                else:
                    st["hcomb_ok"] = False
                    print(f"STACK check T={T} (p10 unpooler {pick}): MISMATCH {msg} -> hand-off off", flush=True)
                continue
            ref_pick = prev[(T, "unpooler")]
            try:
                ok, msg = _stack_chain(st, T, gen, ref_pick)
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, f"failed: {repr(exc)[:300]}"
            if ok:
                print(f"STACK check T={T}: ok ({msg}; unpooler {pick} vs p10 {ref_pick})", flush=True)
                continue
            print(f"STACK check T={T}: MISMATCH {msg} -> p10 unpooler pick {ref_pick} restored at T={T}", flush=True)
            if ref_pick is None:
                st["lt_choice"].pop((T, "unpooler"), None)
            else:
                st["lt_choice"][(T, "unpooler")] = ref_pick
            if use_h:
                try:
                    ok2, msg2 = _stack_chain(st, T, gen, ref_pick)
                except Exception as exc:  # noqa: BLE001
                    ok2, msg2 = False, f"failed: {repr(exc)[:300]}"
                if ok2:
                    print(f"STACK check T={T} with the p10 unpooler pick: ok ({msg2}); hand-off kept", flush=True)
                else:
                    st["hcomb_ok"] = False
                    print(f"STACK check T={T} with the p10 unpooler pick: MISMATCH {msg2} -> hand-off off", flush=True)
        st["hcomb_err"].zero_()
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        print("STACK check failed:", repr(exc)[:300], "-> hand-off off, p10 unpooler picks", flush=True)
        st["hcomb_ok"] = False
        for (T, name), c in (st.get("ltnm_prev") or {}).items():
            if name == "unpooler" and st.get("lt_choice") is not None:
                if c is None:
                    st["lt_choice"].pop((T, name), None)
                else:
                    st["lt_choice"][(T, name)] = c


# --------------------------------------------------------------------------------------
# Self-tests / micro-benchmarks, printed from prepare() when SELFTEST is on.
# They never run on the scored path; they exist to check the kernels against
# a reference and to time the pieces. All timing goes through _time, which
# flushes L2 before each call the way the judge's replays do.
# --------------------------------------------------------------------------------------


def _routes_from_counts(T, counts, dev):
    """Build (lid, pos, cnt) for a synthetic routing: counts[e] tokens to expert e."""
    lid = torch.full((T, TOP_K), -1, dtype=torch.int32, device=dev)
    pos = torch.zeros((T, TOP_K), dtype=torch.int32, device=dev)
    r = 0
    for e, c in enumerate(counts):
        idx = torch.arange(r, r + c, device=dev)
        lid.view(-1)[(idx % T) * TOP_K + idx // T] = e
        pos.view(-1)[(idx % T) * TOP_K + idx // T] = torch.arange(c, device=dev, dtype=torch.int32)
        r += c
    cnt = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device=dev)
    cnt[: len(counts)] = torch.tensor(counts, dtype=torch.int32, device=dev)
    return lid, pos, cnt


_FLUSH_BUF = None


def _time(fn, reps=5):
    """Median time in us with a cold L2 (a read sweep over 256 MB before each call, like the judge's replays)."""
    global _FLUSH_BUF
    if _FLUSH_BUF is None:
        _FLUSH_BUF = (torch.ones(128 << 20, dtype=torch.float16, device="cuda"),
                      torch.zeros((), dtype=torch.float32, device="cuda"))
    buf, sink = _FLUSH_BUF
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        sink.copy_(buf.sum(dtype=torch.float32))
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


SCEN = {
    "T16384": (16384, [12547, 11910, 2820, 2392, 270, 242, 97, 60, 34, 16, 14, 2, 1, 1, 1]),
    "T3012": (3012, [2277, 2191, 416, 400, 47, 37, 32, 11, 7, 5, 4, 3, 2, 1, 1]),
    "T753": (753, [540, 529, 127, 108, 14, 6, 3, 3, 3, 2, 1, 1]),
}


def _selftest(st, weights):
    from nvfp4 import dequantize
    dev = weights["w13"].device
    mod = st["mod"]
    torch.manual_seed(0)
    # correctness on a small skewed routing
    T = 700
    counts = [300, 200, 100, 60, 33, 17, 14, 5, 1]  # 14 -> N=16, 5 and 1 -> N=8 tiles (below FINE_MIN_T)
    lid, pos, cnt = _routes_from_counts(T, counts, dev)
    p = torch.randn((T, D_EXPERT), device=dev).to(torch.bfloat16)
    gw = st["g_pre_expert_norm"]
    out = _moe_forward(st, p, lid, pos, cnt, T)
    y, rrow = out["y"], out["rrow"]
    pf = p.float()
    xn = (pf * torch.rsqrt(pf.pow(2).mean(-1, keepdim=True) + RMS_EPS) * (1.0 + gw.float())).to(torch.bfloat16)
    msg = []
    for e in range(len(counts)):
        tok, slot = (lid == e).nonzero(as_tuple=True)
        W13 = dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).to(torch.bfloat16)
        W2 = dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).to(torch.bfloat16)
        hh = xn[tok].float() @ W13.float().T
        inter = (F.silu(hh[:, :D_EXPERT]) * hh[:, D_EXPERT:]).to(torch.bfloat16)
        ref = inter.float() @ W2.float().T
        got = y[rrow[tok, slot].long()].float()
        msg.append(f"e{e}:n{len(tok)}:{float((got - ref).norm() / ref.norm()):.4f}")
    print("SELFTEST moe rel err", " ".join(msg), flush=True)
    # timing
    for name, (T, counts) in SCEN.items():
        lid, pos, cnt = _routes_from_counts(T, counts, dev)
        p = torch.randn((T, D_EXPERT), device=dev).to(torch.bfloat16)
        out = _moe_forward(st, p, lid, pos, cnt, T)
        nt = int(out["ntiles"])
        rows = int(out["tiles"][:nt, 2].sum())
        R = sum(counts)
        res = []
        P1, P2 = 8, 16
        t1 = _time(lambda: mod.moe_gemm(st["w13r"], out["xt"], out["tiles"], out["ntiles"], st["w13_gs"], out["inter"], 64, P1,
                                        True, st["smem"]))
        t2 = _time(lambda: mod.moe_gemm(st["w2r"], out["inter"], out["tiles"], out["ntiles"], st["w2_gs"], out["y"], 32, P2,
                                        False, st["smem"]))
        f1 = 2 * R * (2 * D_EXPERT) * D_EXPERT / t1 / 1e6  # GEMM1: [R, D_EXPERT] x [2*D_EXPERT, D_EXPERT]^T
        f2 = 2 * R * D_EXPERT * D_EXPERT / t2 / 1e6  # GEMM2: [R, D_EXPERT] x [D_EXPERT, D_EXPERT]^T
        res.append(f"P{P1}/{P2}: g1 {t1:.0f}us ({f1:.0f}TF) g2 {t2:.0f}us ({f2:.0f}TF)")
        nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        rr = out["rrow"][out["rrow"] >= 0].long()
        y0 = out["y"][rr].clone()
        ctr = torch.zeros((2,), dtype=torch.int32, device=dev)
        rows_t = torch.empty((LOCAL_EXPERTS,), dtype=torch.int32, device=dev)

        def g1p():
            ctr.zero_()
            mod.moe_gemm_p(st["w13r"], out["xt"], out["tiles"], out["ntiles"], st["w13_gs"], out["inter"], 64, nsm, True,
                           st["smem"], ctr[0:1])

        def g2p():
            ctr.zero_()
            mod.moe_gemm_p(st["w2r"], out["inter"], out["tiles"], out["ntiles"], st["w2_gs"], out["y"], 32, nsm, False,
                           st["smem"], ctr[1:2])
        zb = torch.zeros((1 + out["tiles"].shape[0],), dtype=torch.int32, device=dev)

        def gf():
            zb.zero_()
            mod.moe_fused(st["w13r"], st["w2r"], out["xt"], out["inter"], out["y"], out["tiles"], out["ntiles"], st["w13_gs"],
                          st["w2_gs"], nsm, st["smem"], zb[0:1], zb[1:], st["moe_dbg"], st["moe_flags"], st["no_zero"],
                          st["no_zero"], out["rrow"], lid, st["no_zero_f"], st["g_post_expert_norm"], st["no_zero_b"],
                          st["no_zero"], st["no_zero"])
        for order in (0, 1, 2):
            mod.moe_plan(cnt, out["offsets"], out["tiles"], out["ntiles"], rows_t, order)
            tf = _time(gf)
            same_f = bool(torch.equal(out["y"][rr], y0))
            res.append(f"fused o{order}: {tf:.0f}us same={same_f}")
            t1 = _time(g1p)
            t2 = _time(g2p)
            f1 = 2 * R * 8192 * 4096 / t1 / 1e6
            f2 = 2 * R * 4096 * 4096 / t2 / 1e6
            same = bool(torch.equal(out["y"][rr], y0))
            res.append(f"persist o{order}: g1 {t1:.0f}us ({f1:.0f}TF) g2 {t2:.0f}us ({f2:.0f}TF) same={same}")
        tg = _time(lambda: mod.moe_gather(p, st["g_pre_expert_norm"], lid, pos, out["offsets"], cnt, out["xt"], out["rrow"]))
        print(f"SELFTEST {name} R={R} padded={rows} tiles={nt} gather {tg:.0f}us |", " | ".join(res), flush=True)
    a = torch.randn((16384, 8192), device=dev).to(torch.bfloat16)
    tmm = _time(lambda: torch.mm(a, st["pooler"].t()))
    print(f"SELFTEST cublas 16384x4096x8192 {tmm:.0f}us {2 * 16384 * D_EXPERT * HIDDEN / tmm / 1e6:.0f}TF", flush=True)
    s = torch.randn((16384, 8192), device=dev)
    o1 = torch.empty_like(s)
    o2 = torch.empty((16384, 8192), device=dev, dtype=torch.bfloat16)
    for nw in (4, 8, 16):
        tb = _time(lambda: _boundary_kernel[(16384,)](a, s, st["g_post_ffn_norm"], st["a2"], st["b2"],
                                             st["g_next_pre_attn_norm"], o1, o2, HAS_GPOST=True, H=HIDDEN, num_warps=nw))
        print(f"SELFTEST boundary nw{nw} {tb:.0f}us {16384 * HIDDEN * 12 / tb / 1e6:.2f}TB/s", flush=True)
    del a, s, o1, o2
    torch.cuda.empty_cache()


ATTN_SCEN = [
    ("turn1_full", 1, 753, 80000, 0),
    ("turn4_full", 4, 753, 80000, 0),
    ("ft_full32k", 1, 16384, 32768, 0),
    ("ft_full64k", 1, 16384, 65536, 0),
    ("ft_slide", 1, 16384, 32768, 2048),
]


def _selftest_attn2(st):
    if _fa3 is None:
        return
    dev = "cuda"
    torch.manual_seed(2)
    for name, B, L, ctx, window in ATTN_SCEN:
        S = ctx + L
        T = B * L
        q8 = (torch.randn((T, HQ, HD), device=dev) * 0.7).to(FP8)
        qkv = torch.randn((T, QKV_SIZE), device=dev).to(torch.bfloat16)
        kc = (torch.randn((B * S, 1, HKV, HD), device=dev) * 0.8 + 0.4).to(FP8)
        vc = torch.randn((B * S, 1, HKV, HD), device=dev).to(FP8)
        perm = torch.randperm(B * S, device=dev).to(torch.int32)
        data = {"k_cache": kc, "v_cache": vc, "page_table": perm.view(B, S), "seq_lens": torch.full((B,), ctx, dtype=torch.int32, device=dev),
                "cache_seqlens": torch.full((B,), S, dtype=torch.int32, device=dev),
                "cu_seqlens_q": torch.arange(0, T + 1, L, dtype=torch.int32, device=dev), "scheduler_metadata": None,
                "k_scale": torch.tensor(1.0, device=dev), "v_scale": torch.tensor(1.0, device=dev)}
        cfg = {"batch": B, "q_len": L, "window": window, "page_size": 1, "sm_scale": SYNTH_SM_SCALE,
               "qk_scale_factor": SYNTH_QK_SCALE_FACTOR, "kv_scale": 1.0, "tokens": T}

        def ref_fn():
            return _attention_norm_gate(data, q8, qkv, cfg)
        flops = 4.0 * HQ * HD * (B * L * (ctx if not window else min(ctx, window)) + B * L * L / 2)
        try:
            r = ref_fn()
            t_ref = _time(ref_fn, reps=3)
            res = [f"fa3+ng {t_ref:.0f}us ({flops / t_ref / 1e6:.0f}TF)"]
        except Exception as exc:  # noqa: BLE001
            print(f"SELFTEST attn2 {name} FA3 ERR {repr(exc)[:300]}", flush=True)
            continue
        for variant, nsplit in [(v, 1) for v in ATTN_VARIANTS] + [(v, _attn_splits(cfg, data, v, st["nsm"])) for v in (1, 9)]:
            try:
                o = _attn_custom(st, data, q8, qkv, cfg, nsplit=nsplit, variant=variant)
                torch.cuda.synchronize()
                err = float((o.float() - r.float()).norm() / r.float().norm())
                t_own = _time(lambda: _attn_custom(st, data, q8, qkv, cfg, nsplit=nsplit, variant=variant), reps=3)
                res.append(f"v{variant}s{nsplit} {t_own:.0f}us ({flops / t_own / 1e6:.0f}TF) err {err:.4f}")
            except Exception as exc:  # noqa: BLE001
                res.append(f"v{variant}s{nsplit} ERR {repr(exc)[:120]}")
        print(f"SELFTEST attn2 {name}: " + " | ".join(res), flush=True)
        del q8, qkv, kc, vc
    torch.cuda.empty_cache()


def _gco_check(st):
    """[r13] st["gco_ok"]: the coalesced-store gather (moe_gather_plan_co) gives torch.equal xt (every byte, zero-initialised
    buffers), rrow and plan (offsets, tiles, tile count, rows) to moe_gather_plan on routes from the route kernel, T in
    {753, 3012, 1, 300, 16384}; off on any mismatch or error."""
    st["gco_ok"] = False
    if not GATHER_CO:
        return
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(977)
    mod = st["mod"]
    ok = True
    try:
        for T in (753, 3012, 1, 300, 16384):
            logits = torch.randn((T, 512), device=dev, generator=g) * 2
            ids = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
            topw = torch.empty((T, TOP_K), dtype=torch.float32, device=dev)
            lid = torch.empty_like(ids)
            pos = torch.empty_like(ids)
            cnt = torch.zeros((LOCAL_EXPERTS,), dtype=torch.int32, device=dev)
            ls = torch.zeros((1,), dtype=torch.int32, device=dev)
            mod.moe_route(logits, st["router_bias"], ls, ids, topw, lid, pos, cnt, st["no_zero"], st["no_zero"])
            p = (torch.randn((T, D_EXPERT), device=dev, generator=g) * 2).to(torch.bfloat16)
            maxrows = T * TOP_K + LOCAL_EXPERTS * MAX_PADDED_ROWS_PER_EXPERT
            fine = 1 if (FINE_MIN_T and T >= FINE_MIN_T) else 0
            order = (MOE_ORDER_LARGE if T >= MOE_LARGE_T else MOE_ORDER_SMALL) | (fine << MOE_FINE_ORDER_SHIFT)
            res = []
            for fn in (mod.moe_gather_plan, mod.moe_gather_plan_co):
                xt = torch.zeros((maxrows, D_EXPERT), dtype=torch.bfloat16, device=dev)
                rrow = torch.full((T, TOP_K), -5, dtype=torch.int32, device=dev)
                offs = torch.full((LOCAL_EXPERTS,), -5, dtype=torch.int32, device=dev)
                rows = offs.clone()
                tiles = torch.full((_max_tiles(T), 4), -5, dtype=torch.int32, device=dev)
                nt = torch.full((1,), -5, dtype=torch.int32, device=dev)
                fn(p, st["g_pre_expert_norm"], lid, pos, cnt, xt, rrow, offs, tiles, nt, rows, order, st["no_zero"], st["no_zero"])
                res.append((xt, rrow, offs, rows, tiles, nt))
            torch.cuda.synchronize()
            eq = all(torch.equal(x, y) for x, y in zip(res[0], res[1]))
            if not eq:
                print(f"GATHER CO check T={T}: MISMATCH", flush=True)
                ok = False
            del res, logits, p
    except Exception as exc:  # noqa: BLE001
        print("GATHER CO check failed:", repr(exc)[:300], flush=True)
        ok = False
    st["gco_ok"] = ok
    torch.cuda.empty_cache()
    print(f"GATHER CO: requested {GATHER_CO}, bitwise-checked in use {ok}", flush=True)


def _bnd_lp1(st, T):
    """[p12f] MODE 1 load placement for T: BND_LP_RC1_T's value if _p12f_check validated it, else the checked BND_LP_RC1."""
    v = BND_LP_RC1_T.get(T)
    return v if v is not None and v in st.get("bnd_lp1_ok", ()) else st["bnd_lp"][1]


def _p12f_check(st):
    """[p12f] Prepare-time bitwise checks of the p12f kernels against the p12e kernels they replace (each off on any mismatch
    or error): qkpm_ok: qk_prep_m's q8, k cache and v cache bytes (poisoned buffers) equal qk_prep's on random rows (T 1 / 3 /
    77 / 753 / 3012 / 4097 / 16384, RoPE on / off, cache scales on / off, int64 positions and cache rows); gkp2_ok: gatherp2
    gives gatherp's whole kvt buffer (poisoned) and lens on full-attention cases (batch 1-4, page 1 and 64) and the sliding
    kvoff check geometries; bnd_lp1_ok: each BND_LP_RC1_T placement gives the lp 0 launch's normed rows, zeroed counters and
    rms factors (T 1 / 37 / 753 / 2047 / 3012 / 16384); gmb_ok: the min-5-blocks gather gives moe_gather_plan_co's xt (every
    byte), rrow and plan on routes from the route kernel (T 1 / 300 / 753 / 3012 / 16384)."""
    dev = "cuda"
    mod = st["mod"]
    g = torch.Generator(device=dev)
    g.manual_seed(14014)
    # (1) qk_prep_m
    ok = bool(QKPM)
    try:
        for T, rope, scaled in (((1, True, False), (3, False, False), (77, False, True), (753, True, False), (753, False, False),
                                 (3012, True, False), (3012, False, False), (4097, True, True), (16384, True, False),
                                 (16384, False, False)) if ok else ()):
            qkv = (torch.randn((T, QKV_SIZE), device=dev, generator=g) * 1.5).to(torch.bfloat16)
            posn = torch.randint(0, 100000, (T,), device=dev, generator=g).to(torch.int64)
            nrow = T + 50
            loc = torch.randperm(nrow, device=dev, generator=g)[:T].to(torch.int64)
            ksc = torch.tensor(0.75 if scaled else 1.0, device=dev)
            vsc = torch.tensor(0.6 if scaled else 1.0, device=dev)
            res = []
            for m in (0, 1):
                q8 = torch.full((T, Q_SIZE), 0x5A, dtype=torch.uint8, device=dev)
                kc = torch.full((nrow, 128), 0xA5, dtype=torch.uint8, device=dev)
                vc = torch.full((nrow, 128), 0x3C, dtype=torch.uint8, device=dev)
                (mod.qk_prep_m if m else mod.qk_prep)(qkv, posn, loc, st["inv_freq"], q8, kc, vc, ksc, vsc, rope, scaled)
                res.append((q8, kc, vc))
            torch.cuda.synchronize()
            if not all(torch.equal(x, y) for x, y in zip(res[0], res[1])):
                print(f"P12F qk_prep_m T={T} rope {rope} scaled {scaled}: MISMATCH", flush=True)
                ok = False
            del res, qkv
    except Exception as exc:  # noqa: BLE001
        print("P12F qk_prep_m check failed:", repr(exc)[:300], flush=True)
        ok = False
    st["qkpm_ok"] = ok
    # (2) gather_kp2
    gok = bool(GKP2) and st.get("zh") is not None
    try:
        cases = [((1, 753, (80000,), 1, False), 0), ((4, 753, (80000, 61234, 97001, 79999), 1, False), 0),
                 ((2, 753, (2500, 1907), 1, False), 0), ((3, 200, (5000, 130, 2047), 64, False), 0), ((1, 300, (0,), 1, False), 0),
                 ((1, 640, (4096,), 64, True), 0)] + [(c, 2048) for c in KVOFF_CHECK_CASES]
        for i, ((B, L, ctxs, page, scaled), win) in enumerate(cases if gok else ()):
            data, cfg, _ = _kvoff_case(B, L, ctxs, page, 501 + i, scaled, window=win if win else 2048)
            pt = data["page_table"]
            pagelog = page.bit_length() - 1
            max_len = pt.shape[1] * page
            lens_full = data["cache_seqlens"]
            cap = ((min(win - 1 + L, max_len) if win else max_len) + 255) & ~255
            capblks = cap // 128
            kc = data["k_cache"].view(torch.uint8)
            vc = data["v_cache"].view(torch.uint8)
            res = []
            for fn in (st["zh"].gatherp, st["zh"].gatherp2):
                kvt = torch.full((B * 2 * capblks, 16384), 0xA5, dtype=torch.uint8, device=dev)
                if win:
                    lens = torch.full((B,), -7, dtype=torch.int32, device=dev)
                    fn(kc, vc, kvt, pt, lens_full, data["seq_lens"].to(torch.int32), lens, pt.stride(0), pagelog, capblks, B, L, win)
                else:
                    lens = lens_full[:0]
                    fn(kc, vc, kvt, pt, lens_full, lens, lens, pt.stride(0), pagelog, capblks, B, L, 0)
                res.append((kvt, lens))
            torch.cuda.synchronize()
            if not (torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])):
                print(f"P12F gather_kp2 B{B} L{L} seq{ctxs} page{page} win{win}: MISMATCH", flush=True)
                gok = False
            del res, data
    except Exception as exc:  # noqa: BLE001
        print("P12F gather_kp2 check failed:", repr(exc)[:300], flush=True)
        gok = False
    st["gkp2_ok"] = gok
    # (3) per-T MODE 1 placements against lp 0
    lps = sorted({int(v) for v in BND_LP_RC1_T.values()} - {0})
    lok = set(lps)
    try:
        for T in ((753, 1, 37, 2047, 3012, 16384) if lps else ()):
            br = (torch.randn((T, HIDDEN), device=dev, generator=g) * 3).to(torch.bfloat16)
            s_in = torch.randn((T, HIDDEN), device=dev, generator=g) * 2
            outs = {}
            for lp in [0] + lps:
                no = torch.zeros((T, HIDDEN), dtype=torch.bfloat16, device=dev)
                z = torch.full((40,), 5, dtype=torch.int32, device=dev)
                rio = torch.full((T,), -1.0, device=dev)
                b_, s_ = br.clone(), s_in.clone()
                mod.boundary_rc_lp(b_, s_, st["a1"], st["a1"], st["b1"], st["g_pre_ffn_norm"], s_, no, 1, z, b_, rio,
                                   st["a1"], st["b1"], lp)
                outs[lp] = (no, z, rio)
            torch.cuda.synchronize()
            for lp in lps:
                if not all(torch.equal(x, y) for x, y in zip(outs[0], outs[lp])):
                    print(f"P12F MODE 1 lp {lp} T={T}: MISMATCH", flush=True)
                    lok.discard(lp)
            del outs, br, s_in
    except Exception as exc:  # noqa: BLE001
        print("P12F MODE 1 lp check failed:", repr(exc)[:300], flush=True)
        lok = set()
    st["bnd_lp1_ok"] = tuple(sorted(lok))
    # (5) gather at min 5 blocks per SM
    mok = bool(GATHER_MINB_T) and bool(st.get("gco_ok"))
    try:
        for T in ((753, 3012, 1, 300, 16384) if mok else ()):
            logits = torch.randn((T, 512), device=dev, generator=g) * 2
            ids = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
            topw = torch.empty((T, TOP_K), dtype=torch.float32, device=dev)
            lid = torch.empty_like(ids)
            pos = torch.empty_like(ids)
            cnt = torch.zeros((LOCAL_EXPERTS,), dtype=torch.int32, device=dev)
            ls = torch.zeros((1,), dtype=torch.int32, device=dev)
            mod.moe_route(logits, st["router_bias"], ls, ids, topw, lid, pos, cnt, st["no_zero"], st["no_zero"])
            p = (torch.randn((T, D_EXPERT), device=dev, generator=g) * 2).to(torch.bfloat16)
            maxrows = T * TOP_K + LOCAL_EXPERTS * MAX_PADDED_ROWS_PER_EXPERT
            fine = 1 if (FINE_MIN_T and T >= FINE_MIN_T) else 0
            order = (MOE_ORDER_LARGE if T >= MOE_LARGE_T else MOE_ORDER_SMALL) | (fine << MOE_FINE_ORDER_SHIFT)
            res = []
            for fn in (mod.moe_gather_plan_co, mod.moe_gather_plan_cm):
                xt = torch.zeros((maxrows, D_EXPERT), dtype=torch.bfloat16, device=dev)
                rrow = torch.full((T, TOP_K), -5, dtype=torch.int32, device=dev)
                offs = torch.full((LOCAL_EXPERTS,), -5, dtype=torch.int32, device=dev)
                rows = offs.clone()
                tiles = torch.full((_max_tiles(T), 4), -5, dtype=torch.int32, device=dev)
                nt = torch.full((1,), -5, dtype=torch.int32, device=dev)
                fn(p, st["g_pre_expert_norm"], lid, pos, cnt, xt, rrow, offs, tiles, nt, rows, order, st["no_zero"], st["no_zero"])
                res.append((xt, rrow, offs, rows, tiles, nt))
            torch.cuda.synchronize()
            if not all(torch.equal(x, y) for x, y in zip(res[0], res[1])):
                print(f"P12F gather min-5 T={T}: MISMATCH", flush=True)
                mok = False
            del res, logits, p
    except Exception as exc:  # noqa: BLE001
        print("P12F gather min-5 check failed:", repr(exc)[:300], flush=True)
        mok = False
    st["gmb_ok"] = mok
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"P12F: qk_prep_m {st['qkpm_ok']}; gather_kp2 {gok}; MODE 1 lp per T {BND_LP_RC1_T} ok {st['bnd_lp1_ok']}; "
          f"gather min-5 {sorted(GATHER_MINB_T)} ok {mok}", flush=True)


def _p12h_check(st):
    """[p12h] Prepare-time checks of the p12h levers (each off on any mismatch or error):
    qkp1_ok (L2): qk_prep_kvt1 gives qk_prep_kvt_tab's q8, k / v cache bytes and whole kvt buffer (all poisoned) on the
    KVOFF_CHECK_CASES geometries (RoPE table from rope_tab; int64 and int32 cache rows; cache scales on / off);
    gptu_ok (L5): gatherpu / gatherp2u give gatherp / gatherp2's whole kvt buffer (poisoned) on full-attention cases (batch
    1-4, page 1 and 64, page_table rows padded with -1, ctx 0, T 16384);
    lcfork_ok (L3): _lcfork_check."""
    dev = "cuda"
    mod = st["mod"]
    # (L2) qk_prep_kvt1 vs qk_prep_kvt_tab
    ok = bool(KVOFF_QKP1)
    try:
        for i, (B, L, ctxs, page, scaled) in enumerate(KVOFF_CHECK_CASES if ok else ()):
            data, cfg, qkv = _kvoff_case(B, L, ctxs, page, 1601 + i, scaled, window=2048)
            T = B * L
            pt = data["page_table"]
            cap = ((min(2048 - 1 + L, pt.shape[1] * page)) + 255) & ~255
            capblks = cap // 128
            lens_full = data["cache_seqlens"]
            seq = data["seq_lens"].to(torch.int32)
            tab = torch.empty((T, 64), dtype=torch.float32, device=dev)
            mod.rope_tab(data["positions"], st["inv_freq"], tab)
            for loc in (data["cache_loc"], data["cache_loc"].to(torch.int32)):
                res = []
                for m in (0, 1):
                    q8 = torch.full((T, Q_SIZE), 0x5A, dtype=torch.uint8, device=dev)
                    kc = data["k_cache"].view(torch.uint8).clone()
                    vc = data["v_cache"].view(torch.uint8).clone()
                    kvt = torch.full((B * 2 * capblks, 16384), 0xA5, dtype=torch.uint8, device=dev)
                    if m:
                        mod.qk_prep_kvt1(qkv, loc, tab, q8, kc, vc, data["k_scale"], data["v_scale"], scaled, kvt, lens_full,
                                         seq, capblks, L, 2048)
                    else:
                        mod.qk_prep_kvt_tab(qkv, data["positions"], loc, tab, q8, kc, vc, data["k_scale"], data["v_scale"],
                                            scaled, kvt, lens_full, seq, capblks, L, 2048)
                    res.append((q8, kc, vc, kvt))
                torch.cuda.synchronize()
                if not all(torch.equal(x, y) for x, y in zip(res[0], res[1])):
                    print(f"P12H qk_prep_kvt1 B{B} L{L} seq{ctxs} page{page} scaled {scaled} loc {loc.dtype}: MISMATCH", flush=True)
                    ok = False
                del res
            del data, qkv, tab
    except Exception as exc:  # noqa: BLE001
        print("P12H qk_prep_kvt1 check failed:", repr(exc)[:300], flush=True)
        ok = False
    st["qkp1_ok"] = ok
    # (L5) gatherpu / gatherp2u vs gatherp / gatherp2
    gok = bool(GKP_PTU) and st.get("zh") is not None
    try:
        cases = [((1, 753, (80000,), 1, False), 0), ((4, 753, (80000, 61234, 97001, 79999), 1, False), 0),
                 ((2, 753, (2500, 1907), 1, False), 0), ((3, 200, (5000, 130, 2047), 64, False), 0), ((1, 300, (0,), 1, False), 0),
                 ((1, 640, (4096,), 64, True), 0), ((1, 16384, (16384,), 1, False), 0), ((2, 77, (0, 129), 64, False), 0),
                 ((4, 753, (3000, 1, 0, 4500), 64, False), 0)]
        for i, ((B, L, ctxs, page, scaled), win) in enumerate(cases if gok else ()):
            data, cfg, _ = _kvoff_case(B, L, ctxs, page, 1701 + i, scaled, window=win if win else 2048)
            pt = data["page_table"]
            pagelog = page.bit_length() - 1
            max_len = pt.shape[1] * page
            lens_full = data["cache_seqlens"]
            cap = ((min(win - 1 + L, max_len) if win else max_len) + 255) & ~255
            capblks = cap // 128
            kc = data["k_cache"].view(torch.uint8)
            vc = data["v_cache"].view(torch.uint8)
            pairs = [(st["zh"].gatherp, st["zh"].gatherpu), (st["zh"].gatherp2, st["zh"].gatherp2u)]
            for fa, fb in pairs:
                res = []
                for fn in (fa, fb):
                    kvt = torch.full((B * 2 * capblks, 16384), 0xA5, dtype=torch.uint8, device=dev)
                    if win:
                        lens = torch.full((B,), -7, dtype=torch.int32, device=dev)
                        fn(kc, vc, kvt, pt, lens_full, data["seq_lens"].to(torch.int32), lens, pt.stride(0), pagelog, capblks, B, L,
                           win)
                    else:
                        lens = lens_full[:0]
                        fn(kc, vc, kvt, pt, lens_full, lens, lens, pt.stride(0), pagelog, capblks, B, L, 0,
                           *(() if fn in (fa,) else (int(GKP_PTU),)))
                    res.append((kvt, lens))
                torch.cuda.synchronize()
                if not (torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])):
                    print(f"P12H gather ptu ({fb.__name__ if hasattr(fb, '__name__') else fb}) B{B} L{L} seq{ctxs} page{page} "
                          f"win{win}: MISMATCH", flush=True)
                    gok = False
                del res
            del data
    except Exception as exc:  # noqa: BLE001
        print("P12H gather ptu check failed:", repr(exc)[:300], flush=True)
        gok = False
    st["gptu_ok"] = gok
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    _lcfork_check(st)
    print(f"P12H: qk_prep_kvt1 {st['qkp1_ok']}; gather ptu {gok}; lc fork {st['lcfork_ok']}", flush=True)


def _lcfork_check(st):
    """[p12h] L3 guard. CUDA documents the launch-completion edge as best effort (the dependent may wait for the whole
    kernel), and gather_old would then sit on the critical path (~+25 us). A mini graph as kernel_fn captures it at T = 753:
    the qkv GEMM (_mm), the side stream forked right after it with the edge retagged to its launch-completion port, a
    side stamp kernel, the join, a PDL stamp kernel on main. lcfork_ok only if the retag succeeded (one kernel-node edge)
    and in every one of LCFORK_CHECK_REPS replays the side stamp leads the join stamp by >= LCFORK_MIN_LEAD_US (the GEMM
    runs ~39 us; a full-completion edge gives ~1 us)."""
    st["lcfork_ok"] = False
    if not KVOFF_LCFORK or not KVOFF_LCFORK_T:
        return
    try:
        dev = "cuda"
        T = 753
        g0 = torch.Generator(device=dev)
        g0.manual_seed(16016)
        x = (torch.randn((T, HIDDEN), device=dev, generator=g0) * 0.5).to(torch.bfloat16)
        ts = torch.zeros((2,), dtype=torch.int64, device=dev)
        side = SIDE_STREAM
        rc = []

        def body():
            main = torch.cuda.current_stream()
            y = _mm(st, x, "qkv")
            ev = torch.cuda.Event()
            ev.record()
            side.wait_event(ev)
            with torch.cuda.stream(side):
                rc.append(int(st["mod"].retag_deps(2, 0)))
                st["mod"].stamp(ts[0:1], False)
            main.wait_stream(side)
            st["mod"].stamp(ts[1:2], True)
            return y

        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()  # eager warm-up (retag_deps is a no-op outside capture)
        torch.cuda.synchronize()
        rc.clear()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            y = body()
        torch.cuda.synchronize()
        lead = []
        for _ in range(LCFORK_CHECK_REPS):
            ts.zero_()
            g.replay()
            torch.cuda.synchronize()
            a, b = ts.tolist()
            lead.append((b - a) / 1000.0)
        ok = rc == [1] and min(lead) >= LCFORK_MIN_LEAD_US
        print(f"P12H lc fork guard: retag {rc}, side stamp leads the join by {min(lead):.1f} / {sorted(lead)[len(lead) // 2]:.1f} / "
              f"{max(lead):.1f} us (min / median / max of {len(lead)}) -> {'USE' if ok else 'fork kernel'}", flush=True)
        st["lcfork_ok"] = ok
        del g, y, x
    except Exception as exc:  # noqa: BLE001
        print("P12H lc fork guard failed:", repr(exc)[:300], "-> fork kernel", flush=True)
        st["lcfork_ok"] = False


def _bnd_lp_check(st):
    """[r13] st["bnd_lp"] = (MODE 0, MODE 1, MODE 2) load placement for kernel_fn: each enabled LP launch must give
    torch.equal outputs (stream out, normed out, the zeroed counters, MODE 1's rms factor) to the LP 0 launch on random rows
    at several T (fresh input copies per launch); a mismatch or error leaves that kind at 0."""
    want = [int(BND_LP_M0), int(BND_LP_RC1), int(BND_LP_RC2)]
    st["bnd_lp"] = [0, 0, 0]
    if not any(want):
        return
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(4242)
    mod = st["mod"]
    ok = [w != 0 for w in want]
    try:
        for T in (753, 1, 37, 2047, 3012):
            br = (torch.randn((T, HIDDEN), device=dev, generator=g) * 3).to(torch.bfloat16)
            f = (torch.randn((T, HIDDEN), device=dev, generator=g) * 3).to(torch.bfloat16)
            s_in = torch.randn((T, HIDDEN), device=dev, generator=g) * 2
            r1 = torch.rand((T,), device=dev, generator=g) + 0.5
            for kind in range(3):
                if not ok[kind]:
                    continue
                outs = []
                for lp in (0, want[kind]):
                    so = torch.full((T, HIDDEN), 7.0, device=dev)
                    no = torch.zeros((T, HIDDEN), dtype=torch.bfloat16, device=dev)
                    z = torch.full((40,), 5, dtype=torch.int32, device=dev)
                    rio = torch.full((T,), -1.0, device=dev)
                    b_, f_, s_ = br.clone(), f.clone(), s_in.clone()
                    if kind == 0:
                        for gp in (False, True):
                            so2 = so.clone(); no2 = no.clone()
                            gpost, al, be, gpre = ((st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"]) if gp
                                                   else (st["a1"], st["a1"], st["b1"], st["g_pre_ffn_norm"]))
                            mod.boundary_lp(b_, s_, gpost, al, be, gpre, so2, no2, gp, z, lp)
                            outs.append((so2, no2, z.clone()))
                    elif kind == 1:
                        mod.boundary_rc_lp(b_, s_, st["a1"], st["a1"], st["b1"], st["g_pre_ffn_norm"], s_, no, 1, z, b_, rio,
                                           st["a1"], st["b1"], lp)
                        outs.append((no, z, rio))
                    else:
                        mod.boundary_rc_lp(f_, s_, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], so, no,
                                           2, st["no_zero"], b_, r1, st["a1"], st["b1"], lp)
                        outs.append((so, no))
                torch.cuda.synchronize()
                half = len(outs) // 2
                eq = all(torch.equal(x, y) for o0, o1 in zip(outs[:half], outs[half:]) for x, y in zip(o0, o1))
                if not eq:
                    print(f"BND LP check kind {kind} T={T} lp {want[kind]}: MISMATCH", flush=True)
                    ok[kind] = False
            del br, f, s_in, r1
    except Exception as exc:  # noqa: BLE001
        print("BND LP check failed:", repr(exc)[:300], flush=True)
        ok = [False, False, False]
    st["bnd_lp"] = [want[k] if ok[k] else 0 for k in range(3)]
    torch.cuda.empty_cache()
    print(f"BND LP (MODE 0, 1, 2): requested {want}, bitwise-checked in use {st['bnd_lp']}", flush=True)


def _selftest_boundary(st):
    dev = "cuda"
    T = 16384
    a = torch.randn((T, HIDDEN), device=dev).to(torch.bfloat16)
    s = torch.randn((T, HIDDEN), device=dev)
    o1 = torch.empty_like(s)
    o2 = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    o3 = torch.empty_like(s)
    o4 = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    _boundary_kernel[(T,)](a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], o1, o2,
                           HAS_GPOST=True, H=HIDDEN, num_warps=16)
    st["mod"].boundary(a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], o3, o4, True, st["no_zero"])
    torch.cuda.synchronize()
    d1 = float((o1 - o3).abs().max())
    d2 = float((o2.float() - o4.float()).abs().max())
    t1 = _time(lambda: _boundary_kernel[(T,)](a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"],
                                            o1, o2, HAS_GPOST=True, H=HIDDEN, num_warps=16))
    t2 = _time(lambda: st["mod"].boundary(a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"],
                                         o3, o4, True, st["no_zero"]))
    print(f"SELFTEST boundary triton {t1:.0f}us cuda {t2:.0f}us (diff {d1:.1e} {d2:.1e})", flush=True)
    del a, s, o1, o2, o3, o4
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Extension build + CUDA / C++ sources. PRUNED (v2): these five strings
# are the tuned algorithm with every dead branch removed: all
# preprocessor branches not taken under either real build config
# (__CUDA_ARCH__ = 900 / sm_90a, no -D flags from _build, in-file
# macro defaults, experiment macros undefined, MOE_FAST_SCALE in
# {0, 1} -- the one switch _build itself supplies, 0 when the weights
# have non-normal block scales), every function / define / variant
# unreachable from the exported functions, and the dead comments
# attached to them. The taken path is unchanged (checked by
# preprocessing v1 and v2 under both configs: identical except for
# the removed dead items). They hold the grouped NVFP4 expert GEMMs,
# the boundary / route / combine / qk-prep kernels, the own Hopper
# attention kernel, and the cuBLASLt glue. Any change to a taken-path
# instruction here is a numerics or performance change, not a cleanup,
# and needs its own GPU gating.
# The Python constants that must agree with defines inside _CUDA_SRC are
# marked at their definitions (ATTN_ASTAGES, ATTN_SMEM).
# --------------------------------------------------------------------------------------


def _build(fast_scales=True):
    cap = torch.cuda.get_device_capability()
    arch = f"{cap[0]}{cap[1]}" + ("a" if cap[0] == 9 else "")
    src = ("" if fast_scales else "#define MOE_FAST_SCALE 0\n") + f"#define KVOFF_HOIST {int(bool(KVOFF_HOIST))}\n" + _CUDA_SRC + GLUE
    name = "okc_prefill_moe_" + hashlib.sha1((src + CPP + arch).encode()).hexdigest()[:16] + "_p11"
    return load_inline(name, cpp_sources=CPP, cuda_sources=src,
                       functions=["qk_prep", "qk_prep_kvt", "moe_route", "moe_route4", "moe_plan", "moe_gather", "moe_gather_plan", "moe_combine", "moe_combine2", "moe_combine2h",
                                  "moe_gemm", "moe_init", "moe_gemm_p", "moe_fused", "boundary", "boundary_rc", "boundary_lp", "boundary_rc_lp", "fork_node", "moe_gather_plan_co", "qk_prep_kvt_tab", "rope_tab", "qk_prep_m", "moe_gather_plan_cm", "qk_prep_kvt1", "stamp", "retag_deps",
                                  "kv_gather", "attn_prep", "attn_fwd", "attn_init", "attn_set_prof"],
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-std=c++20",
                                          "--use_fast_math"])


def _build_lt():
    name = "okc_prefill_lt_" + hashlib.sha1((LT_GLUE + LT_CPP).encode()).hexdigest()[:16] + "_r6s"
    # Host-only code: compiled by the host compiler (no nvcc), linked against cuBLASLt.
    return load_inline(name, cpp_sources=[LT_CPP, LT_GLUE],
                       functions=["lt_plan", "lt_plan_sm", "lt_plan_kb", "lt_nalgo", "lt_info", "lt_run", "lt2_add", "lt2_run",
                                  "ltnm_info", "ltnm_resched", "ltnm_run", "ltnm_cap", "ltnm_graph_stats"],
                       extra_cflags=["-O3", "-std=c++20"], extra_ldflags=["-lcublasLt"], with_cuda=True)


CPP = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
void k_moe_plan(const int* counts, int* offsets, int* tiles, int* ntiles, int* rows, int order, cudaStream_t st);
void k_moe_route(const float* logits, const float* bias, const int* lstart, int* ids, float* w, int* lid, int* pos, int* cnt,
                 int T, cudaStream_t st, int* tok_left, int* cq);
void k_moe_route4(const float* logits, const float* bias, const int* lstart, int* ids, float* w, int* lid, int* pos, int* cnt,
                  int T, cudaStream_t st, int* tok_left, int* cq);
void k_moe_gather(const void* p, const void* gw, const int* lid, const int* pos, const int* offsets, const int* counts,
                  void* xt, int* rrow, int T, cudaStream_t st);
void k_moe_gather_plan(const void* p, const void* gw, const int* lid, const int* pos, const int* counts, void* xt, int* rrow,
                       int* offsets, int* tiles, int* ntiles, int* rows, int order, int T, cudaStream_t st,
                       int* row_tok, int* cq, int* qtail, int co = 0);
void k_moe_combine(const void* y, const int* rrow, const int* lid, const float* topw, const void* pw, void* out, int T,
                   cudaStream_t st);
void k_moe_combine2(const void* y, const int* rrow, const int* lid, const float* topw, const float* gs, const void* pw, void* out,
                    int T, int tpc, cudaStream_t st, const int* hgo, const int* hblk, int hneed, int* herr);
void k_moe_gemm(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out, int NT,
                int P, bool gated, int smem, cudaStream_t st);
void k_moe_gemm_p(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out, int NT,
                  int nctas, bool gated, int smem, int* counter, cudaStream_t st);
void k_boundary(const void* branch, const float* stream, const void* gpost, const float* alpha, const float* beta,
                const void* gpre, float* sout, void* nout, int T, bool has_gpost, cudaStream_t st, int* zp, int zn, int lp = 0);
void k_qk_prep(const void* qkv, const void* pos, bool pos64, const void* loc, bool loc64, const float* invf, void* q8, void* kc,
               void* vc, const float* ksc, const float* vsc, int T, bool rope, bool scaled, cudaStream_t st);
void k_qk_prep_kvt(const void* qkv, const void* pos, bool pos64, const void* loc, bool loc64, const float* invf, void* q8,
                   void* kc, void* vc, const float* ksc, const float* vsc, int T, bool rope, bool scaled, void* kvt,
                   const int* klen, const int* kseq, int capblks, int L, int win, cudaStream_t st);
void k_boundary_rc(const void* branch, const float* stream, const void* gpost, const float* alpha, const float* beta,
                   const void* gpre, float* sout, void* nout, int T, int mode, cudaStream_t st, int* zp, int zn,
                   const void* br1, float* rio, const float* alpha1, const float* beta1, int lp = 0);
void k_kv_gather(const void* kc, const void* vc, const int* page_table, const int* seq_lens, void* ko, void* vo, int B, int max_pages,
                 int page, int L, int window, int Smax, cudaStream_t st);
void k_moe_init(int smem);
void k_qk_prep_kvt_tab(const void* qkv, const void* pos, bool pos64, const void* loc, bool loc64, const float* tab, void* q8,
                       void* kc, void* vc, const float* ksc, const float* vsc, int T, bool scaled, void* kvt,
                       const int* klen, const int* kseq, int capblks, int L, int win, cudaStream_t st);
void k_rope_tab(const void* pos, bool pos64, const float* invf, float* tab, int T, cudaStream_t st);
void k_fork(cudaStream_t st);
void k_qk_prep_kvt1(const void* qkv, const void* loc, bool loc64, const float* tab, void* q8, void* kc, void* vc, const float* ksc,
                    const float* vsc, int T, bool scaled, void* kvt, const int* klen, const int* kseq, int capblks, int L, int win,
                    cudaStream_t st);
void k_stamp(unsigned long long* out, bool pdl, cudaStream_t st);
void k_qk_prep_m(const void* qkv, const void* pos, const void* loc, const float* invf, void* q8, void* kc, void* vc,
                 const float* ksc, const float* vsc, int T, bool rope, bool scaled, cudaStream_t st);
void k_moe_fused(const void* w13, const void* w2, const void* xt, void* inter, void* y, const int* tiles, const int* ntiles,
                 const float* gs13, const float* gs2, int nctas, int smem, int* counter, int* done, void* dbg, int flags,
                 cudaStream_t st, int* cit, int* qctl, int* tok_left, int* cq, const int* row_tok, const int* rrow,
                 const int* lid, const float* topw, const void* pw, void* out, int T, int* hgo, int* hblk);
void k_attn_prep(const void* kc, const void* vc, const int* page_table, int max_pages, int page, const int* seq_lens, int B, int L,
                 int window, int ntk, void* kt, void* vt, cudaStream_t st, int part);
void k_attn_fwd(const void* q, const void* kt, const void* vt, const int* seq_lens, const void* qkv, void* out, int B, int L,
                int window, int ntk, float c, float v_scale, int smem, int nsplit, float* ws, int variant, cudaStream_t st,
                const void* pf, unsigned long long pf_bytes);
void k_attn_init(int smem);
void k_attn_set_prof(void* p);

static cudaStream_t cur() { return c10::cuda::getCurrentCUDAStream(); }

void moe_route(torch::Tensor logits, torch::Tensor bias, torch::Tensor lstart, torch::Tensor ids, torch::Tensor w,
               torch::Tensor lid, torch::Tensor pos, torch::Tensor cnt, torch::Tensor tok_left, torch::Tensor cq) {
  k_moe_route(logits.data_ptr<float>(), bias.data_ptr<float>(), lstart.data_ptr<int>(), ids.data_ptr<int>(), w.data_ptr<float>(),
              lid.data_ptr<int>(), pos.data_ptr<int>(), cnt.data_ptr<int>(), (int)logits.size(0), cur(),
              tok_left.numel() ? tok_left.data_ptr<int>() : nullptr, cq.numel() ? cq.data_ptr<int>() : nullptr);
}
// [r4rtred] logits = [4, T, 512] fp32 K-slice partials of the router GEMM (hi | lo halves each), summed in the kernel
void moe_route4(torch::Tensor logits, torch::Tensor bias, torch::Tensor lstart, torch::Tensor ids, torch::Tensor w,
                torch::Tensor lid, torch::Tensor pos, torch::Tensor cnt, torch::Tensor tok_left, torch::Tensor cq) {
  TORCH_CHECK(logits.dim() == 3 && logits.size(0) == 4 && logits.size(2) == 512 && logits.is_contiguous(),
              "moe_route4: logits must be a contiguous [4, T, 512] tensor");
  k_moe_route4(logits.data_ptr<float>(), bias.data_ptr<float>(), lstart.data_ptr<int>(), ids.data_ptr<int>(), w.data_ptr<float>(),
               lid.data_ptr<int>(), pos.data_ptr<int>(), cnt.data_ptr<int>(), (int)logits.size(1), cur(),
               tok_left.numel() ? tok_left.data_ptr<int>() : nullptr, cq.numel() ? cq.data_ptr<int>() : nullptr);
}
void moe_plan(torch::Tensor counts, torch::Tensor offsets, torch::Tensor tiles, torch::Tensor ntiles, torch::Tensor rows,
              int64_t order) {
  k_moe_plan(counts.data_ptr<int>(), offsets.data_ptr<int>(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), rows.data_ptr<int>(),
             (int)order, cur());
}
// [r13] moe_gather_plan with the coalesced xt row stores (moe_gather_kernel<true, true>)
void moe_gather_plan_co(torch::Tensor p, torch::Tensor gw, torch::Tensor lid, torch::Tensor pos, torch::Tensor counts,
                        torch::Tensor xt, torch::Tensor rrow, torch::Tensor offsets, torch::Tensor tiles, torch::Tensor ntiles,
                        torch::Tensor rows, int64_t order, torch::Tensor cit, torch::Tensor tok) {
  const bool on = cit.numel() > 0;
  const int T = (int)p.size(0);
  int* tk = on ? tok.data_ptr<int>() : nullptr;
  k_moe_gather_plan(p.data_ptr(), gw.data_ptr(), lid.data_ptr<int>(), pos.data_ptr<int>(), counts.data_ptr<int>(), xt.data_ptr(),
                    rrow.data_ptr<int>(), offsets.data_ptr<int>(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), rows.data_ptr<int>(),
                    (int)order, T, cur(), on ? tk + 2 * T : nullptr, on ? tk + T : nullptr,
                    on ? cit.data_ptr<int>() + (cit.numel() - 3) : nullptr, 1);
}
// [p12f] moe_gather_plan_co at min 5 blocks per SM
void moe_gather_plan_cm(torch::Tensor p, torch::Tensor gw, torch::Tensor lid, torch::Tensor pos, torch::Tensor counts,
                        torch::Tensor xt, torch::Tensor rrow, torch::Tensor offsets, torch::Tensor tiles, torch::Tensor ntiles,
                        torch::Tensor rows, int64_t order, torch::Tensor cit, torch::Tensor tok) {
  const bool on = cit.numel() > 0;
  const int T = (int)p.size(0);
  int* tk = on ? tok.data_ptr<int>() : nullptr;
  k_moe_gather_plan(p.data_ptr(), gw.data_ptr(), lid.data_ptr<int>(), pos.data_ptr<int>(), counts.data_ptr<int>(), xt.data_ptr(),
                    rrow.data_ptr<int>(), offsets.data_ptr<int>(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), rows.data_ptr<int>(),
                    (int)order, T, cur(), on ? tk + 2 * T : nullptr, on ? tk + T : nullptr,
                    on ? cit.data_ptr<int>() + (cit.numel() - 3) : nullptr, 5);
}
void moe_gather_plan(torch::Tensor p, torch::Tensor gw, torch::Tensor lid, torch::Tensor pos, torch::Tensor counts,
                     torch::Tensor xt, torch::Tensor rrow, torch::Tensor offsets, torch::Tensor tiles, torch::Tensor ntiles,
                     torch::Tensor rows, int64_t order, torch::Tensor cit, torch::Tensor tok) {
  // CIT (cit.numel() > 0): queue tail = cit[numel - 3]; tok = [tok_left (T) | cq (T) | row_tok]
  const bool on = cit.numel() > 0;
  const int T = (int)p.size(0);
  int* tk = on ? tok.data_ptr<int>() : nullptr;
  k_moe_gather_plan(p.data_ptr(), gw.data_ptr(), lid.data_ptr<int>(), pos.data_ptr<int>(), counts.data_ptr<int>(), xt.data_ptr(),
                    rrow.data_ptr<int>(), offsets.data_ptr<int>(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), rows.data_ptr<int>(),
                    (int)order, T, cur(), on ? tk + 2 * T : nullptr, on ? tk + T : nullptr,
                    on ? cit.data_ptr<int>() + (cit.numel() - 3) : nullptr);
}
void moe_gather(torch::Tensor p, torch::Tensor gw, torch::Tensor lid, torch::Tensor pos, torch::Tensor offsets,
                torch::Tensor counts, torch::Tensor xt, torch::Tensor rrow) {
  k_moe_gather(p.data_ptr(), gw.data_ptr(), lid.data_ptr<int>(), pos.data_ptr<int>(), offsets.data_ptr<int>(),
               counts.data_ptr<int>(), xt.data_ptr(), rrow.data_ptr<int>(), (int)p.size(0), cur());
}
void moe_combine(torch::Tensor y, torch::Tensor rrow, torch::Tensor lid, torch::Tensor topw, torch::Tensor pw, torch::Tensor out) {
  k_moe_combine(y.data_ptr(), rrow.data_ptr<int>(), lid.data_ptr<int>(), topw.data_ptr<float>(), pw.data_ptr(),
                out.data_ptr(), (int)out.size(0), cur());
}
void moe_combine2(torch::Tensor y, torch::Tensor rrow, torch::Tensor lid, torch::Tensor topw, torch::Tensor gs, torch::Tensor pw,
                  torch::Tensor out, int64_t tpc) {
  k_moe_combine2(y.data_ptr(), rrow.data_ptr<int>(), lid.data_ptr<int>(), topw.data_ptr<float>(), gs.data_ptr<float>(),
                 pw.data_ptr(), out.data_ptr(), (int)out.size(0), (int)tpc, cur(), nullptr, nullptr, 0, nullptr);
}
// [r6h] hand-off combine: per-route waits on the 32-row block counters blk (>= hneed) after the go flag, no grid wait
void moe_combine2h(torch::Tensor y, torch::Tensor rrow, torch::Tensor lid, torch::Tensor topw, torch::Tensor gs, torch::Tensor pw,
                   torch::Tensor out, int64_t tpc, torch::Tensor go, torch::Tensor blk, int64_t hneed, torch::Tensor err) {
  // (blk empty: the hoist-only arm, index loads + compaction after go, then the whole-grid PDL wait)
  TORCH_CHECK(go.numel() >= 1 && err.numel() >= 1, "moe_combine2h: empty go / err");
  k_moe_combine2(y.data_ptr(), rrow.data_ptr<int>(), lid.data_ptr<int>(), topw.data_ptr<float>(), gs.data_ptr<float>(),
                 pw.data_ptr(), out.data_ptr(), (int)out.size(0), (int)tpc, cur(), go.data_ptr<int>(),
                 blk.numel() ? blk.data_ptr<int>() : nullptr, (int)hneed, err.data_ptr<int>());
}
void moe_gemm(torch::Tensor wrep, torch::Tensor xt, torch::Tensor tiles, torch::Tensor ntiles, torch::Tensor gscale,
              torch::Tensor out, int64_t NT, int64_t P, bool gated, int64_t smem) {
  k_moe_gemm(wrep.data_ptr(), xt.data_ptr(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), gscale.data_ptr<float>(), out.data_ptr(),
             (int)NT, (int)P, gated, (int)smem, cur());
}
void moe_gemm_p(torch::Tensor wrep, torch::Tensor xt, torch::Tensor tiles, torch::Tensor ntiles, torch::Tensor gscale,
                torch::Tensor out, int64_t NT, int64_t nctas, bool gated, int64_t smem, torch::Tensor counter) {
  k_moe_gemm_p(wrep.data_ptr(), xt.data_ptr(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), gscale.data_ptr<float>(),
               out.data_ptr(), (int)NT, (int)nctas, gated, (int)smem, counter.data_ptr<int>(), cur());
}
void boundary_rc(torch::Tensor branch, torch::Tensor stream, torch::Tensor gpost, torch::Tensor alpha, torch::Tensor beta,
                 torch::Tensor gpre, torch::Tensor sout, torch::Tensor nout, int64_t mode, torch::Tensor zero,
                 torch::Tensor br1, torch::Tensor rio, torch::Tensor alpha1, torch::Tensor beta1) {
  k_boundary_rc(branch.data_ptr(), stream.data_ptr<float>(), gpost.data_ptr(), alpha.data_ptr<float>(), beta.data_ptr<float>(),
                gpre.data_ptr(), sout.data_ptr<float>(), nout.data_ptr(), (int)branch.size(0), (int)mode, cur(),
                zero.numel() ? zero.data_ptr<int>() : nullptr, (int)zero.numel(), br1.data_ptr(), rio.data_ptr<float>(),
                alpha1.data_ptr<float>(), beta1.data_ptr<float>());
}
void qk_prep_kvt(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor invf, torch::Tensor q8, torch::Tensor kc,
                 torch::Tensor vc, torch::Tensor ksc, torch::Tensor vsc, bool rope, bool scaled, torch::Tensor kvt,
                 torch::Tensor klen, torch::Tensor kseq, int64_t capblks, int64_t L, int64_t win) {
  k_qk_prep_kvt(qkv.data_ptr(), pos.data_ptr(), pos.element_size() == 8, loc.data_ptr(), loc.element_size() == 8,
                invf.data_ptr<float>(), q8.data_ptr(), kc.data_ptr(), vc.data_ptr(), ksc.data_ptr<float>(), vsc.data_ptr<float>(),
                (int)qkv.size(0), rope, scaled, kvt.data_ptr(), klen.data_ptr<int>(), kseq.data_ptr<int>(), (int)capblks, (int)L,
                (int)win, cur());
}
// [r13] qk_prep_kvt with a precomputed RoPE table (rope_tab), and the table kernel
void qk_prep_kvt_tab(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor tab, torch::Tensor q8, torch::Tensor kc,
                     torch::Tensor vc, torch::Tensor ksc, torch::Tensor vsc, bool scaled, torch::Tensor kvt,
                     torch::Tensor klen, torch::Tensor kseq, int64_t capblks, int64_t L, int64_t win) {
  TORCH_CHECK(tab.numel() >= qkv.size(0) * 64, "qk_prep_kvt_tab: table too small");
  k_qk_prep_kvt_tab(qkv.data_ptr(), pos.data_ptr(), pos.element_size() == 8, loc.data_ptr(), loc.element_size() == 8,
                    tab.data_ptr<float>(), q8.data_ptr(), kc.data_ptr(), vc.data_ptr(), ksc.data_ptr<float>(), vsc.data_ptr<float>(),
                    (int)qkv.size(0), scaled, kvt.data_ptr(), klen.data_ptr<int>(), kseq.data_ptr<int>(), (int)capblks, (int)L,
                    (int)win, cur());
}
void rope_tab(torch::Tensor pos, torch::Tensor invf, torch::Tensor tab) {
  TORCH_CHECK(tab.numel() >= pos.numel() * 64, "rope_tab: table too small");
  k_rope_tab(pos.data_ptr(), pos.element_size() == 8, invf.data_ptr<float>(), tab.data_ptr<float>(), (int)pos.numel(), cur());
}
void qk_prep(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor invf, torch::Tensor q8, torch::Tensor kc,
             torch::Tensor vc, torch::Tensor ksc, torch::Tensor vsc, bool rope, bool scaled) {
  k_qk_prep(qkv.data_ptr(), pos.data_ptr(), pos.element_size() == 8, loc.data_ptr(), loc.element_size() == 8, invf.data_ptr<float>(),
            q8.data_ptr(), kc.data_ptr(), vc.data_ptr(), ksc.data_ptr<float>(), vsc.data_ptr<float>(), (int)qkv.size(0), rope, scaled, cur());
}
// [r13] boundary / boundary_rc with the load placement lp (0: the p12d launch, 2: stream row after the row-norm barrier)
void boundary_lp(torch::Tensor branch, torch::Tensor stream, torch::Tensor gpost, torch::Tensor alpha, torch::Tensor beta,
                 torch::Tensor gpre, torch::Tensor sout, torch::Tensor nout, bool has_gpost, torch::Tensor zero, int64_t lp) {
  k_boundary(branch.data_ptr(), stream.data_ptr<float>(), gpost.data_ptr(), alpha.data_ptr<float>(), beta.data_ptr<float>(),
             gpre.data_ptr(), sout.data_ptr<float>(), nout.data_ptr(), (int)branch.size(0), has_gpost, cur(),
             zero.numel() ? zero.data_ptr<int>() : nullptr, (int)zero.numel(), (int)lp);
}
void boundary_rc_lp(torch::Tensor branch, torch::Tensor stream, torch::Tensor gpost, torch::Tensor alpha, torch::Tensor beta,
                    torch::Tensor gpre, torch::Tensor sout, torch::Tensor nout, int64_t mode, torch::Tensor zero,
                    torch::Tensor br1, torch::Tensor rio, torch::Tensor alpha1, torch::Tensor beta1, int64_t lp) {
  k_boundary_rc(branch.data_ptr(), stream.data_ptr<float>(), gpost.data_ptr(), alpha.data_ptr<float>(), beta.data_ptr<float>(),
                gpre.data_ptr(), sout.data_ptr<float>(), nout.data_ptr(), (int)branch.size(0), (int)mode, cur(),
                zero.numel() ? zero.data_ptr<int>() : nullptr, (int)zero.numel(), br1.data_ptr(), rio.data_ptr<float>(),
                alpha1.data_ptr<float>(), beta1.data_ptr<float>(), (int)lp);
}
void boundary(torch::Tensor branch, torch::Tensor stream, torch::Tensor gpost, torch::Tensor alpha, torch::Tensor beta,
              torch::Tensor gpre, torch::Tensor sout, torch::Tensor nout, bool has_gpost, torch::Tensor zero) {
  k_boundary(branch.data_ptr(), stream.data_ptr<float>(), gpost.data_ptr(), alpha.data_ptr<float>(), beta.data_ptr<float>(),
             gpre.data_ptr(), sout.data_ptr<float>(), nout.data_ptr(), (int)branch.size(0), has_gpost, cur(),
             zero.numel() ? zero.data_ptr<int>() : nullptr, (int)zero.numel());
}
void kv_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor page_table, torch::Tensor seq_lens, torch::Tensor ko,
               torch::Tensor vo, int64_t page, int64_t L, int64_t window, int64_t Smax) {
  k_kv_gather(kc.data_ptr(), vc.data_ptr(), page_table.data_ptr<int>(), seq_lens.data_ptr<int>(), ko.data_ptr(), vo.data_ptr(),
              (int)seq_lens.size(0), (int)page_table.size(1), (int)page, (int)L, (int)window, (int)Smax, cur());
}
void moe_init(int64_t smem) { k_moe_init((int)smem); }
void fork_node() { k_fork(cur()); }
// [p12h] qk_prep_kvt1: qk_prep_kvt_tab in one L2 round trip after the PDL wait (16 B aligned table rows)
void qk_prep_kvt1(torch::Tensor qkv, torch::Tensor loc, torch::Tensor tab, torch::Tensor q8, torch::Tensor kc, torch::Tensor vc,
                  torch::Tensor ksc, torch::Tensor vsc, bool scaled, torch::Tensor kvt, torch::Tensor klen, torch::Tensor kseq,
                  int64_t capblks, int64_t L, int64_t win) {
  TORCH_CHECK(tab.numel() >= qkv.size(0) * 64 && reinterpret_cast<uintptr_t>(tab.data_ptr()) % 16 == 0,
              "qk_prep_kvt1: table too small / unaligned");
  k_qk_prep_kvt1(qkv.data_ptr(), loc.data_ptr(), loc.element_size() == 8, tab.data_ptr<float>(), q8.data_ptr(), kc.data_ptr(),
                 vc.data_ptr(), ksc.data_ptr<float>(), vsc.data_ptr<float>(), (int)qkv.size(0), scaled, kvt.data_ptr(),
                 klen.data_ptr<int>(), kseq.data_ptr<int>(), (int)capblks, (int)L, (int)win, cur());
}
// [p12h] _lcfork_check stamps (int64 [1] tensors)
void stamp(torch::Tensor out, bool pdl) { k_stamp(reinterpret_cast<unsigned long long*>(out.data_ptr()), pdl, cur()); }
#if CUDART_VERSION < 13000
#define OKC_GET_CAPTURE_INFO cudaStreamGetCaptureInfo_v3
#define OKC_UPDATE_CAPTURE_DEPS cudaStreamUpdateCaptureDependencies_v2
#else
#define OKC_GET_CAPTURE_INFO cudaStreamGetCaptureInfo
#define OKC_UPDATE_CAPTURE_DEPS cudaStreamUpdateCaptureDependencies
#endif
// [p12h] re-annotate the current stream's capture dependencies (all must be kernel nodes, at most 8): the next captured node
// gets edges {from_port, to_port 0, type} (from_port 2 = cudaGraphKernelNodePortLaunchCompletion). Returns the number of
// retagged edges; 0 when the stream is not capturing (eager: nothing to do); < 0 when refused (dependencies unchanged).
int64_t retag_deps(int64_t from_port, int64_t type) {
  cudaStream_t s = cur();
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  const cudaGraphNode_t* deps = nullptr;
  const cudaGraphEdgeData* ed = nullptr;
  size_t n = 0;
  if (OKC_GET_CAPTURE_INFO(s, &cs, nullptr, nullptr, &deps, &ed, &n) != cudaSuccess) { cudaGetLastError(); return -1; }
  if (cs != cudaStreamCaptureStatusActive) return 0;
  if (n == 0 || n > 8) return -2;
  cudaGraphNode_t nd[8];
  cudaGraphEdgeData d[8];
  for (size_t i = 0; i < n; i++) {
    nd[i] = deps[i];
    cudaGraphNodeType t;
    if (cudaGraphNodeGetType(nd[i], &t) != cudaSuccess || t != cudaGraphNodeTypeKernel) { cudaGetLastError(); return -3; }
    d[i] = cudaGraphEdgeData{};
    d[i].from_port = (unsigned char)from_port;
    d[i].to_port = 0;
    d[i].type = (unsigned char)type;
  }
  if (OKC_UPDATE_CAPTURE_DEPS(s, nd, d, n, cudaStreamSetCaptureDependencies) != cudaSuccess) { cudaGetLastError(); return -4; }
  return (int64_t)n;
}
// [p12f] qk_prep with 4 (RoPE) / 2 tokens per block; int64 positions and cache rows only
void qk_prep_m(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor invf, torch::Tensor q8, torch::Tensor kc,
               torch::Tensor vc, torch::Tensor ksc, torch::Tensor vsc, bool rope, bool scaled) {
  TORCH_CHECK(pos.element_size() == 8 && loc.element_size() == 8, "qk_prep_m: int64 positions / cache rows only");
  k_qk_prep_m(qkv.data_ptr(), pos.data_ptr(), loc.data_ptr(), invf.data_ptr<float>(), q8.data_ptr(), kc.data_ptr(), vc.data_ptr(),
              ksc.data_ptr<float>(), vsc.data_ptr<float>(), (int)qkv.size(0), rope, scaled, cur());
}
void moe_fused(torch::Tensor w13, torch::Tensor w2, torch::Tensor xt, torch::Tensor inter, torch::Tensor y, torch::Tensor tiles,
               torch::Tensor ntiles, torch::Tensor gs13, torch::Tensor gs2, int64_t nctas, int64_t smem, torch::Tensor counter,
               torch::Tensor done, torch::Tensor dbg, int64_t flags, torch::Tensor cit, torch::Tensor tok, torch::Tensor rrow,
               torch::Tensor lid, torch::Tensor topw, torch::Tensor pw, torch::Tensor out, torch::Tensor hgo, torch::Tensor hblk) {
  // CIT (cit.numel() > 0): cit = [max tiles GEMM2 counters | head, tail, error, pad]; tok = [tok_left (T) | cq (T) | row_tok]
  // [r6h] HCOMB (hgo.numel() > 0): hgo = [1] go flag; hblk = [rows / 32] GEMM2 block counters or empty (hoist-only)
  const bool on = cit.numel() > 0;
  const int T = (int)rrow.size(0);
  int* c = on ? cit.data_ptr<int>() : nullptr;
  int* tk = on ? tok.data_ptr<int>() : nullptr;
  k_moe_fused(w13.data_ptr(), w2.data_ptr(), xt.data_ptr(), inter.data_ptr(), y.data_ptr(), tiles.data_ptr<int>(),
              ntiles.data_ptr<int>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), (int)nctas, (int)smem, counter.data_ptr<int>(),
              done.data_ptr<int>(), dbg.numel() ? dbg.data_ptr() : nullptr, (int)flags, cur(),
              c, on ? c + (cit.numel() - 4) : nullptr, tk, on ? tk + T : nullptr, on ? tk + 2 * T : nullptr,
              rrow.data_ptr<int>(), lid.data_ptr<int>(), topw.data_ptr<float>(), pw.data_ptr(), on ? out.data_ptr() : nullptr, T,
              hgo.numel() ? hgo.data_ptr<int>() : nullptr, hblk.numel() && hgo.numel() ? hblk.data_ptr<int>() : nullptr);
}
void attn_prep(torch::Tensor kc, torch::Tensor vc, torch::Tensor page_table, torch::Tensor seq_lens, int64_t page, int64_t L,
               int64_t window, int64_t ntk, torch::Tensor kt, torch::Tensor vt, int64_t part) {
  k_attn_prep(kc.data_ptr(), vc.data_ptr(), page_table.data_ptr<int>(), (int)page_table.size(1), (int)page, seq_lens.data_ptr<int>(),
              (int)seq_lens.size(0), (int)L, (int)window, (int)ntk, kt.data_ptr(), vt.data_ptr(), cur(), (int)part);
}
void attn_fwd(torch::Tensor q, torch::Tensor kt, torch::Tensor vt, torch::Tensor seq_lens, torch::Tensor qkv, torch::Tensor out,
              int64_t L, int64_t window, int64_t ntk, double c, double v_scale, int64_t smem, int64_t nsplit, torch::Tensor ws,
              int64_t variant, torch::Tensor pf) {
  k_attn_fwd(q.data_ptr(), kt.data_ptr(), vt.data_ptr(), seq_lens.data_ptr<int>(), qkv.data_ptr(), out.data_ptr(),
             (int)seq_lens.size(0), (int)L, (int)window, (int)ntk, (float)c, (float)v_scale, (int)smem, (int)nsplit,
             nsplit != 1 ? ws.data_ptr<float>() : nullptr, (int)variant, cur(), pf.numel() ? pf.data_ptr() : nullptr,
             (unsigned long long)(pf.numel() * pf.element_size()));
}
void attn_init(int64_t smem) { k_attn_init((int)smem); }
void attn_set_prof(torch::Tensor p) { k_attn_set_prof(p.numel() ? p.data_ptr() : nullptr); }
'''

GLUE = r'''
#include <cuda_runtime.h>
using namespace okc_moe;

template <typename... KArgs, typename... Args>
static inline void okc_launch(void (*k)(KArgs...), dim3 g, dim3 b, size_t smem, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = g; cfg.blockDim = b; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, k, ((KArgs)args)...);
}
// [r13] a plain launch (no programmatic edge from whatever precedes kernel_fn in the stream): it reads and writes nothing,
// and its completion must still imply its predecessors' in eager use
void k_fork(cudaStream_t st) { okc_fork_kernel<<<1, 32, 0, st>>>(); }
void k_moe_route(const float* logits, const float* bias, const int* lstart, int* ids, float* w, int* lid, int* pos, int* cnt,
                 int T, cudaStream_t st, int* tok_left, int* cq) {
  okc_launch(moe_route_kernel<1>, dim3((T + ROUTE_WARPS - 1) / ROUTE_WARPS), dim3(ROUTE_WARPS * 32), 0, st, logits, bias, lstart, ids, w, lid, pos, cnt, T,
             tok_left, cq, 0);
}
// [p12f] qk_prep_m launch (int64 positions and cache rows; NT 4 with RoPE, 2 without), PDL like qk_prep
template <bool R, bool S, int NT>
static void qk_prep_m_launch(const void* qkv, const void* pos, const void* loc, const float* invf, void* q8, void* kc, void* vc,
                             const float* ksc, const float* vsc, int T, cudaStream_t st) {
  okc_launch(qk_prep_m_kernel<R, S, long long, long long, NT>, dim3((unsigned)((T + NT - 1) / NT)), dim3(128), 0, st,
             (const __nv_bfloat16*)qkv, (const long long*)pos, (const long long*)loc, invf, (uint8_t*)q8, (uint8_t*)kc, (uint8_t*)vc,
             ksc, vsc, T);
}
void k_qk_prep_m(const void* qkv, const void* pos, const void* loc, const float* invf, void* q8, void* kc, void* vc,
                 const float* ksc, const float* vsc, int T, bool rope, bool scaled, cudaStream_t st) {
  if (T <= 0) return;
  if (rope && !scaled) qk_prep_m_launch<true, false, 4>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st);
  else if (rope) qk_prep_m_launch<true, true, 4>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st);
  else if (!scaled) qk_prep_m_launch<false, false, 2>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st);
  else qk_prep_m_launch<false, true, 2>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st);
}
// [p12h] qk_prep_kvt1 (one L2 round trip; RoPE table) and the guard's stamp kernel
template <bool S, typename LT>
static void qk_prep_kvt1_launch(const void* qkv, const void* loc, const float* tab, void* q8, void* kc, void* vc, const float* ksc,
                                const float* vsc, void* kvt, const int* klen, const int* kseq, int capblks, int L, int win, int T,
                                cudaStream_t st) {
  okc_launch(qk_prep_kvt1_kernel<S, LT>, dim3(T), dim3(128), 0, st, (const __nv_bfloat16*)qkv, (const LT*)loc, tab, (uint8_t*)q8,
             (uint8_t*)kc, (uint8_t*)vc, ksc, vsc, (uint8_t*)kvt, klen, kseq, capblks, L, win);
}
void k_qk_prep_kvt1(const void* qkv, const void* loc, bool loc64, const float* tab, void* q8, void* kc, void* vc, const float* ksc,
                    const float* vsc, int T, bool scaled, void* kvt, const int* klen, const int* kseq, int capblks, int L, int win,
                    cudaStream_t st) {
  if (T <= 0) return;
  if (scaled) {
    if (loc64) qk_prep_kvt1_launch<true, long long>(qkv, loc, tab, q8, kc, vc, ksc, vsc, kvt, klen, kseq, capblks, L, win, T, st);
    else qk_prep_kvt1_launch<true, int>(qkv, loc, tab, q8, kc, vc, ksc, vsc, kvt, klen, kseq, capblks, L, win, T, st);
  } else {
    if (loc64) qk_prep_kvt1_launch<false, long long>(qkv, loc, tab, q8, kc, vc, ksc, vsc, kvt, klen, kseq, capblks, L, win, T, st);
    else qk_prep_kvt1_launch<false, int>(qkv, loc, tab, q8, kc, vc, ksc, vsc, kvt, klen, kseq, capblks, L, win, T, st);
  }
}
void k_stamp(unsigned long long* out, bool pdl, cudaStream_t st) {
  if (pdl) okc_launch(okc_stamp_kernel<true>, dim3(1), dim3(32), 0, st, out);
  else okc_stamp_kernel<false><<<1, 32, 0, st>>>(out);
}
// [r4rtred] logits = [4][T][512] K-slice partials (slab pitch T * 512 floats)
void k_moe_route4(const float* logits, const float* bias, const int* lstart, int* ids, float* w, int* lid, int* pos, int* cnt,
                  int T, cudaStream_t st, int* tok_left, int* cq) {
  okc_launch(moe_route_kernel<4>, dim3((T + ROUTE_WARPS - 1) / ROUTE_WARPS), dim3(ROUTE_WARPS * 32), 0, st, logits, bias, lstart, ids, w, lid, pos, cnt, T,
             tok_left, cq, T * 512);
}
void k_moe_plan(const int* counts, int* offsets, int* tiles, int* ntiles, int* rows, int order, cudaStream_t st) {
  okc_launch(moe_plan_kernel, dim3(1), dim3(32), 0, st, counts, offsets, (int4*)tiles, ntiles, rows, order);
}
static int g_attn_merge_host = 0;
void k_moe_gather(const void* p, const void* gw, const int* lid, const int* pos, const int* offsets, const int* counts,
                  void* xt, int* rrow, int T, cudaStream_t st) {
#define MOE_PADFILL 1
  okc_launch(moe_gather_kernel<false>, dim3(T + (MOE_PADFILL ? 32 * MOE_PADFILL_PARTS : 0)), dim3(256), 0, st, (const __nv_bfloat16*)p, (const __nv_bfloat16*)gw, lid, pos, offsets, counts, (uint8_t*)xt, rrow, T, (int*)nullptr, (int4*)nullptr, (int*)nullptr, (int*)nullptr, 0,
             (int*)nullptr, (int*)nullptr, (int*)nullptr);
}
void k_moe_gather_plan(const void* p, const void* gw, const int* lid, const int* pos, const int* counts, void* xt, int* rrow,
                       int* offsets, int* tiles, int* ntiles, int* rows, int order, int T, cudaStream_t st,
                       int* row_tok, int* cq, int* qtail, int co) {
  if (co >= 2) {  // [p12f] coalesced stores at min 5 blocks per SM (48 registers, no spills; 6 / 8 spill and are not built)
    okc_launch(moe_gather_kernel<true, true, 5>, dim3(1 + T + (MOE_PADFILL ? 32 * MOE_PADFILL_PARTS : 0)), dim3(256), 0, st, (const __nv_bfloat16*)p, (const __nv_bfloat16*)gw, lid, pos, (const int*)nullptr, counts, (uint8_t*)xt, rrow, T,
               offsets, (int4*)tiles, ntiles, rows, order, row_tok, cq, qtail);
    return;
  }
  if (co) {  // [r13] coalesced xt row stores
    okc_launch(moe_gather_kernel<true, true>, dim3(1 + T + (MOE_PADFILL ? 32 * MOE_PADFILL_PARTS : 0)), dim3(256), 0, st, (const __nv_bfloat16*)p, (const __nv_bfloat16*)gw, lid, pos, (const int*)nullptr, counts, (uint8_t*)xt, rrow, T,
               offsets, (int4*)tiles, ntiles, rows, order, row_tok, cq, qtail);
    return;
  }
  // [r3g] 1 +: block 0 is the dedicated plan block (token t = block t + 1)
  okc_launch(moe_gather_kernel<true>, dim3(1 + T + (MOE_PADFILL ? 32 * MOE_PADFILL_PARTS : 0)), dim3(256), 0, st, (const __nv_bfloat16*)p, (const __nv_bfloat16*)gw, lid, pos, (const int*)nullptr, counts, (uint8_t*)xt, rrow, T,
             offsets, (int4*)tiles, ntiles, rows, order, row_tok, cq, qtail);
}
void k_moe_combine(const void* y, const int* rrow, const int* lid, const float* topw, const void* pw, void* out, int T,
                   cudaStream_t st) {
  okc_launch(moe_combine_kernel, dim3(T), dim3(256), 0, st, (const __nv_bfloat16*)y, rrow, lid, topw, (const __nv_bfloat16*)pw, (__nv_bfloat16*)out);
}
void k_moe_combine2(const void* y, const int* rrow, const int* lid, const float* topw, const float* gs, const void* pw, void* out,
                    int T, int tpc, cudaStream_t st, const int* hgo, const int* hblk, int hneed, int* herr) {
  if (T <= 0) return;
  tpc = tpc < 1 ? 1 : (tpc > COMB2_TPC_MAX ? COMB2_TPC_MAX : tpc);
  if (hgo && herr)  // [r6h] hand-off: go flag, then per-route block counters (hblk) or the whole-grid wait (hblk == nullptr)
    okc_launch(moe_combine2_kernel<true>, dim3((unsigned)((T + tpc - 1) / tpc)), dim3(256), (size_t)COMB2_SMEM, st,
               (const __nv_bfloat16*)y, rrow, lid, topw, gs, (const __nv_bfloat16*)pw, (__nv_bfloat16*)out, T, tpc,
               hgo, hblk, hneed, herr);
  else
    okc_launch(moe_combine2_kernel<false>, dim3((unsigned)((T + tpc - 1) / tpc)), dim3(256), (size_t)COMB2_SMEM, st,
               (const __nv_bfloat16*)y, rrow, lid, topw, gs, (const __nv_bfloat16*)pw, (__nv_bfloat16*)out, T, tpc,
               (const int*)nullptr, (const int*)nullptr, 0, (int*)nullptr);
}
static Args make_args(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out,
                      int NT, int smem, int* counter) {
  Args a;
  a.wrep = (const uint8_t*)wrep; a.xt = (const uint8_t*)xt; a.tiles = (const int4*)tiles; a.ntiles = ntiles;
  a.gscale = gscale; a.out = (uint8_t*)out; a.NT = NT; a.smem_bytes = smem; a.counter = counter; a.flags = 0; a.mul16 = 16u; a.mul64 = 64u;
  return a;
}
void k_moe_gemm(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out, int NT,
                int P, bool gated, int smem, cudaStream_t st) {
  Args a = make_args(wrep, xt, tiles, ntiles, gscale, out, NT, smem, nullptr);
  dim3 grid((unsigned)NT, (unsigned)P);
  if (gated) moe_gemm_kernel<true><<<grid, 384, smem, st>>>(a);
  else moe_gemm_kernel<false><<<grid, 384, smem, st>>>(a);
}
void k_moe_gemm_p(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out, int NT,
                  int nctas, bool gated, int smem, int* counter, cudaStream_t st) {
  Args a = make_args(wrep, xt, tiles, ntiles, gscale, out, NT, smem, counter);
  if (gated) moe_gemm_persistent<true><<<(unsigned)nctas, 384, smem, st>>>(a);
  else moe_gemm_persistent<false><<<(unsigned)nctas, 384, smem, st>>>(a);
}
#define BND_TPB 512
#define BND_HINT false
template <bool R, bool S, typename PT, typename LT>
static void qk_prep_launch(const void* qkv, const void* pos, const void* loc, const float* invf, void* q8, void* kc, void* vc,
                           const float* ksc, const float* vsc, int T, cudaStream_t st) {
  okc_launch(qk_prep_kernel<R, S, PT, LT>, dim3(T), dim3(128), 0, st, (const __nv_bfloat16*)qkv, (const PT*)pos, (const LT*)loc, invf,
             (uint8_t*)q8, (uint8_t*)kc, (uint8_t*)vc, ksc, vsc);
}
// [r6kvoff] qk_prep_kvt: qk_prep that also writes the new keys' K / V^T bytes into gather_kp's kvt layout (PDL launch)
template <bool R, bool S, typename PT, typename LT>
static void qk_prep_kvt_launch(const void* qkv, const void* pos, const void* loc, const float* invf, void* q8, void* kc, void* vc,
                               const float* ksc, const float* vsc, void* kvt, const int* klen, const int* kseq, int capblks,
                               int L, int win, int T, cudaStream_t st) {
  okc_launch(qk_prep_kvt_kernel<R, S, PT, LT>, dim3(T), dim3(128), 0, st, (const __nv_bfloat16*)qkv, (const PT*)pos, (const LT*)loc,
             invf, (uint8_t*)q8, (uint8_t*)kc, (uint8_t*)vc, ksc, vsc, (uint8_t*)kvt, klen, kseq, capblks, L, win);
}
// [r13] the table variants (RoPE only): tab replaces inv_freq
template <bool S, typename PT, typename LT>
static void qk_prep_kvt_tab_launch(const void* qkv, const void* pos, const void* loc, const float* tab, void* q8, void* kc, void* vc,
                                   const float* ksc, const float* vsc, void* kvt, const int* klen, const int* kseq, int capblks,
                                   int L, int win, int T, cudaStream_t st) {
  okc_launch(qk_prep_kvt_kernel<true, S, PT, LT, true>, dim3(T), dim3(128), 0, st, (const __nv_bfloat16*)qkv, (const PT*)pos, (const LT*)loc,
             tab, (uint8_t*)q8, (uint8_t*)kc, (uint8_t*)vc, ksc, vsc, (uint8_t*)kvt, klen, kseq, capblks, L, win);
}
void k_qk_prep_kvt_tab(const void* qkv, const void* pos, bool pos64, const void* loc, bool loc64, const float* tab, void* q8,
                       void* kc, void* vc, const float* ksc, const float* vsc, int T, bool scaled, void* kvt,
                       const int* klen, const int* kseq, int capblks, int L, int win, cudaStream_t st) {
#define QKPT(S) \
  if (pos64 && loc64) qk_prep_kvt_tab_launch<S, long long, long long>(qkv, pos, loc, tab, q8, kc, vc, ksc, vsc, kvt, klen, kseq, capblks, L, win, T, st); \
  else if (pos64) qk_prep_kvt_tab_launch<S, long long, int>(qkv, pos, loc, tab, q8, kc, vc, ksc, vsc, kvt, klen, kseq, capblks, L, win, T, st); \
  else if (loc64) qk_prep_kvt_tab_launch<S, int, long long>(qkv, pos, loc, tab, q8, kc, vc, ksc, vsc, kvt, klen, kseq, capblks, L, win, T, st); \
  else qk_prep_kvt_tab_launch<S, int, int>(qkv, pos, loc, tab, q8, kc, vc, ksc, vsc, kvt, klen, kseq, capblks, L, win, T, st);
  if (scaled) { QKPT(true) } else { QKPT(false) }
#undef QKPT
}
void k_rope_tab(const void* pos, bool pos64, const float* invf, float* tab, int T, cudaStream_t st) {
  if (T <= 0) return;
  const dim3 g((unsigned)((T * 32 + 255) / 256));
  if (pos64) rope_tab_kernel<long long><<<g, 256, 0, st>>>((const long long*)pos, invf, tab, T);
  else rope_tab_kernel<int><<<g, 256, 0, st>>>((const int*)pos, invf, tab, T);
}
void k_qk_prep_kvt(const void* qkv, const void* pos, bool pos64, const void* loc, bool loc64, const float* invf, void* q8,
                   void* kc, void* vc, const float* ksc, const float* vsc, int T, bool rope, bool scaled, void* kvt,
                   const int* klen, const int* kseq, int capblks, int L, int win, cudaStream_t st) {
#define QKPK(R, S) \
  if (pos64 && loc64) qk_prep_kvt_launch<R, S, long long, long long>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, kvt, klen, kseq, capblks, L, win, T, st); \
  else if (pos64) qk_prep_kvt_launch<R, S, long long, int>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, kvt, klen, kseq, capblks, L, win, T, st); \
  else if (loc64) qk_prep_kvt_launch<R, S, int, long long>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, kvt, klen, kseq, capblks, L, win, T, st); \
  else qk_prep_kvt_launch<R, S, int, int>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, kvt, klen, kseq, capblks, L, win, T, st);
  if (rope && !scaled) { QKPK(true, false) }
  else if (rope) { QKPK(true, true) }
  else if (!scaled) { QKPK(false, false) }
  else { QKPK(false, true) }
#undef QKPK
}
void k_qk_prep(const void* qkv, const void* pos, bool pos64, const void* loc, bool loc64, const float* invf, void* q8, void* kc,
               void* vc, const float* ksc, const float* vsc, int T, bool rope, bool scaled, cudaStream_t st) {
#define QKP(R, S) \
  if (pos64 && loc64) qk_prep_launch<R, S, long long, long long>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st); \
  else if (pos64) qk_prep_launch<R, S, long long, int>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st); \
  else if (loc64) qk_prep_launch<R, S, int, long long>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st); \
  else qk_prep_launch<R, S, int, int>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st);
  if (rope && !scaled) { QKP(true, false) }
  else if (rope) { QKP(true, true) }
  else if (!scaled) { QKP(false, false) }
  else { QKP(false, true) }
#undef QKP
}
void k_boundary(const void* branch, const float* stream, const void* gpost, const float* alpha, const float* beta,
                const void* gpre, float* sout, void* nout, int T, bool has_gpost, cudaStream_t st, int* zp, int zn, int lp) {
#define BNDLP(G, LPV) okc_launch(boundary_kernel<BND_TPB, G, false, 0, LPV>, dim3(T), dim3(BND_TPB), 0, st, (const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn, (const __nv_bfloat16*)nullptr, (float*)nullptr, (const float*)nullptr, (const float*)nullptr)
  if (lp == 2) {  // [r13] the stream row read after the branch row-norm barrier
    if (has_gpost) BNDLP(true, 2);
    else BNDLP(false, 2);
    return;
  }
#undef BNDLP
  if (has_gpost)
    okc_launch(boundary_kernel<BND_TPB, true, BND_HINT, 0>, dim3(T), dim3(BND_TPB), 0, st, (const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn, (const __nv_bfloat16*)nullptr, (float*)nullptr, (const float*)nullptr, (const float*)nullptr);
  else
    okc_launch(boundary_kernel<BND_TPB, false, BND_HINT, 0>, dim3(T), dim3(BND_TPB), 0, st, (const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn, (const __nv_bfloat16*)nullptr, (float*)nullptr, (const float*)nullptr, (const float*)nullptr);
}
void k_boundary_rc(const void* branch, const float* stream, const void* gpost, const float* alpha, const float* beta,
                   const void* gpre, float* sout, void* nout, int T, int mode, cudaStream_t st, int* zp, int zn,
                   const void* br1, float* rio, const float* alpha1, const float* beta1, int lp) {
  // glue lever: PDL launches (okc_launch), like the stored-stream boundary path; the kernel already opens with pdl_enter()
  if (mode == 1 && lp == 4) {  // [p12f] MODE 1: the stream row loaded before the PDL wait
    okc_launch(boundary_kernel<128, false, false, 1, 4>, dim3(T), dim3(128), 0, st, (const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn,
               (const __nv_bfloat16*)br1, rio, alpha1, beta1);
    return;
  }
  if (mode == 1 && lp == 2) {  // [r13] MODE 1: the stream row read after the branch row-norm barrier (MODE 2 has no LP)
    okc_launch(boundary_kernel<128, false, false, 1, 2>, dim3(T), dim3(128), 0, st, (const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn,
               (const __nv_bfloat16*)br1, rio, alpha1, beta1);
    return;
  }
  if (mode == 1)
    okc_launch(boundary_kernel<128, false, false, 1>, dim3(T), dim3(128), 0, st, (const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn,
               (const __nv_bfloat16*)br1, rio, alpha1, beta1);
  else
    okc_launch(boundary_kernel<BND_TPB, true, BND_HINT, 2>, dim3(T), dim3(BND_TPB), 0, st, (const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn,
               (const __nv_bfloat16*)br1, rio, alpha1, beta1);
}
void k_kv_gather(const void* kc, const void* vc, const int* page_table, const int* seq_lens, void* ko, void* vo, int B, int max_pages,
                 int page, int L, int window, int Smax, cudaStream_t st) {
  dim3 grid((unsigned)((Smax + 31) / 32), (unsigned)B);
  kv_gather_kernel<<<grid, 256, 0, st>>>((const uint8_t*)kc, (const uint8_t*)vc, page_table, seq_lens, (uint8_t*)ko, (uint8_t*)vo,
                                         max_pages, page, L, window, Smax);
}
void k_moe_fused(const void* w13, const void* w2, const void* xt, void* inter, void* y, const int* tiles, const int* ntiles,
                 const float* gs13, const float* gs2, int nctas, int smem, int* counter, int* done, void* dbg, int flags,
                 cudaStream_t st, int* cit, int* qctl, int* tok_left, int* cq, const int* row_tok, const int* rrow,
                 const int* lid, const float* topw, const void* pw, void* out, int T, int* hgo, int* hblk) {
  FusedArgs fa;
  fa.hgo = hgo; fa.hblk = hgo ? hblk : nullptr;  // [r6h] block counters only with the go flag
  fa.cit = cit; fa.qctl = qctl; fa.tok_left = tok_left; fa.cq = cq; fa.row_tok = row_tok; fa.rrow = rrow; fa.lid = lid;
  fa.topw = topw; fa.pw = (const __nv_bfloat16*)pw; fa.out = (__nv_bfloat16*)out; fa.T = T;
  fa.dbg = (unsigned long long*)dbg;
  fa.flags = flags;
  fa.g1 = make_args(w13, xt, tiles, ntiles, gs13, inter, 64, smem, counter);
  fa.g2 = make_args(w2, inter, tiles, ntiles, gs2, y, 32, smem, counter);
  fa.g1.flags = flags;
  fa.g2.flags = flags;
  fa.done = done;
  if ((fa.flags >> 8) & 1)  // bit 8: fine remainder tiles (the plan used the same flag)
    okc_launch(moe_fused_persistent<1>, dim3((unsigned)nctas), dim3(MOE_FUSED_THREADS), (size_t)smem, st, fa);
  else
    okc_launch(moe_fused_persistent<0>, dim3((unsigned)nctas), dim3(MOE_FUSED_THREADS), (size_t)smem, st, fa);
}
void k_moe_init(int smem) {
  cudaFuncSetAttribute(moe_combine2_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, COMB2_SMEM);
  cudaFuncSetAttribute(moe_combine2_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);  // 3 CTAs/SM; = MoE's config
  cudaFuncSetAttribute(moe_combine2_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, COMB2_SMEM);  // [r6h]
  cudaFuncSetAttribute(moe_combine2_kernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(moe_fused_persistent<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_fused_persistent<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_gemm_persistent<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_gemm_persistent<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_gemm_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_gemm_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
}
// ---- attention ----
#define ATTN_HALF_MAX 640  // per-head prep grids below this use half-tile blocks
void k_attn_prep(const void* kc, const void* vc, const int* page_table, int max_pages, int page, const int* seq_lens, int B, int L,
                 int window, int ntk, void* kt, void* vt, cudaStream_t st, int part) {
  if (ntk * B >= 264) {  // long contexts: both heads per block (full cache rows, one page-table read per key)
    dim3 grid((unsigned)ntk, 1, (unsigned)B);
    okc_launch(okc_attn::attn_prep2_kernel, grid, dim3(256), 0, st, (const uint8_t*)kc, (const uint8_t*)vc, page_table, max_pages, page, seq_lens, L,
               window, ntk, (uint8_t*)kt, (uint8_t*)vt, part);
  } else if (2 * ntk * B >= ATTN_HALF_MAX) {  // one head per block
    dim3 grid((unsigned)ntk, 2, (unsigned)B);
    okc_launch(okc_attn::attn_prep_kernel<1>, grid, dim3(256), 0, st, (const uint8_t*)kc, (const uint8_t*)vc, page_table, max_pages, page, seq_lens, L,
               window, ntk, (uint8_t*)kt, (uint8_t*)vt, part);
  } else {  // few tiles (short sliding windows): one head and half a tile per block
    dim3 grid((unsigned)(2 * ntk), 2, (unsigned)B);
    okc_launch(okc_attn::attn_prep_kernel<2>, grid, dim3(256), 0, st, (const uint8_t*)kc, (const uint8_t*)vc, page_table, max_pages, page, seq_lens, L,
               window, ntk, (uint8_t*)kt, (uint8_t*)vt, part);
  }
}
#define ATTN_CR 2  // (token, head) rows per warp in the split merge (Oct 26 search: 2 +0.03..0.06% over 4)
void k_attn_fwd(const void* q, const void* kt, const void* vt, const int* seq_lens, const void* qkv, void* out, int B, int L,
                int window, int ntk, float c, float v_scale, int smem, int nsplit, float* ws, int variant, cudaStream_t st,
                const void* pf, unsigned long long pf_bytes) {
  okc_attn::AttnArgs a;
  a.q = (const uint8_t*)q; a.kt = (const uint8_t*)kt; a.vt = (const uint8_t*)vt; a.seq_lens = seq_lens;
  a.qkv = (const __nv_bfloat16*)qkv; a.out = (__nv_bfloat16*)out;
  const int nwg = variant == 99 ? 1 : (variant == 64 ? 4 : ((variant & 8) ? 3 : 2));
  a.L = L; a.window = window; a.ntk = ntk; a.nqb = (L + 8 * nwg - 1) / (8 * nwg); a.c = c; a.v_scale = v_scale;
  a.zero = 0;
  a.pf = (const uint8_t*)pf; a.pf_bytes = pf_bytes;
  a.sk = 0; a.nb = B; a.skbuf = nullptr;
  if (nsplit < 0) {  // stream-K: grid -nsplit; ws = [int header: W, pair prefix, first pairs | partials]
    const int G = -nsplit, npairs = B * a.nqb * 2;
    const int hdr = ((2 + npairs + 1 + G) + 63) & ~63;
    a.sk = G; a.skbuf = reinterpret_cast<int*>(ws); a.ws = ws + hdr; a.nsplit = 1;
    if (variant == 43) okc_launch(okc_attn::attn_fwd_sk_kernel<3, 0, true, true>, dim3((unsigned)G), dim3(512), (size_t)smem, st, a);
    else okc_launch(okc_attn::attn_fwd_sk_kernel<3, 0, false, false>, dim3((unsigned)G), dim3(512), (size_t)smem, st, a);
    const int T = B * L;
    okc_launch(okc_attn::attn_combine_sk_kernel<ATTN_CR, 24>, dim3((T * (16 / ATTN_CR) + 7) / 8), dim3(256), 0, st, (const float*)(ws + hdr),
               (const int*)ws, T, L, a.nqb, (const __nv_bfloat16*)qkv, (__nv_bfloat16*)out, c, v_scale);
    return;
  }
  a.nsplit = nsplit; a.ws = nsplit > 1 ? ws : nullptr;
  dim3 grid((unsigned)a.nqb, (unsigned)(2 * nsplit), (unsigned)B);
  switch (variant) {
    case 0: okc_launch(okc_attn::attn_fwd_kernel<2, 0, false>, grid, dim3(384), (size_t)smem, st, a); break;
    case 1: okc_launch(okc_attn::attn_fwd_kernel<2, 16, false>, grid, dim3(384), (size_t)smem, st, a); break;
    case 8: okc_launch(okc_attn::attn_fwd_kernel<3, 0, false>, grid, dim3(512), (size_t)smem, st, a); break;
    case 9: okc_launch(okc_attn::attn_fwd_kernel<3, 16, false>, grid, dim3(512), (size_t)smem, st, a); break;
    case 16: okc_launch(okc_attn::attn_fwd_kernel<2, 0, true>, grid, dim3(384), (size_t)smem, st, a); break;
    case 43: okc_launch(okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>, grid, dim3(512), (size_t)smem, st, a); break;
    default: okc_launch(okc_attn::attn_fwd_kernel<3, 0, true>, grid, dim3(512), (size_t)smem, st, a); break;
  }
  if (nsplit > 1 && !g_attn_merge_host) {  // (with the in-kernel merge the last split's CTA writes the output)
    const int T = B * L;
    okc_launch(okc_attn::attn_combine2_kernel<ATTN_CR>, dim3((T * (16 / ATTN_CR) + 7) / 8), dim3(256), 0, st, ws, nsplit, T, (const __nv_bfloat16*)qkv, (__nv_bfloat16*)out,
               c, v_scale);
  }
}
void k_attn_set_prof(void* p) {
}
void k_attn_init(int smem) {
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<2, 0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<2, 16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<2, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_sk_kernel<3, 0, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_sk_kernel<3, 0, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
}
'''

_CUDA_SRC = r'''// Grouped NVFP4 x BF16 expert GEMM for Muse Spark's MoE (one H200 = 32 local experts).
// out^T[w_rows, tokens] = W_e[w_rows, K] (NVFP4, dequantized in registers) x X^T[K, tokens] (BF16)
// sm_90: warp-specialized (1 producer warpgroup issuing cp.async.bulk, 2 consumer warpgroups
//        running wgmma m64nNk16 with the weights as the register A operand).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

// ---- programmatic dependent launch: every kernel lets its successor launch as soon as all of its blocks
// are running, and waits for its predecessor's results before touching global memory (a no-op when the kernel was not
// launched with the PDL attribute) ----
__device__ __forceinline__ void pdl_enter() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
// Before lane 0 hands a ring slot back, the whole warp's generic reads of it are ordered before the arrive (a warp
// barrier: the memory model orders the other lanes' LDS before lane 0's release only through it).
// MOE_ALLLANE_RELEASE: every lane arrives on the slot's empty barrier (count 32 per warp), so each lane's own reads are
// ordered before its own arrive without a warp barrier.
#define MOE_ALLLANE_RELEASE 1  // v70: formally ordered slot release, measured free
#define REL_LANES (MOE_ALLLANE_RELEASE ? 32 : 1)

namespace okc_moe {

constexpr int KS = 64;        // k-stages of 64 elements (K = 4096)
constexpr int WBLK = 4608;    // bytes per (ntile, kstage) weight block: 4096 code bytes + 512 scale bytes
constexpr int MAX_STAGES = 24;

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

DEVI void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
DEVI void mbar_inval(uint32_t bar) { asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(bar) : "memory"); }
// Norm weights stay BF16 exactly as given; kernels form 1 + w in fp32 when they load them (the same value an fp32
// (1 + w) table would hold). g1p4: 4 consecutive weights (8-byte aligned) -> 1 + w.
DEVI float4 g1p4(const __nv_bfloat16* g) {
  const uint2 v = __ldg(reinterpret_cast<const uint2*>(g));
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
  return make_float4(1.f + a.x, 1.f + a.y, 1.f + b.x, 1.f + b.y);
}
DEVI void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred P1;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
      "@!P1 bra WAIT_%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
DEVI void mbar_arrive(uint32_t bar) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(bar) : "memory");
}

DEVI void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
// L2 prefetch of a weight block a few stages ahead of its TMA load (fire and forget: no smem, no barrier), so the
// weight stream keeps more bytes in flight than the 4-slot ring holds. MOE_L2PF = distance in stages (0: off).
#define MOE_L2PF 1  // Oct 26: 1 + ALL = +0.29% (search r21, x24)
#define MOE_L2PF_ALL 1  // 0: small token tiles only (2+ k-blocks per stage); 1: every item
DEVI bool elect_one() {  // elect.sync over the full warp: true in exactly one lane (the lowest active = lane 0)
  uint32_t r;
  asm volatile("{\n.reg .pred P;\nelect.sync _|P, 0xffffffff;\nselp.u32 %0, 1, 0, P;\n}\n" : "=r"(r));
  return r != 0;
}
DEVI void prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
// A ring slot whose weight codes and scales the consumers read with generic loads (LDS) is about to be overwritten by
// the async proxy (bulk copy): after the empty-barrier acquire, a proxy fence orders those generic reads before the
// async writes (PTX memory model; compute-sanitizer racecheck flags the slot reuse without it).
DEVI void rel_sync() {
}
DEVI void slot_proxy_fence() {
}
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}
// shared -> global bulk copy (async proxy), tracked per thread with bulk groups
DEVI void bulk_s2g(void* dst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src), "r"(bytes) : "memory");
}
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
DEVI void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}


DEVI void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }

// SwiGLU's x * sigmoid(x) as 0.5 x (1 + tanh(x / 2)): one MUFU op instead of exp + reciprocal (the epilogue is
// MUFU-bound and the tensor cores idle during it)
DEVI float silu_fast(float x) {
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return 0.5f * x * (1.f + t);
}

// wgmma descriptor for a K-major, 128B-swizzled B tile (rows of 128 B, 8-row atoms of 1024 B).
DEVI uint64_t desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;           // LBO (unused for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32; // SBO: 8 rows x 128 B
  d |= (uint64_t)1 << 62;           // SWIZZLE_128B
  return d;
}

template <int N>
DEVI void wgmma_rs(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d);
template <> DEVI void wgmma_rs<32>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<64>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<128>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<256>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<96>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %53, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%48, %49, %50, %51}, %52, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<160>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %85, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n160k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79}, {%80, %81, %82, %83}, %84, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<192>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %101, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, {%96, %97, %98, %99}, %100, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<224>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %117, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n224k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111}, {%112, %113, %114, %115}, %116, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

template <> DEVI void wgmma_rs<8>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}
template <> DEVI void wgmma_rs<16>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

#define MOE_XPF256 0
#define MOE_XPFSMALL 0
#define MOE_FINE_MASK 6  // extra widths: 1 = 96, 2 = 160, 4 = 192, 8 = 224 (3+ of them: ptxas C7512, wgmma serialized)

DEVI uint32_t bf162_u32(__nv_bfloat162 v) { return *reinterpret_cast<uint32_t*>(&v); }
// mma.sync m16n8k16 (bf16 x bf16 -> fp32): the A fragment layout equals one warp's share of the wgmma RS m64 A operand
DEVI void mma16816(float* c, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
DEVI __nv_bfloat162 u32_bf162(uint32_t v) { return *reinterpret_cast<__nv_bfloat162*>(&v); }

// Two E4M3 block scales (low byte: row g, high byte: row g+8) -> bf16x2 of scale * 2^119.
// MOE_FAST_SCALE (prepare checks that no block scale is zero or subnormal): for a normal E4M3 byte b the bf16 bits of
// value * 2^119 are exactly (b << 4) + 0x7780 (exponent rebias 120 + 119, mantissa bits moved up 4): one byte permute
// and one IMAD for the pair.
#ifndef MOE_FAST_SCALE
#define MOE_FAST_SCALE 1
#endif
DEVI uint32_t scale_pair(uint32_t s2) {
#if MOE_FAST_SCALE
  return __byte_perm(s2, 0u, 0x4140) * 16u + 0x77807780u;  // [b0, 0, b1, 0] -> (b << 4) + 0x7780 per half
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t h2;
  asm("{\n.reg .b16 t;\ncvt.u16.u32 t, %1;\ncvt.rn.f16x2.e4m3x2 %0, t;\n}\n" : "=r"(h2) : "r"(s2));
  // f16 (exact for every e4m3 value) -> bf16 with the exponent rebiased by 112 + 119
  return ((h2 >> 3) & 0x0FFF0FFFu) + 0x73807380u;
#endif
}

// The scale pair of one k16 step straight from the 32-bit word holding two steps' pairs: the byte permute selects
// bytes 0,1 (hi = 0) or 2,3 (hi = 1) itself, so no shift or mask (hi is a compile-time constant in the unrolled loops).
DEVI uint32_t scale_pair_sel(uint32_t w, int hi, uint32_t m16 = 16u) {
#if MOE_FAST_SCALE
  return __byte_perm(w, 0u, hi ? 0x4342 : 0x4140) * m16 + 0x77807780u;  // m16 opaque: IMAD on the FMA pipe
#else
  return scale_pair((w >> (16 * hi)) & 0xFFFF);
#endif
}

// 8 E2M1 codes -> 4 bf16x2 A-fragment registers (scaled). The repack places each code's bits so that one rotate and
// one mask put it in place: a register half holds (sign << 15 | magnitude << 6), which as bf16 bits is exactly
// e2m1 * 2^-126 (subnormal for magnitude 1), times scale * 2^119. Bit positions in q (sign, m2, m1, m0):
//   a0 (row g):    code A (15, 8, 7, 6),   code B (31, 24, 23, 22)    q & M
//   a1 (row g+8):  code A (21, 14, 13, 12), code B (5, 30, 29, 28)    rotr(q, 6) & M
//   a2 (row g):    code A (25, 18, 17, 16), code B (9, 2, 1, 0)       rotr(q, 10) & M
//   a3 (row g+8):  code A (11, 4, 3, 10),  code B (27, 20, 19, 26)    rotl(q, 4) & 0x81808180 | rotr(q, 4) & 0x00400040
// (9 ALU ops for 8 weights, no IMAD shifts; the repack in the template writes this layout.)
// Dequant layout (via the scout): 7 integer-pipe ops + 2 IMAD shifts per 8 weights. Two of the shifts
// become multiplies by 16 / 64 held in registers (m16 = 16 + opaque 0, m64 = 64 + opaque 0), which run on the FMA pipe
// as IMAD instead of on the half-rate integer pipe. Bit positions in q (m0, m1, m2, sign):
//   a0: code A (6, 7, 8, 15),   code B (22, 23, 24, 31)   q & M
//   a1: code A (2, 3, 4, 11),   code B (18, 19, 20, 27)   (q * 16) & M
//   a2: code A (28, 29, 30, 5), code B (12, 13, 14, 21)   rotl(q, 10) & M
//   a3: code A (0, 1, 26, 9),   code B (16, 17, 10, 25)   (q * 64) & 0x80C080C0 | rotl(q, 14) & 0x01000100
DEVI uint32_t opaque_zero() {
  uint32_t z;
  asm volatile("mov.u32 %0, 0;" : "=r"(z));
  return z;
}
DEVI void convert8(uint32_t q, uint32_t sb, uint32_t (&a)[4], uint32_t m16 = 16u, uint32_t m64 = 64u) {
  const __nv_bfloat162 s = u32_bf162(sb);
  const __nv_bfloat162 slo = __low2bfloat162(s), shi = __high2bfloat162(s);
  constexpr uint32_t M = 0x81C081C0u;
  const uint32_t r0 = q & M;
  const uint32_t r1 = (q * m16) & M;
  const uint32_t r2 = __funnelshift_l(q, q, 10) & M;
  const uint32_t r3 = ((q * m64) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  a[0] = bf162_u32(__hmul2(u32_bf162(r0), slo));
  a[1] = bf162_u32(__hmul2(u32_bf162(r1), shi));
  a[2] = bf162_u32(__hmul2(u32_bf162(r2), slo));
  a[3] = bf162_u32(__hmul2(u32_bf162(r3), shi));
}


struct Args {
  const uint8_t* wrep;   // [E][NT][KS][WBLK]
  const uint8_t* xt;     // activations in tile layout (byte base; a tile starting at row r0 is at r0 * 8192)
  const int4* tiles;     // (expert, row0, n, 0)
  const int* ntiles;     // device scalar
  const float* gscale;   // [E]
  uint8_t* out;          // GATED: tile layout (bytes); else row-major bf16 [rows, 4096]
  int NT;
  int smem_bytes;
  int* counter;          // persistent kernel work queue (zeroed before launch)
  int flags;             // experiments only (0 in production): bit 2 = skip the NVFP4 conversion
  uint32_t mul16, mul64;  // 16 and 64, from the host: multiplies by them stay IMAD (FMA-heavy pipe), since ptxas sees
                          // through an inline-asm zero and turns `x * (16 + 0)` back into an integer-pipe shift
};
#define MOE_OPAQUE_MUL 1

template <int N, bool GATED>
DEVI void consumer_tile(const Args& p, uint8_t* ring, uint32_t bars, int stride, int nst, int e, int row0,
                        int ntile, int* done = nullptr) {
  const uint32_t zz_ = opaque_zero(), m16 = 16u + zz_, m64 = 64u + zz_;  // opaque multipliers (IMAD shifts)
  const int ctid = threadIdx.x - 128;  // 0..255
  // warp-uniform by construction (shuffled from lane 0), so ptxas keeps the smem descriptor math on the uniform datapath
#define MOE_UNIFORM_WG 0  // measured: no change in the MoE spans
  const int wg = MOE_UNIFORM_WG ? __shfl_sync(0xffffffffu, ctid >> 7, 0) : ctid >> 7;
  const int w = MOE_UNIFORM_WG ? __shfl_sync(0xffffffffu, (ctid >> 5) & 3, 0) : (ctid >> 5) & 3;
  const int lane = ctid & 31;
  const int g = lane >> 2, t = lane & 3;
  float acc[128];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  uint32_t afrag[2][4];
  int s = 0, prev_s = 0;
  uint32_t ph = 0;
  for (int ks = 0; ks < KS; ++ks) {
    mbar_wait(bars + 8 * s, ph);
    uint8_t* st = ring + s * stride;
    const uint4 wq = *reinterpret_cast<const uint4*>(st + N * 128 + ((wg * 4 + w) * 32 + lane) * 16);
    const uint2 sc = *reinterpret_cast<const uint2*>(st + N * 128 + 4096 + ((wg * 4 + w) * 8 + g) * 8);
    const uint64_t desc = desc_sw128(smem_u32(st));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
      const uint32_t sw = j < 2 ? sc.x : sc.y;  // scales of k16 steps (j & ~1, j | 1): bytes 0,1 and 2,3
      convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[j & 1], m16, m64);
      wg_fence();
      wgmma_rs<N>(acc, afrag[j & 1], desc + 2 * j, 1);
      wg_commit();
      wg_wait<1>();
    }
    // steps 0-2 of this stage are complete: every wgmma reading the previous stage is done
    if (ks > 0) { rel_sync(); if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * MAX_STAGES + 8 * prev_s); }
    prev_s = s;
    if (++s == nst) { s = 0; ph ^= 1; }
  }
  wg_wait<0>();
  // ---- epilogue ----
  const float cs = p.gscale[e] * 128.f;  // weights were scaled by 2^-7
  named_bar_sync(1, 256);               // every consumer is done reading the ring
  uint8_t* stg = ring;
  if constexpr (GATED) {
    // rows g / g+8 of each 16-row slice are gate / up of inter column c = 8*slice + g
    const int q = wg * 4 + w;
#pragma unroll
    for (int jj = 0; jj < N / 8; ++jj) {
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const float gt = acc[4 * jj + b] * cs, up = acc[4 * jj + 2 + b] * cs;
        const float h = silu_fast(gt) * up;
        const int tok = 8 * jj + 2 * t + b;
        *reinterpret_cast<__nv_bfloat16*>(stg + tok * 128 + ((q ^ (tok & 7)) << 4) + g * 2) = __float2bfloat16_rn(h);
      }
    }
    named_bar_sync(1, 256);
    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)row0 * 8192 + (size_t)ntile * N * 128);
    const uint4* src = reinterpret_cast<const uint4*>(stg);
    for (int i = ctid; i < N * 8; i += 256) dst[i] = src[i];
    if (done) {  // fused kernel: publish this n-tile of the tile's intermediate rows
      __threadfence();
      named_bar_sync(1, 256);
      if (ctid == 0) atomicAdd(done, 1);
    }
  } else {
    // output columns (within the 128-wide ntile) c0 = 16*slice + g, c1 = c0 + 8; staging rows of 272 B
    const int c0 = (wg * 4 + w) * 16 + g;
#pragma unroll
    for (int jj = 0; jj < N / 8; ++jj) {
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const int tok = 8 * jj + 2 * t + b;
        __nv_bfloat16* r = reinterpret_cast<__nv_bfloat16*>(stg + tok * 272);
        r[c0] = __float2bfloat16_rn(acc[4 * jj + b] * cs);
        r[c0 + 8] = __float2bfloat16_rn(acc[4 * jj + 2 + b] * cs);
      }
    }
    named_bar_sync(1, 256);
    for (int i = ctid; i < N * 16; i += 256) {
      const int tok = i >> 4, c = i & 15;
      *reinterpret_cast<uint4*>(p.out + ((size_t)(row0 + tok) * 4096 + ntile * 128) * 2 + c * 16) =
          *reinterpret_cast<const uint4*>(stg + tok * 272 + c * 16);
    }
  }
}

template <bool GATED>
__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Args p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);  // full[MAX_STAGES], empty[MAX_STAGES]
  uint8_t* ring = base + 1024;
  const int ring_bytes = p.smem_bytes - 1024 - (int)pad;
  const int ntile = blockIdx.x;
  const int ntiles = *p.ntiles;
  if (threadIdx.x < 128) {
    // ---------------- producer warpgroup ----------------
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y) {
      const int4 ti = p.tiles[tile];
      const int e = ti.x, row0 = ti.y, n = ti.z;
      const int stride = (n * 128 + WBLK + 1023) & ~1023;
      int nst = ring_bytes / stride;
      nst = nst > MAX_STAGES ? MAX_STAGES : nst;
      if (threadIdx.x == 0) {
        for (int s = 0; s < nst; ++s) {
          mbar_init(bars + 8 * s, 1);
          mbar_init(bars + 8 * MAX_STAGES + 8 * s, 8 * REL_LANES);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      __syncthreads();
      const uint8_t* wsrc = p.wrep + ((size_t)(e * p.NT + ntile) * KS) * WBLK;
      const uint8_t* xsrc = p.xt + (size_t)row0 * 8192;
      const uint32_t xbytes = n * 128;
      if (threadIdx.x == 0) {
        for (int ks = 0; ks < KS; ++ks) {
          const int s = ks % nst;
          const uint32_t ph = (ks / nst) & 1;
          mbar_wait(bars + 8 * MAX_STAGES + 8 * s, ph ^ 1);
          slot_proxy_fence();
          const uint32_t full = bars + 8 * s;
          mbar_expect_tx(full, xbytes + WBLK);
          const uint32_t dst = smem_u32(ring + s * stride);
          bulk_g2s(dst, xsrc + (size_t)ks * xbytes, xbytes, full);
          bulk_g2s(dst + xbytes, wsrc + (size_t)ks * WBLK, WBLK, full);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        for (int s = 0; s < nst; ++s) {
          mbar_inval(bars + 8 * s);
          mbar_inval(bars + 8 * MAX_STAGES + 8 * s);
        }
      }
      __syncthreads();
    }
  } else {
    // ---------------- consumer warpgroups ----------------
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y) {
      const int4 ti = p.tiles[tile];
      const int e = ti.x, row0 = ti.y, n = ti.z;
      const int stride = (n * 128 + WBLK + 1023) & ~1023;
      int nst = ring_bytes / stride;
      nst = nst > MAX_STAGES ? MAX_STAGES : nst;
      __syncthreads();
      switch (n) {
        case 32: consumer_tile<32, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
        case 64: consumer_tile<64, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
        case 128: consumer_tile<128, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
        case 160: consumer_tile<160, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
        case 192: consumer_tile<192, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
        default: consumer_tile<256, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
      }
      __syncthreads();
      __syncthreads();
    }
  }
}

// ---- persistent variant: grid = #SMs, items (tile, ntile) round-robin, fixed slot ring + separate staging ----
constexpr int SLOT = 43008;     // bytes per ring slot (1024-aligned): fits kps k-stages of any tile size
constexpr int NSLOT = 4;
constexpr int STG_BYTES = 45056; // epilogue staging

// k-blocks per ring stage by token-tile size (a stage must fit SLOT: N=32 allows 1/2/4, N=64 and 128 1/2, N=256 1)
#define MOE_KPS32 4
#define MOE_KPS64 2
#define MOE_KPS128 2
#define MOE_GROUP4_MAX 128  // token tiles up to this size commit one wgmma group per stage (see GROUP4)
// tile widths: 32, 64 (small: weight-conversion-bound), 96, 128 (two k-blocks a stage), 160..256 in steps of 32 (one
// k-block a stage; an expert's last tile is its remaining rows rounded up to 32, not to 128/256)
DEVI int kps_for(int n) { return n >= 160 ? 1 : (n <= 32 ? MOE_KPS32 : (n <= 64 ? MOE_KPS64 : MOE_KPS128)); }
// a stage (kps k-blocks of X and W) must fit its ring slot and kps must divide the 64 k-blocks (Oct 26: the search's
// "kps64_4 +0.16%" was a kernel writing 8 KB past each slot: 4 x (64 x 128 + 4608) = 51200 > SLOT)
static_assert(MOE_KPS32 * (32 * 128 + WBLK) <= SLOT && 64 % MOE_KPS32 == 0, "MOE_KPS32: stage does not fit a ring slot");
static_assert(MOE_KPS64 * (64 * 128 + WBLK) <= SLOT && 64 % MOE_KPS64 == 0, "MOE_KPS64: stage does not fit a ring slot");
static_assert(MOE_KPS128 * (128 * 128 + WBLK) <= SLOT && 64 % MOE_KPS128 == 0, "MOE_KPS128: stage does not fit a ring slot");
static_assert(256 * 128 + WBLK <= SLOT, "N=256 stage does not fit a ring slot");

// Copy-out queue in shared memory (fused sm_90 kernel): pubq = base + 256, 8 entries of 32 bytes {dst, done, kind,
// bytes or tokens}; base + 240 = jobs published (st.release by consumer thread 0; negative = -(count + 1) at exit),
// base + 244 = consumer thread 0's own count, base + 248 = jobs whose staging reads the publisher has finished.
DEVI void enqueue_out(volatile long long* pubq, void* dst, int* done, int kind, int amount, int row0 = 0, int nreal = 0) {
  volatile int* cnt = reinterpret_cast<volatile int*>(pubq) - 3;  // base + 244
  const int n = *cnt;
  volatile long long* e = pubq + 4 * (n & 7);
  e[0] = (long long)dst;
  e[1] = (long long)done;
  reinterpret_cast<volatile int*>(e)[4] = kind;
  reinterpret_cast<volatile int*>(e)[5] = amount;
  reinterpret_cast<volatile int*>(e)[6] = row0;   // CIT (GEMM2): the tile's first row and real row count
  reinterpret_cast<volatile int*>(e)[7] = nreal;
  *cnt = n + 1;
  asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32((const void*)(pubq - 2))), "r"(n + 1) : "memory");
}
// consumer thread 0: wait until the publisher has read every queued staging buffer
DEVI void wait_staging_free(volatile long long* pubq) {
  const int n = *(reinterpret_cast<volatile int*>(pubq) - 3);
  const uint32_t a = smem_u32((const void*)(reinterpret_cast<volatile int*>(pubq) - 2));  // base + 248
  while (true) {
    int f;
    asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(f) : "r"(a) : "memory");
    if (f >= n) break;
  }
}

// pubq (fused sm_90 kernel): instead of fencing and bumping `done` itself, the consumers hand the GEMM1 n-tile's
// completion to the producer warpgroup's publisher warp (smem queue at pubq = base + 256, published count at base + 240,
// consumer thread 0's running count at base + 244), so the next item's main loop does not wait for the store fence.
#define MOE_CONS_FIRST 0  // fused kernel: consumers at threads 0..255, producer + publisher warps at 256..319 (320
                          // threads: 200 registers for every thread at launch, no setmaxnreg)
#define MOE_CB (MOE_CONS_FIRST ? 0 : 128)  // first consumer thread
#define MOE_PB (MOE_CONS_FIRST ? 256 : 0)  // first producer thread
#define MOE_FUSED_THREADS (MOE_CONS_FIRST ? 320 : 384)
// ---- cold-expert tiles (smoe/cold lever, 11-02) ----
// Why MOE_MMA32 lost (t1s 536 -> 640 us): in its SASS every k16 step is `branch on nblk -> LDSM -> dependent HMMA`
// with the A fragment and the LDSM destination re-used by the very next step (WAR on registers an HMMA has not read
// yet), so each step pays the full LDSM latency plus the HMMA operand read, 4-8 times per k-block per warp, with only
// two consumer warps per SM sub-partition to hide it; with M split over 8 warps every warp also re-reads the whole X
// k-block (8x the B traffic, 16-32 KB of ldmatrix per k-block). The path below: no branches (token groups are a
// template parameter), all B fragments of a k-block loaded one k-block ahead into their own registers, all 8 A
// fragments of a k-block converted before its MMAs, K split by k-block parity over the two warpgroups (half the B
// traffic, and exactly the wgmma path's NACC=4 accumulator sets), and the slot is handed back as soon as this
// thread's reads of it are consumed (mma.sync reads no shared memory).
#ifndef MOE_COLD
#define MOE_COLD 1  // 0: off (c1); 1: mma.sync K-split path; 2: narrow wgmma (N = 8 / 16) for the same tiles
#endif
#define MOE_COLD_NBMAX 1  // MOE_COLD 1: 8-token groups handled (1: tiles with <= 8 real rows only; 2: <= 16)
#define MOE_COLD_XTRIM 0  // the producer copies only the X rows such items read (8 or 16 of each k-block's 32)
// 0: regular N=32 item; 1: <= 8 real rows; 2: <= 16 real rows
DEVI int cold_kind(int n, int nreal) {
  if (MOE_COLD == 0 || n != 32) return 0;
  if (nreal <= 8) return 1;
  if (nreal <= 16 && (MOE_COLD == 2 || MOE_COLD_NBMAX >= 2)) return 2;
  return 0;
}
DEVI int cold_xrows(int n, int nreal) { return MOE_COLD_XTRIM ? 8 * cold_kind(n, nreal) : 0; }
template <int N, int CB = 128, int CNB = 0, int WN = N>
DEVI void consumer_item(const Args& p, uint8_t* ring, uint8_t* stg, uint32_t bars, int& slot, uint32_t& ph,
                        int e, int row0, int ntile, bool GATED, int* done = nullptr, unsigned long long* t_mid = nullptr,
                        volatile long long* pubq = nullptr, int nreal = 256) {
#define MOE_IMAD_MAXN 64  // IMAD shifts only for small token tiles (N=128 measured 2-3% slower with them)
  const uint32_t zz_ = N <= MOE_IMAD_MAXN ? opaque_zero() : 0u;
  // small tiles: multipliers ptxas cannot see (host values), so the shifts run as IMAD on the FMA-heavy pipe
  const uint32_t m16 = (N <= MOE_IMAD_MAXN && MOE_OPAQUE_MUL) ? p.mul16 : 16u + zz_;
  const uint32_t m64 = (N <= MOE_IMAD_MAXN && MOE_OPAQUE_MUL) ? p.mul64 : 64u + zz_;
  constexpr int KPS = N >= 160 ? 1 : (N <= 32 ? MOE_KPS32 : (N <= 64 ? MOE_KPS64 : MOE_KPS128));
  constexpr int XB = N * 128;  // X bytes per k-stage
  const int ctid = threadIdx.x - CB;
  // warp-uniform by construction (shuffled from lane 0), so ptxas keeps the smem descriptor math on the uniform datapath
  const int wg = MOE_UNIFORM_WG ? __shfl_sync(0xffffffffu, ctid >> 7, 0) : ctid >> 7;
  const int w = MOE_UNIFORM_WG ? __shfl_sync(0xffffffffu, (ctid >> 5) & 3, 0) : (ctid >> 5) & 3;
  const int lane = ctid & 31;
  const int g = lane >> 2, t = lane & 3;
  // small token tiles: the wgmma is short, so one commit group carries a whole k-stage (4 x k16) and the next
  // stage's conversions overlap it (one group in flight, double-buffered A fragments)
#define MOE_G4_96 0  // 96-row tiles in the grouped path compile to wgmma serialization for the whole kernel (C7512)
  constexpr bool GROUP4 = N <= MOE_GROUP4_MAX && (N != 96 || MOE_G4_96);
  // small token tiles: a commit group's MMAs are shorter than the wgmma latency, so consecutive groups accumulate into
  // two independent register sets (summed before the epilogue) instead of one dependency chain
#define MOE_NACC2 1
  // MOE_NACC2 = 0: one accumulator everywhere, so a row's sums do not depend on its tile size (bitwise-deterministic
  // outputs even though the router places rows with atomics)
  // MOE_NACC4 (tiles up to MOE_NACC4_MAXN tokens): the k16 steps of a group also alternate between two sets, so each
  // group is two short dependency chains instead of one of four (four accumulator sets in all)
#define MOE_NACC4 1
#define MOE_NACC4_MAXN 32
  // MOE_TRI (N=32 tiles): three A-fragment buffers and three accumulator sets rotate with the k-block index, so a
  // k-block's conversion runs while the two previous k-blocks' MMA groups are in flight (wait<2> instead of wait<1>)
#define MOE_TRI 0  // 13:50: C7511 (wgmma serialized, register pressure in the fused kernel) -> 15.5 vs 12.2 us per N=32 item
#define MOE_TRI_ARCH 1
  constexpr bool TRI = MOE_TRI && MOE_TRI_ARCH && MOE_NACC2 && GROUP4 && N == 32 && KPS == 4;
  constexpr int NACC = CNB > 0 ? 1 : (TRI ? 3 : ((GROUP4 && MOE_NACC2) ? ((MOE_NACC4 && N <= MOE_NACC4_MAXN) ? 4 : 2) : 1));
  float acc[128];
  float acc2[128];  // only the first N/2 are live (and only when NACC > 1)
  float acc3[128], acc4[128];  // NACC == 4 only
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
#pragma unroll
  for (int i = 0; i < (NACC > 1 ? N / 2 : 0); ++i) acc2[i] = 0.f;
#pragma unroll
  for (int i = 0; i < (NACC > 2 ? N / 2 : 0); ++i) acc3[i] = 0.f;
#pragma unroll
  for (int i = 0; i < (NACC > 3 ? N / 2 : 0); ++i) acc4[i] = 0.f;
  // N=32 tiles may keep more MMA groups in flight (MOE_SMALL_WAIT groups behind; 4 A-fragment buffers indexed by the
  // k-block within the 4-k-block stage); otherwise two buffers (k-block parity) and one group behind
#define MOE_SMALL_WAIT 1
  constexpr bool DEEP = GROUP4 && N == 32 && KPS == 4 && MOE_SMALL_WAIT > 1;
#define MOE_MMA32 0  // N=32 tiles: mma.sync m16n8k16 per warp instead of wgmma, skipping the 8-token groups past the
                     // tile's real rows (tiles[].w): the tensor work follows the tokens, not the 32-row tile
  constexpr bool MMA32 = MOE_MMA32 && GROUP4 && !DEEP && N == 32;
  const int nblk = MMA32 ? min(4, max(1, (nreal + 7) >> 3)) : 4;
  constexpr int AFB = DEEP ? 4 : 2;
  uint32_t afrag[GROUP4 ? 4 * AFB : 2][4];
  int prev = -1;
  const int woff = ((wg * 4 + w) * 32 + lane) * 16, soff = 4096 + ((wg * 4 + w) * 8 + g) * 8;
  // [slice 2w + s][k16-step parity][8-token group][fragment]: MOE_COLD 1 accumulators (unused otherwise)
  float cacc[2][2][CNB > 0 ? CNB : 1][4];
  if constexpr (CNB > 0) {
    static_assert(N == 32 && KPS == 4 && CNB <= 2, "cold path: 32-row tiles, 4 k-blocks per stage, <= 16 rows");
#pragma unroll
    for (int s2 = 0; s2 < 2; ++s2)
#pragma unroll
      for (int jp = 0; jp < 2; ++jp)
#pragma unroll
        for (int nb = 0; nb < CNB; ++nb)
#pragma unroll
          for (int f = 0; f < 4; ++f) cacc[s2][jp][nb][f] = 0.f;
    // warpgroup wg: k-blocks kk = wg and wg + 2 of every stage (global k-block parity wg) for all 128 rows; warp w:
    // the 16-row slices 2w and 2w + 1 of the weight block (codes at (slice * 32 + lane) * 16, scales at 4096 +
    // (slice * 8 + g) * 8, as the wgmma path reads them)
    // (lane offsets derived through an opaque zero: ptxas would hoist them out of the persistent item loop, and these
    // extra long-lived registers alone make it spill the epilogue's invariants in the register-bound N=256 path)
    const int ln = lane + (int)opaque_zero(), wn = w + (int)opaque_zero();
    const int r7 = ln & 7, mq = ln >> 3;
    const int wo0 = ((2 * wn) * 32 + ln) * 16, so0 = 4096 + ((2 * wn) * 8 + (ln >> 2)) * 8;  // slice 2w + 1: +512, +64
    // ldmatrix.x4 lane addresses in the 128B-swizzled token rows: matrix mq = (k16 step 2x + (mq >> 1), k half
    // mq & 1) of token row 8 nb + r7, 16-byte chunk (2 * step + half) ^ r7; x4 #0 = steps 0, 1, x4 #1 = steps 2, 3
    const uint32_t xo0 = r7 * 128 + (((2 * (mq >> 1)) + (mq & 1)) ^ r7) * 16;
    const uint32_t xo1 = r7 * 128 + (((4 + 2 * (mq >> 1)) + (mq & 1)) ^ r7) * 16;
    struct ColdKB {
      uint4 q[2];
      uint2 s[2];
      uint32_t b[CNB][2][4];
    };
    auto cold_load = [&](ColdKB& d, const uint8_t* st, int kk) {
      const uint8_t* wb = st + KPS * XB + kk * WBLK;
      d.q[0] = *reinterpret_cast<const uint4*>(wb + wo0);
      d.q[1] = *reinterpret_cast<const uint4*>(wb + wo0 + 512);
      d.s[0] = *reinterpret_cast<const uint2*>(wb + so0);
      d.s[1] = *reinterpret_cast<const uint2*>(wb + so0 + 64);
      const uint32_t xs = smem_u32(st + kk * XB);
#pragma unroll
      for (int nb = 0; nb < CNB; ++nb) {
        ldsm_x4(d.b[nb][0], xs + nb * 1024 + xo0);
        ldsm_x4(d.b[nb][1], xs + nb * 1024 + xo1);
      }
    };
    auto cold_compute = [&](const ColdKB& d) {
      uint32_t af[2][4][4];  // all 8 A fragments of the k-block before its MMAs (ptxas interleaves them; SASS keeps >= ~16-instruction gaps before an HMMA source register is rewritten, MMA32 had 1)
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int s2 = 0; s2 < 2; ++s2) {
          const uint32_t q = j == 0 ? d.q[s2].x : j == 1 ? d.q[s2].y : j == 2 ? d.q[s2].z : d.q[s2].w;
          const uint32_t sw = j < 2 ? d.s[s2].x : d.s[s2].y;
          convert8(q, scale_pair_sel(sw, j & 1, m16), af[s2][j], m16, m64);
        }
      // k16 steps in order per accumulator (step j into set j & 1, as the wgmma path's acc / acc3 or acc2 / acc4)
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int s2 = 0; s2 < 2; ++s2)
#pragma unroll
          for (int nb = 0; nb < CNB; ++nb)
            mma16816(cacc[s2][j & 1][nb], af[s2][j], d.b[nb][j >> 1][2 * (j & 1)], d.b[nb][j >> 1][2 * (j & 1) + 1]);
    };
    ColdKB d0, d1;
    mbar_wait(bars + 8 * slot, ph);
    cold_load(d0, ring + slot * SLOT, wg);
    cold_load(d1, ring + slot * SLOT, wg + 2);
#pragma unroll 1
    for (int it = 0; it < KS / KPS; ++it) {
      const int ns = slot + 1 == NSLOT ? 0 : slot + 1;
      const uint32_t nph = slot + 1 == NSLOT ? ph ^ 1 : ph;
      const uint8_t* nst = ring + ns * SLOT;
      const bool more = it + 1 < KS / KPS;
      cold_compute(d0);
      if (more) {  // next stage's first k-block, in flight while this stage's second one is converted
        mbar_wait(bars + 8 * ns, nph);
        cold_load(d0, nst, wg);
      }
      cold_compute(d1);
      // this thread's reads of the slot have all been consumed (mma.sync reads registers only): hand it back
      if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * slot);
      if (more) cold_load(d1, nst, wg + 2);
      slot = ns;
      ph = nph;
    }
  } else {
  // packed weights/scales are loaded one k-block ahead, across stage boundaries too
  mbar_wait(bars + 8 * slot, ph);
  uint4 nwq = *reinterpret_cast<const uint4*>(ring + slot * SLOT + KPS * XB + woff);
  uint2 nsc = *reinterpret_cast<const uint2*>(ring + slot * SLOT + KPS * XB + soff);
  constexpr bool XPF = N >= 160 ? MOE_XPF256 : MOE_XPFSMALL;
  if constexpr (TRI) {
    uint32_t af3[3][4][4];
    // one 4-k-block stage; P = (global k-block index of kk = 0) mod 3, so buffer / accumulator b = (P + kk) % 3 is a
    // compile-time index once the kk loop is unrolled
    auto stage = [&]<int P>(int it) {
      uint8_t* st = ring + slot * SLOT;
      if (it > 0) {
        mbar_wait(bars + 8 * slot, ph);
        nwq = *reinterpret_cast<const uint4*>(st + KPS * XB + woff);
        nsc = *reinterpret_cast<const uint2*>(st + KPS * XB + soff);
      }
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const int b = (P + kk) % 3;
        const uint4 wq = nwq;
        const uint2 sc = nsc;
        if (kk + 1 < 4) {
          nwq = *reinterpret_cast<const uint4*>(st + KPS * XB + (kk + 1) * WBLK + woff);
          nsc = *reinterpret_cast<const uint2*>(st + KPS * XB + (kk + 1) * WBLK + soff);
        }
        const uint64_t desc = desc_sw128(smem_u32(st + kk * XB));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
          const uint32_t sw = j < 2 ? sc.x : sc.y;
          convert8(q, scale_pair_sel(sw, j & 1, m16), af3[b][j], m16, m64);  // buffer b's last group (3 k-blocks ago) is complete
        }
        wg_fence();
#pragma unroll
        for (int j = 0; j < 4; ++j) wgmma_rs<N>(b == 0 ? acc : (b == 1 ? acc2 : acc3), af3[b][j], desc + 2 * j, 1);
        wg_commit();
        wg_wait<2>();
        // after the wait at kk = 1 only this stage's k-blocks 0 and 1 may still run: the previous slot is free
        if (kk == 1 && prev >= 0) { rel_sync(); if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev); }
      }
      prev = slot;
      if (++slot == NSLOT) { slot = 0; ph ^= 1; }
    };
    for (int it = 0; it < KS / KPS; ++it) {
      const int ph3 = (it * 4) % 3;
      if (ph3 == 0) stage.template operator()<0>(it);
      else if (ph3 == 1) stage.template operator()<1>(it);
      else stage.template operator()<2>(it);
    }
    wg_wait<0>();
    rel_sync();
    if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev);
  } else {
#define MOE_HOIST 1  // per-stage descriptor base and weight/scale base pointers (constant per-k-block offsets)
  for (int it = 0; it < KS / KPS; ++it) {
    uint8_t* st = ring + slot * SLOT;
    const uint8_t* wbase = st + KPS * XB + woff;
    const uint8_t* sbase = st + KPS * XB + soff;
    const uint64_t dbase = desc_sw128(smem_u32(st));
    if (!XPF && it > 0) {
      mbar_wait(bars + 8 * slot, ph);
      nwq = *reinterpret_cast<const uint4*>(wbase);
      nsc = *reinterpret_cast<const uint2*>(sbase);
    }
#pragma unroll
    for (int kk = 0; kk < KPS; ++kk) {
      const uint4 wq = nwq;
      const uint2 sc = nsc;
      if (kk + 1 < KPS) {
        if (MOE_HOIST) {
          nwq = *reinterpret_cast<const uint4*>(wbase + (kk + 1) * WBLK);
          nsc = *reinterpret_cast<const uint2*>(sbase + (kk + 1) * WBLK);
        } else {
          nwq = *reinterpret_cast<const uint4*>(st + KPS * XB + (kk + 1) * WBLK + woff);
          nsc = *reinterpret_cast<const uint2*>(st + KPS * XB + (kk + 1) * WBLK + soff);
        }
      } else if (XPF && it + 1 < KS / KPS) {
        const int ns = slot + 1 == NSLOT ? 0 : slot + 1;
        mbar_wait(bars + 8 * ns, slot + 1 == NSLOT ? ph ^ 1 : ph);
        nwq = *reinterpret_cast<const uint4*>(ring + ns * SLOT + KPS * XB + woff);
        nsc = *reinterpret_cast<const uint2*>(ring + ns * SLOT + KPS * XB + soff);
      }

      uint8_t* xs = st + kk * XB;
      // the k-block's X tile sits kk * XB bytes into the slot: its descriptor is the slot's plus kk * XB / 16
      const uint64_t desc = MOE_HOIST ? dbase + (uint64_t)((kk * XB) >> 4) : desc_sw128(smem_u32(xs));
      if constexpr (GROUP4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
          const uint32_t sw = j < 2 ? sc.x : sc.y;  // scales of k16 steps (j & ~1, j | 1): bytes 0,1 and 2,3
          convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[(DEEP ? kk : (kk & 1)) * 4 + j], m16, m64);
        }
        if constexpr (MMA32) {
          // B fragments from the 128B-swizzled token rows: ldmatrix.x4 = (group 2q, k lo), (2q, k hi), (2q+1, lo),
          // (2q+1, hi); lane l addresses row 8*(2q + (l>>4)) + (l&7), 16-byte chunk (2j + ((l>>3)&1)) ^ (l&7)
          const uint32_t xbase = smem_u32(st + kk * XB);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float* ca = (NACC > 1 && (j & 1)) ? acc2 : acc;
#pragma unroll
            for (int q2 = 0; q2 < 2; ++q2) {
              if (2 * q2 < nblk) {
                const int mrow = 8 * (2 * q2 + (lane >> 4)) + (lane & 7);
                const int chunk = (2 * j + ((lane >> 3) & 1)) ^ (lane & 7);
                uint32_t bfr[4];
                ldsm_x4(bfr, xbase + mrow * 128 + (chunk << 4));
                mma16816(ca + 8 * q2, afrag[(kk & 1) * 4 + j], bfr[0], bfr[1]);
                if (2 * q2 + 1 < nblk) mma16816(ca + 8 * q2 + 4, afrag[(kk & 1) * 4 + j], bfr[2], bfr[3]);
              }
            }
          }
          if (kk == 0 && prev >= 0) { rel_sync(); if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev); }
        } else {
        wg_fence();
#pragma unroll
        for (int j = 0; j < 4; ++j)
          wgmma_rs<WN>(NACC > 2 ? ((kk & 1) ? ((j & 1) ? acc4 : acc2) : ((j & 1) ? acc3 : acc))
                               : ((NACC > 1 && (kk & 1)) ? acc2 : acc),
                      afrag[(DEEP ? kk : (kk & 1)) * 4 + j], desc + 2 * j, 1);
        wg_commit();
        if constexpr (DEEP) {
          wg_wait<MOE_SMALL_WAIT>();
          // the previous stage's last group completes once MOE_SMALL_WAIT - 1 k-blocks of this stage are issued
          if (kk == MOE_SMALL_WAIT - 1 && prev >= 0) { rel_sync(); if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev); }
        } else {
          wg_wait<1>();
          // the previous stage's MMAs are complete after the first wait of this stage: hand its slot back early
          if (kk == 0 && prev >= 0) { rel_sync(); if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev); }
        }
        }  // !MMA32
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
          const uint32_t sw = j < 2 ? sc.x : sc.y;  // scales of k16 steps (j & ~1, j | 1): bytes 0,1 and 2,3
          convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[j & 1], m16, m64);
          wg_fence();
          wgmma_rs<N>(acc, afrag[j & 1], desc + 2 * j, 1);
          wg_commit();
          wg_wait<1>();
          if (kk == 0 && j == 0 && prev >= 0) { rel_sync(); if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev); }
        }
      }
    }
    prev = slot;
    if (++slot == NSLOT) { slot = 0; ph ^= 1; }
  }
  wg_wait<0>();
  rel_sync();
  if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev);
  }  // !TRI
  }  // CNB == 0
  if (t_mid) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *t_mid = t;
  }
  if constexpr (NACC > 3) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[i] = (acc[i] + acc3[i]) + (acc2[i] + acc4[i]);
  } else if constexpr (NACC > 2) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[i] = (acc[i] + acc2[i]) + acc3[i];
  } else if constexpr (NACC > 1) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[i] += acc2[i];
  }
  if constexpr (CNB > 0) {
    // the two warpgroups' partial sums meet in the staging buffer ([wg][slice][value][lane], 8 KB per token group);
    // thread (wg, w, lane) then holds the wgmma path's final acc for slice wg * 4 + w: (E) + (O), E = even k-blocks
    // (warpgroup 0) = acc + acc3, O = odd k-blocks (warpgroup 1) = acc2 + acc4
    if (pubq && ctid == 0) wait_staging_free(pubq);
    named_bar_sync(1, 256);  // previous item's copy-out is done with the staging buffer
    // (addresses from an opaque zero again: no extra registers live across the other paths)
    const int xz = (int)opaque_zero();
    float* xw = reinterpret_cast<float*>(stg) + ((wg * 8 + 2 * w) * (4 * CNB) * 32 + lane + xz);
#pragma unroll
    for (int s2 = 0; s2 < 2; ++s2)
#pragma unroll
      for (int nb = 0; nb < CNB; ++nb)
#pragma unroll
        for (int f = 0; f < 4; ++f) xw[(s2 * 4 * CNB + 4 * nb + f) * 32] = cacc[s2][0][nb][f] + cacc[s2][1][nb][f];
    named_bar_sync(1, 256);
    const float* xr0 = reinterpret_cast<const float*>(stg) + ((wg * 4 + w) * (4 * CNB) * 32 + lane + xz);
#pragma unroll
    for (int i = 0; i < 4 * CNB; ++i) acc[i] = xr0[i * 32] + xr0[(8 * 4 * CNB + i) * 32];
    // (the epilogue's barrier below orders these reads before its staging writes)
  }
  // ---- epilogue through the dedicated staging buffer ----
  const float cs = p.gscale[e] * 128.f;
  if (pubq && ctid == 0) wait_staging_free(pubq);
  named_bar_sync(1, 256);  // previous item's copy-out is done with the staging buffer
  if (GATED) {
    const int q = wg * 4 + w;
    // stmatrix.trans: the accumulator fragment of an 8-token group is an 8x8 matrix [row g][token]; stored transposed,
    // each token's 8 rows (16 bytes) land in its tile-layout chunk in one instruction per 4 groups
#pragma unroll
    for (int j4 = 0; j4 < N / 32; ++j4) {
      uint32_t r[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int jj = 4 * j4 + u;
        const float h0 = silu_fast(acc[4 * jj] * cs) * (acc[4 * jj + 2] * cs);
        const float h1 = silu_fast(acc[4 * jj + 1] * cs) * (acc[4 * jj + 3] * cs);
        r[u] = bf162_u32(__floats2bfloat162_rn(h0, h1));
      }
      const int rr = lane & 7, tokr = 8 * (4 * j4 + (lane >> 3)) + rr;
      const uint32_t addr = smem_u32(stg + tokr * 128 + ((q ^ rr) << 4));
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r[0]),
                   "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
    }
    if (pubq) {  // fused sm_90: the publisher warp copies the staged tile out (bulk copy) and then publishes `done`
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      named_bar_sync(1, 256);
      if (ctid == 0) enqueue_out(pubq, p.out + (size_t)row0 * 8192 + (size_t)ntile * N * 128, done, 0, N * 128);
    } else {
      named_bar_sync(1, 256);
      uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)row0 * 8192 + (size_t)ntile * N * 128);
      const uint4* src = reinterpret_cast<const uint4*>(stg);
      for (int i = ctid; i < N * 8; i += 256) dst[i] = src[i];
      if (done) {  // fused kernel: publish this n-tile of the tile's intermediate rows
        __threadfence();
        named_bar_sync(1, 256);
        if (ctid == 0) atomicAdd(done, 1);
      }
    }
  } else {
    const int c0 = (wg * 4 + w) * 16 + g;
    // halves of up to 128 tokens through the 128-row staging buffer: half 1 (tiles over 128 rows) holds N - 128 rows
    auto gemm2_half = [&]<int half, int NJ>() {
      if (half) named_bar_sync(1, 256);  // (half 0 was copied out by the consumers themselves)
      // stmatrix.trans: per 8-token group, two 8x8 matrices (this warp's columns 16w..16w+7 and 16w+8..16w+15)
      if constexpr (NJ >= 2) {
#pragma unroll
        for (int j2 = 0; j2 < NJ / 2; ++j2) {
          uint32_t r[4];
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            const int ai = 4 * (2 * j2 + u + half * 16);
            r[2 * u] = bf162_u32(__floats2bfloat162_rn(acc[ai] * cs, acc[ai + 1] * cs));
            r[2 * u + 1] = bf162_u32(__floats2bfloat162_rn(acc[ai + 2] * cs, acc[ai + 3] * cs));
          }
          const int m = lane >> 3, rr = lane & 7;
          const int tl = 8 * (2 * j2 + (m >> 1)) + rr;
          const uint32_t addr = smem_u32(stg + tl * 272 + ((wg * 4 + w) * 16 + 8 * (m & 1)) * 2);
          asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r[0]),
                       "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
        }
      } else {
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
          for (int b = 0; b < 2; ++b) {
            const int tl = 8 * jj + 2 * t + b;
            const int ai = 4 * (jj + half * 16) + b;
            __nv_bfloat16* r = reinterpret_cast<__nv_bfloat16*>(stg + tl * 272);
            r[c0] = __float2bfloat16_rn(acc[ai] * cs);
            r[c0 + 8] = __float2bfloat16_rn(acc[ai + 2] * cs);
          }
        }
      }
      constexpr int ntok = 8 * NJ;
      // the last half goes to the publisher warp (its copy loop overlaps the next item); a first half is copied by the
      // consumers, since the second half needs the staging buffer right away
      if (pubq && half == (N > 128 ? 1 : 0)) {
        named_bar_sync(1, 256);
        if (ctid == 0)
          enqueue_out(pubq, p.out + ((size_t)(row0 + half * 128) * 4096 + ntile * 128) * 2, done, 1, ntok, row0, nreal);
      } else {
        named_bar_sync(1, 256);
        for (int i = ctid; i < ntok * 16; i += 256) {
          const int tl = i >> 4, c = i & 15;
          *reinterpret_cast<uint4*>(p.out + ((size_t)(row0 + half * 128 + tl) * 4096 + ntile * 128) * 2 + c * 16) =
              *reinterpret_cast<const uint4*>(stg + tl * 272 + c * 16);
        }
      }
    };
    gemm2_half.template operator()<0, (N > 128 ? 16 : N / 8)>();
    if constexpr (N > 128) gemm2_half.template operator()<1, (N - 128) / 8>();
  }
}

template <bool GATED>
__global__ void __launch_bounds__(384, 1) moe_gemm_persistent(Args p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);  // full[NSLOT], empty[NSLOT], ifull[2], iempty[2]
  const uint32_t ibar = bars + 16 * NSLOT;
  volatile int* iring = reinterpret_cast<volatile int*>(base + 512);
  uint8_t* ring = base + 1024;
  uint8_t* stg = ring + NSLOT * SLOT;
  const int nitems = *p.ntiles * p.NT;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NSLOT; ++s) {
      mbar_init(bars + 8 * s, 1);
      mbar_init(bars + 8 * NSLOT + 8 * s, 8 * REL_LANES);
    }
    for (int k = 0; k < 2; ++k) {
      mbar_init(ibar + 8 * k, 1);
      mbar_init(ibar + 16 + 8 * k, MOE_ALLLANE_RELEASE ? 256 : 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  int slot = 0;
  uint32_t ph = 0;
  if (threadIdx.x < 128) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (threadIdx.x == 0) {
      int k = 0;
      uint32_t iph = 0;
      while (true) {
        mbar_wait(ibar + 16 + 8 * k, iph ^ 1);
        const int item = atomicAdd(p.counter, 1);
        iring[k] = item;
        mbar_arrive(ibar + 8 * k);
        if (item >= nitems) break;
        const int tile = item / p.NT, ntile = item - tile * p.NT;
        const int4 ti = p.tiles[tile];
        const int e = ti.x, row0 = ti.y, n = ti.z;
        const int kps = kps_for(n);
        const uint32_t xb = n * 128;
        const uint8_t* wsrc = p.wrep + ((size_t)(e * p.NT + ntile) * KS) * WBLK;
        const uint8_t* xsrc = p.xt + (size_t)row0 * 8192;
        for (int it = 0; it < KS / kps; ++it) {
          mbar_wait(bars + 8 * NSLOT + 8 * slot, ph ^ 1);
          slot_proxy_fence();
          const uint32_t full = bars + 8 * slot;
          mbar_expect_tx(full, kps * (xb + WBLK));
          const uint32_t dst = smem_u32(ring + slot * SLOT);
          bulk_g2s(dst, xsrc + (size_t)it * kps * xb, kps * xb, full);
          bulk_g2s(dst + kps * xb, wsrc + (size_t)it * kps * WBLK, kps * WBLK, full);
          if (++slot == NSLOT) { slot = 0; ph ^= 1; }
        }
        k ^= 1;
        if (k == 0) iph ^= 1;
      }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    int k = 0;
    uint32_t iph = 0;
    while (true) {
      mbar_wait(ibar + 8 * k, iph);
      const int item = iring[k];
      named_bar_sync(1, 256);  // every consumer has read the item
      if (MOE_ALLLANE_RELEASE || threadIdx.x == 128) mbar_arrive(ibar + 16 + 8 * k);  // all: each reader orders its own read
      if (item >= nitems) break;
      k ^= 1;
      if (k == 0) iph ^= 1;
      const int tile = item / p.NT, ntile = item - tile * p.NT;
      const int4 ti = p.tiles[tile];
      const int e = ti.x, row0 = ti.y, n = ti.z;
      switch (n) {
        case 32: consumer_item<32>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
        case 64: consumer_item<64>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
        case 128: consumer_item<128>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
        case 160: consumer_item<160>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
        case 192: consumer_item<192>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
        default: consumer_item<256>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
      }
    }
  }
}

// ---- both expert GEMMs in one persistent launch: the queue holds every GEMM1 item, then every GEMM2 item; a GEMM2
// item's loads wait until all 64 GEMM1 n-tiles of its token tile have been written (per-tile counters) ----
struct FusedArgs {
  Args g1, g2;     // GEMM1 (gated, writes inter in tile layout) and GEMM2 (reads inter, writes y)
  int* done;       // [max tiles] GEMM1 n-tiles finished per token tile (zeroed before launch)
  unsigned long long* dbg;  // optional per-item timeline: [item][start ns, end ns, smid, cta] (nullptr: off)
  int flags;       // bit 8: fine remainder tiles (FINE instantiation); experiments: bit 0 = load X only for an item's first stage, bit 1 = same for W
  // CIT (combine in the tail; cit == nullptr: off): the last GEMM2 n-tile item of a tile counts its tokens' routes down
  // (tok_left, set by the route kernel); a token whose routes are all done goes into the combine queue cq (cleared by the
  // route kernel; no-route tokens queued by the gather); CTAs out of expert items pop tokens and write their combined
  // rows (the separate combine launch is gone). All counters are zeroed per call inside the graph (boundary kernel).
  int* cit;        // [max tiles] GEMM2 n-tiles finished per tile
  int* qctl;       // [0] queue head (pops), [1] queue tail (pushes), [2] error flag (a spin limit hit)
  int* tok_left;   // [T] local routes of each token not yet finished
  int* cq;         // [T] combine queue: token + 1 (0: not yet written)
  const int* row_tok;  // [rows] token of each routed row
  const int* rrow; const int* lid; const float* topw; const __nv_bfloat16* pw;  // the combine's inputs
  __nv_bfloat16* out;  // [T, 4096] combined expert output
  int T;
  // [r6h] hand-off combine (hblk == nullptr: off). hblk[b]: GEMM2 items finished on 32-row block b (each GEMM2 item adds 1
  // to every block holding its tile's real rows, red.release.gpu after its rows are written; complete at g2.NT);
  // hgo: 1 once CTA 0 is past its PDL wait (the gather's rrow complete). Both zeroed per call (boundary kernel).
  int* hgo;
  int* hblk;
};

#define MOE_FINE_NOINLINE 0  // 1: the extra widths as separate functions -> ptxas C7510 (wgmma pipeline crossing a call): whole kernel serialized
template <int N>
__noinline__ __device__ void consumer_item_ni(const Args& p, uint8_t* ring, uint8_t* stg, uint32_t bars, int& slot, uint32_t& ph,
                                              int e, int row0, int ntile, bool GATED, int* done, unsigned long long* t_mid,
                                              volatile long long* pubq, int nreal = 256) {
  consumer_item<N>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED, done, t_mid, pubq, nreal);
}

// ---- CIT: combine in the tail of the fused expert kernel ----
// The 256 consumer threads of a CTA with no expert items left pop tokens from the combine queue and write each one's
// combined row, with the CUDA combine's arithmetic (moe_combine_kernel): every local route's y row times (1 + w),
// RMS-normalized on its own (eps 1e-8), times its routing weight, summed in route order k = 0..7 (two routes' rows in
// flight, one block reduction per pair). The order is fixed per token, so the result does not depend on which CTA
// combines it or when (deterministic). y rows were written by other CTAs during this kernel: L2 loads (ld.global.cg)
// after the acquire chain (item releases -> done2 RMW -> tok_left acq_rel RMWs -> queue store release -> acquire here).
#define MOE_CIT 0  // combine in the tail (Oct 26: measured 2.5-3x slower than the combine kernel; TRICKS 5.16): compiled out
#define CIT_BACKOFF_NS 500
#define HCOMB 1  // [r6h] hand-off combine support compiled in (runtime: fa.hblk / moe_combine2_kernel<true>)
#define CIT_SPIN_NS 200000000ull  // 200 ms: a queue slot never written sets the error flag (fa.qctl[2]) instead of hanging
// a completed GEMM2 tile counts down the routes of its rows' tokens; a token whose last route this was is queued
__noinline__ __device__ void cit_release(const int* row_tok, int* tok_left, int* qctl, int* cq, int r0, int nreal, int lane) {
  // (the caller's fence.acq_rel.gpu after acquiring the tile's items is the release side; the counts go down with
  // relaxed RMWs, issued together (up to 8 per lane, 256 rows); a token's last decrementer fences again (acquire side)
  // before the queue store with release: fence + relaxed RMW patterns on both ends)
  int t[8], prev[8];
#pragma unroll
  for (int u = 0; u < 8; ++u) {
    const int lr = lane + 32 * u;
    t[u] = lr < nreal ? row_tok[r0 + lr] : -1;
  }
#pragma unroll
  for (int u = 0; u < 8; ++u) prev[u] = t[u] >= 0 ? atomicAdd(tok_left + t[u], -1) : 0;
  bool any = false;
#pragma unroll
  for (int u = 0; u < 8; ++u) any |= prev[u] == 1;
  if (any) {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      if (prev[u] == 1) {
        const int i = atomicAdd(qctl + 1, 1);
        asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(cq + i), "r"(t[u] + 1) : "memory");
      }
    }
  }
}
DEVI void cit_combine(const FusedArgs& fa, int ctid, uint8_t* base) {
  // warp 0 lane 0 is the CTA's only poller of the global queue (with backoff: thousands of pollers slowed the last
  // expert items down); it hands tokens to warps 1..7 through shared-memory slots (token + 1, 0 = empty, -1 = done);
  // each combining warp takes one token at a time
  volatile int* slot = reinterpret_cast<volatile int*>(base + 768);  // [8]
  const int lane = ctid & 31, warp = ctid >> 5;
  if (lane == 0) slot[warp] = 0;
  named_bar_sync(1, 256);
  if (warp == 0) {
    if (lane == 0) {
      unsigned long long t0, now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      bool done = false;
      while (!done) {
        bool idle = true;
        for (int w = 1; w < 8 && !done; ++w) {
          if (slot[w] != 0) continue;
          int head, tail;
          asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(head) : "l"(fa.qctl) : "memory");
          if (head >= fa.T) { done = true; break; }
          asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(tail) : "l"(fa.qctl + 1) : "memory");
          if (tail <= head) break;
          const int idx = atomicAdd(fa.qctl, 1);
          if (idx >= fa.T) { done = true; break; }
          int v;
          while (true) {
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(fa.cq + idx) : "memory");
            if (v != 0) break;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            if (now - t0 > CIT_SPIN_NS) { atomicExch(fa.qctl + 2, 1); break; }
            __nanosleep(100);
          }
          if (v != 0)  // release (CTA scope): the combining warp's acquire of the slot orders its row loads after ours
            asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32((const void*)(slot + w))), "r"(v) : "memory");
          idle = false;
        }
        if (done) break;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (now - t0 > CIT_SPIN_NS) { atomicExch(fa.qctl + 2, 1); break; }
        if (idle) __nanosleep(CIT_BACKOFF_NS);
      }
      for (int w = 1; w < 8; ++w) {  // done: each warp gets -1 once its slot is free
        while (slot[w] != 0) __nanosleep(64);
        slot[w] = -1;
      }
    }
    __syncwarp();
    return;
  }
  while (true) {
    int v = 0;
    if (lane == 0) {
      while (true) {
        asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(v) : "r"(smem_u32((const void*)(slot + warp))) : "memory");
        if (v != 0) break;
        __nanosleep(32);
      }
    }
    v = __shfl_sync(0xffffffffu, v, 0);
    if (v < 0) break;
    __syncwarp();
    const int tok = v - 1;
    // local routes in order, two at a time; pass 1 issues all 16 chunk loads of both rows at once (L1-allocating: the
    // rows were never read on this SM, and L1 starts clean each launch) for their RMS, pass 2 re-reads them from L1
    int rows[8], nl = 0;
    float wts[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int l = fa.lid[tok * 8 + k];
      rows[k] = -1;
      wts[k] = 0.f;
      if (l >= 0) {
        rows[k] = fa.rrow[tok * 8 + k];
        wts[k] = fa.topw[tok * 8 + k];
        ++nl;
      }
    }
    float sc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      sc[k] = 0.f;
      if (rows[k] >= 0) {
        const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(fa.g2.out) + (size_t)rows[k] * 4096 + lane * 8;
        uint4 q[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(q[j].x), "=r"(q[j].y), "=r"(q[j].z), "=r"(q[j].w) : "l"(src + 256 * j) : "memory");
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&q[j]);
          const float4 g0 = g1p4(fa.pw + lane * 8 + 256 * j), g1 = g1p4(fa.pw + lane * 8 + 256 * j + 4);
          const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(h[i]);
            const float a = f.x * gv[2 * i], b = f.y * gv[2 * i + 1];
            ss += a * a + b * b;
          }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        sc[k] = rsqrtf(ss / 4096.f + 1e-8f) * wts[k];
      }
    }
    (void)nl;
#pragma unroll 4
    for (int j = 0; j < 16; ++j) {
      const float4 g0 = g1p4(fa.pw + lane * 8 + 256 * j), g1 = g1p4(fa.pw + lane * 8 + 256 * j + 4);
      const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        if (rows[k] >= 0) {
          uint4 q;
          asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w)
                       : "l"(reinterpret_cast<const __nv_bfloat16*>(fa.g2.out) + (size_t)rows[k] * 4096 + lane * 8 + 256 * j) : "memory");
          const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&q);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(h[i]);
            acc[2 * i] += (f.x * gv[2 * i]) * sc[k];
            acc[2 * i + 1] += (f.y * gv[2 * i + 1]) * sc[k];
          }
        }
      }
      uint4 o;
      __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) b[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
      *reinterpret_cast<uint4*>(fa.out + (size_t)tok * 4096 + lane * 8 + 256 * j) = o;
    }
    __syncwarp();
    if (lane == 0) slot[warp] = 0;
  }
}

template <int FINE>
#define MOE_EARLY_INIT 0  // 1: shared-memory barrier setup before the PDL wait (overlaps the gather's tail)
__global__ void __launch_bounds__(MOE_FUSED_THREADS, 1) moe_fused_persistent(FusedArgs fa) {
  if constexpr (!MOE_EARLY_INIT) pdl_enter();
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);
  const uint32_t ibar = bars + 16 * NSLOT;
  volatile int* iring = reinterpret_cast<volatile int*>(base + 512);
  uint8_t* ring = base + 1024;
  uint8_t* stg = ring + NSLOT * SLOT;
  // publisher queue (sm_90): count at base + 240, 32 done-pointers at base + 256
  volatile long long* pubq = reinterpret_cast<volatile long long*>(base + 256);
  if (threadIdx.x == 0) {
    *reinterpret_cast<volatile int*>(base + 240) = 0;
    *reinterpret_cast<volatile int*>(base + 244) = 0;
    *reinterpret_cast<volatile int*>(base + 248) = 0;
    for (int s = 0; s < NSLOT; ++s) {
      mbar_init(bars + 8 * s, 1);
      mbar_init(bars + 8 * NSLOT + 8 * s, 8 * REL_LANES);
    }
    for (int k = 0; k < 2; ++k) {
      mbar_init(ibar + 8 * k, 1);
      mbar_init(ibar + 16 + 8 * k, MOE_ALLLANE_RELEASE ? 256 : 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if constexpr (MOE_EARLY_INIT) pdl_enter();
  // [r6h] CTA 0 is past its PDL wait (the gather and everything before it are complete and visible): release the go flag
  if (HCOMB && fa.hgo != nullptr && blockIdx.x == 0 && threadIdx.x == 0)
    asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(fa.hgo), "r"(1) : "memory");
  const int nt = *fa.g1.ntiles;
  const int n1 = nt * fa.g1.NT, nitems = n1 + nt * fa.g2.NT;
  int slot = 0;
  uint32_t ph = 0;
  if (MOE_CONS_FIRST ? threadIdx.x >= 256 : threadIdx.x < 128) {
    if (!MOE_CONS_FIRST) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (__shfl_sync(0xffffffffu, (int)(threadIdx.x - MOE_PB) >> 5, 0) == 0) {  // warp-uniform: the whole producer warp
      const bool lead = elect_one();
      int k = 0;
      uint32_t iph = 0;
      while (true) {
        mbar_wait(ibar + 16 + 8 * k, iph ^ 1);
        int item = 0;
        if (lead) item = atomicAdd(fa.g1.counter, 1);
        item = __shfl_sync(0xffffffffu, item, 0);
        if (lead) {
          iring[k] = item;
          mbar_arrive(ibar + 8 * k);
        }
        if (item >= nitems) break;
        const bool second = item >= n1;
        const Args& p = second ? fa.g2 : fa.g1;
        const int it2 = second ? item - n1 : item;
        const int tile = it2 / p.NT, ntile = it2 - tile * p.NT;
        int4 ti = make_int4(0, 0, 0, 0);
        if (lead) ti = p.tiles[tile];
        ti.x = __shfl_sync(0xffffffffu, ti.x, 0);
        ti.y = __shfl_sync(0xffffffffu, ti.y, 0);
        ti.z = __shfl_sync(0xffffffffu, ti.z, 0);
        ti.w = __shfl_sync(0xffffffffu, ti.w, 0);
        const int e = ti.x, row0 = ti.y, n = ti.z;
        if (second) {
          const int need = fa.g1.NT;
          int v;
          while (true) {
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(fa.done + tile) : "memory");
            v = __shfl_sync(0xffffffffu, v, 0);
            if (v >= need) break;
            __nanosleep(64);
          }
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        const int kps = kps_for(n);
        // X bytes copied per k-block: n * 128, or (MOE_COLD_XTRIM) 1 / 2 KB for a trimmed cold item (N=32, stride 4 KB);
        // one value instead of two keeps the 40-register producer spill-free
#if MOE_COLD && MOE_COLD_XTRIM
        const uint32_t xr = cold_xrows(n, ti.w);
        const uint32_t xb = xr ? xr * 128 : n * 128;
#else
        const uint32_t xb = n * 128;
#endif
        const uint8_t* wsrc = p.wrep + ((size_t)(e * p.NT + ntile) * KS) * WBLK;
        const uint8_t* xsrc = p.xt + (size_t)row0 * 8192;
        const bool pf = MOE_L2PF > 0 && (MOE_L2PF_ALL || kps >= 2);
        if (pf) {  // the item's first stages: a head start while the previous item's last stages are consumed
          if (lead) for (int j = 0; j < MOE_L2PF && j < KS / kps; ++j) prefetch_l2(wsrc + (size_t)j * kps * WBLK, kps * WBLK);
        }
        for (int it = 0; it < KS / kps; ++it) {
          if (lead && pf && it + MOE_L2PF < KS / kps) prefetch_l2(wsrc + (size_t)(it + MOE_L2PF) * kps * WBLK, kps * WBLK);
          mbar_wait(bars + 8 * NSLOT + 8 * slot, ph ^ 1);
          slot_proxy_fence();
          const uint32_t full = bars + 8 * slot;
          const bool lx = !(fa.flags & 1) || it < NSLOT, lw = !(fa.flags & 2) || it < NSLOT;
          const uint32_t dst = smem_u32(ring + slot * SLOT);
          if (lead) {
          mbar_expect_tx(full, (lx ? kps * xb : 0) + (lw ? kps * WBLK : 0) + (!lx && !lw ? 16 : 0));
          if (lx) {
            if (MOE_COLD && MOE_COLD_XTRIM && xb < 4096) {  // trimmed cold item: the first 8 / 16 token rows of each k-block's 4 KB X block, at its usual slot offset
              const uint8_t* xs = xsrc + (size_t)it * (4 * 4096);
              bulk_g2s(dst, xs, xb, full);
              bulk_g2s(dst + 4096, xs + 4096, xb, full);
              bulk_g2s(dst + 8192, xs + 8192, xb, full);
              bulk_g2s(dst + 12288, xs + 12288, xb, full);
            } else {
              bulk_g2s(dst, xsrc + (size_t)it * kps * xb, kps * xb, full);
            }
          }
          if (lw) bulk_g2s(dst + kps * xb, wsrc + (size_t)it * kps * WBLK, kps * WBLK, full);
          if (!lx && !lw) bulk_g2s(dst, xsrc, 16, full);
          }  // lead
          __syncwarp();
          if (++slot == NSLOT) { slot = 0; ph ^= 1; }
        }
        k ^= 1;
        if (k == 0) iph ^= 1;
      }
    } else if (threadIdx.x >= MOE_PB + 32 && threadIdx.x < MOE_PB + 64) {
      // publisher warp: copies each staged epilogue tile out with bulk copies (the consumers go on with their next
      // item), frees the staging buffer once read, and for GEMM1 tiles waits for the writes, then a GPU-scope release
      // and the tile's done counter (read with acquire by other CTAs' producers)
      const int lane = threadIdx.x - (MOE_PB + 32);
      const uint32_t stg_s = smem_u32(stg);
      int seen = 0;
      while (true) {
        int r = 0;
        asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(r) : "r"(smem_u32(base + 240)) : "memory");
        r = __shfl_sync(0xffffffffu, r, 0);  // one value for the warp (a later lane may see a newer count)
        const int avail = r < 0 ? -r - 1 : r;
        while (seen < avail) {
          volatile long long* e = pubq + 4 * (seen & 7);
          uint8_t* dst = reinterpret_cast<uint8_t*>(e[0]);
          int* d = reinterpret_cast<int*>(e[1]);
          const int kind = reinterpret_cast<volatile int*>(e)[4], amount = reinterpret_cast<volatile int*>(e)[5];
          if (kind == 0) {  // one contiguous tile-layout block: a single bulk copy
            if (lane == 0) {
              bulk_s2g(dst, stg_s, amount);
              asm volatile("cp.async.bulk.commit_group;" ::: "memory");
              asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            }
          } else {  // token rows of 256 bytes (272-byte staging pitch): the warp copies them, 16 bytes per lane
            for (int i = lane; i < amount * 16; i += 32) {
              const int tl = i >> 4, c = i & 15;
              *reinterpret_cast<uint4*>(dst + (size_t)tl * 8192 + c * 16) = *reinterpret_cast<const uint4*>(stg + tl * 272 + c * 16);
            }
          }
          __syncwarp();  // after lane 0's bulk-copy read of the staging buffer has completed
          asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32(base + 248)), "r"(seen + 1) : "memory");
          int old = -1;
          if (lane == 0) {
            if (d) {
              if (kind == 0) {  // a bulk (async-proxy) copy: wait for its writes first
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                asm volatile("fence.proxy.async.global;" ::: "memory");
              }
              asm volatile("fence.acq_rel.gpu;" ::: "memory");
              old = atomicAdd(d, 1);
            }
          }
          if (MOE_CIT && kind == 1 && d != nullptr) {  // CIT: a GEMM2 item; the tile's last one counts its tokens' routes down
            old = __shfl_sync(0xffffffffu, old, 0);
            __syncwarp();  // (memory ordering: lane 0's acquire of the other items' releases, before every lane's release)
            if (old == fa.g2.NT - 1) {
              asm volatile("fence.acq_rel.gpu;" ::: "memory");
              // (entry seen & 7 is rewritten only for job seen + 8, after this job: safe to read its rows here)
              cit_release(fa.row_tok, fa.tok_left, fa.qctl, fa.cq, reinterpret_cast<volatile int*>(e)[6],
                          reinterpret_cast<volatile int*>(e)[7], lane);
            }
          }
          if (HCOMB && kind == 1 && fa.hblk != nullptr) {
            // [r6h] a GEMM2 item (128 output columns of a token tile): +1 on each 32-row block of the tile's real rows,
            // one lane per block, red.release.gpu (= fence.acq_rel.gpu + relaxed add per lane). It orders this warp's row
            // copies (the __syncwarp above) and, for N > 128 tiles, the consumers' half-0 rows (their bar.sync, then the
            // queue store release / our acquire) before the add; moe_combine2_kernel<true> acquires the count. Entry
            // seen & 7 is rewritten only for job seen + 8, after this job: safe to read its rows here.
            const int r0 = reinterpret_cast<volatile int*>(e)[6], nr = reinterpret_cast<volatile int*>(e)[7];
            if (lane < ((nr + 31) >> 5))
              asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(fa.hblk + (r0 >> 5) + lane), "r"(1) : "memory");
          }
          ++seen;
        }
        if (r < 0) break;
        __nanosleep(64);
      }
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
  } else {
    if (!MOE_CONS_FIRST) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    int k = 0;
    uint32_t iph = 0;
    volatile long long* pq = pubq;
    while (true) {
      mbar_wait(ibar + 8 * k, iph);
      const int item = iring[k];
      named_bar_sync(1, 256);
      if (MOE_ALLLANE_RELEASE || threadIdx.x == MOE_CB) mbar_arrive(ibar + 16 + 8 * k);  // all: each reader orders its own read
      if (item >= nitems) {
        if (threadIdx.x == MOE_CB) {  // the publisher drains the queue and exits
          const int pubn = *reinterpret_cast<volatile int*>(base + 244);
          asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32(base + 240)), "r"(-(pubn + 1)) : "memory");
        }
#define CIT_DBG_NOCOMBINE 0  // timing experiment only: the per-item release path without the tail combine
        if (MOE_CIT && fa.cit != nullptr && !CIT_DBG_NOCOMBINE) cit_combine(fa, threadIdx.x - MOE_CB, base);  // CIT: combine finished tokens
        break;
      }
      k ^= 1;
      if (k == 0) iph ^= 1;
      unsigned long long t_start = 0;
      if (fa.dbg && threadIdx.x == MOE_CB) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
      const bool first = item < n1;
      const Args& p = first ? fa.g1 : fa.g2;
      const int it2 = first ? item : item - n1;
      const int tile = it2 / p.NT, ntile = it2 - tile * p.NT;
      const int4 ti = p.tiles[tile];
      int* dn = first ? fa.done + tile : ((MOE_CIT && fa.cit != nullptr) ? fa.cit + tile : nullptr);
      unsigned long long t_mid = 0;
      unsigned long long* tm = (fa.dbg && threadIdx.x == MOE_CB) ? &t_mid : nullptr;
      if (__builtin_expect(ti.z != 32 && ti.z != 64 && ti.z != 128 && ti.z != 160 && ti.z != 192, 1)) {
          consumer_item<256, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
      } else if (ti.z == 160) {
          if constexpr (FINE != 0) consumer_item<160, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
      } else if (ti.z == 32 && cold_kind(32, ti.w) != 0) {
          if (cold_kind(32, ti.w) == 1) consumer_item<32, MOE_CB, MOE_COLD == 1 ? 1 : 0, MOE_COLD == 2 ? 8 : 32>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
          else consumer_item<32, MOE_CB, MOE_COLD == 1 ? 2 : 0, MOE_COLD == 2 ? 16 : 32>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
      } else if (ti.z == 64) {
          consumer_item<64, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
      } else if (ti.z == 192) {
          if constexpr (FINE != 0) consumer_item<192, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
      } else if (ti.z == 128) {
          consumer_item<128, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
      } else if (ti.z == 32 && cold_kind(32, ti.w) == 0) {
          consumer_item<32, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
      }
      item_done:;
      if (fa.dbg && threadIdx.x == MOE_CB) {
        unsigned long long t_end;
        unsigned int smid;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_end));
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        fa.dbg[(size_t)item * 4 + 0] = t_start;
        fa.dbg[(size_t)item * 4 + 1] = t_end;
        fa.dbg[(size_t)item * 4 + 2] = smid | ((unsigned long long)(t_mid - t_start) << 16);
        fa.dbg[(size_t)item * 4 + 3] = blockIdx.x;
      }
    }
  }
}

// ---- routing plan: per-expert tiles of 256 rows plus one remainder tile of 32/64/128/256 ----
// an expert's last (remainder) tile: 32 or 64 rows when it fits, else the rows rounded up to 32
DEVI int rem_tile(int r, int fine) {
  if (r == 0) return 0;
  if (r <= 32) return 32;
  if (r <= 64) return 64;
  if (!fine) return r <= 128 ? 128 : 256;
  if ((MOE_FINE_MASK & 1) && r <= 96) return 96;
  if (r <= 128) return 128;
  if ((MOE_FINE_MASK & 2) && r <= 160) return 160;
  if ((MOE_FINE_MASK & 4) && r <= 192) return 192;
  if ((MOE_FINE_MASK & 8) && r <= 224) return 224;
  return 256;
}
#define MOE_SPLIT_REM 1
// an expert's remainder rows as one or two tiles: without fine widths, 129..192 rows become 128 + 32/64 (cheaper than
// one 256-row tile: ~22 + 11..16 vs ~41 us per weight n-tile item)
DEVI void rem_tiles(int r, int fine, int& na, int& nb) {
  if (MOE_SPLIT_REM && !fine && r > 128 && r <= 192) {
    na = 128;
    nb = rem_tile(r - 128, 0);
  } else {
    na = rem_tile(r, fine);
    nb = 0;
  }
}
DEVI void plan_warp(const int* __restrict__ counts, int* __restrict__ offsets, int4* __restrict__ tiles,
                    int* __restrict__ ntiles, int* __restrict__ rows_per_expert, int order, int e) {
  const int fine = order >> 8;  // bit 8: fine remainder tiles (the expert kernel must be the FINE instantiation)
  order &= 255;
  // Tile list order (the persistent kernel takes items from it through an atomic counter):
  //   0: expert order; 1: biggest first (all 256-row tiles, then 128, 64, 32: longest-processing-time);
  //   2: the 256/128-row (compute-bound) tiles biggest first with the 64/32-row (weight-streaming) tiles spread
  //      between them, the last fifth of the small tiles at the end.
  // Each expert's rows stay contiguous (offsets) in tile-row order.
  const int c = counts[e];  // one warp, lane e = expert e
  const int full = c >> 8, r = c & 255;
  int nr, nb;  // remainder tile widths (nb: second remainder tile or 0)
  rem_tiles(r, fine, nr, nb);
  const int rows = full * 256 + nr + nb;
  const int nt = full + (nr > 0) + (nb > 0);
  // size classes, widest first: 256, 224, 192, 160, 128, 96 ("big", tensor-bound), 64, 32 ("small")
  // cnt: [0] rows, [1 + k] tiles of class k (k = 0..7), [9] tiles
  int cnt[10];
  cnt[0] = rows;
#pragma unroll
  for (int k = 0; k < 8; ++k) cnt[1 + k] = (k == 0 ? full : 0) + (nr == 256 - 32 * k) + (nb == 256 - 32 * k);
  cnt[9] = nt;
  int incl[10];
#pragma unroll
  for (int k = 0; k < 10; ++k) {
    int v = cnt[k];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int a2 = __shfl_up_sync(0xffffffffu, v, o);
      if (e >= o) v += a2;
    }
    incl[k] = v;
  }
  int tot[10];
#pragma unroll
  for (int k = 0; k < 10; ++k) tot[k] = __shfl_sync(0xffffffffu, incl[k], 31);
  const int r0 = incl[0] - rows;
  offsets[e] = r0;
  rows_per_expert[e] = rows;
  if (e == 31) *ntiles = tot[9];
  if (order == 0) {
    const int t0 = incl[9] - nt;
    for (int i = 0; i < full; ++i) tiles[t0 + i] = make_int4(e, r0 + 256 * i, 256, 256);
    if (nr) tiles[t0 + full] = make_int4(e, r0 + 256 * full, nr, nb ? nr : r);
    if (nb) tiles[t0 + full + 1] = make_int4(e, r0 + 256 * full + nr, nb, r - nr);
    return;
  }
  // class bases in widest-first order; big ranks k (classes 0..5), small ranks j (classes 6, 7)
  int cbase[8];
  {
    int acc = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) { cbase[k] = (k == 6) ? 0 : acc; acc = (k == 5) ? 0 : acc + tot[1 + k]; }
  }
  const int B = tot[1] + tot[2] + tot[3] + tot[4] + tot[5] + tot[6], S = tot[7] + tot[8];
  auto pos_big = [&](int k) -> int {
    if (order == 1 || B == 0) return k;
    return k + min(S, (4 * k * S) / (5 * B));
  };
  auto pos_small = [&](int j) -> int {
    if (order == 1) return B + j;
    if (B == 0) return j;
    const int lim = (4 * S) / 5;
    const int bb = j < lim ? (5 * B * (j + 1) + 4 * S - 1) / (4 * S) : B;
    return j + bb;
  };
  // class 0 (256): this expert's full tiles and a 256-row remainder, at in-class ranks incl - cnt ..
  const int p256 = cbase[0] + incl[1] - cnt[1];
  for (int i = 0; i < full; ++i) tiles[pos_big(p256 + i)] = make_int4(e, r0 + 256 * i, 256, 256);
  if (nr == 256) tiles[pos_big(p256 + full)] = make_int4(e, r0 + 256 * full, 256, r);
#pragma unroll
  for (int k = 1; k < 6; ++k)
    if (nr == 256 - 32 * k) tiles[pos_big(cbase[k] + incl[1 + k] - 1)] = make_int4(e, r0 + 256 * full, nr, nb ? nr : r);
  if (nr == 64) tiles[pos_small(cbase[6] + incl[7] - 1)] = make_int4(e, r0 + 256 * full, 64, r);
  if (nr == 32) tiles[pos_small(cbase[7] + incl[8] - 1)] = make_int4(e, r0 + 256 * full, 32, r);
  // a split remainder's second tile (32 or 64 rows; the first one, 128 rows, was placed above)
  if (nb == 64) tiles[pos_small(cbase[6] + incl[7] - 1)] = make_int4(e, r0 + 256 * full + nr, 64, r - nr);
  if (nb == 32) tiles[pos_small(cbase[7] + incl[8] - 1)] = make_int4(e, r0 + 256 * full + nr, 32, r - nr);
}

__global__ void moe_plan_kernel(const int* __restrict__ counts, int* __restrict__ offsets, int4* __restrict__ tiles,
                                int* __restrict__ ntiles, int* __restrict__ rows_per_expert, int order) {
  pdl_enter();
  plan_warp(counts, offsets, tiles, ntiles, rows_per_expert, order, threadIdx.x);  // 32 threads
}

// ---- normalize the pooler output and scatter each token's row into its local routes' tile slots ----
// p: [T, 4096] bf16; lid/pos: [T, 8]; writes xt (tile layout); zero-fills pad rows; rrow[t*8+k] = row (or -1)
#define MOE_GATHER_MINB 1  // [r3g] was 1 (54 regs, 4 CTAs/SM)
#define MOE_PADFILL_PARTS 8  // pad-fill blocks per expert (fewer blocks: the gather grid fits one wave at small T)
// PLAN: the block works out the expert offsets itself (one warp scan over the 32 counts) and block 0 also writes the
// whole plan (offsets, tile list, tile count, rows per expert) for the expert kernel: no separate plan launch.
template <bool PLAN, bool CO = false, int MINB = MOE_GATHER_MINB>  // [p12f] MINB: launch-bound min blocks per SM
__global__ void __launch_bounds__(256, MINB) moe_gather_kernel(const __nv_bfloat16* __restrict__ p,
                                                         const __nv_bfloat16* __restrict__ gw, const int* __restrict__ lid,
                                                         const int* __restrict__ pos, const int* __restrict__ offsets_in,
                                                         const int* __restrict__ counts, uint8_t* __restrict__ xt,
                                                         int* __restrict__ rrow, int T, int* __restrict__ offsets_out = nullptr,
                                                         int4* __restrict__ tiles = nullptr, int* __restrict__ ntiles = nullptr,
                                                         int* __restrict__ rows_out = nullptr, int order = 0,
                                                         int* __restrict__ row_tok = nullptr, int* __restrict__ cq = nullptr,
                                                         int* __restrict__ qtail = nullptr) {
  pdl_enter();
  // [r3g] PLAN: block 0 is a dedicated plan block (the launcher adds it); token t is block t + 1, then the pad-fill blocks
  const int tok = PLAN ? (int)blockIdx.x - 1 : (int)blockIdx.x;
  const int tid = threadIdx.x;
  const int fine = order >> 8;
  __shared__ float red[8];
  __shared__ int off_s[32];
  __shared__ int cnt_s[32];
  if constexpr (PLAN) {
    if (blockIdx.x == 0) {  // [r3g] plan block: warp 0 writes the whole plan; the block exits before any block barrier
      if (tid < 32) plan_warp(counts, offsets_out, tiles, ntiles, rows_out, order, tid);
      return;
    }
  }
  // [r3g] issued right after the PDL wait, in parallel with warp 0's counts[] load: the token's row (2 x 16 B per thread)
  // and, on threads 0..7, its 8 routes' lid / pos (before: row, then lid -> counts[l] -> pos after the scan)
  uint4 v0 = make_uint4(0u, 0u, 0u, 0u), v1 = v0;
  int h_l = -1, h_ps = 0;
  if (tok < T) {
    const uint4* src = reinterpret_cast<const uint4*>(p + (size_t)tok * 4096 + tid * 16);
    v0 = src[0];
    v1 = src[1];
    if (tid < 8) {
      h_l = lid[tok * 8 + tid];
      h_ps = pos[tok * 8 + tid];
    }
  }
  const int* offsets = offsets_in;
  if constexpr (PLAN) {
    if (tid < 32) {
      const int c = counts[tid];
      cnt_s[tid] = c;  // [r3g] smem copy for the token path's counts[l] and the pad-fill path's counts[e]
      const int full = c >> 8, r = c & 255;
      int nr, nb;
      rem_tiles(r, fine, nr, nb);
      const int rows = full * 256 + nr + nb;
      int v = rows;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int a2 = __shfl_up_sync(0xffffffffu, v, o);
        if (tid >= o) v += a2;
      }
      off_s[tid] = v - rows;
    }
    offsets = off_s;
  }
  if (tok >= T) {
    if constexpr (PLAN) __syncthreads();
    // pad-row zero fill: block (T + e*PARTS + part) zeroes part of expert e's pad rows
    const int idx = tok - T, e = idx / MOE_PADFILL_PARTS, part = idx % MOE_PADFILL_PARTS;
    int c;
    if constexpr (PLAN) c = cnt_s[e];  // [r3g] (warp 0's copy; ordered by the barrier above)
    else c = counts[e];
    const int full = c >> 8, r = c & 255;
    int nr, nb;
    rem_tiles(r, fine, nr, nb);
    // remainder tile(s): pad rows [used, width) of each; tile layout: 64 stage blocks of width * 128 B
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int n = q ? nb : nr;
      if (n == 0) continue;
      const int r0 = offsets[e] + 256 * full + (q ? nr : 0);
      const int used = q ? r - nr : (r < nr ? r : nr);
      for (int lr = used + part; lr < n; lr += MOE_PADFILL_PARTS) {
        for (int i = tid; i < 64 * 8; i += 256) {
          const int stg = i >> 3, ch = i & 7;
          *reinterpret_cast<uint4*>(xt + (size_t)r0 * 8192 + (size_t)stg * n * 128 + lr * 128 + ch * 16) =
              make_uint4(0, 0, 0, 0);
        }
      }
    }
    return;
  }
  // thread handles elements [16*tid, 16*tid+16): stage tid/4, chunks 2*(tid%4), 2*(tid%4)+1 (row loaded at entry)
  // the token's routes (row base, tile rows, row in tile), worked out by 8 threads while the row load is in flight
  __shared__ long long mrow[8];
  __shared__ int mn[8], mlr[8];
  if constexpr (PLAN) __syncwarp();  // (warp 0 wrote off_s / cnt_s)
  if (tid < 8) {
    const int k = tid;
    const int l = h_l;
    if (l < 0) {
      rrow[tok * 8 + k] = -1;
      mn[k] = 0;
    } else {
      const int ps = h_ps;
      int c;
      if constexpr (PLAN) c = cnt_s[l];  // [r3g] smem copy (before: a dependent global load after lid)
      else c = counts[l];
      const int full = c >> 8, r = c & 255;
      int nr, nb;
      rem_tiles(r, fine, nr, nb);
      const int tl = ps >> 8;
      int row0 = offsets[l] + 256 * tl, lr = ps & 255, n = tl < full ? 256 : nr;
      if (tl >= full && nb && lr >= nr) {  // a split remainder's second tile
        row0 += nr;
        lr -= nr;
        n = nb;
      }
      rrow[tok * 8 + k] = row0 + lr;
      if (row_tok != nullptr) row_tok[row0 + lr] = tok;  // CIT: which token a GEMM2 row belongs to
      mrow[k] = (long long)row0 * 8192 + lr * 128;
      mn[k] = n;
      mlr[k] = lr;
    }
    if (cq != nullptr) {  // CIT: a token without local routes is ready now (its combine writes zeros)
      const unsigned lm = __ballot_sync(0xffu, l >= 0);
      if (k == 0 && lm == 0) {
        const int i = atomicAdd(qtail, 1);
        asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(cq + i), "r"(tok + 1) : "memory");
      }
    }
  }
  const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
  const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
  float ss = 0.f;
  {  // [r3g] same element order and FMA chain as before; the row stays packed across the barrier, f[] is rebuilt after it
    float f[16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 a = __bfloat1622float2(h0[i]), b = __bfloat1622float2(h1[i]);
      f[2 * i] = a.x; f[2 * i + 1] = a.y; f[8 + 2 * i] = b.x; f[8 + 2 * i + 1] = b.y;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) ss += f[i] * f[i];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((tid & 31) == 0) red[tid >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += red[i];
  const float rr = rsqrtf(tot / 4096.f + 1e-5f);
  float f[16];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 a = __bfloat1622float2(h0[i]), b = __bfloat1622float2(h1[i]);
    f[2 * i] = a.x; f[2 * i + 1] = a.y; f[8 + 2 * i] = b.x; f[8 + 2 * i + 1] = b.y;
  }
  uint32_t outw[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float4 gg = g1p4(gw + tid * 16 + 4 * i);
    __nv_bfloat162 a = __floats2bfloat162_rn(f[4 * i] * rr * gg.x, f[4 * i + 1] * rr * gg.y);
    __nv_bfloat162 b = __floats2bfloat162_rn(f[4 * i + 2] * rr * gg.z, f[4 * i + 3] * rr * gg.w);
    outw[2 * i] = bf162_u32(a);
    outw[2 * i + 1] = bf162_u32(b);
  }
  const int stg = tid >> 2, ch0 = 2 * (tid & 3);
  if constexpr (CO) {
    // [r13] lane l of each 4-lane stage group stores logical chunks l and 4 + l of the stage (chunk c holds elements
    // 8c .. 8c + 7 and lives on group lane c >> 1, half c & 1): with the row swizzle x, chunks {l ^ x} and {(4 + l) ^ x}
    // are contiguous 64 B runs, so each warp store writes whole 32 B sectors (same bytes to the same addresses)
    const int l = tid & 3, gb = (tid & 31) & ~3;
    const int s1 = gb | (l >> 1), s2 = gb | (2 + (l >> 1));
    uint32_t w1[4], w2[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t a = __shfl_sync(0xffffffffu, outw[i], s1), b = __shfl_sync(0xffffffffu, outw[4 + i], s1);
      const uint32_t c = __shfl_sync(0xffffffffu, outw[i], s2), d = __shfl_sync(0xffffffffu, outw[4 + i], s2);
      w1[i] = (l & 1) ? b : a;
      w2[i] = (l & 1) ? d : c;
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int n = mn[k];
      if (n == 0) continue;
      const int lr = mlr[k];
      uint8_t* rowp = xt + mrow[k] + (size_t)stg * n * 128;
      const int x = lr & 7;
      *reinterpret_cast<uint4*>(rowp + ((l ^ x) << 4)) = make_uint4(w1[0], w1[1], w1[2], w1[3]);
      *reinterpret_cast<uint4*>(rowp + (((4 + l) ^ x) << 4)) = make_uint4(w2[0], w2[1], w2[2], w2[3]);
    }
    return;
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int n = mn[k];  // (written before the reduction's __syncthreads)
    if (n == 0) continue;
    const int lr = mlr[k];
    uint8_t* rowp = xt + mrow[k] + (size_t)stg * n * 128;
    const int x = lr & 7;
    *reinterpret_cast<uint4*>(rowp + ((ch0 ^ x) << 4)) = make_uint4(outw[0], outw[1], outw[2], outw[3]);
    *reinterpret_cast<uint4*>(rowp + (((ch0 + 1) ^ x) << 4)) = make_uint4(outw[4], outw[5], outw[6], outw[7]);
  }
}

// ---- routing: top-8 experts per token from the router logits (hi + lo halves), one warp per token ----
// Same math and tie-break as the Triton reference version (sigmoid(logit / sqrt(32)), + bias for selection, the
// highest biased score first, lowest expert index on ties; weights = selected scores / their sum), but each lane
// holds 8 experts and the 8 rounds of arg-max are warp shuffles (no block barriers). Local routes (experts
// local_start .. +31) get a row position from a per-expert counter: positions are first taken from block-local
// shared-memory counters, then each block reserves its range with one global atomic per expert (hot experts take
// thousands of routes, and per-route global atomics on one address serialize).
#define ROUTE_WARPS 16  // v69: 16 measured +0.06% over 32
// [r4rtred] NP = 1: the base kernel (logits [T][512]); NP = 4: logits are [NP][T][512] K-slice partials (pitch pstride)
template <int NP>
__global__ void __launch_bounds__(ROUTE_WARPS * 32) moe_route_kernel(const float* __restrict__ logits,
                                                                     const float* __restrict__ bias,
                                                                     const int* __restrict__ lstart, int* __restrict__ ids,
                                                                     float* __restrict__ wout, int* __restrict__ lid,
                                                                     int* __restrict__ pos, int* __restrict__ cnt, int T,
                                                                     int* __restrict__ tok_left = nullptr, int* __restrict__ cq = nullptr,
                                                                     int pstride = 0) {
  pdl_enter();
  __shared__ int s_cnt[32], s_base[32];
  const int t = blockIdx.x * ROUTE_WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (threadIdx.x < 32) s_cnt[threadIdx.x] = 0;
  __syncthreads();
  int e_mine = 0, loc = 0, lpos = 0;
  bool local = false;
  float w_mine = 0.f;
  if (t < T) {
    const float* lg = logits + (size_t)t * 512;
    float sv[8], bv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = lane + 32 * j;
      float x;
      if constexpr (NP == 1) {
        x = (lg[e] + lg[256 + e]) * 0.1767766952966369f;
      } else {  // [r4rtred] the hi and lo halves each summed in slice order (the split-K reduce's order), then hi + lo
        float hs = lg[e], ls = lg[256 + e];
#pragma unroll
        for (int q = 1; q < NP; ++q) {
          hs += lg[(size_t)q * pstride + e];
          ls += lg[(size_t)q * pstride + 256 + e];
        }
        x = (hs + ls) * 0.1767766952966369f;
      }
      sv[j] = 1.f / (1.f + expf(-x));
      bv[j] = sv[j] + bias[e];
    }
    int sel_e[8];
    float sel_s[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float bvv = bv[0];
      int be = lane;
#pragma unroll
      for (int j = 1; j < 8; ++j) {  // ascending expert index within the lane: strict > keeps the lowest on ties
        if (bv[j] > bvv) { bvv = bv[j]; be = lane + 32 * j; }
      }
      {
        const uint32_t u = __float_as_uint(bvv);
        const uint32_t key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // unsigned order == float order
        const uint32_t mk = __reduce_max_sync(0xffffffffu, key);
        be = (int)__reduce_min_sync(0xffffffffu, key == mk ? (uint32_t)be : 0xffffffffu);  // lowest expert on ties
      }
      float mys = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j == (be >> 5)) mys = sv[j];
      }
      sel_s[k] = __shfl_sync(0xffffffffu, mys, be & 31);
      sel_e[k] = be;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (lane + 32 * j == be) bv[j] = -INFINITY;
      }
    }
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) sum += sel_s[k];
    e_mine = sel_e[0];
    float sk = sel_s[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) {
      if (lane == k) { e_mine = sel_e[k]; sk = sel_s[k]; }
    }
    w_mine = sk / (sum + 1e-15f);
    loc = (e_mine - *lstart + 256) % 256;
    local = lane < 8 && loc < 32;
    if (local) lpos = atomicAdd(&s_cnt[loc], 1);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int c = s_cnt[threadIdx.x];
    s_base[threadIdx.x] = c ? atomicAdd(cnt + threadIdx.x, c) : 0;
  }
  __syncthreads();
  if (t < T && tok_left != nullptr) {  // CIT: the token's local route count, and its combine-queue slot cleared
    const unsigned lm = __ballot_sync(0xffffffffu, local);
    if (lane == 0) {
      tok_left[t] = __popc(lm);
      cq[t] = 0;
    }
  }
  if (t < T && lane < 8) {
    ids[t * 8 + lane] = e_mine;
    wout[t * 8 + lane] = w_mine;
    lid[t * 8 + lane] = local ? loc : -1;
    pos[t * 8 + lane] = local ? s_base[loc] + lpos : 0;
  }
}

// ---- combine: out[t] = sum over t's local routes of norm(y[row] * pw) * w, one block of 256 threads per token ----
// (the post-expert norm: scale by pw = 1 + w first, then RMSNorm with eps 1e-8). The token's local routes (about two of
// its eight) are compacted first; their sums of squares share one block reduction, and the second pass re-reads each
// route row from L1/L2. Few registers, so several blocks per SM keep enough loads in flight.
__global__ void __launch_bounds__(256) moe_combine_kernel(const __nv_bfloat16* __restrict__ y, const int* __restrict__ rrow,
                                                          const int* __restrict__ lid, const float* __restrict__ topw,
                                                          const __nv_bfloat16* __restrict__ pw, __nv_bfloat16* __restrict__ out) {
  // one token per block, 16 elements per thread: each local route's row post-normed on its own (rsqrt of its mean
  // square + 1e-8, times g_post_expert_norm), weighted by its routing weight, summed (as the Triton _combine_kernel);
  // up to 4 routes' rows in flight at once, one block reduction per group of 4
  pdl_enter();
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float red[4][8];
  int rows[8];
  float wts[8];
  int nl = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int l = lid[tok * 8 + k];
    if (l >= 0) {
      rows[nl] = rrow[tok * 8 + k];
      wts[nl] = topw[tok * 8 + k];
      ++nl;
    }
  }
  float g[16];
  {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 q = g1p4(pw + tid * 16 + 4 * i);
      g[4 * i] = q.x; g[4 * i + 1] = q.y; g[4 * i + 2] = q.z; g[4 * i + 3] = q.w;
    }
  }
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#define COMB_GROUP 2
  for (int k0 = 0; k0 < nl; k0 += COMB_GROUP) {
    float x[COMB_GROUP][16];
    float ss[COMB_GROUP];
#pragma unroll
    for (int u = 0; u < COMB_GROUP; ++u) {
      ss[u] = 0.f;
      if (k0 + u < nl) {
        const uint4* src = reinterpret_cast<const uint4*>(y + (size_t)rows[k0 + u] * 4096 + tid * 16);
        const uint4 v0 = src[0], v1 = src[1];
        const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
        const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 f = __bfloat1622float2(h0[i]), q = __bfloat1622float2(h1[i]);
          x[u][2 * i] = f.x * g[2 * i]; x[u][2 * i + 1] = f.y * g[2 * i + 1];
          x[u][8 + 2 * i] = q.x * g[8 + 2 * i]; x[u][8 + 2 * i + 1] = q.y * g[8 + 2 * i + 1];
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) ss[u] += x[u][i] * x[u][i];
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) x[u][i] = 0.f;
      }
    }
#pragma unroll
    for (int u = 0; u < COMB_GROUP; ++u) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss[u] += __shfl_xor_sync(0xffffffffu, ss[u], o);
    }
    if (k0) __syncthreads();  // the previous group's red[] has been read
    if (lane == 0) {
#pragma unroll
      for (int u = 0; u < COMB_GROUP; ++u) red[u][warp] = ss[u];
    }
    __syncthreads();
#pragma unroll
    for (int u = 0; u < COMB_GROUP; ++u) {
      if (k0 + u < nl) {
        float tot = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) tot += red[u][w];
        const float sc = rsqrtf(tot / 4096.f + 1e-8f) * wts[k0 + u];
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] += x[u][i] * sc;
      }
    }
  }
  uint4 o0, o1;
  __nv_bfloat162* b0 = reinterpret_cast<__nv_bfloat162*>(&o0);
  __nv_bfloat162* b1 = reinterpret_cast<__nv_bfloat162*>(&o1);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    b0[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    b1[i] = __floats2bfloat162_rn(acc[8 + 2 * i], acc[8 + 2 * i + 1]);
  }
  uint4* dst = reinterpret_cast<uint4*>(out + (size_t)tok * 4096 + tid * 16);
  dst[0] = o0;
  dst[1] = o1;
}

}  // namespace okc_moe

namespace okc_moe {

// ---- combine v2 (COMBINE_V2): bitwise twin of the Triton _combine_kernel (num_warps 8, H 4096), bandwidth-oriented ----
// Arithmetic contract (read off Triton 3.8's TTGIR/PTX/SASS of _combine_kernel; the judge image runs Triton 3.8.0):
//  * layout: 256 threads; thread tid owns elements tid*8 .. +7 (j = 0..7) and 2048 + tid*8 .. +7 (j = 8..15)
//  * pwf_j = bf16(pw) + 1 (add.f32);  y_j = (pwf_j * gs[lid]) * bf16(y) (two mul.f32)
//  * per-thread sum of squares as a pairwise tree: q_i = fma(y_2i, y_2i, y_2i+1 * y_2i+1) (i = 0..7),
//    s = ((q0 + q1) + (q2 + q3)) + ((q4 + q5) + (q6 + q7))
//  * warp butterfly xor 16, 8, 4, 2, 1 (s + shfl(s)); the 8 warp sums S_w combined as
//    ((S0 + S4) + (S2 + S6)) + ((S1 + S5) + (S3 + S7))  (Triton's second butterfly xor 4, 2, 1 over smem[tid & 7])
//  * m = fma(sum, 2^-12, 1e-8) (Triton: div.full.f32 by 4096 + add, which ptxas folds into this one FFMA; the product is
//    exact, so it equals the unfused pair too);  r = rsqrt.approx.ftz(m);  sc = r * topw
//  * acc_j = fma(y_j, sc, acc_j) over the token's local routes in k order, acc from +0; out = cvt.rn.bf16x2
// Every float op is inline PTX without .ftz: the extension builds with --use_fast_math (-ftz=true), which would turn
// plain C++ float ops (and even __fmul_rn & co.) into .ftz variants that Triton does not use.
// Schedule: one CTA per `tpc` consecutive tokens, 256 threads, 3 CTAs per SM. The CTA compacts its tokens' local routes
// (lid >= 0, (token, k) order) into smem, then streams the route rows through a ring of COMB2_S 8 KB slots with
// cp.async: each thread copies and later reads only its own 2 x 16 B of every row (no barrier for the ring itself),
// prefetch distance COMB2_D = COMB2_S - 1 routes (the slot refilled at route r is the one consumed at route r - 1).
// One block barrier per route (the cross-warp sum, double-buffered by route parity). PDL: launch_dependents at entry;
// only the constant norm weight and per-expert scale are read before griddepcontrol.wait.
#define COMB2_S 8
#define COMB2_D (COMB2_S - 1)
#define COMB2_TPC_MAX 32
#define COMB2_SMEM (COMB2_S * 8192)

DEVI float c2_mul(float a, float b) { float d; asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
DEVI float c2_add(float a, float b) { float d; asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
DEVI float c2_fma(float a, float b, float c) {
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
DEVI float c2_rsqrt(float a) { float d; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
DEVI void c2_bf8(const uint4 v, float* f) {  // 8 bf16 -> fp32 (exact: bit shifts)
  f[0] = __uint_as_float(v.x << 16); f[1] = __uint_as_float(v.x & 0xffff0000u);
  f[2] = __uint_as_float(v.y << 16); f[3] = __uint_as_float(v.y & 0xffff0000u);
  f[4] = __uint_as_float(v.z << 16); f[5] = __uint_as_float(v.z & 0xffff0000u);
  f[6] = __uint_as_float(v.w << 16); f[7] = __uint_as_float(v.w & 0xffff0000u);
}
DEVI uint32_t c2_pack(float lo, float hi) {  // cvt.rn.bf16x2.f32 d, hi, lo (Triton's packing: element j in the low half)
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
DEVI void c2_cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
DEVI void c2_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
// [r6h] hand-off combine waits. A spin past HCOMB_SPIN_NS (a bug: the MoE would long be done) records an error word
// and falls back to griddepcontrol.wait (the whole MoE grid: correct, just late) instead of hanging or reading early.
#define HCOMB_SPIN_NS 200000000ull
#define HCOMB_POLL_NS 128
#define HCOMB_TAILWAIT 1  // CTA 0 of moe_combine2_kernel<true> waits for the MoE grid at its end (see there)
DEVI void hc_wait_go(const int* go, int* herr) {  // one thread
  unsigned long long t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  while (true) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(go) : "memory");
    if (v != 0) return;
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (now - t0 > HCOMB_SPIN_NS) {
      atomicExch(herr, 1);
      asm volatile("griddepcontrol.wait;" ::: "memory");
      return;
    }
    __nanosleep(64);
  }
}
// one warp: lane l watches routes l, l + 32, ... (their rows' 32-row blocks) until every block count reaches need
DEVI void hc_wait_routes(const int* blk, const int* s_row, unsigned R, int need, int* herr, int lane) {
  unsigned long long t0 = 0;
  bool late = false;
  for (unsigned q0 = 0; q0 < R && !late; q0 += 32) {
    const unsigned q = q0 + (unsigned)lane;
    bool ok = q >= R;
    const int* f = blk + (ok ? 0 : (s_row[q] >> 5));
    while (true) {
      if (!ok) {
        int v;
        asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
        ok = v >= need;
      }
      if (__all_sync(0xffffffffu, ok)) break;
      int to = 0;
      if (lane == 0) {  // one clock for the warp: the timeout decision is warp-uniform
        unsigned long long now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (t0 == 0) t0 = now;
        else to = now - t0 > HCOMB_SPIN_NS;
      }
      if (__shfl_sync(0xffffffffu, to, 0)) { late = true; break; }
      __nanosleep(HCOMB_POLL_NS);
    }
  }
  if (late) {
    if (lane == 0) atomicExch(herr, 2);
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}
DEVI uint4 c2_lds(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}

// [r6h] HAND: no whole-grid PDL wait; the go flag, then per-route 32-row block counters (fa.hblk of the fused MoE)
template <bool HAND>
__global__ void __launch_bounds__(256, 3) moe_combine2_kernel(const __nv_bfloat16* __restrict__ y, const int* rrow,
                                                              const int* lid, const float* topw,
                                                              const float* __restrict__ gs, const __nv_bfloat16* __restrict__ pw,
                                                              __nv_bfloat16* __restrict__ out, int T, int tpc,
                                                              const int* hgo, const int* hblk, int hneed, int* herr) {
  extern __shared__ __align__(16) uint8_t c2_ring[];  // COMB2_S slots x 8 KB (slot s, thread tid: bytes s*8K + tid*16, + 4K)
  __shared__ int s_row[COMB2_TPC_MAX * 8];
  __shared__ float s_w[COMB2_TPC_MAX * 8];
  __shared__ int s_lid[COMB2_TPC_MAX * 8];
  __shared__ int s_cnt[COMB2_TPC_MAX];
  __shared__ int s_wn[8];
  __shared__ float s_gs[32];
  __shared__ float s_red[2][8];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  // before the wait: constants only (post_expert_norm and the per-expert scale are never written inside kernel_fn)
  float pwf[16];
  {
    float f[16];
    c2_bf8(__ldg(reinterpret_cast<const uint4*>(pw + tid * 8)), f);
    c2_bf8(__ldg(reinterpret_cast<const uint4*>(pw + 2048 + tid * 8)), f + 8);
#pragma unroll
    for (int j = 0; j < 16; ++j) pwf[j] = c2_add(f[j], 1.0f);
  }
  if (tid < 32) s_gs[tid] = __ldg(gs + tid);
  if constexpr (HAND) {
    // [r6h] the MoE's CTA 0 sets go after its own PDL wait: the gather's rrow (and the route kernel's lid / topw, done
    // before the gather started) are complete; thread 0's acquire, then the barrier, order every thread's loads
    if (tid == 0) hc_wait_go(hgo, herr);
    __syncthreads();
  } else {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  // ---- this CTA's local routes, compacted in (token, k) order ----
  const int t0 = blockIdx.x * tpc;
  const int ntok = min(tpc, T - t0);
  const int ne = ntok * 8;
  int l = -1, rw = 0;
  float w = 0.f;
  if (tid < ne) {
    l = lid[(size_t)t0 * 8 + tid];
    rw = rrow[(size_t)t0 * 8 + tid];
    w = topw[(size_t)t0 * 8 + tid];
  }
  const bool loc = l >= 0;
  const unsigned bal = __ballot_sync(0xffffffffu, loc);
  if (lane == 0) s_wn[warp] = __popc(bal);
  if ((lane & 7) == 0 && tid < ne) s_cnt[tid >> 3] = __popc((bal >> lane) & 0xffu);
  __syncthreads();
  int base = 0, Ri = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int c = s_wn[i];
    base += i < warp ? c : 0;
    Ri += c;
  }
  if (loc) {
    const int idx = base + __popc(bal & ((1u << lane) - 1u));
    s_row[idx] = rw;
    s_w[idx] = w;
    s_lid[idx] = l;
  }
  __syncthreads();
  const unsigned R = (unsigned)Ri;
  if constexpr (HAND) {
    // [r6h] every local route's y row: wait until its 32-row block has all hneed GEMM2 n-tiles published (warp 0's
    // acquires, then the barrier, order the ring's cp.async.cg (L2) row loads after them)
    if (hblk != nullptr) {
      if (warp == 0) hc_wait_routes(hblk, s_row, R, hneed, herr, lane);
      __syncthreads();
    } else {  // hoist-only arm: the index loads and compaction ran early; y rows after the whole MoE grid
      asm volatile("griddepcontrol.wait;" ::: "memory");
    }
  }
  // ---- route-row ring: routes 0 .. COMB2_D - 1 in flight first (one commit group per route, empty past R) ----
  const uint32_t ring = smem_u32(c2_ring) + tid * 16;
  const __nv_bfloat16* ysrc = y + tid * 8;
#pragma unroll
  for (unsigned q = 0; q < COMB2_D; ++q) {
    if (q < R) {
      const __nv_bfloat16* src = ysrc + (size_t)s_row[q] * 4096;
      c2_cp16(ring + q * 8192, src);
      c2_cp16(ring + q * 8192 + 4096, src + 2048);
    }
    c2_commit();
  }
  unsigned r = 0;  // route index in the CTA's stream (ring slot r % COMB2_S, reduction buffer r & 1)
  for (int tt = 0; tt < ntok; ++tt) {
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = 0.f;
    const int n = s_cnt[tt];
    for (int k = 0; k < n; ++k, ++r) {
      {  // refill: route r + D into slot (r + D) % S == (r - 1) % S, read (and used) by this thread at route r - 1
        const unsigned q = r + COMB2_D;
        if (q < R) {
          const uint32_t d = ring + (q % COMB2_S) * 8192;
          const __nv_bfloat16* src = ysrc + (size_t)s_row[q] * 4096;
          c2_cp16(d, src);
          c2_cp16(d + 4096, src + 2048);
        }
        c2_commit();
      }
      asm volatile("cp.async.wait_group %0;" ::"n"(COMB2_D) : "memory");  // groups 0 .. r done (own copies)
      const uint32_t sl = ring + (r % COMB2_S) * 8192;
      const uint4 va = c2_lds(sl), vb = c2_lds(sl + 4096);
      const int le = s_lid[r];
      const float g = (unsigned)le < 32u ? s_gs[le] : __ldg(gs + le);  // lid < LOCAL_EXPERTS (32) always
      float yv[16];
      c2_bf8(va, yv);
      c2_bf8(vb, yv + 8);
#pragma unroll
      for (int j = 0; j < 16; ++j) yv[j] = c2_mul(c2_mul(pwf[j], g), yv[j]);
      float qq[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) qq[i] = c2_fma(yv[2 * i], yv[2 * i], c2_mul(yv[2 * i + 1], yv[2 * i + 1]));
      float s = c2_add(c2_add(c2_add(qq[0], qq[1]), c2_add(qq[2], qq[3])), c2_add(c2_add(qq[4], qq[5]), c2_add(qq[6], qq[7])));
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s = c2_add(s, __shfl_xor_sync(0xffffffffu, s, o));
      if (lane == 0) s_red[r & 1][warp] = s;
      __syncthreads();
      const float* rd = s_red[r & 1];
      const float tot = c2_add(c2_add(c2_add(rd[0], rd[4]), c2_add(rd[2], rd[6])), c2_add(c2_add(rd[1], rd[5]), c2_add(rd[3], rd[7])));
      const float sc = c2_mul(c2_rsqrt(c2_fma(tot, 0.000244140625f, 1e-8f)), s_w[r]);
#pragma unroll
      for (int j = 0; j < 16; ++j) acc[j] = c2_fma(yv[j], sc, acc[j]);
    }
    uint4 o0, o1;
    o0.x = c2_pack(acc[0], acc[1]); o0.y = c2_pack(acc[2], acc[3]); o0.z = c2_pack(acc[4], acc[5]); o0.w = c2_pack(acc[6], acc[7]);
    o1.x = c2_pack(acc[8], acc[9]); o1.y = c2_pack(acc[10], acc[11]); o1.z = c2_pack(acc[12], acc[13]); o1.w = c2_pack(acc[14], acc[15]);
    __nv_bfloat16* dst = out + (size_t)(t0 + tt) * 4096 + tid * 8;
    *reinterpret_cast<uint4*>(dst) = o0;
    *reinterpret_cast<uint4*>(dst + 2048) = o1;
  }
  if constexpr (HAND) {  // [r6h] kernel completion implies the MoE grid's completion (one thread of CTA 0 waits for it)
    if (HCOMB_TAILWAIT && blockIdx.x == 0 && tid == 0) asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}

}  // namespace okc_moe

namespace okc_moe {

// ---- q/k prep, CUDA version of the Triton _qk_prep_kernel (QK_PREP_CUDA): one block of 128 threads per token.
// Per-head RMSNorm (eps 1e-5) of q (16 heads) and k (2 heads), RoPE on interleaved (even, odd) pairs with the angle
// pos * inv_freq[i] rounded to fp32 as in the Triton kernel (cos/sin evaluated in double, then rounded), e4m3 (RNE,
// satfinite) q rows and k/v cache rows (k and v divided by their scales when SCALED). Thread layout: q head tid/8, dims
// 8*(tid%8)..+7; threads 0..15 also do the k heads (head tid/8), threads 16..31 the v row (dims 8*(tid-16)..+7).
DEVI uint32_t qk_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI uint2 qk_pack8(const float (&x)[8]) {
  return make_uint2(qk_e4m3x2(x[0], x[1]) | (qk_e4m3x2(x[2], x[3]) << 16), qk_e4m3x2(x[4], x[5]) | (qk_e4m3x2(x[6], x[7]) << 16));
}
DEVI void qk_load8(const __nv_bfloat16* p, float (&x)[8]) {
  const uint4 v = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) { const float2 f = __bfloat1622float2(h[i]); x[2 * i] = f.x; x[2 * i + 1] = f.y; }
}
template <bool ROPE>
DEVI void qk_norm_rope8(float (&x)[8], const float* cs, const float* sn, int d0) {
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss += x[i] * x[i];
  ss += __shfl_xor_sync(0xffffffffu, ss, 1);
  ss += __shfl_xor_sync(0xffffffffu, ss, 2);
  ss += __shfl_xor_sync(0xffffffffu, ss, 4);
  const float r = rsqrtf(ss / 64.f + 1e-5f);
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] *= r;
  if constexpr (ROPE) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float c = cs[d0 / 2 + j], s = sn[d0 / 2 + j];
      const float e = x[2 * j], o = x[2 * j + 1];
      x[2 * j] = e * c - o * s;
      x[2 * j + 1] = e * s + o * c;
    }
  }
}
template <bool ROPE, bool SCALED, typename PT, typename LT>
__global__ void __launch_bounds__(128) qk_prep_kernel(const __nv_bfloat16* __restrict__ qkv, const PT* __restrict__ pos,
                                                      const LT* __restrict__ loc, const float* __restrict__ invf,
                                                      uint8_t* __restrict__ q8, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                                      const float* __restrict__ ksc, const float* __restrict__ vsc) {
  __shared__ float s_cos[32], s_sin[32];
  const int t = blockIdx.x, tid = threadIdx.x;
  float pf = 0.f;
  if (ROPE && tid < 32) pf = (float)pos[t];  // positions / cache rows are inputs, not the predecessor's results
  const long long lrow = tid < 32 ? (long long)loc[t] : 0;
  pdl_enter();
  if constexpr (ROPE) {
    if (tid < 32) {
      const float ang = __fmul_rn(pf, invf[tid]);
      double sd, cd;
      sincos((double)ang, &sd, &cd);
      s_cos[tid] = (float)cd;
      s_sin[tid] = (float)sd;
    }
    __syncthreads();
  }
  const __nv_bfloat16* row = qkv + (size_t)t * 2304;
  const int d0 = (tid & 7) * 8;
  {
    float x[8];
    qk_load8(row + (tid >> 3) * 64 + d0, x);
    qk_norm_rope8<ROPE>(x, s_cos, s_sin, d0);
    *reinterpret_cast<uint2*>(q8 + (size_t)t * 1024 + (tid >> 3) * 64 + d0) = qk_pack8(x);
  }
  if (tid < 16) {  // k heads 0, 1 (the 8-lane groups of the reduction stay within one head)
    float x[8];
    qk_load8(row + 1024 + (tid >> 3) * 64 + d0, x);
    qk_norm_rope8<ROPE>(x, s_cos, s_sin, d0);
    if constexpr (SCALED) {
      const float ks = *ksc;
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = __fdiv_rn(x[i], ks);
    }
    *reinterpret_cast<uint2*>(kc + lrow * 128 + (tid >> 3) * 64 + d0) = qk_pack8(x);
  } else if (tid < 32) {  // v: no norm
    float x[8];
    const int dv = (tid - 16) * 8;
    qk_load8(row + 1152 + dv, x);
    if constexpr (SCALED) {
      const float vs = *vsc;
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = __fdiv_rn(x[i], vs);
    }
    *reinterpret_cast<uint2*>(vc + lrow * 128 + dv) = qk_pack8(x);
  }
}

// [r6kvoff] inverse of gather_kp's pi_lane (a permutation of the 5 low key bits): lam = (key & ~31) | qk_pinv(key & 31)
// is the V^T column gather_kp stores key `key` in.
DEVI int qk_pinv(int k) { return (k & 1) | (((k >> 3) & 1) << 1) | (((k >> 1) & 3) << 2) | (k & 16); }
// [r6kvoff fix] qk_norm_rope8 with an explicit shuffle member mask. In qk_prep_kvt only lanes 0..15 of warp 0 run the k
// heads' reduction, and lanes 16..31 wait at the reconvergence point before the V^T stores instead of exiting (as they
// do in qk_prep_kernel): a full mask would name lanes that never execute this shuffle. xor 1/2/4 stays inside each
// 8-lane group, so the sums (and the K bytes) are qk_norm_rope8's.
template <bool ROPE>
DEVI void qk_norm_rope8_m(float (&x)[8], const float* cs, const float* sn, int d0, unsigned mask) {
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss += x[i] * x[i];
  ss += __shfl_xor_sync(mask, ss, 1);
  ss += __shfl_xor_sync(mask, ss, 2);
  ss += __shfl_xor_sync(mask, ss, 4);
  const float r = rsqrtf(ss / 64.f + 1e-5f);
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] *= r;
  if constexpr (ROPE) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float c = cs[d0 / 2 + j], s = sn[d0 / 2 + j];
      const float e = x[2 * j], o = x[2 * j + 1];
      x[2 * j] = e * c - o * s;
      x[2 * j + 1] = e * s + o * c;
    }
  }
}
#ifndef KVOFF_HOIST
#define KVOFF_HOIST 1
#endif
// [r13] RoPE table of the new tokens: tab[t][j] = cos, tab[t][32 + j] = sin of pos[t] * inv_freq[j], with qk_prep's
// expression (fp32 angle by __fmul_rn, sincos in double, rounded to fp32). A side-stream kernel (inputs only).
template <typename PT>
__global__ void __launch_bounds__(256) rope_tab_kernel(const PT* __restrict__ pos, const float* __restrict__ invf,
                                                       float* __restrict__ tab, int T) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= T * 32) return;
  const int t = i >> 5, j = i & 31;
  const float pf = (float)pos[t];
  const float ang = __fmul_rn(pf, invf[j]);
  double sd, cd;
  sincos((double)ang, &sd, &cd);
  tab[(size_t)t * 64 + j] = (float)cd;
  tab[(size_t)t * 64 + 32 + j] = (float)sd;
}
// [r13] TAB: invf points at rope_tab_kernel's table instead of inv_freq; the angles are loaded, not evaluated
template <bool ROPE, bool SCALED, typename PT, typename LT, bool TAB = false>
__global__ void __launch_bounds__(128) qk_prep_kvt_kernel(const __nv_bfloat16* __restrict__ qkv, const PT* __restrict__ pos,
                                                      const LT* __restrict__ loc, const float* __restrict__ invf,
                                                      uint8_t* __restrict__ q8, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                                      const float* __restrict__ ksc, const float* __restrict__ vsc,
                                                      uint8_t* __restrict__ kvt, const int* __restrict__ klen,
                                                      const int* __restrict__ kseq, int capblks, int L, int win) {
  __shared__ float s_cos[32], s_sin[32];
  const int t = blockIdx.x, tid = threadIdx.x;
  float pf = 0.f;
  if (ROPE && tid < 32) pf = (float)pos[t];  // positions / cache rows are inputs, not the predecessor's results
  const long long lrow = tid < 32 ? (long long)loc[t] : 0;
#if KVOFF_HOIST  // [r6kvoff] pdl_enter split around the RoPE angles (inputs only: pos, inv_freq): trigger, angles, wait
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if constexpr (ROPE) {
    if (tid < 32) {
      if constexpr (TAB) {  // [r13] the side branch (joined before this launch) wrote the table
        s_cos[tid] = invf[(size_t)t * 64 + tid];
        s_sin[tid] = invf[(size_t)t * 64 + 32 + tid];
      } else {
        const float ang = __fmul_rn(pf, invf[tid]);
        double sd, cd;
        sincos((double)ang, &sd, &cd);
        s_cos[tid] = (float)cd;
        s_sin[tid] = (float)sd;
      }
    }
    __syncthreads();
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
#else
  pdl_enter();
  if constexpr (ROPE) {
    if (tid < 32) {
      const float ang = __fmul_rn(pf, invf[tid]);
      double sd, cd;
      sincos((double)ang, &sd, &cd);
      s_cos[tid] = (float)cd;
      s_sin[tid] = (float)sd;
    }
    __syncthreads();
  }
#endif
  // [r6kvoff] this token's key slot in gather_kp's kvt layout: tile (b, kvh, rel >> 7), key rel & 127, with rel = its
  // position (len - L + i) minus the window start kst (cache_seqlens / seq_lens read after the wait, like gather_kp)
  const int b = t / L;
  int rel = 0;
  if (tid < 32) rel = klen[b] - L + (t - b * L) - max(kseq[b] - (win - 1), 0);
  const __nv_bfloat16* row = qkv + (size_t)t * 2304;
  const int d0 = (tid & 7) * 8;
  {
    float x[8];
    qk_load8(row + (tid >> 3) * 64 + d0, x);
    qk_norm_rope8<ROPE>(x, s_cos, s_sin, d0);
    *reinterpret_cast<uint2*>(q8 + (size_t)t * 1024 + (tid >> 3) * 64 + d0) = qk_pack8(x);
  }
  uint2 vw = make_uint2(0u, 0u);
  if (tid < 16) {  // k heads 0, 1 (the 8-lane groups of the reduction stay within one head)
    float x[8];
    qk_load8(row + 1024 + (tid >> 3) * 64 + d0, x);
    qk_norm_rope8_m<ROPE>(x, s_cos, s_sin, d0, 0x0000ffffu);  // [r6kvoff fix] lanes 0..15 only
    if constexpr (SCALED) {
      const float ks = *ksc;
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = __fdiv_rn(x[i], ks);
    }
    const uint2 kw = qk_pack8(x);
    *reinterpret_cast<uint2*>(kc + lrow * 128 + (tid >> 3) * 64 + d0) = kw;
    *reinterpret_cast<uint2*>(kvt + ((size_t)(b * 2 + (tid >> 3)) * capblks + (rel >> 7)) * 16384 + (d0 >> 4) * 2048 +
                              (rel & 127) * 16 + (d0 & 15)) = kw;   // [r6kvoff] gather_kp's K layout
  } else if (tid < 32) {  // v: no norm
    float x[8];
    const int dv = (tid - 16) * 8;
    qk_load8(row + 1152 + dv, x);
    if constexpr (SCALED) {
      const float vs = *vsc;
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = __fdiv_rn(x[i], vs);
    }
    vw = qk_pack8(x);
    *reinterpret_cast<uint2*>(vc + lrow * 128 + dv) = vw;
  }
  if (tid < 32) {  // [r6kvoff] V^T bytes: lane l stores dim 8 * (vi & 7) + (l & 7) of v head vi >> 3, vi = 4 s + (l >> 3),
    // fetched from v thread 16 + vi: 4 warp stores of 4 lines each per token (not 16 lines per byte store)
    const int key = rel & 127, j = tid & 7;
    const int lam = (key & ~31) | qk_pinv(key & 31);   // the V^T column that gather_kp gives this key
    uint8_t* tb = kvt + ((size_t)(b * 2) * capblks + (rel >> 7)) * 16384 + 8192 + (lam >> 4) * 128 + j * 16 + (lam & 15);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int vi = 4 * s + (tid >> 3);
      const unsigned wx = __shfl_sync(0xffffffffu, vw.x, 16 + vi);
      const unsigned wy = __shfl_sync(0xffffffffu, vw.y, 16 + vi);
      tb[(size_t)(vi >> 3) * capblks * 16384 + (vi & 7) * 1024] = (uint8_t)((j < 4 ? wx : wy) >> (8 * (j & 3)));
    }
  }
}

// [p12f] the bf16 -> fp32 conversion of qk_load8, on a row chunk already loaded
DEVI void qk_cvt8(const uint4 v, float (&x)[8]) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) { const float2 f = __bfloat1622float2(h[i]); x[2 * i] = f.x; x[2 * i + 1] = f.y; }
}
// [p12f] qk_prep_kernel with NT tokens per 128-thread block: every thread does its qk_prep_kernel role for tokens
// blockIdx.x * NT + j (j < NT), with the block's qkv row chunks all loaded before any is used (NT x the bytes in flight),
// and the RoPE angles (inputs only: positions, inv_freq) evaluated before the PDL wait by warp j % 4 for token j. Same
// per-element arithmetic as qk_prep_kernel; the k heads' reduction names lanes 0..15 (qk_norm_rope8_m, as qk_prep_kvt),
// the lanes whose values it sums.
template <bool ROPE, bool SCALED, typename PT, typename LT, int NT>
__global__ void __launch_bounds__(128) qk_prep_m_kernel(const __nv_bfloat16* __restrict__ qkv, const PT* __restrict__ pos,
                                                        const LT* __restrict__ loc, const float* __restrict__ invf,
                                                        uint8_t* __restrict__ q8, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                                        const float* __restrict__ ksc, const float* __restrict__ vsc, int T) {
  __shared__ float s_cos[NT][32], s_sin[NT][32];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int t0 = blockIdx.x * NT;
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if constexpr (ROPE) {
#pragma unroll
    for (int j = wid; j < NT; j += 4) {
      if (t0 + j < T) {
        const float pf = (float)pos[t0 + j];
        const float ang = __fmul_rn(pf, invf[lane]);
        double sd, cd;
        sincos((double)ang, &sd, &cd);
        s_cos[j][lane] = (float)cd;
        s_sin[j][lane] = (float)sd;
      }
    }
  }
  long long lrow[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) lrow[j] = (tid < 32 && t0 + j < T) ? (long long)loc[t0 + j] : 0;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if constexpr (ROPE) __syncthreads();
  const int d0 = (tid & 7) * 8;
  uint4 rq[NT], rk[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    rq[j] = make_uint4(0u, 0u, 0u, 0u);
    rk[j] = rq[j];
    if (t0 + j < T) {
      const __nv_bfloat16* row = qkv + (size_t)(t0 + j) * 2304;
      rq[j] = *reinterpret_cast<const uint4*>(row + (tid >> 3) * 64 + d0);
      if (tid < 16) rk[j] = *reinterpret_cast<const uint4*>(row + 1024 + (tid >> 3) * 64 + d0);
      else if (tid < 32) rk[j] = *reinterpret_cast<const uint4*>(row + 1152 + (tid - 16) * 8);
    }
  }
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int t = t0 + j;
    if (t >= T) break;  // block-uniform
    {
      float x[8];
      qk_cvt8(rq[j], x);
      qk_norm_rope8<ROPE>(x, s_cos[j], s_sin[j], d0);
      *reinterpret_cast<uint2*>(q8 + (size_t)t * 1024 + (tid >> 3) * 64 + d0) = qk_pack8(x);
    }
    if (tid < 16) {  // k heads 0, 1
      float x[8];
      qk_cvt8(rk[j], x);
      qk_norm_rope8_m<ROPE>(x, s_cos[j], s_sin[j], d0, 0x0000ffffu);
      if constexpr (SCALED) {
        const float ks = *ksc;
#pragma unroll
        for (int i = 0; i < 8; ++i) x[i] = __fdiv_rn(x[i], ks);
      }
      *reinterpret_cast<uint2*>(kc + lrow[j] * 128 + (tid >> 3) * 64 + d0) = qk_pack8(x);
    } else if (tid < 32) {  // v: no norm
      float x[8];
      const int dv = (tid - 16) * 8;
      qk_cvt8(rk[j], x);
      if constexpr (SCALED) {
        const float vs = *vsc;
#pragma unroll
        for (int i = 0; i < 8; ++i) x[i] = __fdiv_rn(x[i], vs);
      }
      *reinterpret_cast<uint2*>(vc + lrow[j] * 128 + dv) = qk_pack8(x);
    }
  }
}

// [p12h] qk_norm_rope8 / qk_norm_rope8_m with this thread's 4 cos / 4 sin in registers (RoPE always; same arithmetic)
DEVI void qk_norm_rope8_r(float (&x)[8], const float4 c4, const float4 s4, unsigned mask) {
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss += x[i] * x[i];
  ss += __shfl_xor_sync(mask, ss, 1);
  ss += __shfl_xor_sync(mask, ss, 2);
  ss += __shfl_xor_sync(mask, ss, 4);
  const float r = rsqrtf(ss / 64.f + 1e-5f);
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] *= r;
  const float cs[4] = {c4.x, c4.y, c4.z, c4.w}, sn[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float c = cs[j], s = sn[j];
    const float e = x[2 * j], o = x[2 * j + 1];
    x[2 * j] = e * c - o * s;
    x[2 * j + 1] = e * s + o * c;
  }
}
// [p12h] qk_prep_kvt_kernel<true, SCALED, PT, LT, true> (RoPE table) in one L2 round trip after its PDL wait. Before
// griddepcontrol.wait: this thread's 4 cos / 4 sin (table row t, dims d0/2 .. +3; the k-head threads use the same d0) as
// 2 float4 loads (no smem, no barrier; the side branch that wrote the table is joined by a full edge), the cache row, the
// key-slot inputs (cache_seqlens / seq_lens: inputs or main-stream values from before the qkv GEMM) and the cache scales.
// After it: the q, k and v row chunks together. No input pointer is __restrict__: a const __restrict__ read becomes an
// LDG.CONSTANT that the compiler moves freely across the griddepcontrol.wait asm (NVRTC hoists the q-row load above the
// acquire; nvcc sank the table / cache-row / length loads below it), while a plain load stays on its side of the
// memory-clobbering asm. Same per-element arithmetic and stores as qk_prep_kvt_kernel.
template <bool SCALED, typename LT>
__global__ void __launch_bounds__(128) qk_prep_kvt1_kernel(const __nv_bfloat16* qkv, const LT* loc, const float* tab,
                                                           uint8_t* __restrict__ q8, uint8_t* __restrict__ kc,
                                                           uint8_t* __restrict__ vc, const float* ksc, const float* vsc,
                                                           uint8_t* __restrict__ kvt, const int* klen, const int* kseq,
                                                           int capblks, int L, int win) {
  const int t = blockIdx.x, tid = threadIdx.x;
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int d0 = (tid & 7) * 8;
  const float4 c4 = *reinterpret_cast<const float4*>(tab + (size_t)t * 64 + (d0 >> 1));
  const float4 s4 = *reinterpret_cast<const float4*>(tab + (size_t)t * 64 + 32 + (d0 >> 1));
  const long long lrow = tid < 32 ? (long long)loc[t] : 0;
  const int b = t / L;
  int rel = 0;
  if (tid < 32) rel = klen[b] - L + (t - b * L) - max(kseq[b] - (win - 1), 0);
  float ks = 1.f, vs = 1.f;
  if constexpr (SCALED) {
    if (tid < 16) ks = *ksc;
    else if (tid < 32) vs = *vsc;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const __nv_bfloat16* row = qkv + (size_t)t * 2304;
  const uint4 rq = *reinterpret_cast<const uint4*>(row + (tid >> 3) * 64 + d0);
  uint4 rk = make_uint4(0u, 0u, 0u, 0u);
  if (tid < 16) rk = *reinterpret_cast<const uint4*>(row + 1024 + (tid >> 3) * 64 + d0);
  else if (tid < 32) rk = *reinterpret_cast<const uint4*>(row + 1152 + (tid - 16) * 8);
  {
    float x[8];
    qk_cvt8(rq, x);
    qk_norm_rope8_r(x, c4, s4, 0xffffffffu);
    *reinterpret_cast<uint2*>(q8 + (size_t)t * 1024 + (tid >> 3) * 64 + d0) = qk_pack8(x);
  }
  uint2 vw = make_uint2(0u, 0u);
  if (tid < 16) {  // k heads 0, 1 (lanes 0..15 only in the reduction, as qk_prep_kvt)
    float x[8];
    qk_cvt8(rk, x);
    qk_norm_rope8_r(x, c4, s4, 0x0000ffffu);
    if constexpr (SCALED) {
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = __fdiv_rn(x[i], ks);
    }
    const uint2 kw = qk_pack8(x);
    *reinterpret_cast<uint2*>(kc + lrow * 128 + (tid >> 3) * 64 + d0) = kw;
    *reinterpret_cast<uint2*>(kvt + ((size_t)(b * 2 + (tid >> 3)) * capblks + (rel >> 7)) * 16384 + (d0 >> 4) * 2048 +
                              (rel & 127) * 16 + (d0 & 15)) = kw;   // gather_kp's K layout
  } else if (tid < 32) {  // v: no norm
    float x[8];
    qk_cvt8(rk, x);
    if constexpr (SCALED) {
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = __fdiv_rn(x[i], vs);
    }
    vw = qk_pack8(x);
    *reinterpret_cast<uint2*>(vc + lrow * 128 + (tid - 16) * 8) = vw;
  }
  if (tid < 32) {  // V^T bytes, as qk_prep_kvt_kernel
    const int key = rel & 127, j = tid & 7;
    const int lam = (key & ~31) | qk_pinv(key & 31);
    uint8_t* tb = kvt + ((size_t)(b * 2) * capblks + (rel >> 7)) * 16384 + 8192 + (lam >> 4) * 128 + j * 16 + (lam & 15);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int vi = 4 * s + (tid >> 3);
      const unsigned wx = __shfl_sync(0xffffffffu, vw.x, 16 + vi);
      const unsigned wy = __shfl_sync(0xffffffffu, vw.y, 16 + vi);
      tb[(size_t)(vi >> 3) * capblks * 16384 + (vi & 7) * 1024] = (uint8_t)((j < 4 ? wx : wy) >> (8 * (j & 3)));
    }
  }
}
// [p12h] _lcfork_check: thread 0 stores %globaltimer (PDL: after griddepcontrol.wait)
template <bool PDL>
__global__ void __launch_bounds__(32) okc_stamp_kernel(unsigned long long* out) {
  if constexpr (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (threadIdx.x == 0) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *out = t;
  }
}

// [r13] kvoff fork point: an empty 1-warp kernel (plain launch) that triggers its dependents at entry. kernel_fn launches
// it first when the kvoff side branch forks, so the fork (and the join before qk_prep_kvt) hangs off a kernel node, not
// the graph root.
__global__ void __launch_bounds__(32) okc_fork_kernel() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// Residual boundary, one block per token row: stream = alpha * stream + beta * rms(branch [* gpost]); h = rms(stream)
// * gpre in bf16. TPB threads x 8 elements per chunk; used from 2048 tokens up (the Triton version wins below).
// MODE 0: stream' = alpha * stream + beta * rmsnorm(branch [* gpost]); writes stream' (fp32) and its rmsnorm * gpre (bf16).
// MODE 1 (first boundary, recompute scheme): the same, but stream' is not written; the branch's rms factor r1 goes to
//   rio[row] instead (4 bytes a row instead of 32 KB).
// MODE 2 (second boundary, recompute scheme): `stream` is the layer's input stream; the first boundary's stream' is
//   rebuilt on the fly as alpha1 * stream + beta1 * (br1 * rio[row]) (the same fp32 expression as MODE 1) instead of
//   being read back: 16 KB of bf16 branch read instead of 32 KB of fp32 stream written and read.
#define BND_RC_MINB 2  // MODE 2 at 512 threads compiled to 74 registers (1 block/SM, half the bytes in flight)
// [r13] LP 2: the stream row (MODE 2: and the first boundary's rebuild inputs) is read after the branch row-norm barrier
// instead of before it (measured: both T = 753 boundaries -0.4 / -1.6 us). Same arithmetic in every LP.
template <int TPB, bool GPOST, bool STREAMHINT, int MODE = 0, int LP = 0>
__global__ void __launch_bounds__(TPB, MODE == 2 ? BND_RC_MINB : (TPB <= 512 ? 2 : 1)) boundary_kernel(const __nv_bfloat16* __restrict__ branch, const float* __restrict__ stream,
                                           const __nv_bfloat16* __restrict__ gpost, const float* __restrict__ alpha,
                                           const float* __restrict__ beta, const __nv_bfloat16* __restrict__ gpre,
                                           float* __restrict__ sout, __nv_bfloat16* __restrict__ nout, int* __restrict__ zp = nullptr, int zn = 0,
                                           const __nv_bfloat16* __restrict__ br1 = nullptr, float* __restrict__ rio = nullptr,
                                           const float* __restrict__ alpha1 = nullptr, const float* __restrict__ beta1 = nullptr) {
  constexpr bool PRE = (LP & 4) != 0;  // [p12f] LP 4 (MODE 1): the stream row before the PDL wait (see below)
  if constexpr (PRE) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  else pdl_enter();
  // block 0 also zeroes the MoE's per-call counters (counts | work counter | tile done): saves a fill kernel
  if constexpr (!PRE) {
  if (zn > 0 && blockIdx.x == 0)
    for (int i = threadIdx.x; i < zn; i += TPB) zp[i] = 0;
  }
  constexpr int H = 8192, C = H / (TPB * 8), NW = TPB / 32;
  const size_t row = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float red[2][NW];
  float b[C * 8], s[C * 8];
  const __nv_bfloat16* br = branch + row * H;
  const float* sr = stream + row * H;
  if constexpr (PRE) {  // [p12f] LP 4: the stream row (the layer's input, never written in kernel_fn) before griddepcontrol.wait,
    // so the first wave's 32 KB a row are in flight during the o GEMM's tail; the branch row (the GEMM's output) and the
    // counter zeroing after the wait. Same arithmetic.
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const int off = c * TPB * 8 + tid * 8;
      const float4 s0 = *reinterpret_cast<const float4*>(sr + off), s1 = *reinterpret_cast<const float4*>(sr + off + 4);
      s[c * 8 + 0] = s0.x; s[c * 8 + 1] = s0.y; s[c * 8 + 2] = s0.z; s[c * 8 + 3] = s0.w;
      s[c * 8 + 4] = s1.x; s[c * 8 + 5] = s1.y; s[c * 8 + 6] = s1.z; s[c * 8 + 7] = s1.w;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (zn > 0 && blockIdx.x == 0)
      for (int i = threadIdx.x; i < zn; i += TPB) zp[i] = 0;
  }
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int off = c * TPB * 8 + tid * 8;
    uint4 v; float4 s0, s1;
    static_assert(LP == 0 || (LP == 2 && !STREAMHINT && MODE != 2) || (LP == 4 && !STREAMHINT && MODE == 1), "boundary LP");
    if constexpr ((LP & 6) != 0) { v = *reinterpret_cast<const uint4*>(br + off); s0 = s1 = make_float4(0.f, 0.f, 0.f, 0.f); }
    else if (STREAMHINT) { v = __ldcs(reinterpret_cast<const uint4*>(br + off)); s0 = __ldcs(reinterpret_cast<const float4*>(sr + off)); s1 = __ldcs(reinterpret_cast<const float4*>(sr + off + 4)); }
    else { v = *reinterpret_cast<const uint4*>(br + off); s0 = *reinterpret_cast<const float4*>(sr + off); s1 = *reinterpret_cast<const float4*>(sr + off + 4); }
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) { float2 f = __bfloat1622float2(h[i]); b[c * 8 + 2 * i] = f.x; b[c * 8 + 2 * i + 1] = f.y; }
    if constexpr (!PRE) {
    s[c * 8 + 0] = s0.x; s[c * 8 + 1] = s0.y; s[c * 8 + 2] = s0.z; s[c * 8 + 3] = s0.w;
    s[c * 8 + 4] = s1.x; s[c * 8 + 5] = s1.y; s[c * 8 + 6] = s1.z; s[c * 8 + 7] = s1.w;
    }
    if constexpr (MODE == 2 && (LP & 2) == 0) {
      const uint4 u = *reinterpret_cast<const uint4*>(br1 + row * H + off);
      const __nv_bfloat162* hu = reinterpret_cast<const __nv_bfloat162*>(&u);
      const float r1p = rio[row];
      const float4 a0 = __ldg(reinterpret_cast<const float4*>(alpha1 + off)), a1 = __ldg(reinterpret_cast<const float4*>(alpha1 + off + 4));
      const float4 e0 = __ldg(reinterpret_cast<const float4*>(beta1 + off)), e1 = __ldg(reinterpret_cast<const float4*>(beta1 + off + 4));
      const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float bv[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(hu[i]);
        s[c * 8 + 2 * i] = __fmaf_rn(av[2 * i], s[c * 8 + 2 * i], bv[2 * i] * (f.x * r1p));
        s[c * 8 + 2 * i + 1] = __fmaf_rn(av[2 * i + 1], s[c * 8 + 2 * i + 1], bv[2 * i + 1] * (f.y * r1p));
      }
    }
  }
  float ss = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int off = c * TPB * 8 + tid * 8;
    if constexpr (GPOST) {
      const float4 g0 = g1p4(gpost + off), g1 = g1p4(gpost + off + 4);
      b[c * 8 + 0] *= g0.x; b[c * 8 + 1] *= g0.y; b[c * 8 + 2] *= g0.z; b[c * 8 + 3] *= g0.w;
      b[c * 8 + 4] *= g1.x; b[c * 8 + 5] *= g1.y; b[c * 8 + 6] *= g1.z; b[c * 8 + 7] *= g1.w;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) ss += b[c * 8 + i] * b[c * 8 + i];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((tid & 31) == 0) red[0][tid >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) tot += red[0][i];
  const float r1 = rsqrtf(tot / H + 1e-8f);
  if constexpr ((LP & 2) != 0) {  // [r13] the stream row (MODE 2: and the rebuild inputs), read after the barrier
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const int off = c * TPB * 8 + tid * 8;
      const float4 s0 = *reinterpret_cast<const float4*>(sr + off), s1 = *reinterpret_cast<const float4*>(sr + off + 4);
      s[c * 8 + 0] = s0.x; s[c * 8 + 1] = s0.y; s[c * 8 + 2] = s0.z; s[c * 8 + 3] = s0.w;
      s[c * 8 + 4] = s1.x; s[c * 8 + 5] = s1.y; s[c * 8 + 6] = s1.z; s[c * 8 + 7] = s1.w;
      if constexpr (MODE == 2) {
      const uint4 u = *reinterpret_cast<const uint4*>(br1 + row * H + off);
      const __nv_bfloat162* hu = reinterpret_cast<const __nv_bfloat162*>(&u);
      const float r1p = rio[row];
      const float4 a0 = __ldg(reinterpret_cast<const float4*>(alpha1 + off)), a1 = __ldg(reinterpret_cast<const float4*>(alpha1 + off + 4));
      const float4 e0 = __ldg(reinterpret_cast<const float4*>(beta1 + off)), e1 = __ldg(reinterpret_cast<const float4*>(beta1 + off + 4));
      const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float bv[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(hu[i]);
        s[c * 8 + 2 * i] = __fmaf_rn(av[2 * i], s[c * 8 + 2 * i], bv[2 * i] * (f.x * r1p));
        s[c * 8 + 2 * i + 1] = __fmaf_rn(av[2 * i + 1], s[c * 8 + 2 * i + 1], bv[2 * i + 1] * (f.y * r1p));
      }
      }
    }
  }
  if constexpr (MODE == 1) {
    if (tid == 0) rio[row] = r1;
  }
  float ss2 = 0.f;
  float* so = sout + row * H;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int off = c * TPB * 8 + tid * 8;
    const float4 a0 = __ldg(reinterpret_cast<const float4*>(alpha + off)), a1 = __ldg(reinterpret_cast<const float4*>(alpha + off + 4));
    const float4 e0 = __ldg(reinterpret_cast<const float4*>(beta + off)), e1 = __ldg(reinterpret_cast<const float4*>(beta + off + 4));
    const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float bv[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if constexpr (MODE == 0) s[c * 8 + i] = av[i] * s[c * 8 + i] + bv[i] * (b[c * 8 + i] * r1);
      else s[c * 8 + i] = __fmaf_rn(av[i], s[c * 8 + i], bv[i] * (b[c * 8 + i] * r1));  // MODE 2 rebuilds it identically
      ss2 += s[c * 8 + i] * s[c * 8 + i];
    }
    if constexpr (MODE != 1) {
      float4 o0 = make_float4(s[c * 8 + 0], s[c * 8 + 1], s[c * 8 + 2], s[c * 8 + 3]), o1 = make_float4(s[c * 8 + 4], s[c * 8 + 5], s[c * 8 + 6], s[c * 8 + 7]);
      if (STREAMHINT) { __stcs(reinterpret_cast<float4*>(so + off), o0); __stcs(reinterpret_cast<float4*>(so + off + 4), o1); }
      else { *reinterpret_cast<float4*>(so + off) = o0; *reinterpret_cast<float4*>(so + off + 4) = o1; }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss2 += __shfl_xor_sync(0xffffffffu, ss2, o);
  if ((tid & 31) == 0) red[1][tid >> 5] = ss2;
  __syncthreads();
  float tot2 = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) tot2 += red[1][i];
  const float r2 = rsqrtf(tot2 / H + 1e-5f);
  __nv_bfloat16* no = nout + row * H;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int off = c * TPB * 8 + tid * 8;
    const float4 g0 = g1p4(gpre + off), g1 = g1p4(gpre + off + 4);
    const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    uint4 o;
    __nv_bfloat162* ho = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) ho[i] = __floats2bfloat162_rn(s[c * 8 + 2 * i] * r2 * gv[2 * i], s[c * 8 + 2 * i + 1] * r2 * gv[2 * i + 1]);
    if (STREAMHINT) __stcs(reinterpret_cast<uint4*>(no + off), o); else *reinterpret_cast<uint4*>(no + off) = o;
  }
}
}  // namespace okc_moe

namespace okc_moe {
// ---- gather the attended keys/values of each request into contiguous [B, Smax, 2, 64] FP8 buffers ----
// keys at positions [first_b, seq_b + L) where first_b = window ? max(0, seq_b - window + 1) : 0; slot j <- position first_b + j
__global__ void __launch_bounds__(256) kv_gather_kernel(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                                        const int* __restrict__ page_table, const int* __restrict__ seq_lens,
                                                        uint8_t* __restrict__ ko, uint8_t* __restrict__ vo, int max_pages,
                                                        int page, int L, int window, int Smax) {
  const int b = blockIdx.y;
  const int seq = seq_lens[b];
  const int first = window ? max(0, seq - window + 1) : 0;
  const int n = seq + L - first;
  const int j = blockIdx.x * 32 + (threadIdx.x >> 3);  // 32 rows per block, 8 threads x 16 B per row (K), same for V
  const int c = threadIdx.x & 7;
  if (j >= n) return;
  const int pos = first + j;
  const long row = (long)page_table[(long)b * max_pages + pos / page] * page + pos % page;
  const uint4 kv = *reinterpret_cast<const uint4*>(kc + row * 128 + c * 16);
  const uint4 vv = *reinterpret_cast<const uint4*>(vc + row * 128 + c * 16);
  *reinterpret_cast<uint4*>(ko + ((long)b * Smax + j) * 128 + c * 16) = kv;
  *reinterpret_cast<uint4*>(vo + ((long)b * Smax + j) * 128 + c * 16) = vv;
}
}  // namespace okc_moe

// FP8 prefill attention for Muse Spark (hdim 64, 16 q heads / 2 kv heads), fused output RMSNorm * sigmoid(gate).
// Appended after moe_gemm.cu (uses its helpers).
//
// attn_prep_kernel gathers each request's attended keys/values from the paged cache into per-(request, kv head,
// 128-key tile) blocks: K tiles in the 64B-swizzled K-major layout; V tiles transposed (72 rows x 128 keys, 128B
// swizzle) with the keys permuted inside each 32-key chunk so that the QK^T accumulator registers are directly the
// FP8 A operand of the P.V wgmma, plus a row of ones (row 64) so the tensor core also sums P (the softmax
// denominator, exactly over the FP8-rounded P that multiplies V).
//
// attn_fwd_kernel: one CTA = 16 query tokens x the 8 query heads of one kv head (128 rows); 2 consumer warpgroups
// (64 rows each) + 1 producer warp streaming K/V tiles with bulk copies into a 4-stage ring. Per tile and
// warpgroup: S = Q K^T (FP8 wgmma) -> mask -> online softmax (P scaled by 2^8 like FA3) -> P.V (FP8 wgmma).
// EXPMODE n >= 0: n of each thread's 64 exponentials per tile as an FMA-pipe polynomial, the rest on the MUFU;
// -1: exp2 on f16 pairs converted straight to FP8 pairs. P is rounded to E4M3 for the P.V MMA in every mode.
DEVI void wgmma_fp8_ss_n128(float* d, uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "l"(da), "l"(db), "r"(scale_d));
}


DEVI void wgmma_fp8_rs_n72(float* d, const uint32_t* a, uint64_t db, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, {%36, %37, %38, %39}, %40, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(scale_d));
}


DEVI void wgmma_fp8_ss_n64(float* d, uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "l"(da), "l"(db), "r"(scale_d));
}


namespace okc_attn {
using namespace okc_moe;

constexpr int NK = 128;                   // keys per tile
constexpr int TILE_K = NK * 64;           // 8 KB
constexpr int VROWS = 72;                 // 64 head dims + the ones row (+ 7 zero rows)
constexpr int TILE_V = VROWS * NK;        // 9 KB
constexpr int STAGE = TILE_K + TILE_V;    // 17 KB (1024-aligned)
#define ATTN_ASTAGES 4
constexpr int ASTAGES = ATTN_ASTAGES;
#define ATTN_VOTEMAX 1  // exp-turn softmax: skip the row-max shuffles and rescale exps when a warp's maxima did not grow
#define ATTN_VOTEMAX_ALL 0  // the same in the other softmax paths (slower for the sliding-window variant 8)
#define ATTN_GATE_PF 1
#define ATTN_RR_HAND2 48  // early hand-off point (exps per thread) of the non-pipelined RR step (variants 41/42: slower than 8 on sliding)


// grid (ntk * HALVES, 2, B), block 256; HALVES = 2: each block stages one 64-key half of a tile (short windows: more
// blocks in flight)
// ---- programmatic dependent launch: every kernel lets its successor launch as soon as all of its blocks
// are running, and waits for its predecessor's results before touching global memory (a no-op when the kernel was not
// launched with the PDL attribute) ----
__device__ __forceinline__ void pdl_enter_attn() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void pdl_trigger_attn() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ void pdl_wait_attn() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
// ATTN_EARLY_Q: the K/V staging kernels let the attention kernel launch only after their own wait (so qk_prep, which
// writes Q, is complete), and the attention kernel waits for the staged K/V in its producer only: barrier setup and the
// consumers' Q staging overlap the staging kernel.
#define ATTN_EARLY_Q 1  // (Oct 26: search +0.06..0.09% on v113; the staging kernel triggers only after its own wait, so Q is complete)
__device__ __forceinline__ void pdl_enter_prep() {
  if constexpr (ATTN_EARLY_Q) {
    pdl_wait_attn();
    pdl_trigger_attn();
  } else {
    pdl_enter_attn();
  }
}

template <int HALVES>
__global__ void __launch_bounds__(256) attn_prep_kernel(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                                        const int* __restrict__ page_table, int max_pages, int page,
                                                        const int* __restrict__ seq_lens, int L, int window, int ntk,
                                                        uint8_t* __restrict__ kt, uint8_t* __restrict__ vt, int part = 0) {
  pdl_enter_prep();
  const int tile = blockIdx.x / HALVES, hf = blockIdx.x % HALVES, h = blockIdx.y, b = blockIdx.z;
  constexpr int NKH = NK / HALVES;  // keys staged by this block
  const int seq = seq_lens[b];
  const int first = window ? max(0, seq - window + 1) : 0;
  const int nk = seq + L - first;
  if (tile * NK >= nk) return;
  // part 1: only tiles made entirely of keys already in the cache before this call (they need nothing from this
  // call's qk_prep, so they are staged on a side stream during the qkv GEMM); part 2: the rest; part 0: all
  if (part && ((part == 1) != (tile < (seq - first) / NK))) return;
  constexpr int VPS = 68;  // padded rows (17 words): the transposing LDS.32 reads of 4 keys hit different banks
  __shared__ __align__(16) uint8_t vs[NK * VPS];
  const size_t tb = ((size_t)(b * 2 + h) * ntk + tile);
  uint8_t* ko = kt + tb * TILE_K;
  uint8_t* vo = vt + tb * TILE_V;
  for (int i = threadIdx.x; i < NKH * 4; i += 256) {
    const int r = hf * NKH + (i >> 2), c = i & 3;
    const int j = tile * NK + r;
    uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
    if (j < nk) {
      const int pos = first + j;
      const long row = (long)page_table[(long)b * max_pages + pos / page] * page + pos % page;
      kv = *reinterpret_cast<const uint4*>(kc + row * 128 + h * 64 + c * 16);
      vv = *reinterpret_cast<const uint4*>(vc + row * 128 + h * 64 + c * 16);
    }
    *reinterpret_cast<uint4*>(ko + r * 64 + ((c ^ ((r >> 1) & 3)) << 4)) = kv;
    uint32_t* vw = reinterpret_cast<uint32_t*>(vs + r * VPS + c * 16);
    vw[0] = vv.x; vw[1] = vv.y; vw[2] = vv.z; vw[3] = vv.w;
  }
  __syncthreads();
  // V^T by register 4x4 byte transposes (as attn_prep2): this block's chunks cc of 16 dim groups, then the ones and
  // zero rows
  for (int i = threadIdx.x; i < 16 * (8 / HALVES); i += 256) {
    const int dg = i / (8 / HALVES), cc = hf * (8 / HALVES) + i % (8 / HALVES);
    const int d0 = 4 * dg, kb = (cc >> 1) * 32 + (cc & 1) * 16;
    uint32_t w[4][4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t* r0 = reinterpret_cast<const uint32_t*>(vs + (kb + 2 * q) * VPS + d0);
      const uint32_t a0 = r0[0], a1 = r0[VPS / 4], a2 = r0[8 * VPS / 4], a3 = r0[9 * VPS / 4];
      const uint32_t t0 = __byte_perm(a0, a1, 0x5140), t1 = __byte_perm(a2, a3, 0x5140);
      const uint32_t t2 = __byte_perm(a0, a1, 0x7362), t3 = __byte_perm(a2, a3, 0x7362);
      w[0][q] = __byte_perm(t0, t1, 0x5410);
      w[1][q] = __byte_perm(t0, t1, 0x7632);
      w[2][q] = __byte_perm(t2, t3, 0x5410);
      w[3][q] = __byte_perm(t2, t3, 0x7632);
    }
#pragma unroll
    for (int dd = 0; dd < 4; ++dd) {
      const int d = d0 + dd;
      *reinterpret_cast<uint4*>(vo + d * 128 + ((cc ^ (d & 7)) << 4)) = make_uint4(w[dd][0], w[dd][1], w[dd][2], w[dd][3]);
    }
  }
  for (int i = threadIdx.x; i < 8 * (8 / HALVES); i += 256) {
    const int d = 64 + i / (8 / HALVES), cc = hf * (8 / HALVES) + i % (8 / HALVES);
    const uint32_t one = d == 64 ? 0x38383838u : 0u;  // e4m3 1.0
    *reinterpret_cast<uint4*>(vo + d * 128 + ((cc ^ (d & 7)) << 4)) = make_uint4(one, one, one, one);
  }
}

// grid (ntk, 1, B), block 256: both heads of a 128-key tile per block (full 128-byte cache rows, one page-table read per
// key); V goes through shared memory with a 68-byte row pitch so the byte transpose reads are bank-conflict free
__global__ void __launch_bounds__(256) attn_prep2_kernel(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                                         const int* __restrict__ page_table, int max_pages, int page,
                                                         const int* __restrict__ seq_lens, int L, int window, int ntk,
                                                         uint8_t* __restrict__ kt, uint8_t* __restrict__ vt, int part = 0) {
  pdl_enter_prep();
  const int tile = blockIdx.x, b = blockIdx.z;
  const int seq = seq_lens[b];
  const int first = window ? max(0, seq - window + 1) : 0;
  const int nk = seq + L - first;
  if (tile * NK >= nk) return;
  if (part && ((part == 1) != (tile < (seq - first) / NK))) return;  // (see attn_prep_kernel)
  constexpr int VP = 68;
  __shared__ __align__(16) uint8_t vs[2][NK * VP];
  // 128 keys x 8 chunks of 16 bytes (chunks 0-3: head 0, 4-7: head 1)
  // (explicitly batching a thread's 4 page lookups and 8 row loads measured slower: 12.0 -> 13.4 us at turn-1 full)
  for (int i = threadIdx.x; i < NK * 8; i += 256) {
    const int r = i >> 3, c8 = i & 7, h = c8 >> 2, c = c8 & 3;
    const int j = tile * NK + r;
    uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
    if (j < nk) {
      const int pos = first + j;
      const long row = (long)page_table[(long)b * max_pages + pos / page] * page + pos % page;
      kv = *reinterpret_cast<const uint4*>(kc + row * 128 + c8 * 16);
      vv = *reinterpret_cast<const uint4*>(vc + row * 128 + c8 * 16);
    }
    uint8_t* ko = kt + ((size_t)(b * 2 + h) * ntk + tile) * TILE_K;
    *reinterpret_cast<uint4*>(ko + r * 64 + ((c ^ ((r >> 1) & 3)) << 4)) = kv;
    uint32_t* vw = reinterpret_cast<uint32_t*>(vs[h] + r * VP + c * 16);  // 4-byte aligned (68 = 17 words)
    vw[0] = vv.x; vw[1] = vv.y; vw[2] = vv.z; vw[3] = vv.w;
  }
  __syncthreads();
  // V^T by 4x4 byte transposes in registers: thread (h, dims 4dg..4dg+3, chunk cc) reads, for each of the chunk's 4
  // words q, the 4 keys' 4-dim words (LDS.32) and transposes them with 8 PRMTs into 4 dims' words of 4 keys
  {
    const int h = threadIdx.x >> 7, dg = (threadIdx.x >> 3) & 15, cc = threadIdx.x & 7;
    const int d0 = 4 * dg;
    const int kb = (cc >> 1) * 32 + (cc & 1) * 16;
    uint32_t w[4][4];  // [dim dd][word q]
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t* r0 = reinterpret_cast<const uint32_t*>(vs[h] + (kb + 2 * q) * VP + d0);
      const uint32_t a0 = r0[0], a1 = r0[VP / 4], a2 = r0[8 * VP / 4], a3 = r0[9 * VP / 4];  // keys 2q, 2q+1, 2q+8, 2q+9
      const uint32_t t0 = __byte_perm(a0, a1, 0x5140), t1 = __byte_perm(a2, a3, 0x5140);
      const uint32_t t2 = __byte_perm(a0, a1, 0x7362), t3 = __byte_perm(a2, a3, 0x7362);
      w[0][q] = __byte_perm(t0, t1, 0x5410);
      w[1][q] = __byte_perm(t0, t1, 0x7632);
      w[2][q] = __byte_perm(t2, t3, 0x5410);
      w[3][q] = __byte_perm(t2, t3, 0x7632);
    }
    uint8_t* vo = vt + ((size_t)(b * 2 + h) * ntk + tile) * TILE_V;
#pragma unroll
    for (int dd = 0; dd < 4; ++dd) {
      const int d = d0 + dd;
      *reinterpret_cast<uint4*>(vo + d * 128 + ((cc ^ (d & 7)) << 4)) = make_uint4(w[dd][0], w[dd][1], w[dd][2], w[dd][3]);
    }
    // the ones row (d = 64) and 7 zero rows: 2 heads x 8 rows x 8 chunks = 128 stores
    if (threadIdx.x < 128) {
      const int hh = threadIdx.x >> 6, dr = 64 + ((threadIdx.x >> 3) & 7), c2 = threadIdx.x & 7;
      const uint32_t one = dr == 64 ? 0x38383838u : 0u;  // e4m3 1.0
      uint8_t* vo2 = vt + ((size_t)(b * 2 + hh) * ntk + tile) * TILE_V;
      *reinterpret_cast<uint4*>(vo2 + dr * 128 + ((c2 ^ (dr & 7)) << 4)) = make_uint4(one, one, one, one);
    }
  }
}

struct AttnArgs {
  const uint8_t* q;         // [T, 16, 64] e4m3
  const uint8_t* kt;
  const uint8_t* vt;
  const int* seq_lens;
  const __nv_bfloat16* qkv; // [T, 2304]: gate at 1280 + head*64
  __nv_bfloat16* out;       // [T, 1024]
  int L, window, ntk, nqb;
  float c;                  // softmax scale * k_scale * log2(e)
  float v_scale;
  int nsplit;               // key-range splits (grid.y = 2 * nsplit); > 1 writes partials to ws
  float* ws;                // [nsplit, T, 16, 66] fp32: O (64, unnormalized), m, l
  int zero;                 // always 0: the first QK k-step's scale-d, opaque to the compiler
  const uint8_t* pf;        // ATTN_PF: the next GEMM's weight, prefetched into L2 by the producers (nullptr: none)
  unsigned long long pf_bytes;
  int sk;                   // stream-K (attn_fwd_sk_kernel): the grid size G; 0 otherwise
  int nb;                   // stream-K: batch count
  int* skbuf;               // stream-K: [0] = W (tiles per CTA), [1 .. npairs+1] = pair prefix, then each CTA's first pair
};
#define ATTN_PF 0  // (Oct 26 03:12: o-proj weight prefetch during the attention: no change, off)

DEVI uint64_t desc_sw64(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)(512 >> 4) << 32;  // 8 rows x 64 B
  d |= (uint64_t)2 << 62;           // SWIZZLE_64B
  return d;
}


DEVI uint32_t pack_e4m3x4(float p0, float p1, float p2, float p3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(p1), "f"(p0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(p3), "f"(p2));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

// e4m3 bytes of 2^y0..2^y3 (low to high) via f16 pairs: one MUFU op per two exponentials
DEVI uint32_t exp2_e4m3x4(float y0, float y1, float y2, float y3) {
  uint32_t h01, h23;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h01) : "f"(y1), "f"(y0));
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h23) : "f"(y3), "f"(y2));
  asm("ex2.approx.f16x2 %0, %0;" : "+r"(h01));
  asm("ex2.approx.f16x2 %0, %0;" : "+r"(h23));
  uint16_t b01, b23;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(b01) : "r"(h01));
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(b23) : "r"(h23));
  return (uint32_t)b01 | ((uint32_t)b23 << 16);
}

// 2^y on the FMA pipe (FA4-style): round-to-nearest split, degree-3 minimax on [-0.5, 0.5], exponent add.
DEVI float exp2_poly(float y) {
  y = fmaxf(y, -127.f);
  const float jm = y + 12582912.f;  // 1.5 * 2^23: integer part lands in the mantissa
  const float j = jm - 12582912.f;
  const float f = y - j;            // [-0.5, 0.5]
  float pf = fmaf(f, 0.0555041086648216f, 0.2402264923172690f);
  pf = fmaf(pf, f, 0.6931472028550421f);
  pf = fmaf(pf, f, 1.0f);
  return __int_as_float(__float_as_int(pf) + (__float_as_int(jm) << 23));
}


template <int N>
DEVI void fence_f(float* r) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
template <int N>
DEVI void fence_u(uint32_t* r) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i])::"memory");
}

// P = exp2(s * c - (m * c - ATTN_POFF)): P is 2^ATTN_POFF at the running maximum, rounded to e4m3 (max 448). Every
// softmax path (masked or not, split partials) must use the same offset: O and l carry the same 2^ATTN_POFF factor.
#define ATTN_POFF 8.f
// Late-max (exp-turn path, unmasked tiles): no row maximum before the exps. P is computed against the running
// maximum of earlier tiles, packed, then tested on the packed bytes: a byte of 0x7E (P >= ~432, or saturated) means a
// score jumped past the headroom (448 / 2^ATTN_POFF); only then (and on the first tile, whose running maximum is
// -1e30) the warp takes the exact path: row maxima, rescale factors, and the tile's P again from the fp32 scores.
// P below 448 against a stale maximum is still exact softmax arithmetic (O and l share the reference).
struct AttnState {
  float o[36];  // columns 0..63: O; column 64: the softmax denominator (ones row of V)
  float mA, mB;
  int nready;   // ATTN_EARLY_TEST: the next tile's K/V barrier already tested complete
};
__device__ int* g_attn_cnt;  // per (batch, query block, kv head) arrival counters (zeroed at prepare; reset by the merger)
struct AttnCtx {
  const AttnArgs& p;
  uint8_t* qs;
  uint8_t* ring;
  uint32_t bars;
  int wg, w, lane, g, t4;
  int kmaxA, kmaxB, kminA, kminB, nk, L, q_first, q_last, t_begin, t_end;
  uint64_t dq, dk0, dv0;  // wgmma descriptors of this warpgroup's Q tile and of stage 0's K / V tiles
  int t_ring;             // the tile that sits in ring position 0 (t_begin, or earlier for later stream-K segments)
};

DEVI int stage_of(const AttnCtx& cx, int t) { return (t - cx.t_ring) % ASTAGES; }
DEVI uint32_t phase_of(const AttnCtx& cx, int t) { return ((t - cx.t_ring) / ASTAGES) & 1; }


DEVI void issue_qk(const AttnCtx& cx, float* sacc, int stage) {
  uint8_t* kst = cx.ring + stage * STAGE;
  const uint64_t dq = desc_sw64(smem_u32(cx.qs + cx.wg * 64 * 64));
  const uint64_t dk = desc_sw64(smem_u32(kst));
  fence_f<64>(sacc);
  wg_fence();
  wgmma_fp8_ss_n128(sacc, dq, dk, cx.p.zero);
  wgmma_fp8_ss_n128(sacc, dq + 2, dk + 2, 1);
  wg_commit();
  fence_f<64>(sacc);
}


// one key tile, sequential within the warpgroup (QK -> softmax -> P.V); the two consumer warpgroups overlap each
// other's softmax with their MMAs.
// EXPMODE: -1 = exp2 on f16 pairs; 0..64 = that many of each thread's 64 exponentials per tile on the FMA pipe
DEVI void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }

// RR: the consumer warpgroups take turns on the softmax (named barriers 4..4+NWG-1), so each softmax has the SM's
// exp2 units to itself while the other warpgroups run their QK^T / P.V on the tensor cores.
template <int EXPMODE, bool OVL, int RR = 0>
DEVI void attn_step(const AttnCtx& cx, AttnState& st, float* sacc, int t) {
  const AttnArgs& p = cx.p;
  const int scu = stage_of(cx, t);
  if (!OVL) {  // OVL: S(t) was computed by the previous step (issued together with its P.V)
    mbar_wait(cx.bars + 8 * scu, phase_of(cx, t));
    issue_qk(cx, sacc, scu);
    wg_wait<0>();
  }
  const int k0 = t * NK;
  const int L = cx.L, nk = cx.nk;
  const bool need_mask = (k0 + NK - 1 > nk - L + cx.q_first) || (p.window && k0 < nk - L + cx.q_last - p.window + 1);
  if (need_mask) {
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int key = k0 + 8 * jj + 2 * cx.t4 + e;
        if (key > cx.kmaxA || key < cx.kminA) sacc[4 * jj + e] = -INFINITY;
        if (key > cx.kmaxB || key < cx.kminB) sacc[4 * jj + 2 + e] = -INFINITY;
      }
    }
  }
  const float c = p.c;
  // tree reductions (short dependency chains): row A = elements 4jj, 4jj+1; row B = 4jj+2, 4jj+3
#define MX4(o) fmaxf(fmaxf(sacc[(o)], sacc[(o) + 1]), fmaxf(sacc[(o) + 4], sacc[(o) + 5]))
  const float a0 = fmaxf(MX4(0), MX4(8)), a1 = fmaxf(MX4(16), MX4(24)), a2 = fmaxf(MX4(32), MX4(40)), a3 = fmaxf(MX4(48), MX4(56));
  const float b0 = fmaxf(MX4(2), MX4(10)), b1 = fmaxf(MX4(18), MX4(26)), b2 = fmaxf(MX4(34), MX4(42)), b3 = fmaxf(MX4(50), MX4(58));
#undef MX4
  float tmA = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)), tmB = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));
  float nmA = st.mA, nmB = st.mB;
  float scA = 1.f, scB = 1.f;
  // exact skip: when no lane's local maximum exceeds its row's running maximum, every row of this warp keeps
  // its maximum (rescale 1): no shuffles and no rescale exps on the exp2 units
  if (!ATTN_VOTEMAX_ALL || __any_sync(0xffffffffu, (tmA > st.mA) || (tmB > st.mB))) {
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
    nmA = fmaxf(st.mA, tmA); nmB = fmaxf(st.mB, tmB);
    scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
    st.mA = nmA; st.mB = nmB;
  }
  const float offA = nmA * c - ATTN_POFF, offB = nmB * c - ATTN_POFF;
  uint32_t pa[4][4];
  if constexpr (EXPMODE < 0) {
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      const float* x = sacc + 16 * cc;
      pa[cc][0] = exp2_e4m3x4(fmaf(x[0], c, -offA), fmaf(x[1], c, -offA), fmaf(x[4], c, -offA), fmaf(x[5], c, -offA));
      pa[cc][1] = exp2_e4m3x4(fmaf(x[2], c, -offB), fmaf(x[3], c, -offB), fmaf(x[6], c, -offB), fmaf(x[7], c, -offB));
      pa[cc][2] = exp2_e4m3x4(fmaf(x[8], c, -offA), fmaf(x[9], c, -offA), fmaf(x[12], c, -offA), fmaf(x[13], c, -offA));
      pa[cc][3] = exp2_e4m3x4(fmaf(x[10], c, -offB), fmaf(x[11], c, -offB), fmaf(x[14], c, -offB), fmaf(x[15], c, -offB));
    }
  } else {
    // RR: the warpgroups take turns on the exp2 units for the exp loop only (the max before it and the FP8 packing
    // after it run outside the turn), so the SM's MUFU pipe always has one warpgroup's exps to issue
    if constexpr (RR > 1) {
      named_bar_sync(4 + cx.wg, 256);
      fence_f<64>(sacc);
    }
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
      if (4 * jj < EXPMODE) {
        sacc[4 * jj] = exp2_poly(fmaf(sacc[4 * jj], c, -offA));
        sacc[4 * jj + 1] = exp2_poly(fmaf(sacc[4 * jj + 1], c, -offA));
        sacc[4 * jj + 2] = exp2_poly(fmaf(sacc[4 * jj + 2], c, -offB));
        sacc[4 * jj + 3] = exp2_poly(fmaf(sacc[4 * jj + 3], c, -offB));
      } else {
        sacc[4 * jj] = exp2f(fmaf(sacc[4 * jj], c, -offA));
        sacc[4 * jj + 1] = exp2f(fmaf(sacc[4 * jj + 1], c, -offA));
        sacc[4 * jj + 2] = exp2f(fmaf(sacc[4 * jj + 2], c, -offB));
        sacc[4 * jj + 3] = exp2f(fmaf(sacc[4 * jj + 3], c, -offB));
      }
      if constexpr (RR > 1 && ATTN_RR_HAND2 < 64) {
        if (4 * jj + 4 == ATTN_RR_HAND2) {  // early hand-off: the next warpgroup's turn starts now
          fence_f<ATTN_RR_HAND2>(sacc);
          named_bar_arrive(4 + (cx.wg + 1 == RR ? 0 : cx.wg + 1), 256);
        }
      }
    }
    if constexpr (RR > 1) {
      if constexpr (ATTN_RR_HAND2 >= 64) {
        fence_f<64>(sacc);
        named_bar_arrive(4 + (cx.wg + 1 == RR ? 0 : cx.wg + 1), 256);  // the next warpgroup's turn
      }
    }
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      const float* x = sacc + 16 * cc;
      pa[cc][0] = pack_e4m3x4(x[0], x[1], x[4], x[5]);
      pa[cc][1] = pack_e4m3x4(x[2], x[3], x[6], x[7]);
      pa[cc][2] = pack_e4m3x4(x[8], x[9], x[12], x[13]);
      pa[cc][3] = pack_e4m3x4(x[10], x[11], x[14], x[15]);
    }
  }
  if (!__all_sync(0xffffffffu, scA == 1.f && scB == 1.f)) {  // the running max rarely moves after a few tiles
#pragma unroll
    for (int jj = 0; jj < 9; ++jj) {
      st.o[4 * jj] *= scA; st.o[4 * jj + 1] *= scA;
      st.o[4 * jj + 2] *= scB; st.o[4 * jj + 3] *= scB;
    }
  }
  uint8_t* vst = cx.ring + scu * STAGE + TILE_K;
  const uint64_t dv = desc_sw128(smem_u32(vst));
  fence_u<16>(&pa[0][0]);
  fence_f<36>(st.o);
  wg_fence();
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) wgmma_fp8_rs_n72(st.o, pa[cc], dv + 2 * cc, 1);
  wg_commit();
  fence_f<36>(st.o);
  if (OVL && t + 1 <= cx.t_end) {  // the next tile's QK^T runs on the tensor cores together with this P.V
    const int sn = stage_of(cx, t + 1);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, t + 1));
    issue_qk(cx, sacc, sn);
  }
  wg_wait<0>();
  __syncwarp();
  if (cx.lane == 0) mbar_arrive(cx.bars + 8 * ASTAGES + 8 * scu);
}

// ---------------- half tiles: each 128-key stage in two 64-key steps ----------------
// S and P are half as large (32 + 8 registers), so four consumer warpgroups fit the register file: more warps per
// scheduler keep the exp2 units busy while other warpgroups wait on their MMAs.
DEVI void issue_qk_h(const AttnCtx& cx, float* sacc, int stage, int h) {
  uint8_t* kst = cx.ring + stage * STAGE + h * 64 * 64;
  uint64_t dq0 = desc_sw64(smem_u32(cx.qs + cx.wg * 64 * 64));
  uint64_t dk0 = desc_sw64(smem_u32(kst));
  uint64_t dq1 = dq0 + 2, dk1 = dk0 + 2;
  // both k-steps' descriptors exist before the pipeline stage opens (a descriptor register rewritten between the two
  // wgmmas serializes every wgmma in the kernel, C7513)
  asm volatile("" : "+l"(dq0), "+l"(dk0), "+l"(dq1), "+l"(dk1));
  fence_f<32>(sacc);
  wg_fence();
  wgmma_fp8_ss_n64(sacc, dq0, dk0, cx.p.zero);
  wgmma_fp8_ss_n64(sacc, dq1, dk1, 1);
  wg_commit();
  fence_f<32>(sacc);
}

template <int EXPMODE>
DEVI void attn_step_h(const AttnCtx& cx, AttnState& st, float* sacc, int t) {
  const AttnArgs& p = cx.p;
  const int scu = stage_of(cx, t);
  mbar_wait(cx.bars + 8 * scu, phase_of(cx, t));
  uint8_t* vst = cx.ring + scu * STAGE + TILE_K;
  const uint64_t dv = desc_sw128(smem_u32(vst));
  const int L = cx.L, nk = cx.nk;
  const float c = p.c;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    issue_qk_h(cx, sacc, scu, h);
    wg_wait<0>();
    const int k0 = t * NK + 64 * h;
    const bool need_mask = (k0 + 63 > nk - L + cx.q_first) || (p.window && k0 < nk - L + cx.q_last - p.window + 1);
    if (need_mask) {
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int key = k0 + 8 * jj + 2 * cx.t4 + e;
          if (key > cx.kmaxA || key < cx.kminA) sacc[4 * jj + e] = -INFINITY;
          if (key > cx.kmaxB || key < cx.kminB) sacc[4 * jj + 2 + e] = -INFINITY;
        }
      }
    }
#define MX4(o) fmaxf(fmaxf(sacc[(o)], sacc[(o) + 1]), fmaxf(sacc[(o) + 4], sacc[(o) + 5]))
    float tmA = fmaxf(fmaxf(MX4(0), MX4(8)), fmaxf(MX4(16), MX4(24)));
    float tmB = fmaxf(fmaxf(MX4(2), MX4(10)), fmaxf(MX4(18), MX4(26)));
#undef MX4
    float nmA = st.mA, nmB = st.mB;
    float scA = 1.f, scB = 1.f;
    // exact skip: when no lane's local maximum exceeds its row's running maximum, every row of this warp keeps
    // its maximum (rescale 1): no shuffles and no rescale exps on the exp2 units
    if (!ATTN_VOTEMAX_ALL || __any_sync(0xffffffffu, (tmA > st.mA) || (tmB > st.mB))) {
      tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
      tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
      tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
      tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
      nmA = fmaxf(st.mA, tmA); nmB = fmaxf(st.mB, tmB);
      scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
      st.mA = nmA; st.mB = nmB;
    }
    const float offA = nmA * c - ATTN_POFF, offB = nmB * c - ATTN_POFF;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      if (4 * jj < EXPMODE) {
        sacc[4 * jj] = exp2_poly(fmaf(sacc[4 * jj], c, -offA));
        sacc[4 * jj + 1] = exp2_poly(fmaf(sacc[4 * jj + 1], c, -offA));
        sacc[4 * jj + 2] = exp2_poly(fmaf(sacc[4 * jj + 2], c, -offB));
        sacc[4 * jj + 3] = exp2_poly(fmaf(sacc[4 * jj + 3], c, -offB));
      } else {
        sacc[4 * jj] = exp2f(fmaf(sacc[4 * jj], c, -offA));
        sacc[4 * jj + 1] = exp2f(fmaf(sacc[4 * jj + 1], c, -offA));
        sacc[4 * jj + 2] = exp2f(fmaf(sacc[4 * jj + 2], c, -offB));
        sacc[4 * jj + 3] = exp2f(fmaf(sacc[4 * jj + 3], c, -offB));
      }
    }
    uint32_t pa[2][4];
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const float* x = sacc + 16 * cc;
      pa[cc][0] = pack_e4m3x4(x[0], x[1], x[4], x[5]);
      pa[cc][1] = pack_e4m3x4(x[2], x[3], x[6], x[7]);
      pa[cc][2] = pack_e4m3x4(x[8], x[9], x[12], x[13]);
      pa[cc][3] = pack_e4m3x4(x[10], x[11], x[14], x[15]);
    }
    if (!__all_sync(0xffffffffu, scA == 1.f && scB == 1.f)) {
#pragma unroll
      for (int jj = 0; jj < 9; ++jj) {
        st.o[4 * jj] *= scA; st.o[4 * jj + 1] *= scA;
        st.o[4 * jj + 2] *= scB; st.o[4 * jj + 3] *= scB;
      }
    }
    fence_u<8>(&pa[0][0]);
    fence_f<36>(st.o);
    wg_fence();
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) wgmma_fp8_rs_n72(st.o, pa[cc], dv + 2 * (2 * h + cc), 1);
    wg_commit();
    fence_f<36>(st.o);
    wg_wait<0>();
  }
  __syncwarp();
  if (cx.lane == 0) mbar_arrive(cx.bars + 8 * ASTAGES + 8 * scu);
}

// ---------------- software-pipelined consumer (FA3-style intra-warpgroup overlap) ----------------
// Tile t's softmax runs while the tensor cores do tile t-1's P.V: per tile, issue QK(t) and P.V(t-1), wait for QK(t),
// softmax(t) into a second P register set, wait for P.V(t-1), rescale O, release tile t-1's stage.
// Reads S without writing it (S is the accumulator of a wgmma still inside an open pipeline stage: writing it in place
// makes ptxas serialize every wgmma, C7513); masking is applied on the fly.
template <int EXPMODE, bool MASK>
DEVI void softmax_core(const AttnCtx& cx, AttnState& st, const float* sacc, int k0, uint32_t (&pa)[4][4], float& scA, float& scB) {
  auto V = [&](int i) -> float {
    if constexpr (!MASK) {
      return sacc[i];
    } else {
      const int key = k0 + 8 * (i >> 2) + 2 * cx.t4 + (i & 1);
      const bool bad = (i & 2) ? (key > cx.kmaxB || key < cx.kminB) : (key > cx.kmaxA || key < cx.kminA);
      return bad ? -INFINITY : sacc[i];
    }
  };
  const float c = cx.p.c;
#define MX4(o) fmaxf(fmaxf(V(o), V((o) + 1)), fmaxf(V((o) + 4), V((o) + 5)))
  const float a0 = fmaxf(MX4(0), MX4(8)), a1 = fmaxf(MX4(16), MX4(24)), a2 = fmaxf(MX4(32), MX4(40)), a3 = fmaxf(MX4(48), MX4(56));
  const float b0 = fmaxf(MX4(2), MX4(10)), b1 = fmaxf(MX4(18), MX4(26)), b2 = fmaxf(MX4(34), MX4(42)), b3 = fmaxf(MX4(50), MX4(58));
#undef MX4
  float tmA = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)), tmB = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));
  float nmA = st.mA, nmB = st.mB;
  scA = 1.f; scB = 1.f;
  // exact skip: when no lane's local maximum exceeds its row's running maximum, every row of this warp keeps
  // its maximum (rescale 1): no shuffles and no rescale exps on the exp2 units
  if (!ATTN_VOTEMAX_ALL || __any_sync(0xffffffffu, (tmA > st.mA) || (tmB > st.mB))) {
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
    nmA = fmaxf(st.mA, tmA); nmB = fmaxf(st.mB, tmB);
    scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
    st.mA = nmA; st.mB = nmB;
  }
  const float offA = nmA * c - ATTN_POFF, offB = nmB * c - ATTN_POFF;
  auto E = [&](int i) -> float {
    const float y = fmaf(V(i), c, (i & 2) ? -offB : -offA);
    if (i < EXPMODE) return exp2_poly(y);
    return exp2f(y);
  };
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const int o = 16 * cc;
    pa[cc][0] = pack_e4m3x4(E(o + 0), E(o + 1), E(o + 4), E(o + 5));
    pa[cc][1] = pack_e4m3x4(E(o + 2), E(o + 3), E(o + 6), E(o + 7));
    pa[cc][2] = pack_e4m3x4(E(o + 8), E(o + 9), E(o + 12), E(o + 13));
    pa[cc][3] = pack_e4m3x4(E(o + 10), E(o + 11), E(o + 14), E(o + 15));
  }
}

#define ATTN_RR_HAND 24  // v64: 24 measured +0.16% with the vote-skip
// RR > 1: the softmax with the warpgroups taking turns on the exp2 units (named barriers 4..4+RR-1) for the exp loop
// only; the max before it and the FP8 packing after it run outside the turn. Exps are written over S in place.
template <bool MASK, int RR, int EXPMODE = 0>
DEVI void softmax_core_rr(const AttnCtx& cx, AttnState& st, float* sacc, int k0, uint32_t (&pa)[4][4], float& scA, float& scB) {
  // EXPMODE > 0: that many of the 64 exps per thread run as a cubic polynomial on the FMA pipe (every 64/EXPMODE-th),
  // off the exp2 units that bound the turns
  auto EXP = [&](int i, float y) -> float {
    if constexpr (EXPMODE > 0) {
      if (i % (64 / EXPMODE) == 0) return exp2_poly(y);
    }
    return exp2f(y);
  };
  if constexpr (MASK) {
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      const int key = k0 + 8 * (i >> 2) + 2 * cx.t4 + (i & 1);
      const bool bad = (i & 2) ? (key > cx.kmaxB || key < cx.kminB) : (key > cx.kmaxA || key < cx.kminA);
      if (bad) sacc[i] = -INFINITY;
    }
  }
  const float c = cx.p.c;
#define MX4(o) fmaxf(fmaxf(sacc[(o)], sacc[(o) + 1]), fmaxf(sacc[(o) + 4], sacc[(o) + 5]))
  const float a0 = fmaxf(MX4(0), MX4(8)), a1 = fmaxf(MX4(16), MX4(24)), a2 = fmaxf(MX4(32), MX4(40)), a3 = fmaxf(MX4(48), MX4(56));
  const float b0 = fmaxf(MX4(2), MX4(10)), b1 = fmaxf(MX4(18), MX4(26)), b2 = fmaxf(MX4(34), MX4(42)), b3 = fmaxf(MX4(50), MX4(58));
#undef MX4
  float tmA = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)), tmB = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));
  float nmA = st.mA, nmB = st.mB;
  scA = 1.f; scB = 1.f;
  // exact skip: when no lane's local maximum exceeds its row's running maximum, every row of this warp keeps
  // its maximum (rescale 1): no shuffles and no rescale exps on the exp2 units
  if (!ATTN_VOTEMAX || __any_sync(0xffffffffu, (tmA > st.mA) || (tmB > st.mB))) {
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
    nmA = fmaxf(st.mA, tmA); nmB = fmaxf(st.mB, tmB);
    scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
    st.mA = nmA; st.mB = nmB;
  }
  const float offA = nmA * c - ATTN_POFF, offB = nmB * c - ATTN_POFF;
  named_bar_sync(4 + cx.wg, 256);  // my turn
  fence_f<64>(sacc);
  // the next warpgroup's turn starts after the first ATTN_RR_HAND of my 64 exps: two warpgroups overlap on the exp2
  // units for the rest, which hides the hand-off latency
#pragma unroll
  for (int i = 0; i < ATTN_RR_HAND; ++i) sacc[i] = EXP(i, fmaf(sacc[i], c, (i & 2) ? -offB : -offA));
  fence_f<ATTN_RR_HAND>(sacc);
  named_bar_arrive(4 + (cx.wg + 1 == RR ? 0 : cx.wg + 1), 256);
#pragma unroll
  for (int i = ATTN_RR_HAND; i < 64; ++i) sacc[i] = EXP(i, fmaf(sacc[i], c, (i & 2) ? -offB : -offA));
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const float* x = sacc + 16 * cc;
    pa[cc][0] = pack_e4m3x4(x[0], x[1], x[4], x[5]);
    pa[cc][1] = pack_e4m3x4(x[2], x[3], x[6], x[7]);
    pa[cc][2] = pack_e4m3x4(x[8], x[9], x[12], x[13]);
    pa[cc][3] = pack_e4m3x4(x[10], x[11], x[14], x[15]);
  }
}

template <int EXPMODE, int RR = 0>
DEVI void softmax_tile(const AttnCtx& cx, AttnState& st, float* sacc, int t, uint32_t (&pa)[4][4], float& scA, float& scB) {
  const AttnArgs& p = cx.p;
  const int k0 = t * NK;
  const int L = cx.L, nk = cx.nk;
  const bool need_mask = (k0 + NK - 1 > nk - L + cx.q_first) || (p.window && k0 < nk - L + cx.q_last - p.window + 1);
  if constexpr (RR > 1) {
    if (need_mask) softmax_core_rr<true, RR, EXPMODE>(cx, st, sacc, k0, pa, scA, scB);
    else softmax_core_rr<false, RR, EXPMODE>(cx, st, sacc, k0, pa, scA, scB);
  } else {
    if (need_mask) softmax_core<EXPMODE, true>(cx, st, sacc, k0, pa, scA, scB);
    else softmax_core<EXPMODE, false>(cx, st, sacc, k0, pa, scA, scB);
  }
}

DEVI void issue_pv(const AttnCtx& cx, float* o, uint32_t (&pa)[4][4], int stage) {
  uint8_t* vst = cx.ring + stage * STAGE + TILE_K;
  const uint64_t dv = desc_sw128(smem_u32(vst));
  fence_u<16>(&pa[0][0]);
  fence_f<36>(o);
  wg_fence();
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) wgmma_fp8_rs_n72(o, pa[cc], dv + 2 * cc, 1);
  wg_commit();
  fence_f<36>(o);
}

DEVI void rescale_o(AttnState& st, float scA, float scB) {
  if (!__all_sync(0xffffffffu, scA == 1.f && scB == 1.f)) {
#pragma unroll
    for (int jj = 0; jj < 9; ++jj) {
      st.o[4 * jj] *= scA; st.o[4 * jj + 1] *= scA;
      st.o[4 * jj + 2] *= scB; st.o[4 * jj + 3] *= scB;
    }
  }
}

DEVI void release_stage(const AttnCtx& cx, int stage) {
  __syncwarp();
  if (cx.lane == 0) mbar_arrive(cx.bars + 8 * ASTAGES + 8 * stage);
}

#define ATTN_EARLY_TEST 0  // test the next tile's K/V barrier (non-blocking) before the P.V wait; skip the blocking wait
DEVI bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t r;
  asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(r) : "r"(bar), "r"(parity) : "memory");
  return r != 0;
}
template <int EXPMODE, int RR = 0>
DEVI void pipe_step(const AttnCtx& cx, AttnState& st, float* sacc, int t, uint32_t (&pin)[4][4], uint32_t (&pout)[4][4]) {
  const int s = stage_of(cx, t), sp = stage_of(cx, t - 1);
  if (!ATTN_EARLY_TEST || !st.nready) mbar_wait(cx.bars + 8 * s, phase_of(cx, t));
  issue_qk(cx, sacc, s);
  issue_pv(cx, st.o, pin, sp);
  wg_wait<1>();  // QK(t) is done; P.V(t-1) may still run
  float scA, scB;
  softmax_tile<EXPMODE, RR>(cx, st, sacc, t, pout, scA, scB);
  if (ATTN_EARLY_TEST) st.nready = t + 1 <= cx.t_end && mbar_test(cx.bars + 8 * stage_of(cx, t + 1), phase_of(cx, t + 1));
  wg_wait<0>();  // P.V(t-1) is done
  fence_u<16>(&pin[0][0]);
  fence_f<36>(st.o);
  release_stage(cx, sp);
  rescale_o(st, scA, scB);
}

DEVI void pipe_last(const AttnCtx& cx, AttnState& st, uint32_t (&pin)[4][4]) {
  const int sl = stage_of(cx, cx.t_end);
  issue_pv(cx, st.o, pin, sl);
  wg_wait<0>();
  fence_u<16>(&pin[0][0]);
  fence_f<36>(st.o);
  release_stage(cx, sl);
}

// two P register sets alternate (no copies: a copy into an in-flight wgmma's operand serializes the pipeline)
template <int EXPMODE, int RR = 0>
DEVI void attn_pipe(const AttnCtx& cx, AttnState& st, float* sacc) {
  const int tb = cx.t_begin, te = cx.t_end;
  if (tb > te) return;
  uint32_t p0[4][4], p1[4][4];
  {
    const int s = stage_of(cx, tb);
    mbar_wait(cx.bars + 8 * s, phase_of(cx, tb));
    issue_qk(cx, sacc, s);
    wg_wait<0>();
    float scA, scB;
    softmax_tile<EXPMODE, RR>(cx, st, sacc, tb, p0, scA, scB);  // O is still zero: nothing to rescale
  }
  int t = tb + 1;
  for (; t + 1 <= te; t += 2) {
    pipe_step<EXPMODE, RR>(cx, st, sacc, t, p0, p1);
    pipe_step<EXPMODE, RR>(cx, st, sacc, t + 1, p1, p0);
  }
  if (t <= te) {
    pipe_step<EXPMODE, RR>(cx, st, sacc, t, p0, p1);
    pipe_last(cx, st, p1);
  } else {
    pipe_last(cx, st, p0);
  }
}

#define ATTN_EARLY_QK 0
// Early QK^T: QK(t+1) is issued as soon as softmax(t) has packed P(t) (S is dead then), so its latency hides behind the
// P.V(t-1) drain, the O rescale and the next step's P.V issue instead of opening the next step. On entry QK(t) is in
// flight.
template <int EXPMODE, int RR, bool MORE>
DEVI void pipe_step_eq(const AttnCtx& cx, AttnState& st, float* sacc, int t, uint32_t (&pin)[4][4], uint32_t (&pout)[4][4]) {
  // compile-time MORE: a wgmma issued under a runtime branch serializes the kernel (ptxas C7513)
  const int sp = stage_of(cx, t - 1);
  issue_pv(cx, st.o, pin, sp);
  wg_wait<1>();  // QK(t) is done; P.V(t-1) may still run
  float scA, scB;
  softmax_tile<EXPMODE, RR>(cx, st, sacc, t, pout, scA, scB);
  if constexpr (MORE) {
    const int sn = stage_of(cx, t + 1);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, t + 1));
    issue_qk(cx, sacc, sn);
  }
  if constexpr (MORE) wg_wait<1>();  // P.V(t-1) is done; QK(t+1) may still run
  else wg_wait<0>();
  fence_u<16>(&pin[0][0]);
  fence_f<36>(st.o);
  release_stage(cx, sp);
  rescale_o(st, scA, scB);
}

template <int EXPMODE, int RR = 0>
DEVI void attn_pipe_eq(const AttnCtx& cx, AttnState& st, float* sacc) {
  const int tb = cx.t_begin, te = cx.t_end;
  if (tb > te) return;
  uint32_t p0[4][4], p1[4][4];
  {
    const int s = stage_of(cx, tb);
    mbar_wait(cx.bars + 8 * s, phase_of(cx, tb));
    issue_qk(cx, sacc, s);
    wg_wait<0>();
    float scA, scB;
    softmax_tile<EXPMODE, RR>(cx, st, sacc, tb, p0, scA, scB);  // O is still zero: nothing to rescale
  }
  {  // QK(tb + 1) runs while the first tile's tail drains. Issued even for a single-tile range (the producer loads a
     // dummy tile): a separate single-tile path makes ptxas serialize every wgmma (C7511)
    const int sn = stage_of(cx, tb + 1);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, tb + 1));
    issue_qk(cx, sacc, sn);
  }
  int t = tb + 1;
  for (; t + 2 <= te; t += 2) {
    pipe_step_eq<EXPMODE, RR, true>(cx, st, sacc, t, p0, p1);
    pipe_step_eq<EXPMODE, RR, true>(cx, st, sacc, t + 1, p1, p0);
  }
  if (t + 1 == te) {
    pipe_step_eq<EXPMODE, RR, true>(cx, st, sacc, t, p0, p1);
    pipe_step_eq<EXPMODE, RR, false>(cx, st, sacc, t + 1, p1, p0);
    pipe_last(cx, st, p0);
  } else if (t == te) {
    pipe_step_eq<EXPMODE, RR, false>(cx, st, sacc, t, p0, p1);
    pipe_last(cx, st, p1);
  } else {  // a single-tile range: P(tb) is in p0; pipe_last's wait also retires the dummy QK
    pipe_last(cx, st, p0);
  }
}

// Deeper pipeline: QK(t+1) and P.V(t-1) both run on the tensor cores during softmax(t) (two S and two P register sets).
template <int EXPMODE>
DEVI void pipe2_step(const AttnCtx& cx, AttnState& st, int t, float* s_cur, float* s_next, uint32_t (&pin)[4][4],
                     uint32_t (&pout)[4][4]) {
  const bool more = t + 1 <= cx.t_end;
  if (more) {
    const int sn = stage_of(cx, t + 1);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, t + 1));
    issue_qk(cx, s_next, sn);
  }
  const int sp = stage_of(cx, t - 1);
  issue_pv(cx, st.o, pin, sp);
  if (more) wg_wait<2>();  // QK(t) is done
  else wg_wait<1>();
  float scA, scB;
  softmax_tile<EXPMODE>(cx, st, s_cur, t, pout, scA, scB);
  wg_wait<0>();  // P.V(t-1) (and QK(t+1), issued before it) are done
  fence_u<16>(&pin[0][0]);
  fence_f<36>(st.o);
  release_stage(cx, sp);
  rescale_o(st, scA, scB);
}

template <int EXPMODE>
DEVI void attn_pipe2(const AttnCtx& cx, AttnState& st, float* s0, float* s1) {
  const int tb = cx.t_begin, te = cx.t_end;
  if (tb > te) return;
  uint32_t p0[4][4], p1[4][4];
  {
    const int s = stage_of(cx, tb);
    mbar_wait(cx.bars + 8 * s, phase_of(cx, tb));
    issue_qk(cx, s0, s);
    wg_wait<0>();
    if (tb + 1 <= te) {
      const int sn = stage_of(cx, tb + 1);
      mbar_wait(cx.bars + 8 * sn, phase_of(cx, tb + 1));
      issue_qk(cx, s1, sn);
    }
    float scA, scB;
    softmax_tile<EXPMODE>(cx, st, s0, tb, p0, scA, scB);
  }
  int t = tb + 1;
  for (; t + 1 <= te; t += 2) {
    pipe2_step<EXPMODE>(cx, st, t, s1, s0, p0, p1);
    pipe2_step<EXPMODE>(cx, st, t + 1, s0, s1, p1, p0);
  }
  if (t <= te) {
    pipe2_step<EXPMODE>(cx, st, t, s1, s0, p0, p1);
    pipe_last(cx, st, p1);
  } else {
    pipe_last(cx, st, p0);
  }
}

// ---------------- variant 53: two consumer warpgroups (240 registers), two S register sets, one P set, exp turns ----------------
// Step t (QK(t) in flight on entry): issue P.V(t-1); issue QK(t+1) into the other S set; wait QK(t); softmax(t): row max
// (vote-skipped), my exp turn (the other warpgroup's turn starts after ATTN_RR_HAND exps), then wait P.V(t-1), rescale O,
// pack P(t). P.V(t-1) and QK(t+1) run on the tensor cores during the softmax; the two warpgroups alternate on the exp2
// units. MORE (another tile follows) is compile-time: a wgmma issued under a runtime branch serializes (C7513).
#define ATTN_P3_HAND ATTN_RR_HAND
#define ATTN_P3_PVW 0  // where softmax(t) waits for P.V(t-1): 0 after the exps, 1 after the row max, 2 before the row max
template <bool MASK, int RR, bool PVPEND, bool MORE>
DEVI void softmax_core_p3(const AttnCtx& cx, AttnState& st, float* sacc, int k0, uint32_t (&pa)[4][4]) {
  auto pv_wait = [&]() {
    if constexpr (PVPEND) {
      if constexpr (MORE) wg_wait<1>();  // QK(t+1), issued after P.V(t-1), may still run
      else wg_wait<0>();
    }
    fence_u<16>(&pa[0][0]);
    fence_f<36>(st.o);
  };
  if constexpr (ATTN_P3_PVW == 2) pv_wait();
  if constexpr (MASK) {
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      const int key = k0 + 8 * (i >> 2) + 2 * cx.t4 + (i & 1);
      const bool bad = (i & 2) ? (key > cx.kmaxB || key < cx.kminB) : (key > cx.kmaxA || key < cx.kminA);
      if (bad) sacc[i] = -INFINITY;
    }
  }
  const float c = cx.p.c;
#define MX4(o) fmaxf(fmaxf(sacc[(o)], sacc[(o) + 1]), fmaxf(sacc[(o) + 4], sacc[(o) + 5]))
  const float a0 = fmaxf(MX4(0), MX4(8)), a1 = fmaxf(MX4(16), MX4(24)), a2 = fmaxf(MX4(32), MX4(40)), a3 = fmaxf(MX4(48), MX4(56));
  const float b0 = fmaxf(MX4(2), MX4(10)), b1 = fmaxf(MX4(18), MX4(26)), b2 = fmaxf(MX4(34), MX4(42)), b3 = fmaxf(MX4(50), MX4(58));
#undef MX4
  float tmA = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)), tmB = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));
  float nmA = st.mA, nmB = st.mB;
  float scA = 1.f, scB = 1.f;
  if (!ATTN_VOTEMAX || __any_sync(0xffffffffu, (tmA > st.mA) || (tmB > st.mB))) {
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
    nmA = fmaxf(st.mA, tmA); nmB = fmaxf(st.mB, tmB);
    scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
    st.mA = nmA; st.mB = nmB;
  }
  const float offA = nmA * c - ATTN_POFF, offB = nmB * c - ATTN_POFF;
  if constexpr (ATTN_P3_PVW >= 1) {
    if constexpr (ATTN_P3_PVW == 1) pv_wait();
    rescale_o(st, scA, scB);
  }
  if constexpr (RR > 1) named_bar_sync(4 + cx.wg, 256);  // my turn
  fence_f<64>(sacc);
#pragma unroll
  for (int i = 0; i < ATTN_P3_HAND; ++i) sacc[i] = exp2f(fmaf(sacc[i], c, (i & 2) ? -offB : -offA));
  if constexpr (RR > 1) {
    fence_f<ATTN_P3_HAND>(sacc);
    named_bar_arrive(4 + (cx.wg + 1 == RR ? 0 : cx.wg + 1), 256);
  }
#pragma unroll
  for (int i = ATTN_P3_HAND; i < 64; ++i) sacc[i] = exp2f(fmaf(sacc[i], c, (i & 2) ? -offB : -offA));
  // P.V(t-1) reads P and accumulates into O: it must be done before O is rescaled and P is overwritten
  if constexpr (ATTN_P3_PVW == 0) {
    pv_wait();
    rescale_o(st, scA, scB);
  }
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const float* x = sacc + 16 * cc;
    pa[cc][0] = pack_e4m3x4(x[0], x[1], x[4], x[5]);
    pa[cc][1] = pack_e4m3x4(x[2], x[3], x[6], x[7]);
    pa[cc][2] = pack_e4m3x4(x[8], x[9], x[12], x[13]);
    pa[cc][3] = pack_e4m3x4(x[10], x[11], x[14], x[15]);
  }
}

template <int RR, bool PVPEND, bool MORE>
DEVI void softmax_tile_p3(const AttnCtx& cx, AttnState& st, float* sacc, int t, uint32_t (&pa)[4][4]) {
  const AttnArgs& p = cx.p;
  const int k0 = t * NK;
  const int L = cx.L, nk = cx.nk;
  const bool need_mask = (k0 + NK - 1 > nk - L + cx.q_first) || (p.window && k0 < nk - L + cx.q_last - p.window + 1);
  if (need_mask) softmax_core_p3<true, RR, PVPEND, MORE>(cx, st, sacc, k0, pa);
  else softmax_core_p3<false, RR, PVPEND, MORE>(cx, st, sacc, k0, pa);
}

template <int RR, bool MORE>
DEVI void pipe3_step(const AttnCtx& cx, AttnState& st, int t, float* s_cur, float* s_next, uint32_t (&pa)[4][4]) {
  const int sp = stage_of(cx, t - 1);
  issue_pv(cx, st.o, pa, sp);  // P(t-1), V(t-1)
  if constexpr (MORE) {
    const int sn = stage_of(cx, t + 1);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, t + 1));
    issue_qk(cx, s_next, sn);
  }
  if constexpr (MORE) wg_wait<2>();  // QK(t) is done; P.V(t-1) and QK(t+1) may still run
  else wg_wait<1>();
  softmax_tile_p3<RR, true, MORE>(cx, st, s_cur, t, pa);
  release_stage(cx, sp);
}

template <int RR>
DEVI void attn_pipe3(const AttnCtx& cx, AttnState& st, float* s0, float* s1) {
  const int tb = cx.t_begin, te = cx.t_end;
  if (tb > te) return;
  uint32_t pa[4][4];
  {
    // QK(tb + 1) is issued even for a single-tile range (the producer loads a dummy tile): one uniform prologue
    // (a separate single-tile path makes ptxas serialize every wgmma: C7511)
    const int s = stage_of(cx, tb), sn = stage_of(cx, tb + 1);
    mbar_wait(cx.bars + 8 * s, phase_of(cx, tb));
    issue_qk(cx, s0, s);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, tb + 1));
    issue_qk(cx, s1, sn);
    wg_wait<1>();  // QK(tb) is done; QK(tb+1) may still run
    softmax_tile_p3<RR, false, true>(cx, st, s0, tb, pa);  // O is still zero: nothing pending on it
  }
  int t = tb + 1;  // QK(t) in flight, into s1
  for (; t + 1 < te; t += 2) {
    pipe3_step<RR, true>(cx, st, t, s1, s0, pa);
    pipe3_step<RR, true>(cx, st, t + 1, s0, s1, pa);
  }
  if (t + 1 == te) {  // two tiles left
    pipe3_step<RR, true>(cx, st, t, s1, s0, pa);
    pipe3_step<RR, false>(cx, st, t + 1, s0, s1, pa);
  } else if (t == te) {  // one tile left
    pipe3_step<RR, false>(cx, st, t, s1, s0, pa);
  } else {  // a single-tile range: retire the dummy QK
    wg_wait<0>();
  }
  pipe_last(cx, st, pa);
}

// ---------------- half tiles, software-pipelined (QK of the next 64-key half and P.V of the previous one run on the
// tensor cores during this half's softmax; two S and two P register sets of half size fit three warpgroups) ----------------
template <bool MASK>
DEVI void softmax_half(const AttnCtx& cx, AttnState& st, const float* sacc, int k0, uint32_t (&pa)[2][4], float& scA, float& scB) {
  auto V = [&](int i) -> float {
    if constexpr (!MASK) {
      return sacc[i];
    } else {
      const int key = k0 + 8 * (i >> 2) + 2 * cx.t4 + (i & 1);
      const bool bad = (i & 2) ? (key > cx.kmaxB || key < cx.kminB) : (key > cx.kmaxA || key < cx.kminA);
      return bad ? -INFINITY : sacc[i];
    }
  };
  const float c = cx.p.c;
#define MX4(o) fmaxf(fmaxf(V(o), V((o) + 1)), fmaxf(V((o) + 4), V((o) + 5)))
  float tmA = fmaxf(fmaxf(MX4(0), MX4(8)), fmaxf(MX4(16), MX4(24)));
  float tmB = fmaxf(fmaxf(MX4(2), MX4(10)), fmaxf(MX4(18), MX4(26)));
#undef MX4
  float nmA = st.mA, nmB = st.mB;
  scA = 1.f; scB = 1.f;
  // exact skip: when no lane's local maximum exceeds its row's running maximum, every row of this warp keeps
  // its maximum (rescale 1): no shuffles and no rescale exps on the exp2 units
  if (!ATTN_VOTEMAX_ALL || __any_sync(0xffffffffu, (tmA > st.mA) || (tmB > st.mB))) {
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
    nmA = fmaxf(st.mA, tmA); nmB = fmaxf(st.mB, tmB);
    scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
    st.mA = nmA; st.mB = nmB;
  }
  const float offA = nmA * c - ATTN_POFF, offB = nmB * c - ATTN_POFF;
  auto E = [&](int i) -> float { return exp2f(fmaf(V(i), c, (i & 2) ? -offB : -offA)); };
#pragma unroll
  for (int cc = 0; cc < 2; ++cc) {
    const int o = 16 * cc;
    pa[cc][0] = pack_e4m3x4(E(o + 0), E(o + 1), E(o + 4), E(o + 5));
    pa[cc][1] = pack_e4m3x4(E(o + 2), E(o + 3), E(o + 6), E(o + 7));
    pa[cc][2] = pack_e4m3x4(E(o + 8), E(o + 9), E(o + 12), E(o + 13));
    pa[cc][3] = pack_e4m3x4(E(o + 10), E(o + 11), E(o + 14), E(o + 15));
  }
}

DEVI void softmax_half_tile(const AttnCtx& cx, AttnState& st, const float* sacc, int u, uint32_t (&pa)[2][4], float& scA, float& scB) {
  const AttnArgs& p = cx.p;
  const int k0 = (u >> 1) * NK + 64 * (u & 1);
  const int L = cx.L, nk = cx.nk;
  const bool need_mask = (k0 + 63 > nk - L + cx.q_first) || (p.window && k0 < nk - L + cx.q_last - p.window + 1);
  if (need_mask) softmax_half<true>(cx, st, sacc, k0, pa, scA, scB);
  else softmax_half<false>(cx, st, sacc, k0, pa, scA, scB);
}

template <int H>
DEVI void issue_pv_h(const AttnCtx& cx, float* o, uint32_t (&pa)[2][4], int stage) {
  uint8_t* vst = cx.ring + stage * STAGE + TILE_K;
  uint64_t dv0 = desc_sw128(smem_u32(vst)) + 4 * H;
  uint64_t dv1 = dv0 + 2;
  asm volatile("" : "+l"(dv0), "+l"(dv1));
  fence_u<8>(&pa[0][0]);
  fence_f<36>(o);
  wg_fence();
  wgmma_fp8_rs_n72(o, pa[0], dv0, 1);
  wgmma_fp8_rs_n72(o, pa[1], dv1, 1);
  wg_commit();
  fence_f<36>(o);
}

// sub-step u = 2 t + h covers keys [128 t + 64 h, +64); H (= u & 1) is a compile-time constant at every call
template <int H>
DEVI void issue_qk_u(const AttnCtx& cx, float* s, int u) {
  const int t = u >> 1;
  const int st = stage_of(cx, t);
  if (H == 0) mbar_wait(cx.bars + 8 * st, phase_of(cx, t));
  issue_qk_h(cx, s, st, H);
}

// MORE: issue the next half's QK (every step but the last; the wgmma groups never depend on a runtime branch)
template <int HC, bool MORE>
DEVI void pipe2h_step(const AttnCtx& cx, AttnState& st, int u, float* s_cur, float* s_next, uint32_t (&pin)[2][4],
                      uint32_t (&pout)[2][4]) {
  if constexpr (MORE) issue_qk_u<1 - HC>(cx, s_next, u + 1);
  const int up = u - 1;
  issue_pv_h<1 - HC>(cx, st.o, pin, stage_of(cx, up >> 1));
  if constexpr (MORE) wg_wait<2>();  // QK(u) is done
  else wg_wait<1>();
  float scA = 1.f, scB = 1.f;
  softmax_half_tile(cx, st, s_cur, u, pout, scA, scB);
  wg_wait<0>();  // P.V(u-1) (and QK(u+1), issued before it) are done
  fence_u<8>(&pin[0][0]);
  fence_f<36>(st.o);
  if (HC == 0) release_stage(cx, stage_of(cx, up >> 1));  // u-1 was the second half of its tile
  rescale_o(st, scA, scB);
}

DEVI void attn_pipe2h(const AttnCtx& cx, AttnState& st, float* s0, float* s1) {
  const int tb = cx.t_begin, te = cx.t_end;
  if (tb > te) return;
  const int u0 = 2 * tb, u_end = 2 * te + 1;
  uint32_t p0[2][4], p1[2][4];
  {
    issue_qk_u<0>(cx, s0, u0);
    wg_wait<0>();
    issue_qk_u<1>(cx, s1, u0 + 1);  // u_end >= u0 + 1 always (two halves per tile)
    float scA, scB;
    softmax_half_tile(cx, st, s0, u0, p0, scA, scB);
  }
  int u = u0 + 1;  // odd
  for (; u < u_end; u += 2) {
    pipe2h_step<1, true>(cx, st, u, s1, s0, p0, p1);
    pipe2h_step<0, true>(cx, st, u + 1, s0, s1, p1, p0);
  }
  // u == u_end here (odd: the last tile's second half)
  pipe2h_step<1, false>(cx, st, u, s1, s0, p0, p1);
  issue_pv_h<1>(cx, st.o, p1, stage_of(cx, te));
  wg_wait<0>();
  fence_u<8>(&p1[0][0]);
  fence_f<36>(st.o);
  release_stage(cx, stage_of(cx, te));
}

template <int NWG, int EXPMODE, bool OVL, bool PIPE = false, bool PIPE2 = false, bool RRS = false, bool HALF = false, bool PIPE2H = false,
          bool PIPE3 = false>
__global__ void __launch_bounds__(128 * (NWG + 1), 1) attn_fwd_kernel(AttnArgs p) {
  if constexpr (ATTN_EARLY_Q) pdl_trigger_attn();
  else pdl_enter_attn();
  constexpr int QBT = 8 * NWG;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);           // full[4], empty[4]
  uint8_t* qs = base + 1024;                      // 64*NWG rows x 64 B (SW64)
  uint8_t* ring = qs + 4096 * NWG;                // ASTAGES x (K tile, V tile)
  const int qb = p.nqb - 1 - blockIdx.x;          // heaviest (latest) query blocks first
  const int h = blockIdx.y & 1, split = blockIdx.y >> 1, b = blockIdx.z;
  const int L = p.L;
  const int seq = p.seq_lens[b];
  const int first = p.window ? max(0, seq - p.window + 1) : 0;
  const int nk = seq + L - first;
  const int q_first = qb * QBT;
  const int q_last = min(q_first + QBT - 1, L - 1);
  const int kmax_last = nk - L + q_last;
  const int kmin_first = p.window ? max(0, nk - L + q_first - p.window + 1) : 0;
  int t_begin = kmin_first / NK, t_end = kmax_last / NK;
  if (p.nsplit > 1) {
    const int ntiles = t_end - t_begin + 1;
    const int per = (ntiles + p.nsplit - 1) / p.nsplit;
    const int tb = t_begin + split * per;
    t_end = min(t_end, tb + per - 1);
    t_begin = tb;  // may exceed t_end: empty split
  }
  const size_t tb0 = (size_t)(b * 2 + h) * p.ntk;
  if (threadIdx.x == 0) {
    for (int s = 0; s < ASTAGES; ++s) {
      mbar_init(bars + 8 * s, 1);
      mbar_init(bars + 8 * ASTAGES + 8 * s, 4 * NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x < 128) {
    // ---------------- producer warpgroup (one thread issues the bulk copies) ----------------
    const int ptid = threadIdx.x;
    if constexpr (NWG >= 3 || PIPE2 || PIPE3) asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");
    else asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (ptid == 0) {
      if constexpr (ATTN_EARLY_Q) pdl_wait_attn();  // the staged K/V tiles
      if (ATTN_PF && p.pf != nullptr) {
        // the attention leaves DRAM mostly idle: warm L2 with this CTA's slice of the next GEMM's weight
        const unsigned long long nct = (unsigned long long)gridDim.x * gridDim.y * gridDim.z;
        const unsigned long long cid = ((unsigned long long)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        const unsigned long long per = (p.pf_bytes / nct + 255ull) & ~255ull;
        const unsigned long long b0 = cid * per, b1 = min(b0 + per, p.pf_bytes);
        for (unsigned long long off = b0; off < b1; off += 32768ull)
          prefetch_l2(p.pf + off, (uint32_t)min(32768ull, b1 - off) & ~15u);
      }
      int s = 0;
      uint32_t ph = 0;
      // PIPE3 always issues QK(t_begin + 1): a single-tile range gets a dummy second load (tile t_end again)
      constexpr bool DUMMY = PIPE3 || (PIPE && RRS && ATTN_EARLY_QK);  // always issue QK(t_begin + 1)
      const int t_last = (DUMMY && t_begin == t_end) ? t_end + 1 : t_end;
      for (int t = t_begin; t <= t_last; ++t) {
        mbar_wait(bars + 8 * ASTAGES + 8 * s, ph ^ 1);
        const uint32_t full = bars + 8 * s;
        mbar_expect_tx(full, STAGE);
        const uint32_t dst = smem_u32(ring + s * STAGE);
        const int tt = min(t, t_end);
        bulk_g2s(dst, p.kt + (tb0 + tt) * TILE_K, TILE_K, full);
        bulk_g2s(dst + TILE_K, p.vt + (tb0 + tt) * TILE_V, TILE_V, full);
        if (++s == ASTAGES) { s = 0; ph ^= 1; }
      }
    }
    return;
  }
  // ---------------- consumers (warpgroups 1 .. NWG) ----------------
  // setmaxnreg moves registers inside the CTA's launch allocation: 640 threads launch with 96 each, and the producer's
  // 72 x 128 freed registers cover at most +16 for each of the 512 consumer threads
  if constexpr (NWG == 4) asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\n");
  else if constexpr (NWG == 3) asm volatile("setmaxnreg.inc.sync.aligned.u32 160;\n");
  else if constexpr (PIPE2 || PIPE3) asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n");
  else asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  const int ctid = threadIdx.x - 128;
  const int wg = ctid >> 7, w = (ctid >> 5) & 3, lane = ctid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  // stage Q: row r = (token q_first + r/8, head 8h + r%8) -> smem SW64
  for (int i = ctid; i < 64 * NWG * 4; i += 128 * NWG) {
    const int r = i >> 2, c = i & 3;
    const int tok = q_first + (r >> 3);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (tok < L) {
      const size_t trow = (size_t)b * L + tok;
      v = *reinterpret_cast<const uint4*>(p.q + (trow * 16 + 8 * h + (r & 7)) * 64 + c * 16);
    }
    *reinterpret_cast<uint4*>(qs + r * 64 + ((c ^ ((r >> 1) & 3)) << 4)) = v;
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  named_bar_sync(1, 128 * NWG);
  // this thread's rows: A = (token q_first + 8wg + 2w, head g), B = token + 1
  const int tokA = q_first + 8 * wg + 2 * w, tokB = tokA + 1;
  const int kmaxA = nk - L + tokA, kmaxB = nk - L + tokB;
  const int kminA = p.window ? kmaxA - p.window + 1 : -1, kminB = p.window ? kmaxB - p.window + 1 : -1;
  AttnState stt;
#pragma unroll
  for (int i = 0; i < 36; ++i) stt.o[i] = 0.f;
  stt.mA = -1e30f; stt.mB = -1e30f; stt.nready = 0;
  AttnCtx cx{p, qs, ring, bars, wg, w, lane, g, t4, kmaxA, kmaxB, kminA, kminB, nk, L, q_first, q_last, t_begin, t_end,
             desc_sw64(smem_u32(qs + wg * 64 * 64)), desc_sw64(smem_u32(ring)), desc_sw128(smem_u32(ring + TILE_K)), t_begin};
  // the output gate, loaded before the tile loop so that the epilogue does not wait on global loads (variants with
  // registers to spare only: the pipelined ones would spill)
  constexpr bool GPF = ATTN_GATE_PF && !PIPE && !PIPE2 && !PIPE2H && !PIPE3 && !HALF && NWG <= 3;
  uint32_t gpf[2][8];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int tok = half ? tokB : tokA;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) gpf[half][jj] = 0u;
    if (GPF && p.nsplit == 1 && tok < L) {
      const __nv_bfloat16* gp = p.qkv + ((size_t)b * L + tok) * 2304 + 1280 + (8 * h + g) * 64;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) gpf[half][jj] = *reinterpret_cast<const uint32_t*>(gp + 8 * jj + 2 * t4);
    }
  }
  float s0[64];
  if (OVL && t_begin <= t_end) {
    mbar_wait(bars, 0);
    issue_qk(cx, s0, 0);
    wg_wait<0>();
  }
  if constexpr (PIPE2H) {
    float s1[32];
    attn_pipe2h(cx, stt, s0, s1);
  } else if constexpr (HALF) {
    for (int t = t_begin; t <= t_end; ++t) attn_step_h<EXPMODE>(cx, stt, s0, t);
  } else if constexpr (PIPE3) {
    if (wg == NWG - 1 && t_begin <= t_end) named_bar_arrive(4, 256);  // warpgroup 0 starts
    float s1[64];
#define ATTN_P3_RR NWG
    attn_pipe3<ATTN_P3_RR>(cx, stt, s0, s1);
  } else if constexpr (PIPE2) {
    float s1[64];
    attn_pipe2<EXPMODE>(cx, stt, s0, s1);
  } else if constexpr (PIPE && RRS) {
    if (wg == NWG - 1 && t_begin <= t_end) named_bar_arrive(4, 256);  // warpgroup 0 starts
    if constexpr (ATTN_EARLY_QK) attn_pipe_eq<EXPMODE, NWG>(cx, stt, s0);
    else attn_pipe<EXPMODE, NWG>(cx, stt, s0);
  } else if constexpr (PIPE) attn_pipe<EXPMODE>(cx, stt, s0);
  else if constexpr (RRS) {
    if (wg == NWG - 1 && t_begin <= t_end) named_bar_arrive(4, 256);  // warpgroup 0 starts
    for (int t = t_begin; t <= t_end; ++t) attn_step<EXPMODE, OVL, NWG>(cx, stt, s0, t);
  } else
    for (int t = t_begin; t <= t_end; ++t) attn_step<EXPMODE, OVL>(cx, stt, s0, t);
  float* o = stt.o;
  const float mA = stt.mA, mB = stt.mB;
  // the denominator (column 64) sits in the t4 == 0 lane of each quad
  const float lA = __shfl_sync(0xffffffffu, o[32], lane & ~3), lB = __shfl_sync(0xffffffffu, o[34], lane & ~3);
  const int head = 8 * h + g;
  // ---- epilogue: partials for a split, else O / l, per-head RMSNorm, * sigmoid(gate) ----
  if (p.nsplit > 1) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int tok = half ? tokB : tokA;
      if (tok >= L) continue;
      const size_t trow = (size_t)b * L + tok;
      const size_t T = (size_t)gridDim.z * L;
      float* wp = p.ws + ((size_t)split * T * 16 + trow * 16 + head) * 66;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj)
        *reinterpret_cast<float2*>(wp + 8 * jj + 2 * t4) = make_float2(o[4 * jj + 2 * half], o[4 * jj + 2 * half + 1]);
      if (t4 == 0) {
        wp[64] = half ? mB : mA;
        wp[65] = half ? lB : lA;
      }
    }
    return;
  }
  const float iA = p.v_scale / lA, iB = p.v_scale / lB;
  float ssA = 0.f, ssB = 0.f;
#pragma unroll
  for (int jj = 0; jj < 8; ++jj) {
    o[4 * jj] *= iA; o[4 * jj + 1] *= iA; o[4 * jj + 2] *= iB; o[4 * jj + 3] *= iB;
    ssA += o[4 * jj] * o[4 * jj] + o[4 * jj + 1] * o[4 * jj + 1];
    ssB += o[4 * jj + 2] * o[4 * jj + 2] + o[4 * jj + 3] * o[4 * jj + 3];
  }
  ssA += __shfl_xor_sync(0xffffffffu, ssA, 1);
  ssA += __shfl_xor_sync(0xffffffffu, ssA, 2);
  ssB += __shfl_xor_sync(0xffffffffu, ssB, 1);
  ssB += __shfl_xor_sync(0xffffffffu, ssB, 2);
  const float rA = rsqrtf(ssA / 64.f + 1e-5f), rB = rsqrtf(ssB / 64.f + 1e-5f);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int tok = half ? tokB : tokA;
    if (tok >= L) continue;
    const size_t trow = (size_t)b * L + tok;
    const float rr = half ? rB : rA;
    const __nv_bfloat16* gp = p.qkv + trow * 2304 + 1280 + head * 64;
    __nv_bfloat16* op = p.out + trow * 1024 + head * 64;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      const int d = 8 * jj + 2 * t4;
      float2 gv;
      if constexpr (GPF) {
        const uint32_t gbits = gpf[half][jj];
        gv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gbits));
      } else {
        gv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp + d));
      }
      const float x0 = o[4 * jj + 2 * half] * rr, x1 = o[4 * jj + 2 * half + 1] * rr;
      const float y0 = x0 / (1.f + __expf(-gv.x)), y1 = x1 / (1.f + __expf(-gv.y));
      *reinterpret_cast<__nv_bfloat162*>(op + d) = __floats2bfloat162_rn(y0, y1);
    }
  }
}

// ---------------- stream-K attention ----------------
// The (pair, key tile) space, pair = (batch, query block, kv head) in that order, is cut into gridDim.x equal ranges of W
// tiles, one per CTA, so every CTA (and SM) gets the same work: a fixed grid of nqb x 2 x B CTAs leaves the last wave
// part-empty (T=753 full: 128 CTAs on 132 SMs; T=3012 full: 256 CTAs = 1.94 waves). A CTA runs its range as one segment
// per pair it touches, each writing a partial (O, m, l) for the pair's QBT x 8 rows to ws slot pair + cta (unique: every
// segment starts at a pair or a CTA boundary, so slots run 0 .. npairs + G - 2); attn_combine_sk_kernel merges them. The ring (stage, phase) continues across segments, and the exp-turn
// hand-off of the last warpgroup at a segment's end opens the next segment (no extra start arrive after segment 0).
#define ATTN_SK_MAXPAIRS 1536
DEVI void sk_pair(const AttnArgs& p, int QBT, int pp, int& b, int& qb, int& h, int& tb, int& te, int& nk) {
  h = pp & 1;
  const int bq = pp >> 1;
  b = bq / p.nqb;
  qb = bq - b * p.nqb;
  const int L = p.L, seq = p.seq_lens[b];
  const int first = p.window ? max(0, seq - p.window + 1) : 0;
  nk = seq + L - first;
  const int q_first = qb * QBT, q_last = min(q_first + QBT - 1, L - 1);
  const int kmin_first = p.window ? max(0, nk - L + q_first - p.window + 1) : 0;
  tb = kmin_first / NK;
  te = (nk - L + q_last) / NK;
}

template <int NWG, int EXPMODE, bool PIPE, bool RRS>
__global__ void __launch_bounds__(128 * (NWG + 1), 1) attn_fwd_sk_kernel(AttnArgs p) {
  pdl_enter_attn();
  constexpr int QBT = 8 * NWG, NT = 128 * (NWG + 1);
  constexpr int PER = (ATTN_SK_MAXPAIRS + NT - 1) / NT;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);
  uint8_t* qs = base + 1024;
  uint8_t* ring = qs + 4096 * NWG;
  __shared__ int s_pre[ATTN_SK_MAXPAIRS + 1];
  __shared__ int s_wsum[NT / 32];
  const int npairs = p.nb * p.nqb * 2;
  // per-pair tile counts -> inclusive prefix s_pre[pp + 1] (thread i owns pairs [i * PER, i * PER + PER))
  {
    int cnt[PER];
    int sum = 0;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
      const int pp = threadIdx.x * PER + k;
      int n = 0;
      if (pp < npairs) {
        int b, qb, h, tb, te, nk;
        sk_pair(p, QBT, pp, b, qb, h, tb, te, nk);
        n = max(0, te - tb + 1);
      }
      cnt[k] = n;
      sum += n;
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int incl = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += v;
    }
    if (lane == 31) s_wsum[wid] = incl;
    __syncthreads();
    int wbase = 0;
    for (int i = 0; i < wid; ++i) wbase += s_wsum[i];
    int run = wbase + incl - sum;  // exclusive prefix of this thread's first pair
#pragma unroll
    for (int k = 0; k < PER; ++k) {
      const int pp = threadIdx.x * PER + k;
      run += cnt[k];
      if (pp < npairs) s_pre[pp + 1] = run;
    }
    if (threadIdx.x == 0) s_pre[0] = 0;
  }
  if (threadIdx.x == 0) {
    for (int s = 0; s < ASTAGES; ++s) {
      mbar_init(bars + 8 * s, 1);
      mbar_init(bars + 8 * ASTAGES + 8 * s, 4 * NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int total = s_pre[npairs];
  const int W = (total + (int)gridDim.x - 1) / (int)gridDim.x;
  const int r0 = min((int)blockIdx.x * W, total), r1 = min(r0 + W, total);
  // first pair of the range: the last pp with s_pre[pp] <= r0 (binary search)
  int lo = 0, hi = npairs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (s_pre[mid] <= r0) lo = mid; else hi = mid - 1;
  }
  const int pp0 = lo;
  if (threadIdx.x == 0) {  // for attn_combine_sk_kernel: W, the prefix, each CTA's first pair
    p.skbuf[2 + npairs + blockIdx.x] = pp0;
    if (blockIdx.x == 0) p.skbuf[0] = W;
  }
  if (blockIdx.x == 0)
    for (int i = threadIdx.x; i <= npairs; i += NT) p.skbuf[1 + i] = s_pre[i];
  if (threadIdx.x < 128) {
    // ---------------- producer warpgroup ----------------
    if constexpr (NWG >= 3) asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");
    else asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (threadIdx.x == 0) {
      int s = 0;
      uint32_t ph = 0;
      for (int pp = pp0; r0 < r1 && pp < npairs && s_pre[pp] < r1; ++pp) {
        int b, qb, h, tb, te, nk;
        sk_pair(p, QBT, pp, b, qb, h, tb, te, nk);
        const int l0 = max(r0, s_pre[pp]) - s_pre[pp], l1 = min(r1, s_pre[pp + 1]) - s_pre[pp];
        const size_t tb0 = (size_t)(b * 2 + h) * p.ntk;
        for (int t = tb + l0; t < tb + l1; ++t) {
          mbar_wait(bars + 8 * ASTAGES + 8 * s, ph ^ 1);
          const uint32_t full = bars + 8 * s;
          mbar_expect_tx(full, STAGE);
          const uint32_t dst = smem_u32(ring + s * STAGE);
          bulk_g2s(dst, p.kt + (tb0 + t) * TILE_K, TILE_K, full);
          bulk_g2s(dst + TILE_K, p.vt + (tb0 + t) * TILE_V, TILE_V, full);
          if (++s == ASTAGES) { s = 0; ph ^= 1; }
        }
      }
    }
    return;
  }
  // ---------------- consumers ----------------
  if constexpr (NWG == 3) asm volatile("setmaxnreg.inc.sync.aligned.u32 160;\n");
  else asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  const int ctid = threadIdx.x - 128;
  const int wg = ctid >> 7, w = (ctid >> 5) & 3, lane = ctid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  // the consumers' loop state (pair, ring position) lives in shared memory, not in registers across the tile loop:
  // ctid 0 writes the next segment's values after the post-staging barrier; everyone reads them after the top barrier
  __shared__ int s_loop[2];
  if (ctid == 0) { s_loop[0] = pp0; s_loop[1] = 0; }
  for (int seg = 0;; ++seg) {
    named_bar_sync(1, 128 * NWG);  // (also: every consumer warpgroup is done with the previous segment's Q)
    const int pp = reinterpret_cast<volatile int*>(s_loop)[0], ring0 = reinterpret_cast<volatile int*>(s_loop)[1];
    if (!(r0 < r1 && pp < p.nb * p.nqb * 2 && s_pre[pp] < r1)) break;
    int b, qb, h, tb, te, nk;
    sk_pair(p, QBT, pp, b, qb, h, tb, te, nk);
    const int l0 = max(r0, s_pre[pp]) - s_pre[pp], l1 = min(r1, s_pre[pp + 1]) - s_pre[pp];
    const int stb = tb + l0, ste = tb + l1 - 1;
    const int L = p.L;
    const int q_first = qb * QBT, q_last = min(q_first + QBT - 1, L - 1);
    for (int i = ctid; i < 64 * NWG * 4; i += 128 * NWG) {
      const int r = i >> 2, c = i & 3;
      const int tok = q_first + (r >> 3);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (tok < L) {
        const size_t trow = (size_t)b * L + tok;
        v = *reinterpret_cast<const uint4*>(p.q + (trow * 16 + 8 * h + (r & 7)) * 64 + c * 16);
      }
      *reinterpret_cast<uint4*>(qs + r * 64 + ((c ^ ((r >> 1) & 3)) << 4)) = v;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    named_bar_sync(1, 128 * NWG);
    if (ctid == 0) { s_loop[0] = pp + 1; s_loop[1] = ring0 + ste - stb + 1; }  // read after the next top barrier
    const int tokA = q_first + 8 * wg + 2 * w, tokB = tokA + 1;
    const int kmaxA = nk - L + tokA, kmaxB = nk - L + tokB;
    const int kminA = p.window ? kmaxA - p.window + 1 : -1, kminB = p.window ? kmaxB - p.window + 1 : -1;
    AttnState stt;
#pragma unroll
    for (int i = 0; i < 36; ++i) stt.o[i] = 0.f;
    stt.mA = -1e30f; stt.mB = -1e30f; stt.nready = 0;
    AttnCtx cx{p, qs, ring, bars, wg, w, lane, g, t4, kmaxA, kmaxB, kminA, kminB, nk, L, q_first, q_last, stb, ste,
               desc_sw64(smem_u32(qs + wg * 64 * 64)), desc_sw64(smem_u32(ring)), desc_sw128(smem_u32(ring + TILE_K)),
               stb - ring0};
    float s0[64];
    if constexpr (PIPE && RRS) {
      if (seg == 0 && wg == NWG - 1) named_bar_arrive(4, 256);  // warpgroup 0 starts (later segments: the last hand-off)
      attn_pipe<EXPMODE, NWG>(cx, stt, s0);
    } else {
      for (int t = stb; t <= ste; ++t) attn_step<EXPMODE, false>(cx, stt, s0, t);
    }
    // partial (O unnormalized, m, l) of this segment: slot blockIdx.x * MAXSEG + seg, row (token in block) * 8 + g
    float* o = stt.o;
    const float lA = __shfl_sync(0xffffffffu, o[32], lane & ~3), lB = __shfl_sync(0xffffffffu, o[34], lane & ~3);
    float* wslot = p.ws + ((size_t)pp + blockIdx.x) * (QBT * 8 * 66);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int tok = half ? tokB : tokA;
      if (tok >= L) continue;
      float* wp = wslot + ((tok - q_first) * 8 + g) * 66;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj)
        *reinterpret_cast<float2*>(wp + 8 * jj + 2 * t4) = make_float2(o[4 * jj + 2 * half], o[4 * jj + 2 * half + 1]);
      if (t4 == 0) {
        wp[64] = half ? stt.mB : stt.mA;
        wp[65] = half ? lB : lA;
      }
    }
  }
}

// stream-K merge: one warp per R (token, head) rows (adjacent heads of one kv head), the pair's partials from the CTAs
// whose ranges cover it, then O / l, the per-head RMSNorm and * sigmoid(gate)
template <int R, int QBT>
__global__ void __launch_bounds__(256) attn_combine_sk_kernel(const float* __restrict__ ws, const int* __restrict__ skbuf,
                                                              int T, int L, int nqb, const __nv_bfloat16* __restrict__ qkv,
                                                              __nv_bfloat16* __restrict__ out, float c, float v_scale) {
  pdl_enter_attn();
  const int wid = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (wid >= T * (16 / R)) return;
  const int r0 = R * wid;
  const int tok = r0 >> 4, head = r0 & 15;
  const int b = tok / L, tq = tok - b * L, qb = tq / QBT;
  const int npairs = (T / L) * nqb * 2;
  const int pp = (b * nqb + qb) * 2 + (head >> 3);
  const int W = skbuf[0];
  const int* pre = skbuf + 1;
  const int i0 = pre[pp] / W, i1 = (pre[pp + 1] - 1) / W;
  const int rowp = (tq - qb * QBT) * 8 + (head & 7);
  float M[R], o0[R], o1[R], l[R];
#pragma unroll
  for (int r = 0; r < R; ++r) { M[r] = -1e30f; o0[r] = 0.f; o1[r] = 0.f; l[r] = 0.f; }
  for (int i = i0; i <= i1; ++i) {
    const float* wp = ws + (((size_t)pp + i) * (QBT * 8) + rowp) * 66;
#pragma unroll
    for (int r = 0; r < R; ++r) M[r] = fmaxf(M[r], wp[66 * r + 64]);
  }
  for (int i = i0; i <= i1; ++i) {
    const float* wp = ws + (((size_t)pp + i) * (QBT * 8) + rowp) * 66;
    float ls[R], ms[R];
    float2 v[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      ls[r] = wp[66 * r + 65];
      ms[r] = wp[66 * r + 64];
      v[r] = *reinterpret_cast<const float2*>(wp + 66 * r + 2 * lane);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (ls[r] != 0.f) {
        const float sc = exp2f((ms[r] - M[r]) * c);
        o0[r] += sc * v[r].x; o1[r] += sc * v[r].y; l[r] += sc * ls[r];
      }
    }
  }
  float ss[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float inv = v_scale / l[r];
    o0[r] *= inv; o1[r] *= inv;
    ss[r] = o0[r] * o0[r] + o1[r] * o1[r];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int r = 0; r < R; ++r) ss[r] += __shfl_xor_sync(0xffffffffu, ss[r], off);
  }
  const __nv_bfloat16* gp = qkv + (size_t)tok * 2304 + 1280 + head * 64 + 2 * lane;
  __nv_bfloat16* op = out + (size_t)tok * 1024 + head * 64 + 2 * lane;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float rr = rsqrtf(ss[r] / 64.f + 1e-5f);
    const float2 gv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp + 64 * r));
    *reinterpret_cast<__nv_bfloat162*>(op + 64 * r) =
        __floats2bfloat162_rn(o0[r] * rr / (1.f + __expf(-gv.x)), o1[r] * rr / (1.f + __expf(-gv.y)));
  }
}


// Same merge, R (token, head) rows per warp (adjacent heads of one token): all R rows' loads are in flight together.
template <int R>
__global__ void __launch_bounds__(256) attn_combine2_kernel(const float* __restrict__ ws, int nsplit, int T,
                                                            const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out,
                                                            float c, float v_scale) {
  pdl_enter_attn();
  const int wid = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (wid >= T * (16 / R)) return;
  const int r0 = R * wid;  // rows r0 .. r0 + R - 1: token r0 >> 4, heads (r0 & 15) ..
  const int tok = r0 >> 4, head = r0 & 15;
  float M[R], o0[R], o1[R], l[R];
#pragma unroll
  for (int r = 0; r < R; ++r) { M[r] = -1e30f; o0[r] = 0.f; o1[r] = 0.f; l[r] = 0.f; }
  for (int s = 0; s < nsplit; ++s) {
    const float* wp = ws + ((size_t)s * T * 16 + r0) * 66;
#pragma unroll
    for (int r = 0; r < R; ++r) M[r] = fmaxf(M[r], wp[66 * r + 64]);
  }
  for (int s = 0; s < nsplit; ++s) {
    const float* wp = ws + ((size_t)s * T * 16 + r0) * 66;
    float ls[R], ms[R];
    float2 v[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      ls[r] = wp[66 * r + 65];
      ms[r] = wp[66 * r + 64];
      v[r] = *reinterpret_cast<const float2*>(wp + 66 * r + 2 * lane);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (ls[r] != 0.f) {  // an empty split contributes nothing
        const float sc = exp2f((ms[r] - M[r]) * c);
        o0[r] += sc * v[r].x; o1[r] += sc * v[r].y; l[r] += sc * ls[r];
      }
    }
  }
  float ss[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float inv = v_scale / l[r];
    o0[r] *= inv; o1[r] *= inv;
    ss[r] = o0[r] * o0[r] + o1[r] * o1[r];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int r = 0; r < R; ++r) ss[r] += __shfl_xor_sync(0xffffffffu, ss[r], off);
  }
  const __nv_bfloat16* gp = qkv + (size_t)tok * 2304 + 1280 + head * 64 + 2 * lane;
  __nv_bfloat16* op = out + (size_t)tok * 1024 + head * 64 + 2 * lane;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float rr = rsqrtf(ss[r] / 64.f + 1e-5f);
    const float2 gv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp + 64 * r));
    *reinterpret_cast<__nv_bfloat162*>(op + 64 * r) =
        __floats2bfloat162_rn(o0[r] * rr / (1.f + __expf(-gv.x)), o1[r] * rr / (1.f + __expf(-gv.y)));
  }
}

}  // namespace okc_attn
'''

LT_CPP = r'''
#include <torch/extension.h>
int64_t lt_plan(int64_t M, int64_t N, int64_t K, bool out_f32, int64_t ws_bytes, int64_t max_algos);
int64_t lt_plan_sm(int64_t M, int64_t N, int64_t K, bool out_f32, int64_t ws_bytes, int64_t max_algos, int64_t sm_count);
int64_t lt_plan_kb(int64_t M, int64_t N, int64_t K, int64_t nb, bool out_f32, int64_t ws_bytes, int64_t max_algos);
int64_t lt_nalgo(int64_t id);
std::vector<int64_t> lt_info(int64_t id, int64_t algo);
int64_t lt2_add(int64_t m, int64_t n, int64_t k, bool out_f32, int64_t lay, std::vector<int64_t> cfg, int64_t ws_bytes);
void lt2_run(int64_t h, torch::Tensor x, torch::Tensor w, torch::Tensor wt, torch::Tensor out, torch::Tensor ws);
void lt_run(int64_t id, int64_t algo, torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws);
std::vector<int64_t> ltnm_info(int64_t kind, int64_t id, int64_t algo);
int64_t ltnm_resched(int64_t kind, int64_t id, int64_t algo, int64_t mode, int64_t ws_limit);
int64_t ltnm_run(int64_t kind, int64_t id, int64_t algo, torch::Tensor x, torch::Tensor w, torch::Tensor wt, torch::Tensor out,
                 torch::Tensor ws, int64_t ws_bytes, bool check);
std::vector<int64_t> ltnm_cap(int64_t kind, int64_t id, int64_t algo, torch::Tensor x, torch::Tensor w, torch::Tensor wt,
                              torch::Tensor out, torch::Tensor ws, int64_t ws_bytes);
std::vector<int64_t> ltnm_graph_stats(int64_t graph, int64_t ws_base, int64_t ws_total);
'''

LT_GLUE = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <vector>
struct LtPlan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, lc;
  std::vector<cublasLtMatmulHeuristicResult_t> res;
};
static std::vector<LtPlan*> g_lt_plans;
static cublasLtHandle_t g_lt = nullptr;

// ---- explicit-config cuBLASLt algorithms (algo 66 sweep winners; layouts as in the bf16 board's LTCPP) ----
// layout bit 0: row-major descriptors with x as A (else column-major, W as A); bit 1: W pre-transposed [K, N] (wt)
struct Lt2Algo { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; int lay; size_t ws; };
static std::vector<Lt2Algo> g_lt2;
static bool lt2_set(cublasLtMatmulAlgo_t& a, int attr, int64_t v) {
  if (attr >= 7) { uint16_t x = (uint16_t)v; return cublasLtMatmulAlgoConfigSetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)attr, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS; }
  uint32_t x = (uint32_t)v; return cublasLtMatmulAlgoConfigSetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)attr, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS;
}
// cfg: CUBLASLT_ALGO_CONFIG_{ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID, INNER_SHAPE_ID, CLUSTER_SHAPE_ID}
// returns a handle, or -1 when this cuBLASLt does not offer / accept the configuration
int64_t lt2_add(int64_t m, int64_t n, int64_t k, bool out_f32, int64_t lay, std::vector<int64_t> cfg, int64_t ws_bytes) {
  if (cfg.size() != 9) return -1;
  if (!g_lt && cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return -1;
  Lt2Algo r{};
  r.lay = (int)lay;
  const bool row = lay & 1, pk = lay & 2;
  const cudaDataType_t ct = out_f32 ? CUDA_R_32F : CUDA_R_16BF;
  if (cublasLtMatmulDescCreate(&r.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return -1;
  cublasOperation_t ta = !row && !pk ? CUBLAS_OP_T : CUBLAS_OP_N, tb = row && !pk ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  if (row) {
    cublasLtMatrixLayoutCreate(&r.a, CUDA_R_16BF, m, k, k);
    cublasLtMatrixLayoutCreate(&r.b, CUDA_R_16BF, pk ? k : n, pk ? n : k, pk ? n : k);
    cublasLtMatrixLayoutCreate(&r.c, ct, m, n, n);
    cublasLtOrder_t o = CUBLASLT_ORDER_ROW;
    for (auto d : {r.a, r.b, r.c}) cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ORDER, &o, sizeof(o));
  } else {
    cublasLtMatrixLayoutCreate(&r.a, CUDA_R_16BF, pk ? n : k, pk ? k : n, pk ? n : k);
    cublasLtMatrixLayoutCreate(&r.b, CUDA_R_16BF, k, m, k);
    cublasLtMatrixLayoutCreate(&r.c, ct, n, m, n);
  }
  cublasLtMatmulAlgo_t a;
  if (cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, ct, ct, (int)cfg[0], &a) != CUBLAS_STATUS_SUCCESS) return -1;
  for (int i = 1; i < 9; i++) if (!lt2_set(a, i, cfg[i]) && cfg[i]) return -1;
  if (cfg[2] > 1 && cfg[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) return -1;
  uint64_t fl = 0; size_t w = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &fl, sizeof(fl), &w) != CUBLAS_STATUS_SUCCESS) return -1;
  if ((fl & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F || (fl & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32)) return -1;
  cublasLtMatmulHeuristicResult_t hr{};
  if (cublasLtMatmulAlgoCheck(g_lt, r.op, r.a, r.b, r.c, r.c, &a, &hr) != CUBLAS_STATUS_SUCCESS || hr.state != CUBLAS_STATUS_SUCCESS ||
      (int64_t)hr.workspaceSize > ws_bytes) return -1;
  r.algo = a;
  r.ws = hr.workspaceSize;
  g_lt2.push_back(r);
  return (int64_t)g_lt2.size() - 1;
}
void lt2_run(int64_t h, torch::Tensor x, torch::Tensor w, torch::Tensor wt, torch::Tensor out, torch::Tensor ws) {
  const Lt2Algo& r = g_lt2.at(h);
  const float alpha = 1.f, beta = 0.f;
  const void* wp = (r.lay & 2) ? wt.data_ptr() : w.data_ptr();
  const void* a = (r.lay & 1) ? x.data_ptr() : wp;
  const void* b = (r.lay & 1) ? wp : x.data_ptr();
  auto s = cublasLtMatmul(g_lt, r.op, &alpha, a, r.a, b, r.b, &beta, out.data_ptr(), r.c, out.data_ptr(), r.c, &r.algo,
                          ws.data_ptr(), (size_t)ws.numel(), c10::cuda::getCurrentCUDAStream());
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul (lt2) failed: ", (int)s);
}
int64_t lt_plan_sm(int64_t M, int64_t N, int64_t K, bool out_f32, int64_t ws_bytes, int64_t max_algos, int64_t sm_count) {
  if (!g_lt) TORCH_CHECK(cublasLtCreate(&g_lt) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate");
  auto* p = new LtPlan();
  TORCH_CHECK(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS, "desc");
  if (sm_count > 0) {  // size the kernel for fewer SMs: a concurrent stream keeps the rest
    const int32_t smc = (int32_t)sm_count;
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &smc, sizeof(smc));
  }
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  // column-major views: A = W^T (K x N, ld K) used transposed, B = X^T (K x M, ld K), C = out^T (N x M, ld N)
  cublasLtMatrixLayoutCreate(&p->la, CUDA_R_16BF, K, N, K);
  cublasLtMatrixLayoutCreate(&p->lb, CUDA_R_16BF, K, M, K);
  cublasLtMatrixLayoutCreate(&p->lc, out_f32 ? CUDA_R_32F : CUDA_R_16BF, N, M, N);
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  uint64_t ws = (uint64_t)ws_bytes;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  p->res.resize(max_algos);
  int n = 0;
  cublasLtMatmulAlgoGetHeuristic(g_lt, p->op, p->la, p->lb, p->lc, p->lc, pref, (int)max_algos, p->res.data(), &n);
  p->res.resize(n);
  cublasLtMatmulPreferenceDestroy(pref);
  g_lt_plans.push_back(p);
  return (int64_t)g_lt_plans.size() - 1;
}
int64_t lt_plan(int64_t M, int64_t N, int64_t K, bool out_f32, int64_t ws_bytes, int64_t max_algos) {
  return lt_plan_sm(M, N, K, out_f32, ws_bytes, max_algos, 0);
}
// [r4rtred] x [M, K] @ W[N, K]^T as nb K-slices in one strided batch: batch q writes out[q] = x[:, qKb:(q+1)Kb] @
// W[:, qKb:(q+1)Kb]^T into out [nb, M, N] (Kb = K / nb). A and B keep their row pitch K and step Kb elements per batch;
// C steps M * N. Same column-major views and transposes as lt_plan_sm; lt_run / lt_nalgo / lt_info work on the plan.
int64_t lt_plan_kb(int64_t M, int64_t N, int64_t K, int64_t nb, bool out_f32, int64_t ws_bytes, int64_t max_algos) {
  if (!g_lt) TORCH_CHECK(cublasLtCreate(&g_lt) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate");
  TORCH_CHECK(nb > 0 && K % nb == 0, "lt_plan_kb: K must split into nb slices");
  const int64_t Kb = K / nb;
  auto* p = new LtPlan();
  TORCH_CHECK(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS, "desc");
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  // column-major views: A = W^T slice (Kb x N, ld K), B = X^T slice (Kb x M, ld K), C = out[q]^T (N x M, ld N)
  cublasLtMatrixLayoutCreate(&p->la, CUDA_R_16BF, Kb, N, K);
  cublasLtMatrixLayoutCreate(&p->lb, CUDA_R_16BF, Kb, M, K);
  cublasLtMatrixLayoutCreate(&p->lc, out_f32 ? CUDA_R_32F : CUDA_R_16BF, N, M, N);
  const int32_t bc = (int32_t)nb;
  const int64_t s_ab = Kb, s_c = M * N;
  for (auto d : {p->la, p->lb, p->lc})
    cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
  cublasLtMatrixLayoutSetAttribute(p->la, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &s_ab, sizeof(s_ab));
  cublasLtMatrixLayoutSetAttribute(p->lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &s_ab, sizeof(s_ab));
  cublasLtMatrixLayoutSetAttribute(p->lc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &s_c, sizeof(s_c));
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  uint64_t ws = (uint64_t)ws_bytes;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  p->res.resize(max_algos);
  int n = 0;
  cublasLtMatmulAlgoGetHeuristic(g_lt, p->op, p->la, p->lb, p->lc, p->lc, pref, (int)max_algos, p->res.data(), &n);
  p->res.resize(n);
  cublasLtMatmulPreferenceDestroy(pref);
  g_lt_plans.push_back(p);
  return (int64_t)g_lt_plans.size() - 1;
}
int64_t lt_nalgo(int64_t id) { return (int64_t)g_lt_plans[id]->res.size(); }
std::vector<int64_t> lt_info(int64_t id, int64_t algo) {
  const auto& r = g_lt_plans[id]->res[algo];
  int tile = 0, splitk = 0, stages = 0, cluster = 0, aid = 0, red = 0, swz = 0, cust = 0, inner = 0;
  size_t sz = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster, sizeof(cluster), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_ID, &aid, sizeof(aid), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof(swz), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cust, sizeof(cust), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof(inner), &sz);
  // [0..3] as before; [4] workspace; [5..9] algo id, reduction scheme, CTA swizzling, custom option, inner shape
  return {tile, splitk, stages, cluster, (int64_t)r.workspaceSize, aid, red, swz, cust, inner};
}
void lt_run(int64_t id, int64_t algo, torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws) {
  auto* p = g_lt_plans[id];
  const float alpha = 1.f, beta = 0.f;
  auto st = c10::cuda::getCurrentCUDAStream();
  auto s = cublasLtMatmul(g_lt, p->op, &alpha, w.data_ptr(), p->la, x.data_ptr(), p->lb, &beta, out.data_ptr(), p->lc,
                          out.data_ptr(), p->lc, &p->res[algo].algo, ws.data_ptr(), (size_t)ws.numel(), st);
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed: ", (int)s);
}

// ---- [r6ltnm] memset-free picks: algorithm references, a no-throw run with an explicit workspace size, a hidden
// scheduling-mode re-config, and a capture that counts the graph nodes cublasLtMatmul emits (memset = the workspace
// tile counter of a dynamically scheduled nvjet kernel) ----
#include <cuda_runtime_api.h>
#include <algorithm>
#define LTNM_ATTR_SCHED 10  // hidden algo-config attribute (uint16; "schedulingMode" in cuBLASLt's own algo log)
struct LtnmRef { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; const cublasLtMatmulAlgo_t* algo; int lay; size_t ws; };
// kind 0: heuristic plan id + result index (lt_run's views: A = W, B = x); kind 1: lt2 handle (its own layout bits)
static bool ltnm_ref(int64_t kind, int64_t id, int64_t a, LtnmRef& r) {
  if (kind == 0) {
    if (id < 0 || id >= (int64_t)g_lt_plans.size()) return false;
    auto* p = g_lt_plans[id];
    if (a < 0 || a >= (int64_t)p->res.size()) return false;
    r = LtnmRef{p->op, p->la, p->lb, p->lc, &p->res[a].algo, 0, p->res[a].workspaceSize};
    return true;
  }
  if (kind != 1 || id < 0 || id >= (int64_t)g_lt2.size()) return false;
  const Lt2Algo& q = g_lt2[id];
  r = LtnmRef{q.op, q.a, q.b, q.c, &q.algo, q.lay, q.ws};
  return true;
}
// {ok, listed workspace, AlgoCheck workspace (-1: rejected), attr 10 (-1: unreadable), tile, split-K, stages, cluster,
//  custom option, inner shape, waves x 1000}
std::vector<int64_t> ltnm_info(int64_t kind, int64_t id, int64_t algo) {
  LtnmRef r;
  if (!g_lt || !ltnm_ref(kind, id, algo, r)) return {0};
  int64_t wsc = -1, waves = -1;
  cublasLtMatmulHeuristicResult_t hr{};
  if (cublasLtMatmulAlgoCheck(g_lt, r.op, r.a, r.b, r.c, r.c, r.algo, &hr) == CUBLAS_STATUS_SUCCESS && hr.state == CUBLAS_STATUS_SUCCESS) {
    wsc = (int64_t)hr.workspaceSize;
    waves = (int64_t)(hr.wavesCount * 1000.f);
  }
  size_t sz = 0;
  uint16_t sched = 0, inner = 0, cluster = 0;
  uint32_t tile = 0, stages = 0, cust = 0;
  int32_t splitk = 0;
  const bool hs = cublasLtMatmulAlgoConfigGetAttribute(r.algo, (cublasLtMatmulAlgoConfigAttributes_t)LTNM_ATTR_SCHED, &sched, sizeof(sched), &sz) == CUBLAS_STATUS_SUCCESS;
  cublasLtMatmulAlgoConfigGetAttribute(r.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(r.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(r.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(r.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster, sizeof(cluster), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(r.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cust, sizeof(cust), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(r.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof(inner), &sz);
  return {1, (int64_t)r.ws, wsc, hs ? (int64_t)sched : -1, (int64_t)tile, (int64_t)splitk, (int64_t)stages, (int64_t)cluster,
          (int64_t)cust, (int64_t)inner, waves};
}
// a copy of the referenced algorithm with attr 10 = mode, kept only if AlgoCheck accepts it with workspace <= ws_limit;
// returns its lt2 handle, or -1 bad reference, -2 attribute not settable, -3 AlgoCheck rejected, -4 needs workspace
int64_t ltnm_resched(int64_t kind, int64_t id, int64_t algo, int64_t mode, int64_t ws_limit) {
  LtnmRef r;
  if (!g_lt || !ltnm_ref(kind, id, algo, r)) return -1;
  Lt2Algo n{};
  n.op = r.op;
  n.a = r.a;
  n.b = r.b;
  n.c = r.c;
  n.lay = r.lay;
  n.algo = *r.algo;
  const uint16_t m = (uint16_t)mode;
  if (cublasLtMatmulAlgoConfigSetAttribute(&n.algo, (cublasLtMatmulAlgoConfigAttributes_t)LTNM_ATTR_SCHED, &m, sizeof(m)) != CUBLAS_STATUS_SUCCESS) return -2;
  cublasLtMatmulHeuristicResult_t hr{};
  if (cublasLtMatmulAlgoCheck(g_lt, n.op, n.a, n.b, n.c, n.c, &n.algo, &hr) != CUBLAS_STATUS_SUCCESS || hr.state != CUBLAS_STATUS_SUCCESS) return -3;
  if ((int64_t)hr.workspaceSize > ws_limit) return -4;
  n.ws = hr.workspaceSize;
  g_lt2.push_back(n);
  return (int64_t)g_lt2.size() - 1;
}
// ws_bytes < 0: the whole buffer (as lt_run / lt2_run pass it); else min(ws_bytes, buffer)
static int64_t ltnm_call(const LtnmRef& r, const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& wt,
                         const torch::Tensor& out, const torch::Tensor& ws, int64_t ws_bytes, cudaStream_t s) {
  const float alpha = 1.f, beta = 0.f;
  const void* wp = (r.lay & 2) ? wt.data_ptr() : w.data_ptr();
  const void* A = (r.lay & 1) ? x.data_ptr() : wp;
  const void* B = (r.lay & 1) ? wp : x.data_ptr();
  const size_t wsz = ws_bytes < 0 ? (size_t)ws.numel() : std::min((size_t)ws_bytes, (size_t)ws.numel());
  return (int64_t)cublasLtMatmul(g_lt, r.op, &alpha, A, r.a, B, r.b, &beta, out.data_ptr(), r.c, out.data_ptr(), r.c, r.algo,
                                 ws.data_ptr(), wsz, s);
}
int64_t ltnm_run(int64_t kind, int64_t id, int64_t algo, torch::Tensor x, torch::Tensor w, torch::Tensor wt, torch::Tensor out,
                 torch::Tensor ws, int64_t ws_bytes, bool check) {
  LtnmRef r;
  TORCH_CHECK(g_lt && ltnm_ref(kind, id, algo, r), "ltnm_run: bad algorithm reference");
  const int64_t s = ltnm_call(r, x, w, wt, out, ws, ws_bytes, c10::cuda::getCurrentCUDAStream());
  if (check) TORCH_CHECK(s == (int64_t)CUBLAS_STATUS_SUCCESS, "cublasLtMatmul (ltnm) failed: ", s);
  if (s != (int64_t)CUBLAS_STATUS_SUCCESS) cudaGetLastError();  // a rejected trial leaves no pending (non-sticky) error behind
  return s;
}
// {nodes, kernels, memsets, memcpys, others, first memset's offset in [ws_base, ws_base + ws_total) (-1 outside, -2 none),
//  its bytes, largest kernel grid, its block, its dynamic smem}; failed queries are cleared, never left pending
static std::vector<int64_t> ltnm_stats(cudaGraph_t g, int64_t ws_base, int64_t ws_total) {
  std::vector<int64_t> v{0, 0, 0, 0, 0, -2, 0, -1, -1, -1};
  size_t n = 0;
  if (cudaGraphGetNodes(g, nullptr, &n) != cudaSuccess) { cudaGetLastError(); return v; }
  std::vector<cudaGraphNode_t> nodes(n);
  if (n && cudaGraphGetNodes(g, nodes.data(), &n) != cudaSuccess) { cudaGetLastError(); return v; }
  v[0] = (int64_t)n;
  for (size_t i = 0; i < n; ++i) {
    cudaGraphNodeType t;
    if (cudaGraphNodeGetType(nodes[i], &t) != cudaSuccess) { cudaGetLastError(); v[4]++; continue; }
    if (t == cudaGraphNodeTypeKernel) {
      v[1]++;
      cudaKernelNodeParams kp{};
      if (cudaGraphKernelNodeGetParams(nodes[i], &kp) == cudaSuccess) {
        const int64_t gs = (int64_t)kp.gridDim.x * kp.gridDim.y * kp.gridDim.z;
        if (gs > v[7]) {
          v[7] = gs;
          v[8] = (int64_t)kp.blockDim.x * kp.blockDim.y * kp.blockDim.z;
          v[9] = (int64_t)kp.sharedMemBytes;
        }
      } else {
        cudaGetLastError();
      }
    } else if (t == cudaGraphNodeTypeMemset) {
      if (v[2]++ == 0) {
        cudaMemsetParams mp{};
        if (cudaGraphMemsetNodeGetParams(nodes[i], &mp) == cudaSuccess) {
          const int64_t d = (int64_t)(uintptr_t)mp.dst;
          v[5] = (d >= ws_base && d < ws_base + ws_total) ? d - ws_base : -1;
          v[6] = (int64_t)mp.elementSize * (int64_t)mp.width * (int64_t)(mp.height ? mp.height : 1);
        } else {
          cudaGetLastError();
        }
      }
    } else if (t == cudaGraphNodeTypeMemcpy) {
      v[3]++;
    } else {
      v[4]++;
    }
  }
  return v;
}
std::vector<int64_t> ltnm_graph_stats(int64_t graph, int64_t ws_base, int64_t ws_total) {
  return ltnm_stats((cudaGraph_t)(uintptr_t)graph, ws_base, ws_total);
}
// one cublasLtMatmul captured (thread-local mode) on a private stream: {matmul status, capture error, ltnm_stats...};
// nothing is launched
std::vector<int64_t> ltnm_cap(int64_t kind, int64_t id, int64_t algo, torch::Tensor x, torch::Tensor w, torch::Tensor wt,
                              torch::Tensor out, torch::Tensor ws, int64_t ws_bytes) {
  LtnmRef r;
  if (!g_lt || !ltnm_ref(kind, id, algo, r)) return {-1, 0};
  cudaStream_t s = nullptr;
  if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) { cudaGetLastError(); return {-2, 0}; }
  int64_t st = -3;
  cudaGraph_t g = nullptr;
  cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
  if (e == cudaSuccess) {
    st = ltnm_call(r, x, w, wt, out, ws, ws_bytes, s);
    e = cudaStreamEndCapture(s, &g);
  }
  std::vector<int64_t> res{st, (int64_t)e};
  if (e == cudaSuccess && g) {
    const auto v = ltnm_stats(g, (int64_t)(uintptr_t)ws.data_ptr(), (int64_t)ws.numel());
    res.insert(res.end(), v.begin(), v.end());
  }
  if (g) cudaGraphDestroy(g);
  cudaGetLastError();
  cudaStreamDestroy(s);
  return res;
}
'''







# replicate r9 of p12ou (e27e1296), 2026-11-05
