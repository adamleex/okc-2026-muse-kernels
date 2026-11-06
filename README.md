# OKC 2026 Muse Spark kernels: team adamleex

These are the final [Open Kernel Challenge](https://openkernelchallenge.dev) kernel-track submissions of team adamleex, on one H200 (sm_90a).
- There is one self-contained submission file per problem (`<problem>.py`), exactly as the judge scored the board row.
- Every file starts with a header naming its lineage, credits and change log.

| Problem | File | Row job | Judge speedup vs SGLang | Board time | Provisional rank |
|---|---|---|---|---|---|
| muse_prefill_layer | `muse_prefill_layer.py` | 9f2d2daca12748ab | 2.005× | 193.24 ms | 20 |
| muse_decode_layer | `muse_decode_layer.py` | 533769e4756b4e5e | 2.412× | 9,378.9 µs | 43 |
| muse_attention_decode | `muse_attention_decode.py` | cc62de0415bf4ee6 | 2.895× | 1,301.0 µs | 28 |
| muse_moe_w4a16 | `muse_moe_w4a16.py` | 2613deff75ad4721 | 2.498× | 5,383.2 µs | 69 |
| muse_bf16_proj | `muse_bf16_proj.py` | cc53f10248fa4c78 | 1.601× | 3,018.3 µs | 61 |

Notes on the table:
- The overall board (decode layer) rank was 43.
- Ranks were read on 2026-11-06 after kernel submissions closed. They stay provisional until the organizers rerun the top entries (medians of reruns, held-out data included).
- The prefill and decode row files carry one trailing replicate comment line. The code is identical to the base versions:
  - prefill p12ou, md5 e27e1296;
  - decode r36FIN ach_hg, md5 872ee2f0.

## Main ideas per file
- **muse_prefill_layer**:
  - A fused persistent NVFP4 MoE kernel (dequant to BF16, wgmma).
  - FP8 attention (fmha7/fmha7p) with byte-domain exp.
  - CUDA-graph glue with PDL edges and fork points.
  - Recompute-scheme residual boundaries.
  - Prepare-time cuBLASLt picks pinned per shape.
  - An explicit T <= 8 sliced-unpooler path.
- **muse_decode_layer**: alygg77's decode layer, plus:
  - ATT8 q_len 8 attention (E4M3 query, FP8 P, byte exp);
  - HS2G;
  - MoE route-prefetch (rp) and balanced deal (balt) ports;
  - the bf16 dead-diag strip.
- **muse_attention_decode**: vjeux's attention, plus QDIR/fpart and an FMA-pipe exp share for MUFU-bound q8 kernels, and the r10 load balancing (LBAL2).
- **muse_moe_w4a16**: alygg77/vjeux MoE, plus:
  - TINY/TINY2/MID small-batch builds;
  - L1-prefetch removal (NOPF);
  - the r15 route/balance changes and the r16d barrier removal (r17).
- **muse_bf16_proj**: alygg77's bf16 projections (m14p route set; cuBLASLt via ctypes from the judge image for the large-K m=256 cells).

## License
Apache License 2.0 (see `LICENSE`). Third-party sources and notices are in `NOTICE` and in each file's header.
