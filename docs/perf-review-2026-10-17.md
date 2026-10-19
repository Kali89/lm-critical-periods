# Performance review, 2026-10-17 (read-only; baselines grid in flight)

Measured on a finished baselines run (m, 600M tokens, M3 Max 30-core GPU, fp32):

| What | Value |
|---|---|
| Training throughput | 42.7k tok/s median, p10 41.4k (steady) |
| Eval time | 3.1% of wall time (26 evals) |
| Plasticity metrics | 0.7% |
| Pure training share | ~96% |
| Estimated compute | ~97 MFLOP/token x 42.7k tok/s = ~4.1 TFLOP/s, ~35-40% of fp32 peak |

LM code path is already the right shape: fused SDPA (`mx.fast.scaled_dot_product_attention`), fused RoPE and
RMSNorm, fused QKV projection, tied unembedding as a matmul, whole step under `mx.compile`, one `mx.eval` per
step. Remaining levers, none worth taking mid-programme:

1. bf16 compute (the only lever above ~5%). Apple GPUs do not run fp16 at double rate, so the gain is memory
   bandwidth only; expect 1.2-1.5x, must be measured. It changes numerics, so it would force the baseline
   ladder to be re-run for comparability. Defer to the rented-GPU scale point, where it is the default anyway.
2. `mask="causal"` string instead of a dense (T,T) additive array for the clean path: ~1-3%. The short_context
   arm needs an array mask regardless, so keep one code path until the pilot is designed.
3. `mx.async_eval` to overlap CPU batch assembly with the GPU step: ~1%. The DocReader loop is 64 Python
   iterations per 0.78 s step; negligible.

Vision (ResNet-18 CIFAR, 690 img/s): ~1.15 TFLOP/s, roughly 11% of peak. The cost is MLX's Metal conv
kernels, not our code (compiled step, NHWC, standard BatchNorm; augmentation and test eval are <4% combined).
PyTorch-MPS typically runs this model 2-3x faster, but switching frameworks mid-grid would make the 18 runs
incomparable and the time saved on the remaining runs roughly equals the time already spent. Leave the grid
in MLX. If more vision seeds are ever needed, benchmark torch-MPS on an idle machine first and run them as a
separately labelled replication.

Decision: no changes. Nothing here lengthens the wait by more than a few percent.
