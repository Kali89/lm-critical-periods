# Vision positive control (Achille, Rovere & Soatto 2019)

Purpose: show the harness can detect a known critical period before trusting a null on language models.
Code: `lmcp/vision/`. Grids: `configs/grid/vision_smoke.yaml` (pipeline check), `configs/grid/vision_control.yaml`.

## Data

    .venv/bin/python -m lmcp.vision.data     # one-off download to data/cifar10/, md5-checked, cached as .npy

The official host (cs.toronto.edu) ran at ~30 KB/s on 2026-10-16, so the downloader tries `dl.fbaipublicfiles.com`
first and falls back to the official URL. Either way the tarball must match md5 `c58f30108f718f92721af3b95e74349a`.
Training code never touches the network.

## Paper protocol as recorded (arXiv 1711.08856, read via ar5iv, 2026-10-16)

Source quality: the text was obtained through a summarising fetcher, not read as raw text; verify the numbers below
against the PDF before citing. Items marked NOT STATED were not found in the paper text.

| Item | Paper |
|---|---|
| Deficit "blur" (cataract) | "for the first t0 epochs the images in the dataset are downsampled to 8x8 and then upsampled back to 32x32 using bilinear interpolation" (downsampling kernel not stated; we use area/box averaging) |
| Deficit removal | deficit applied from epoch 0 to t0, then clean; "After that, the training continues for 160 more epochs" (so total = t0 + 160 epochs; stated for the main All-CNN experiment, ResNet total NOT STATED) |
| Other deficits | vertical flip and random permutation of output labels (high-level deficits, "no lasting effect"); Gaussian/white-noise input replacement (less severe than blur) |
| Architecture | ResNet-18 (He et al. 2016) on CIFAR-10 (Fig. 3); main experiments use an All-CNN |
| Optimiser | SGD; ResNet-18 initial lr 0.1, exponential decay 0.97 per epoch ("exponential annealing") |
| Weight decay | 5e-4 for ResNet-18 (All-CNN 1e-3) |
| Batch size | 128 (stated for the general setup) |
| Momentum | NOT STATED in what we read (we use 0.9, per task spec) |
| Augmentation (CIFAR) | NOT STATED for CIFAR (the appendix mentions random translations of up to 4 px and horizontal flips for the MNIST setup); we use random crop (4 px pad) + horizontal flip |
| Constant-lr variant | critical period also exists with a fixed lr (All-CNN lr 0.001), and with Adam (lr 0.001, wd 1e-4) |
| Headline numbers | **No numeric ResNet-18 accuracies found in the text**; results are plots (Fig. 1 CNN, Fig. 3 ResNet-18). Text-level claims only: ResNet-18 critical period is "remarkably sharper" than the standard CNN; for the All-CNN, a blur not removed within the first ~40-60 epochs gives up to ~3x test error; the deficit window of 40 epochs in the sensitivity plot (Fig. 1 right) shows the same pattern. |

Consequence for this repo: the expected outcome is qualitative. Blur removed early (window `[0, 0.25]`) should recover to
near the clean baseline; blur kept longer (`[0, 0.5]`) should leave a clear final-accuracy gap; the same-length blur
later (`[0.5, 0.75]`) should not; `label_permute` and `vflip` should leave no lasting gap (negative controls).
The deficit-removal curve (final accuracy vs removal epoch) can be extended with more windows `[0, f]`.

## Implementation notes and departures from the paper

- Window semantics as in the LM trainer: fractions of total steps; `[0, f]` is the paper's "deficit removed at epoch f x epochs".
- Blur is applied to the training images only (test images are always clean), before crop/flip augmentation.
- `label_permute` is one fixed random derangement of the 10 labels per seed for the whole window.
- Learning rate: `train.schedule: exp` (local to the vision trainer) = `train.lr * vision.lr_decay ** epoch`; `const`,
  `cosine`, `wsd` go through `lmcp.schedule.lr_at`. No warmup on `exp`.
- Weight decay is `vision.weight_decay` (default 5e-4), not `train.weight_decay`: the base config's 0.1 is an AdamW value and would be destructive as SGD L2.
- BatchNorm statistics are part of the compiled step state; test eval uses running stats, full 10k test set.
- Epoch count: 60 rather than 160+t0 for compute reasons (see below), with the per-epoch decay rescaled to 0.914.
  This is a departure from the paper and should be stated in any write-up.

## Compute

Measured on the M3 Max, fp32, batch 128: ~690 img/s steady state (GPU-bound; CPU augmentation is 2.4 ms/batch).
That is ~73 s per full CIFAR epoch plus ~6 s per test eval. 160 epochs = ~3.5 h/run (18 runs ~63 h); 60 epochs =
~78 min/run, 18 runs ~23-24 h. Throughput was measured with a 5000-image smoke run and a short microbenchmark, not a full epoch.

## Confirmed against the PDF (docs/refs/1711.08856v3.pdf, 2026-10-16)

Appendix A: "For the ResNet experiments, we use the ResNet-18 architecture from He et al. (2016) with initial
learning rate 0.1, learning rate decay .97 per epoch, and weight decay 0.0005." Batch size 128, SGD. Deficit:
"for the first t0 epochs the images in the dataset are downsampled to 8x8 and then upsampled back to 32x32
using bilinear interpolation ... After that, the training continues for 160 more epochs". Augmentation
(stated for the other architectures): random translations up to 4 pixels and horizontal flipping. Figure 3
(left), ResNet-18 on CIFAR-10: test accuracy spans roughly 94% (deficit removed at epoch 0) down to about 82%
(removed at epoch 180), i.e. a deficit of order 10 points, "remarkably sharper" than the All-CNN. Figure 2:
vertical flip and label permutation show no lasting deficit; noise a smaller one. `configs/grid/vision_control.yaml`
matches these settings.
