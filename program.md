# program.md — how this lab is run

This file is the operating manual for anyone (human or agent) working in this repo. CLAUDE.md points here.

## What we are testing

Whether corrupting a language model's *input structure* during an early window of pretraining leaves a
permanent, ability-specific deficit that later clean training cannot repair (a "critical period"), and if so,
which window hurts which ability. Background and design rationale: README.md.

## Phases and gates (each gate must pass before the next phase is queued)

1. **Setup** — `bin/lmcp doctor` all green; `configs/grid/smoke.yaml` completes through the queue.
2. **Instrument check (positive control)** — `configs/grid/vision_control.yaml`: CIFAR-10 ResNet-18 blur must
   reproduce Achille et al. (deficit for early blur, none for label permutation). If it does not, stop: the
   harness cannot be trusted to detect a sensitive period and no LM result is reported either way.
3. **Clean baselines** — `configs/grid/baselines.yaml` (3 seeds, size m). Then `bin/lmcp analyze baseline`,
   `bin/lmcp analyze landmarks`, `bin/lmcp prereg --write`.
4. **Pre-registration** — fill `docs/preregistration.md` from the prereg output: primary endpoint, seed count
   from the power table, predicted window from the Fisher-trace curve. Commit. Record the hash on OSF.
   The pilot grid is edited to the registered seed count and windows and committed in the same commit.
5. **Pilot grid** (`configs/grid/pilot.yaml`) — includes the matched-token, late-window, vocabulary-permutation
   and extended-recovery controls and two dose levels. Analyse with `bin/lmcp analyze mixed` / `dose`.
6. **Sliding-window grid** — only if the pilot deficit survives the controls.
7. **Prior-null replication** — `configs/grid/constantinescu.yaml` (English then Dutch) shows the harness
   reproduces the TACL 2025 null with the same code.
8. **Mechanism and anchors** — `tools/export_tl.py` + `induction_heads.py` + `llc.py` on saved checkpoints;
   `tools/pythia_probe.py` landmarks for comparison; `tools/export_nanogpt.py` for a CUDA rerun by others.
9. **Write-up** — notebooks/; figures follow the dataviz conventions in ~/.claude/CLAUDE.md.

## Rules of control (these are hard rules for agents)

- **Never start a training run unless Matthew asked for it in this session.** "Set up", "prepare", "check"
  are not requests to train. Queueing (`bin/lmcp grid`) is fine; draining (`scripts/run-queue.sh`) is not.
- **One run per machine at a time**, always via `scripts/run-queue.sh` (caffeinate, AC-power and thermal
  checks, lock file). Never launch `bin/lmcp run` for anything longer than a smoke test.
- **Do not edit `lmcp/` while a grid is in flight.** Each run records the git rev and a dirty flag; a mid-grid
  code change makes the runs incomparable. Fix bugs, then re-run the grid under a new name.
- **Never delete `runs/`, `results/`, or `data/` contents.** Disk is cheap; re-running is not.
- **Never `git commit` or `git push` on Matthew's behalf** unless asked. Grid pre-registration commits are his.
- **No network calls from training code.** All downloads go through `bin/lmcp data` / `prepare-all`.
- A `STOP` file in the repo root stops the queue after the current run. Use it instead of killing processes.

## Model tiers (use the cheapest tier that can do the job)

| Tier | Use for | Subagent |
|---|---|---|
| Haiku | queue status, disk/thermal checks, tailing logs, formatting results tables, collecting runs | `.claude/agents/lab-tech.md` |
| Sonnet | writing or editing eval probes, corruption functions, plotting code, fixing tracebacks, unit tests | `.claude/agents/engineer.md` |
| Opus / Fable | experiment design, choosing windows from baseline curves, interpreting results, write-up | main session or `.claude/agents/analyst.md` |

Delegate down, not up: the main session should hand routine checks to lab-tech rather than doing them itself.
Nothing in the training loop calls a model; the loop is plain Python and costs no tokens.

## Hardware rules

- MacBook Pro M3 Max, 36 GB. Memory is not the constraint; thermals and sleep are.
- Plugged in, lid open or clamshell with a cooling surface. `run-queue.sh` refuses to start on battery and
  pauses when macOS reports CPU throttling.
- Measured throughput (M3 Max, 2026-10): `xs` ~300k tok/s, `s` ~105k, `m` ~44k, `l` ~23k. A 100M-token run is
  ~40 min at `m`, ~75 min at `l`. If a run is far below that,
  something else is using the GPU; check before continuing.
- Several machines: clone the repo on each, run `bin/lmcp prepare-all` once per machine (or rsync `data/`),
  split the grid by seed (`axes: train.seed: [0]` on one machine, `[1]` on another), rsync `runs/` back to
  one machine for `bin/lmcp collect`. No cross-machine coordination is needed.

## Everyday commands

    bin/lmcp doctor                      health check: env, data, queue, git
    bin/lmcp streams                     which token streams exist (and whether they carry a document index)
    bin/lmcp tok encode --kind shuffle_sentence --intensity 0.5     build a corrupted stream (dose 0.5)
    bin/lmcp tok train --corpora babylm100m_clean,babylm_nl_clean --name en_nl_v8192   shared bilingual tokenizer
    bin/lmcp tok encode --name en_nl_v8192 --corpus babylm_nl --split both --as babylm_nl   second-language streams
    bin/lmcp grid configs/grid/pilot.yaml --dry           show what a grid would queue
    bin/lmcp grid configs/grid/pilot.yaml                 queue it
    bin/lmcp queue status
    scripts/run-queue.sh                 drain the queue (MAX_RUNS=n to cap)
    bin/lmcp collect && bin/lmcp report  results/results.jsonl, results/curves.jsonl, mean/sd table
    bin/lmcp analyze baseline|landmarks|mixed|dose|window   the real analysis (mixed model, CIs, dose, Fisher window)
    bin/lmcp prereg --write              numbers for docs/preregistration.md
    .venv-torch/bin/python tools/export_tl.py runs/<name> --check      export a checkpoint, verify logits
    .venv-torch/bin/python tools/induction_heads.py exports/tl/<name>
    .venv-torch/bin/python tools/llc.py exports/tl/<name> --data data/tok/babylm100m_clean_v8192/val.clean.bin
    .venv-torch/bin/python tools/pythia_probe.py --model pythia-14m --steps log
    bin/lmcp run --set model.size=xs --set train.tokens=2_000_000 --set corruption.kind=clean   one-off smoke run

## Validity rules (why the design looks the way it does)

- Document order is a per-seed permutation (`data.doc_order_seed`, default = train.seed), saved as
  runs/<name>/doc_order.npy. Without it a window is a fixed slice of the corpus and onset is confounded with content.
- Every corrupted arm is compared with: the clean baseline, the matched-clean-tokens control, a late window
  of the same length, and the vocabulary-permutation arm (a predicted null). A deficit that any of these
  explains is not a sensitive period.
- Constant learning rate in critical-period arms. Plasticity (Fisher trace, norms, effective rank) is logged
  at every eval so a deficit can be related to measured plasticity, not asserted.
- Seeds per cell come from `bin/lmcp prereg` (simulated power on baseline seed SD), not from habit. Three
  seeds detect only large effects (about 3 seed SDs at 70% power).
- The pre-registration commit precedes the first corrupted run. Amendments are dated additions, never edits.

## Provenance

Every run directory holds the resolved config, git rev, host, MLX version, parameter count, the full loss curve
and every eval. `bin/lmcp collect` flattens these to results/. Results files are committed; runs/ is not.
