# program.md — how this lab is run

This file is the operating manual for anyone (human or agent) working in this repo. CLAUDE.md points here.

## What we are testing

Whether corrupting a language model's *input structure* during an early window of pretraining leaves a
permanent, ability-specific deficit that later clean training cannot repair (a "critical period"), and if so,
which window hurts which ability. Background and design rationale: README.md.

## Phases and gates

1. **Setup** — `bin/lmcp doctor` is all green; the smoke grid completes; results/results.jsonl has two rows.
2. **Baselines** — clean runs at the pilot scale with 3 seeds. Inspect train_curve.jsonl and metrics.jsonl
   to locate developmental landmarks (induction jump, BLiMP rise). Decide window boundaries from these.
3. **Pilot grid** (`configs/grid/pilot.yaml`) — commit the grid file *before* running it. That commit is the
   pre-registration. Do not edit a grid file after any of its runs have started; make a new grid.
4. **Sliding-window grid** — only if the pilot shows a deficit that survives the matched-token control.
5. **Controls** — learning-rate restart arm, extended-recovery arm (2x tokens), then scale (`l`, then GPU).
6. **Analysis and write-up** — notebooks/ ; figures follow the dataviz conventions in ~/.claude/CLAUDE.md.

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
    bin/lmcp streams                     which token streams exist
    bin/lmcp tok encode --kind char_noise --param 0.1     build a corrupted stream
    bin/lmcp grid configs/grid/pilot.yaml --dry           show what a grid would queue
    bin/lmcp grid configs/grid/pilot.yaml                 queue it
    bin/lmcp queue status
    scripts/run-queue.sh               drain the queue (MAX_RUNS=n to cap)
    bin/lmcp collect && bin/lmcp report    results/results.jsonl, results/curves.jsonl, summary table
    bin/lmcp run --set model.size=xs --set train.tokens=2_000_000 --set corruption.kind=clean   one-off smoke run

## Provenance

Every run directory holds the resolved config, git rev, host, MLX version, parameter count, the full loss curve
and every eval. `bin/lmcp collect` flattens these to results/. Results files are committed; runs/ is not.
