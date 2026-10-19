# lm-critical-periods

Critical-period experiments on tiny language models, trained from scratch in MLX on Apple silicon, with the
controls and measurements needed for the result to be credible. Read `program.md` before doing anything; it
holds the phases and gates, the rules of control, the model tiers, the hardware rules and the validity rules.

Non-negotiable:
- Do not start training runs unless asked in this session. Queueing (`bin/lmcp grid`) is fine; draining the
  queue (`scripts/run-queue.sh`) is not. Smoke-scale checks (xs model, a few million tokens) count as runs too.
- Do not edit training code under `lmcp/` while a grid is running (analysis-only code, `lmcp/analysis.py`,
  `lmcp/stats.py` and the analyze/prereg subcommands, is exempt; see program.md). Do not delete anything under `runs/`, `results/`, `data/`.
- Do not edit a grid file after any of its runs started; do not edit `docs/preregistration.md` after it is
  committed except by dated amendment.
- Use the `lab-tech` subagent (Haiku) for status checks and log triage, `engineer` (Sonnet) for code changes,
  `analyst` (Opus) for design and interpretation; the main session coordinates.
- All commands go through `bin/lmcp`, `scripts/run-queue.sh`, and `tools/*.py` under `.venv-torch`.
