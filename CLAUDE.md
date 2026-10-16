# lm-critical-periods

Critical-period experiments on tiny language models, trained from scratch in MLX on Apple silicon.
Read `program.md` before doing anything; it holds the rules of control, the model tiers, and the hardware rules.

Non-negotiable:
- Do not start training runs unless asked in this session. Queueing is fine, draining the queue is not.
- Do not edit `lmcp/` while a grid is running. Do not delete anything under `runs/`, `results/`, `data/`.
- Use the `lab-tech` subagent (Haiku) for status checks and log triage, `engineer` (Sonnet) for code changes,
  and reserve the main session for design and interpretation.
- All commands go through `bin/lmcp` and `scripts/run-queue.sh`.
