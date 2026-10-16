"""Learning-rate schedules and the corruption window."""
import math


def lr_at(step, total, tr):
    lr, min_lr = tr["lr"], tr["lr"] * tr["min_lr_frac"]
    warm = max(1, int(total * tr["warmup_frac"]))
    if step < warm:
        return lr * (step + 1) / warm
    s = tr["schedule"]
    if s == "const":
        return lr
    if s == "cosine":
        p = (step - warm) / max(1, total - warm)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))
    if s == "wsd":
        decay_start = int(total * (1 - tr["decay_frac"]))
        if step < decay_start:
            return lr
        return lr - (lr - min_lr) * (step - decay_start) / max(1, total - decay_start)
    raise ValueError(f"unknown schedule {s}")


def in_window(frac, window):
    return window is not None and window[0] <= frac < window[1]
