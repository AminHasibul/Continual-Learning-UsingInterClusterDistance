"""
recompute_from_checkpoints.py
=============================
Regenerates raw_numbers.txt, lambda_sweep.txt, and Figure 0 from the
per-seed .npz checkpoints written by car_reproduce.py — using the
CORRECT statistic:

    for each seed: final_avg = mean over seen tasks of the final-row accuracy
    report mean and POPULATION std (ddof=0) across the three per-seed scalars

This is the statistic the paper reports (Table II / Table III). The
original car_reproduce.py instead averaged per-task standard deviations,
which produced larger, inconsistent stds. No retraining is needed.

Usage (same session/dir as the run that produced ./car_results):
    python recompute_from_checkpoints.py
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "./car_results"
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
SEEDS       = [42, 123, 456]
NUM_TASKS   = 5
LAMBDA_SWEEP = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]

ABLATION = [
    ("finetune",    "Fine-tuning (no replay, lambda=0)"),
    ("replay_only", "Replay only (lambda=0)"),
    ("icf_only",    "ICF only (no replay, lambda=0.1)"),
    ("full_car",    "CAR (replay + ICF, lambda=0.1)"),
]


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────
def load_matrix(phase, label, seed):
    """Load one seed's accuracy matrix from its checkpoint."""
    path = os.path.join(CKPT_DIR, f"{phase}_{label}_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing checkpoint: {path}\n"
            f"Run car_reproduce.py first (in this session) to create it.")
    data = np.load(path, allow_pickle=True)
    return data["acc_matrix"]          # (5,5), fractions in [0,1], -1 = unset


def load_variant(phase, label):
    """Return list of per-seed (5,5) matrices for a variant."""
    return [load_matrix(phase, label, s) for s in SEEDS]


# ─────────────────────────────────────────────
# Correct statistics
# ─────────────────────────────────────────────
def per_seed_final_avg(matrices):
    """One final-average-accuracy scalar (%) per seed."""
    vals = []
    for m in matrices:
        final_row = m[NUM_TASKS - 1]
        seen = final_row[final_row >= 0]
        vals.append(seen.mean() * 100.0)
    return np.array(vals)


def summary(matrices):
    """Mean and POPULATION std of the per-seed final-average accuracy."""
    finals = per_seed_final_avg(matrices)
    return finals.mean(), finals.std(), finals   # np.std → ddof=0


def per_cell(matrices):
    """Per-cell mean and population std across seeds, in %."""
    stack = np.stack(matrices, axis=0)             # (n_seeds,5,5)
    mask  = stack[0] >= 0
    mean  = np.where(mask, stack.mean(0), -1.0) * 100.0
    std   = np.where(mask, stack.std(0),  -1.0) * 100.0   # ddof=0
    return mean, std


# ─────────────────────────────────────────────
# Gather everything
# ─────────────────────────────────────────────
def gather():
    sweep = {}
    for lam in LAMBDA_SWEEP:
        mats = load_variant("sweep", f"sweep_lam{lam}")
        mean, std, finals = summary(mats)
        sweep[lam] = dict(mean=mean, std=std, finals=finals, mats=mats)

    best_lam = max(sweep, key=lambda l: sweep[l]["mean"])

    abl = {}
    for key, label in ABLATION:
        mats = load_variant("ablation", key)
        mean, std, finals = summary(mats)
        abl[key] = dict(label=label, mean=mean, std=std, mats=mats)

    return sweep, best_lam, abl


# ─────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────
def write_lambda_sweep(sweep, best_lam):
    path = os.path.join(RESULTS_DIR, "lambda_sweep.txt")
    with open(path, "w") as f:
        f.write("Lambda Sweep Results\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write("Std = population std (ddof=0) of per-seed "
                "final-average accuracy\n\n")
        f.write(f"{'lambda':<8} {'Avg Acc (%)':>12} {'Std (%)':>10}\n")
        f.write("-" * 35 + "\n")
        for lam in LAMBDA_SWEEP:
            d = sweep[lam]
            mark = "  <- BEST" if lam == best_lam else ""
            f.write(f"{lam:<8} {d['mean']:>12.1f} {d['std']:>10.1f}{mark}\n")
    print(f"  wrote {path}")


def write_raw_numbers(sweep, best_lam, abl):
    car_mean, car_std = per_cell(sweep[best_lam]["mats"])
    path = os.path.join(RESULTS_DIR, "raw_numbers.txt")
    with open(path, "w") as f:
        f.write("CAR Reproduction Results\n")
        f.write(f"Seeds       : {SEEDS}\n")
        f.write(f"Best lambda : {best_lam}\n")
        f.write("All '±' = population std (ddof=0) across the three "
                "per-seed final scalars\n\n")

        f.write("LAMBDA SWEEP\n")
        f.write(f"{'lambda':<8} {'Avg Acc (%)':>12} {'Std':>8}\n")
        f.write("-" * 32 + "\n")
        for lam in LAMBDA_SWEEP:
            d = sweep[lam]
            mark = "  <- BEST" if lam == best_lam else ""
            f.write(f"{lam:<8} {d['mean']:>12.1f} {d['std']:>8.1f}{mark}\n")

        f.write("\nTABLE I — mean (%)   [best lambda]\n")
        for i in range(NUM_TASKS):
            row = [("   -  " if car_mean[i][j] < 0 else f"{car_mean[i][j]:5.1f}")
                   for j in range(NUM_TASKS)]
            f.write("  ".join(row) + "\n")

        f.write("\nTABLE I — std (%)\n")
        for i in range(NUM_TASKS):
            row = [("   -  " if car_std[i][j] < 0 else f"{car_std[i][j]:5.1f}")
                   for j in range(NUM_TASKS)]
            f.write("  ".join(row) + "\n")

        f.write("\nTABLE III — Ablation\n")
        for key, _ in ABLATION:
            d = abl[key]
            f.write(f"  {d['label']}: {d['mean']:.1f} ± {d['std']:.1f}\n")
    print(f"  wrote {path}")


def save_figure0(sweep, best_lam):
    lams = LAMBDA_SWEEP
    means = [sweep[l]["mean"] for l in lams]
    stds  = [sweep[l]["std"]  for l in lams]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(range(len(lams)), means, yerr=stds, fmt="o-",
                color="steelblue", linewidth=2, capsize=5, markersize=7)
    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels([str(l) for l in lams])
    ax.set_xlabel(r"$\lambda$ (ICF weight)")
    ax.set_ylabel("Final average accuracy (%)")
    ax.set_title("Lambda sweep (population std error bars)")
    ax.axvline(x=lams.index(best_lam), color="red", linestyle="--",
               alpha=0.6, label=fr"Best $\lambda$={best_lam}")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "figure0_lambda_sweep.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  wrote {path}")


# ─────────────────────────────────────────────
# Console verification
# ─────────────────────────────────────────────
def print_verification(sweep, best_lam, abl):
    print("\nRecomputed with population std of per-seed final scalars:")
    print(f"  {'lambda':<8}{'mean':>8}{'std':>8}   per-seed finals")
    for lam in LAMBDA_SWEEP:
        d = sweep[lam]
        finals = ", ".join(f"{v:.1f}" for v in d["finals"])
        star = "  *best" if lam == best_lam else ""
        print(f"  {lam:<8}{d['mean']:>8.1f}{d['std']:>8.1f}   [{finals}]{star}")
    print("\n  Ablation:")
    for key, _ in ABLATION:
        d = abl[key]
        print(f"    {d['label']:<38} {d['mean']:5.1f} ± {d['std']:.1f}")
    print("\n  These should equal your paper's Table II / Table III.")


def main():
    if not os.path.isdir(CKPT_DIR):
        raise SystemExit(
            f"No checkpoint dir at {CKPT_DIR}. Run car_reproduce.py first.")
    sweep, best_lam, abl = gather()
    print(f"Best lambda = {best_lam}")
    write_lambda_sweep(sweep, best_lam)
    write_raw_numbers(sweep, best_lam, abl)
    save_figure0(sweep, best_lam)
    print_verification(sweep, best_lam, abl)
    print("\nDone. All three artifacts now use the paper's statistic.")


if __name__ == "__main__":
    main()
