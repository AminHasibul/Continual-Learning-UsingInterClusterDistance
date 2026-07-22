"""
car_reproduce.py
================
Reproduction script for arXiv:2510.07648
Cluster-Aware Replay (CAR) on Split CIFAR-10

Checkpoint-aware: saves after every (lambda, seed) run. If the process is
interrupted, re-run the script and it resumes from the last checkpoint.

Produces:
  - Lambda sweep : finds best λ ∈ {0.1, 1, 5, 10, 20, 50}
  - Table I      : per-task accuracy matrix using best λ
  - Table III    : ablation (fine-tuning / replay-only / ICF-only / full CAR)
  - Figure 0     : lambda sweep plot
  - Figure 1     : final per-task accuracy bar chart
  - Figure 2     : average accuracy curve over tasks
  - Figure 3     : per-task forgetting
  - Figure 4     : training loss curve (Task 1, full CAR)

Usage (Google Colab T4):
  !pip install torch torchvision matplotlib numpy --quiet
  !python car_reproduce.py

All results saved to ./car_results/
Resume after disconnect: just re-run the same command.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 0. PATHS AND REPRODUCIBILITY
# ─────────────────────────────────────────────
SEEDS        = [42, 123, 456]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR  = "./car_results"
CKPT_DIR     = "./car_results/checkpoints"
PROGRESS_FILE = "./car_results/progress.json"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

print(f"Device : {DEVICE}")
print(f"Seeds  : {SEEDS}")
print(f"Results: {RESULTS_DIR}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# 1. PROGRESS TRACKER
# Saves/loads a JSON that records every completed run.
# Key format:  "sweep|lam=10.0|seed=42"
#              "ablation|Fine-tuning|seed=42"
# ─────────────────────────────────────────────
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def run_key(phase, label, seed):
    """Unique string key for one (phase, label, seed) run."""
    return f"{phase}|{label}|seed={seed}"


def ckpt_path(phase, label, seed):
    """File path for the numpy result of one run."""
    safe = label.replace(" ", "_").replace("/", "-") \
                .replace("=", "").replace("(", "").replace(")", "") \
                .replace(",", "").replace("ω", "w")
    return os.path.join(CKPT_DIR, f"{phase}_{safe}_seed{seed}.npz")


def save_run_result(phase, label, seed, acc_matrix, loss_curves):
    """Save acc_matrix and loss_curves for one completed run."""
    path = ckpt_path(phase, label, seed)
    # loss_curves: list of lists — convert to ragged object array
    lc_array = np.array(loss_curves, dtype=object)
    np.savez(path,
             acc_matrix=acc_matrix,
             loss_curves=lc_array)
    print(f"    ✓ Checkpoint saved → {os.path.basename(path)}")


def load_run_result(phase, label, seed):
    """Load a previously saved run. Returns (acc_matrix, loss_curves)."""
    path = ckpt_path(phase, label, seed)
    data = np.load(path, allow_pickle=True)
    acc_matrix  = data['acc_matrix']
    loss_curves = data['loss_curves'].tolist()
    return acc_matrix, loss_curves


def run_is_done(progress, phase, label, seed):
    return run_key(phase, label, seed) in progress


def mark_done(progress, phase, label, seed):
    progress[run_key(phase, label, seed)] = True
    save_progress(progress)


# ─────────────────────────────────────────────
# 2. HYPERPARAMETERS  (exactly as in paper)
# ─────────────────────────────────────────────
NUM_TASKS           = 5
CLASSES_PER_TASK    = 2
EPOCHS_PER_TASK     = 20
BATCH_SIZE          = 32
LR                  = 0.001
EXEMPLARS_PER_CLASS = 20

LAMBDA_SWEEP = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]


def final_scalar_stats(all_matrices):
    """Compute one final-average-accuracy scalar per seed (mean over seen
    tasks of that seed's final-row accuracy), then the population std
    (ddof=0) across those per-seed scalars. Returns (mean, std) in percent."""
    finals = []
    for m in all_matrices:
        fr = m[NUM_TASKS - 1]
        finals.append(fr[fr >= 0].mean() * 100.0)
    finals = np.array(finals)
    return finals.mean(), finals.std()


# ─────────────────────────────────────────────
# 3. DATA
# ─────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


def get_cifar10():
    train = torchvision.datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=TRANSFORM)
    test  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=TRANSFORM)
    return train, test


class ClassSubset(Dataset):
    def __init__(self, base_dataset, classes):
        self.base    = base_dataset
        targets      = np.array(base_dataset.targets)
        self.indices = np.where(np.isin(targets, classes))[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


class ListDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y = self.items[i]
        return x, y


# ─────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────
def build_model():
    model         = resnet18(weights=None, num_classes=10)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(DEVICE)


def get_features(model, x):
    x = model.conv1(x);   x = model.bn1(x)
    x = model.relu(x);    x = model.maxpool(x)
    x = model.layer1(x);  x = model.layer2(x)
    x = model.layer3(x);  x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


# ─────────────────────────────────────────────
# 5. REPLAY BUFFER
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, exemplars_per_class=20):
        self.n    = exemplars_per_class
        self.data = {}

    def add(self, dataset, classes):
        targets = np.array(dataset.targets)
        for c in classes:
            idx    = np.where(targets == c)[0]
            chosen = np.random.choice(
                idx, size=min(self.n, len(idx)), replace=False)
            self.data[c] = [(dataset[i][0], dataset[i][1])
                            for i in chosen]

    def as_dataset(self):
        items = []
        for v in self.data.values():
            items.extend(v)
        return items

    def __len__(self):
        return sum(len(v) for v in self.data.values())


# ─────────────────────────────────────────────
# 6. ICF LOSS  (paper eq. 1–2)
# ─────────────────────────────────────────────
class ICFLoss(nn.Module):
    def __init__(self, lam):
        super().__init__()
        self.lam       = lam
        self.centroids = {}

    @torch.no_grad()
    def update_centroids(self, model, dataset, classes):
        model.eval()
        loader          = DataLoader(dataset, batch_size=256, shuffle=False)
        feats_per_class = {c: [] for c in classes}
        for x, y in loader:
            x = x.to(DEVICE)
            f = F.normalize(get_features(model, x), p=2, dim=1)
            for c in classes:
                mask = (y == c)
                if mask.any():
                    feats_per_class[c].append(f[mask].cpu())
        for c in classes:
            if feats_per_class[c]:
                cat = torch.cat(feats_per_class[c], dim=0)
                mu  = cat.mean(0)
                self.centroids[c] = F.normalize(mu, p=2, dim=0).to(DEVICE)
        model.train()

    def forward(self, features):
        if not self.centroids:
            return torch.tensor(0.0, device=features.device)
        f_norm = F.normalize(features, p=2, dim=1)
        loss   = torch.tensor(0.0, device=features.device)
        for mu in self.centroids.values():
            dist  = torch.norm(f_norm - mu.unsqueeze(0), dim=1)
            loss += -dist.mean()
        loss /= len(self.centroids)
        return self.lam * loss


# ─────────────────────────────────────────────
# 7. TRAINING / EVAL
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion_ce, icf_loss=None):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        feats  = get_features(model, x)
        logits = model.fc(feats)
        loss   = criterion_ce(logits, y)
        if icf_loss is not None:
            loss = loss + icf_loss(feats)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, dataset, classes):
    subset = ClassSubset(dataset, classes)
    loader = DataLoader(subset, batch_size=256, shuffle=False)
    correct, total = 0, 0
    model.eval()
    for x, y in loader:
        x, y    = x.to(DEVICE), y.to(DEVICE)
        preds   = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────
# 8. SINGLE CL RUN
# ─────────────────────────────────────────────
def run_cl(train_ds, test_ds, task_classes,
           use_replay, use_icf, icf_lambda, seed):
    set_seed(seed)
    model     = build_model()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    buffer    = ReplayBuffer(EXEMPLARS_PER_CLASS) if use_replay else None
    icf       = ICFLoss(lam=icf_lambda)           if use_icf   else None

    acc_matrix  = np.full((NUM_TASKS, NUM_TASKS), -1.0)
    loss_curves = []

    for t, classes in enumerate(task_classes):
        print(f"      Task {t+1}/5  classes={classes}", flush=True)

        current_ds = ClassSubset(train_ds, classes)
        if buffer and len(buffer) > 0:
            loader = DataLoader(
                ConcatDataset([current_ds, ListDataset(buffer.as_dataset())]),
                batch_size=BATCH_SIZE, shuffle=True,
                num_workers=2, pin_memory=True)
        else:
            loader = DataLoader(
                current_ds, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=2, pin_memory=True)

        epoch_losses = []
        for ep in range(EPOCHS_PER_TASK):
            loss = train_one_epoch(
                model, loader, optimizer, criterion, icf)
            epoch_losses.append(loss)
            # Print every 5 epochs so you can see progress
            if (ep + 1) % 5 == 0:
                print(f"        epoch {ep+1}/{EPOCHS_PER_TASK} "
                      f"loss={loss:.4f}", flush=True)
        loss_curves.append(epoch_losses)

        if icf is not None:
            icf.update_centroids(
                model, ClassSubset(train_ds, classes), classes)
        if buffer is not None:
            buffer.add(train_ds, classes)

        for j in range(t + 1):
            acc_matrix[t][j] = evaluate(model, test_ds, task_classes[j])

        # Print accuracy row after each task
        seen = [f"{acc_matrix[t][j]*100:.1f}"
                for j in range(t+1)]
        print(f"      → acc after task {t+1}: [{', '.join(seen)}]",
              flush=True)

    return acc_matrix, loss_curves


# ─────────────────────────────────────────────
# 9. CHECKPOINT-AWARE MULTI-SEED RUNNER
# ─────────────────────────────────────────────
def run_variant_with_ckpt(train_ds, test_ds, task_classes,
                          use_replay, use_icf, icf_lambda,
                          phase, label, progress):
    """
    Runs all seeds for one variant.
    Skips seeds already completed (loaded from checkpoint).
    Returns (mean_mat, std_mat, all_curves).
    """
    print(f"\n  {'─'*55}")
    print(f"  Variant : {label}")
    print(f"  λ       : {icf_lambda if use_icf else 'N/A'}")
    print(f"  {'─'*55}")

    all_matrices, all_curves = [], []

    for seed in SEEDS:
        key = run_key(phase, label, seed)

        if run_is_done(progress, phase, label, seed):
            # Load from checkpoint
            acc_matrix, loss_curves = load_run_result(phase, label, seed)
            print(f"    Seed {seed} → loaded from checkpoint ✓")
        else:
            # Run fresh
            print(f"    Seed {seed} → running...", flush=True)
            acc_matrix, loss_curves = run_cl(
                train_ds, test_ds, task_classes,
                use_replay=use_replay,
                use_icf=use_icf,
                icf_lambda=icf_lambda,
                seed=seed)
            # Save immediately
            save_run_result(phase, label, seed, acc_matrix, loss_curves)
            mark_done(progress, phase, label, seed)

        all_matrices.append(acc_matrix)
        all_curves.append(loss_curves)

    mean_mat = np.mean(all_matrices, axis=0)
    std_mat  = np.std(all_matrices,  axis=0)

    # Print summary for this variant
    final_row = mean_mat[NUM_TASKS - 1]
    seen      = final_row[final_row >= 0]
    avg_acc   = seen.mean() * 100
    print(f"\n  ✓ {label}")
    print(f"    Final avg accuracy: {avg_acc:.1f}%")

    return mean_mat, std_mat, all_curves, all_matrices


# ─────────────────────────────────────────────
# 10. PHASE 1 — LAMBDA SWEEP
# ─────────────────────────────────────────────
def run_lambda_sweep(train_ds, test_ds, task_classes, progress):
    print("\n" + "█"*60)
    print("  PHASE 1 — LAMBDA SWEEP")
    print(f"  λ values: {LAMBDA_SWEEP}")
    print("  Each λ: 3 seeds × 5 tasks × 20 epochs")
    print("█"*60)

    sweep_results = {}

    for lam in LAMBDA_SWEEP:
        label = f"sweep_lam{lam}"
        mean_mat, std_mat, _, all_matrices = run_variant_with_ckpt(
            train_ds, test_ds, task_classes,
            use_replay=True, use_icf=True,
            icf_lambda=lam,
            phase="sweep", label=label,
            progress=progress)
        sweep_results[lam] = (mean_mat, std_mat, all_matrices)

    # Pick best lambda
    best_lam = max(
        sweep_results.keys(),
        key=lambda l: (
            sweep_results[l][0][NUM_TASKS-1][
                sweep_results[l][0][NUM_TASKS-1] >= 0].mean()))

    best_mean = sweep_results[best_lam][0]
    best_seen = best_mean[NUM_TASKS-1][best_mean[NUM_TASKS-1] >= 0]

    print(f"\n{'='*60}")
    print(f"  SWEEP COMPLETE")
    print(f"  Best λ = {best_lam}  "
          f"(avg acc = {best_seen.mean()*100:.1f}%)")
    print(f"{'='*60}")

    # Save sweep summary text
    sweep_path = os.path.join(RESULTS_DIR, "lambda_sweep.txt")
    with open(sweep_path, 'w') as f:
        f.write("Lambda Sweep Results — arXiv:2510.07648\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        f.write(f"{'λ':<8} {'Avg Acc (%)':>12} {'Std (%)':>10}\n")
        f.write("-"*35 + "\n")
        for lam in LAMBDA_SWEEP:
            _, _, am  = sweep_results[lam]
            avg, std  = final_scalar_stats(am)
            marker = "  ← BEST" if lam == best_lam else ""
            f.write(f"{lam:<8} {avg:>12.1f} {std:>10.1f}{marker}\n")
    print(f"  Saved {sweep_path}")

    # Figure 0
    lams = LAMBDA_SWEEP
    avgs, stds = [], []
    for lam in lams:
        _, _, am = sweep_results[lam]
        a, s = final_scalar_stats(am)
        avgs.append(a)
        stds.append(s)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(range(len(lams)), avgs, yerr=stds,
                fmt='o-', color='steelblue', linewidth=2,
                capsize=5, markersize=7)
    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels([str(l) for l in lams])
    ax.set_xlabel('λ (ICF weight)')
    ax.set_ylabel('Final average accuracy (%)')
    ax.set_title('Figure 0 — Lambda sweep: avg accuracy vs λ')
    ax.axvline(x=lams.index(best_lam), color='red',
               linestyle='--', alpha=0.6,
               label=f'Best λ={best_lam}')
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    fig0_path = os.path.join(RESULTS_DIR, "figure0_lambda_sweep.png")
    plt.savefig(fig0_path, dpi=150); plt.close()
    print(f"  Saved {fig0_path}")

    return best_lam, sweep_results


# ─────────────────────────────────────────────
# 11. PHASE 2 — FULL ABLATION
# ─────────────────────────────────────────────
def run_ablation(train_ds, test_ds, task_classes, best_lam, progress):
    print("\n" + "█"*60)
    print(f"  PHASE 2 — FULL ABLATION  (best λ={best_lam})")
    print("█"*60)

    variants = {
        "finetune":    ("Fine-tuning (no replay, λ=0)", False, False),
        "replay_only": (f"Replay only (λ=0)",           True,  False),
        "icf_only":    (f"ICF only (no replay, λ={best_lam})", False, True),
        "full_car":    (f"CAR (replay + ICF, λ={best_lam})",   True,  True),
    }

    results = {}
    for vkey, (label, use_r, use_i) in variants.items():
        mean_mat, std_mat, curves, all_matrices = run_variant_with_ckpt(
            train_ds, test_ds, task_classes,
            use_replay=use_r, use_icf=use_i,
            icf_lambda=best_lam,
            phase="ablation", label=vkey,
            progress=progress)
        results[label] = (mean_mat, std_mat, curves, all_matrices)

    return results


# ─────────────────────────────────────────────
# 12. PRINT TABLE I
# ─────────────────────────────────────────────
def print_table1(mean_mat, std_mat, best_lam):
    print("\n" + "="*70)
    print(f"TABLE I — Per-task accuracy (%) after each task  [λ={best_lam}]")
    print("  Rows = after training task i | Cols = accuracy on task j")
    print("="*70)
    header = "After \\ Task  " + "".join(
        [f"    T{j+1}     " for j in range(NUM_TASKS)])
    print(header)
    print("-"*70)
    avgs = []
    for i in range(NUM_TASKS):
        row       = f"  Task {i+1}      "
        seen_accs = []
        for j in range(NUM_TASKS):
            if mean_mat[i][j] < 0:
                row += "     –     "
            else:
                v = mean_mat[i][j] * 100
                s = std_mat[i][j]  * 100
                row += f" {v:4.1f}±{s:3.1f} "
                seen_accs.append(v)
        avg = np.mean(seen_accs)
        avgs.append(avg)
        row += f"  avg={avg:.1f}"
        print(row)
    print("-"*70)
    print(f"  Final average accuracy: {avgs[-1]:.1f}%")
    print("="*70)


# ─────────────────────────────────────────────
# 13. PRINT TABLE III
# ─────────────────────────────────────────────
def print_table3(results):
    print("\n" + "="*60)
    print("TABLE III — Ablation Study")
    print("="*60)
    print(f"  {'Method':<40} {'Avg Acc (%)':>12}")
    print("-"*60)
    for name, (mean_mat, std_mat, _, all_matrices) in results.items():
        avg, std = final_scalar_stats(all_matrices)
        print(f"  {name:<40} {avg:>8.1f} ± {std:.1f}")
    print("="*60)


# ─────────────────────────────────────────────
# 14. FIGURES 1–4
# ─────────────────────────────────────────────
def save_figure1(mean_mat, std_mat, ft_mean_mat=None):
    """Final per-task accuracy. If the fine-tuning ablation matrix is
    provided, plot it as the comparison; otherwise plot CAR alone."""
    final  = mean_mat[NUM_TASKS - 1]
    errs   = std_mat[NUM_TASKS - 1]
    tasks  = [f"T{i+1}" for i in range(NUM_TASKS)]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(NUM_TASKS); w = 0.35
    ax.bar(x - w/2, [v*100 for v in final], w,
           yerr=[e*100 for e in errs],
           label='Replay + ICF', color='steelblue', capsize=4)
    if ft_mean_mat is not None:
        ft_final = ft_mean_mat[NUM_TASKS - 1]
        ax.bar(x + w/2, [v*100 for v in ft_final], w,
               label='Fine-tuning (measured)', color='lightcoral')
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 105)
    ax.set_title('Figure 1 — Final per-task accuracy after all 5 tasks')
    ax.legend(); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "figure1_final_accuracy.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


def save_figure2(mean_mat):
    avgs = []
    for i in range(NUM_TASKS):
        row = mean_mat[i]; seen = row[row >= 0]
        avgs.append(seen.mean() * 100)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, NUM_TASKS+1), avgs, 'o-',
            color='steelblue', linewidth=2)
    ax.set_xlabel('Task index (after training Tk)')
    ax.set_ylabel('Average accuracy over seen tasks (%)')
    ax.set_title('Figure 2 — Average accuracy curve')
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "figure2_avg_accuracy.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


def save_figure3(mean_mat):
    forgetting = []
    for j in range(NUM_TASKS - 1):
        col  = mean_mat[:, j]; seen = col[col >= 0]
        if len(seen) > 0:
            forgetting.append(
                (f"T{j+1}", seen.max()*100 - seen[-1]*100))
    tasks  = [f[0] for f in forgetting]
    values = [f[1] for f in forgetting]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tasks, values, color='salmon')
    ax.set_ylabel('Forgetting (percentage points)')
    ax.set_title('Figure 3 — Per-task forgetting')
    ax.set_ylim(0, max(values)*1.2 if values else 10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "figure3_forgetting.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


def save_figure4(all_curves):
    curve = all_curves[0][0]   # seed 42, task 1
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(curve)+1), curve,
            color='steelblue', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')
    ax.set_title('Figure 4 — Training convergence (Task 1, full CAR)')
    ax.grid(True, alpha=0.3); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "figure4_loss_curve.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


# ─────────────────────────────────────────────
# 15. SAVE ALL RAW NUMBERS
# ─────────────────────────────────────────────
def save_raw_numbers(results, car_mean, car_std,
                     best_lam, sweep_results):
    path = os.path.join(RESULTS_DIR, "raw_numbers.txt")
    with open(path, 'w') as f:
        f.write("CAR Reproduction Results\n")
        f.write("arXiv:2510.07648\n")
        f.write(f"Seeds      : {SEEDS}\n")
        f.write(f"Device     : {DEVICE}\n")
        f.write(f"Best lambda: {best_lam}\n\n")

        f.write("LAMBDA SWEEP\n")
        f.write(f"{'λ':<8} {'Avg Acc (%)':>12} {'Std':>8}\n")
        f.write("-"*32 + "\n")
        for lam in LAMBDA_SWEEP:
            _, _, am  = sweep_results[lam]
            avg, std  = final_scalar_stats(am)
            marker = "  ← BEST" if lam == best_lam else ""
            f.write(f"{lam:<8} {avg:>12.1f} {std:>8.1f}{marker}\n")

        f.write("\nTABLE I — mean (%)\n")
        f.write("Rows=after task i  Cols=task j accuracy\n")
        for i in range(NUM_TASKS):
            row = []
            for j in range(NUM_TASKS):
                if car_mean[i][j] < 0:
                    row.append("   –  ")
                else:
                    row.append(f"{car_mean[i][j]*100:5.1f}")
            f.write("  ".join(row) + "\n")

        f.write("\nTABLE I — std (%)\n")
        for i in range(NUM_TASKS):
            row = []
            for j in range(NUM_TASKS):
                if car_std[i][j] < 0:
                    row.append("   –  ")
                else:
                    row.append(f"{car_std[i][j]*100:5.1f}")
            f.write("  ".join(row) + "\n")

        f.write("\nTABLE III — Ablation\n")
        for name, (mean_mat, std_mat, _, all_matrices) in results.items():
            avg, std = final_scalar_stats(all_matrices)
            f.write(f"  {name}: {avg:.1f} ± {std:.1f}\n")

    print(f"\nRaw numbers → {path}")


# ─────────────────────────────────────────────
# 16. PRINT RESUME STATUS
# ─────────────────────────────────────────────
def print_status(progress):
    if not progress:
        print("  No checkpoints found — starting fresh.")
        return
    done = list(progress.keys())
    print(f"\n  Resuming — {len(done)} run(s) already completed:")
    for k in done:
        print(f"    ✓ {k}")


# ─────────────────────────────────────────────
# 17. MAIN
# ─────────────────────────────────────────────
def main():
    print("Loading CIFAR-10...")
    train_ds, test_ds = get_cifar10()

    task_classes = [list(range(i*2, i*2+2)) for i in range(NUM_TASKS)]
    print(f"Task splits : {task_classes}")
    print(f"Lambda sweep: {LAMBDA_SWEEP}")

    # Load existing progress
    progress = load_progress()
    print_status(progress)

    # ── PHASE 1: lambda sweep ─────────────────────────────────
    best_lam, sweep_results = run_lambda_sweep(
        train_ds, test_ds, task_classes, progress)

    # Save best lambda so we can verify on resume
    best_lam_path = os.path.join(RESULTS_DIR, "best_lambda.txt")
    with open(best_lam_path, 'w') as f:
        f.write(str(best_lam))
    print(f"Best λ={best_lam} saved to {best_lam_path}")

    # ── PHASE 2: full ablation ────────────────────────────────
    results = run_ablation(
        train_ds, test_ds, task_classes, best_lam, progress)

    # ── Extract full CAR results ──────────────────────────────
    car_key               = f"CAR (replay + ICF, λ={best_lam})"
    car_mean, car_std, car_curves, _ = results[car_key]

    # ── Print tables ──────────────────────────────────────────
    print_table1(car_mean, car_std, best_lam)
    print_table3(results)

    # ── Save figures ──────────────────────────────────────────
    print("\nGenerating figures...")
    ft_key = "Fine-tuning (no replay, λ=0)"
    ft_mean = results[ft_key][0] if ft_key in results else None
    save_figure1(car_mean, car_std, ft_mean_mat=ft_mean)
    save_figure2(car_mean)
    save_figure3(car_mean)
    save_figure4(car_curves)

    # ── Save raw numbers ──────────────────────────────────────
    save_raw_numbers(results, car_mean, car_std,
                     best_lam, sweep_results)

    print("\n" + "="*60)
    print("  ALL DONE")
    print(f"  Best λ = {best_lam}")
    print(f"  Outputs in {RESULTS_DIR}/")
    print("  → raw_numbers.txt  : all aggregate numbers")
    print("  → figure0–4 .png   : figures")
    print("  → lambda_sweep.txt : lambda sweep summary")
    print("="*60)


if __name__ == "__main__":
    main()