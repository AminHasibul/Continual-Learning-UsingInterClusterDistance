"""
car_debug.py
============
Verification script for car_reproduce.py.

Part A — structural checks (fresh single-seed run, seed 42, lambda 0.1):
  1. Replay buffer size and per-class counts after each task
  2. Current/replay/combined dataset sizes
  3. Centroid count and unit norms after each task
  4. CE vs weighted-ICF objective magnitudes per epoch
  5. Batch labels restricted to classes seen so far

Part B — numeric check against the saved checkpoint:
  Loads car_results/checkpoints/ablation_full_car_seed42.npz (written by
  car_reproduce.py) and prints its accuracy matrix next to this run's.

NOTE on stochasticity: this script iterates the DataLoader once for the
batch-label check before training, which advances the RNG stream. A fresh
run here is therefore a *separate stochastic rerun* and will NOT match the
checkpoint bit-for-bit; agreement within a few points per cell is expected.
The authoritative numbers are always the checkpoint / raw_numbers.txt.
The objective-component trace logged here corresponds to Fig. 2 of the
paper (also a separate instrumented rerun, as stated there).

Usage:
  python car_debug.py            # structural checks + checkpoint comparison
  python car_debug.py --no-train # checkpoint inspection only
"""

import argparse
import os
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

# ─────────────────────────────────────────────
# CONFIG — fixed for the debug run
# ─────────────────────────────────────────────
SEED                = 42
ICF_LAMBDA          = 0.1
NUM_TASKS           = 5
EPOCHS_PER_TASK     = 20
BATCH_SIZE          = 32
LR                  = 0.001
EXEMPLARS_PER_CLASS = 20
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT    = "./car_results/checkpoints/ablation_full_car_seed42.npz"

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
class ClassSubset(Dataset):
    def __init__(self, base_dataset, classes):
        self.base = base_dataset
        targets = np.array(base_dataset.targets)
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
# MODEL
# ─────────────────────────────────────────────
def build_model():
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                            stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(DEVICE)


def get_features(model, x):
    x = model.conv1(x); x = model.bn1(x)
    x = model.relu(x);  x = model.maxpool(x)
    x = model.layer1(x); x = model.layer2(x)
    x = model.layer3(x); x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, exemplars_per_class=20):
        self.n = exemplars_per_class
        self.data = {}

    def add(self, dataset, classes):
        targets = np.array(dataset.targets)
        for c in classes:
            idx = np.where(targets == c)[0]
            chosen = np.random.choice(
                idx, size=min(self.n, len(idx)), replace=False)
            self.data[c] = [(dataset[i][0], dataset[i][1]) for i in chosen]

    def as_dataset(self):
        items = []
        for v in self.data.values():
            items.extend(v)
        return items

    def __len__(self):
        return sum(len(v) for v in self.data.values())


# ─────────────────────────────────────────────
# ICF LOSS (Eq. 1–2 of the paper)
# ─────────────────────────────────────────────
class ICFLoss(nn.Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam
        self.centroids = {}

    @torch.no_grad()
    def update_centroids(self, model, dataset, classes):
        model.eval()
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
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
                mu = cat.mean(0)
                self.centroids[c] = F.normalize(mu, p=2, dim=0).to(DEVICE)
        model.train()

    def forward(self, features):
        if not self.centroids:
            return torch.tensor(0.0, device=features.device)
        f_norm = F.normalize(features, p=2, dim=1)
        loss = torch.tensor(0.0, device=features.device)
        for mu in self.centroids.values():
            dist = torch.norm(f_norm - mu.unsqueeze(0), dim=1)
            loss += -dist.mean()
        loss /= len(self.centroids)
        return self.lam * loss


# ─────────────────────────────────────────────
# EVAL
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, dataset, classes):
    subset = ClassSubset(dataset, classes)
    loader = DataLoader(subset, batch_size=256, shuffle=False)
    correct, total = 0, 0
    model.eval()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


def sep(title=""):
    print(f"\n{'─'*55}")
    if title:
        print(f"  {title}")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────
# CHECKPOINT LOADING
# ─────────────────────────────────────────────
def load_checkpoint_matrix():
    if not os.path.exists(CKPT):
        return None
    data = np.load(CKPT, allow_pickle=True)
    return data["acc_matrix"]


def print_matrix(m, label):
    print(f"\n  {label}")
    print("  After\\Task " + "".join(f"    T{j+1} " for j in range(NUM_TASKS)))
    for i in range(NUM_TASKS):
        row = f"  Task {i+1}   "
        for j in range(NUM_TASKS):
            row += ("    –  " if m[i][j] < 0
                    else f" {m[i][j]*100:5.1f} ")
        print(row)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(run_training=True):
    print("=" * 55)
    print("  CAR DEBUG VERIFICATION")
    print(f"  Seed={SEED}  λ={ICF_LAMBDA}  device={DEVICE}")
    print("=" * 55)

    ckpt_matrix = load_checkpoint_matrix()
    if ckpt_matrix is None:
        print(f"\n  NOTE: no checkpoint at {CKPT}")
        print("  Run car_reproduce.py first for the numeric comparison.")

    if not run_training:
        if ckpt_matrix is not None:
            print_matrix(ckpt_matrix, "Saved checkpoint (authoritative):")
        return

    set_seed(SEED)

    print("\nLoading CIFAR-10...")
    train_ds = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=TRANSFORM)
    test_ds = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=TRANSFORM)

    task_classes = [list(range(i * 2, i * 2 + 2)) for i in range(NUM_TASKS)]

    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    buffer = ReplayBuffer(EXEMPLARS_PER_CLASS)
    icf = ICFLoss(lam=ICF_LAMBDA)

    acc_matrix = np.full((NUM_TASKS, NUM_TASKS), -1.0)
    all_ok = True

    for t, classes in enumerate(task_classes):
        print(f"\n{'█'*55}")
        print(f"  TASK {t+1}/5   classes={classes}")
        print(f"{'█'*55}")

        current_ds = ClassSubset(train_ds, classes)
        if len(buffer) > 0:
            replay_ds = ListDataset(buffer.as_dataset())
            combined = ConcatDataset([current_ds, replay_ds])
        else:
            replay_ds = None
            combined = current_ds

        # CHECK 2 — dataset sizes
        sep("CHECK 2 — Dataset sizes")
        expected_replay = t * EXEMPLARS_PER_CLASS * 2
        ok = len(buffer) == expected_replay
        all_ok &= ok
        print(f"  Current={len(current_ds)}  "
              f"Replay={len(replay_ds) if replay_ds else 0} "
              f"(expected {expected_replay})  "
              f"Combined={len(combined)}  {'✓' if ok else '✗'}")

        loader = DataLoader(combined, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2, pin_memory=True)

        # CHECK 5 — batch labels (consumes one batch; see stochasticity note)
        sep("CHECK 5 — Batch labels (first batch)")
        for x_b, y_b in loader:
            batch_labels = sorted(set(y_b.tolist()))
            allowed = set(classes) | set(buffer.data.keys())
            ok = set(batch_labels).issubset(allowed)
            all_ok &= ok
            print(f"  Labels={batch_labels}  "
                  f"Allowed={sorted(allowed)}  {'✓' if ok else '✗'}")
            break

        # Training with CHECK 4 — objective magnitudes
        print(f"\n  Training {EPOCHS_PER_TASK} epochs...")
        for ep in range(EPOCHS_PER_TASK):
            model.train()
            ce_samples, icf_samples = [], []
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                feats = get_features(model, x)
                logits = model.fc(feats)
                ce_val = criterion(logits, y)
                icf_val = icf(feats)
                loss = ce_val + icf_val
                loss.backward()
                optimizer.step()
                ce_samples.append(ce_val.item())
                icf_samples.append(icf_val.item())
            if ep == 0 or (ep + 1) % 5 == 0:
                avg_ce, avg_icf = np.mean(ce_samples), np.mean(icf_samples)
                note = ("ICF exceeds CE (expected late-task; see paper §V-D)"
                        if abs(avg_icf) > abs(avg_ce) else "balanced")
                print(f"  Epoch {ep+1:>2}/{EPOCHS_PER_TASK}  "
                      f"CE={avg_ce:.4f}  ICF={avg_icf:.4f}  "
                      f"total={avg_ce+avg_icf:.4f}  [{note}]")

        # Centroid update (full current-task training subset; Eq. 1)
        icf.update_centroids(model, ClassSubset(train_ds, classes), classes)

        # CHECK 3 — centroid state
        sep("CHECK 3 — Centroid state after update")
        expected_cent = (t + 1) * 2
        norms = [icf.centroids[c].norm().item() for c in icf.centroids]
        ok = (len(icf.centroids) == expected_cent
              and max(abs(n - 1.0) for n in norms) < 1e-4)
        all_ok &= ok
        print(f"  Count={len(icf.centroids)} (expected {expected_cent})  "
              f"Norms in [{min(norms):.4f}, {max(norms):.4f}]  "
              f"{'✓' if ok else '✗'}")

        # Buffer update, then CHECK 1
        buffer.add(train_ds, classes)
        sep("CHECK 1 — Replay buffer state")
        expected_total = (t + 1) * EXEMPLARS_PER_CLASS * 2
        per_class_ok = all(len(v) == EXEMPLARS_PER_CLASS
                           for v in buffer.data.values())
        ok = len(buffer) == expected_total and per_class_ok
        all_ok &= ok
        print(f"  Classes={sorted(buffer.data.keys())}  "
              f"Total={len(buffer)} (expected {expected_total})  "
              f"{'✓' if ok else '✗'}")

        # Accuracy on all seen tasks
        sep("Accuracy after this task")
        for j in range(t + 1):
            acc = evaluate(model, test_ds, task_classes[j])
            acc_matrix[t][j] = acc
            print(f"  Task {j+1}: {acc*100:.1f}%")

    # Summary
    print("\n" + "=" * 55)
    print("  STRUCTURAL CHECKS: "
          + ("ALL PASSED ✓" if all_ok else "SOME FAILED ✗"))
    print("=" * 55)
    print_matrix(acc_matrix, "This debug run (separate stochastic rerun):")
    if ckpt_matrix is not None:
        print_matrix(ckpt_matrix, "Saved checkpoint (authoritative):")
        diffs = []
        for i in range(NUM_TASKS):
            for j in range(i + 1):
                diffs.append(abs(acc_matrix[i][j] - ckpt_matrix[i][j]) * 100)
        print(f"\n  Mean |difference| vs checkpoint: {np.mean(diffs):.1f} pp "
              f"(nonzero differences are expected; see stochasticity note "
              f"in the module docstring)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-train", action="store_true",
                   help="only inspect the saved checkpoint")
    args = p.parse_args()
    main(run_training=not args.no_train)
