# Failure Modes of Always-On Inter-Cluster Repulsion in Replay-Based Continual Learning

Code to reproduce the paper:

> **Failure Modes of Always-On Inter-Cluster Repulsion in Replay-Based Continual Learning**
> Md Hasibul Amin, Tamzid Tanvi Alam
> arXiv:2510.07648

---

## What is this project about?

When a neural network learns several tasks one after another, it tends to
**forget** what it learned earlier. This is called *catastrophic forgetting*,
and the field that studies how to prevent it is called **continual learning**.

A common and simple fix is **replay**: keep a small memory of a few old
examples and mix them back into training when learning a new task. This
reminds the network of the past so it forgets less.

This project tests one extra idea on top of replay. The idea is to also push
the features of new classes **away** from the features of old classes, so the
classes stay well separated in the network's internal space. We call this the
**Inter-Cluster Force (ICF)**, and the full method **Cluster-Aware Replay
(CAR)**. Intuitively, keeping classes further apart might make them easier to
tell apart and reduce forgetting.

**The honest result: it does not help.** This is a *negative-results study*.
We show that, in the setup we tested, always keeping this repulsion turned on
gives **no measurable improvement** over plain replay. The main value of this
repository is a careful, reproducible demonstration of that finding, so others
do not spend time on the same idea without knowing its limits.

---

## How the method works (in plain terms)

1. **Split the data into tasks.** We use CIFAR-10 (10 image classes) and split
   it into 5 tasks, each with 2 classes. The network sees one task at a time.
2. **Train with a small memory (replay).** For each class we keep 20 example
   images. When training a new task, those saved examples are mixed in so the
   network keeps practising the old classes.
3. **Add the repulsion term (ICF).** After finishing a task, we compute a
   "centre point" (centroid) for each class it just learned, based on the
   network's features. During later training, an extra term in the loss pushes
   every image's features **away** from these old centroids.
4. **Keep the repulsion always on.** The centroids are computed once and never
   updated afterwards. This "always-on" design is exactly the part we show is
   problematic — the centroids go stale and the extra term stops helping.

---

## Results

Final average accuracy on Split CIFAR-10 (5 tasks, ResNet-18 trained from
scratch, 3 random seeds). The "±" is the population standard deviation across
the three per-seed final scores.

| Configuration | Accuracy (%) |
|---|---|
| Fine-tuning (no replay, no repulsion) | 19.0 ± 0.1 |
| Replay only | 23.1 ± 2.5 |
| Repulsion only (no replay) | 19.2 ± 0.1 |
| Replay + Repulsion (full CAR) | 22.5 ± 1.4 |

Reading the table:

- **Replay clearly helps** (23.1% vs 19.0%).
- **Repulsion alone does nothing useful** (19.2% ≈ 19.0%).
- **Adding repulsion to replay does not help** (22.5% ≈ 23.1%, and if
  anything slightly lower).

We also swept the repulsion strength λ over {0.1, 1, 5, 10, 20, 50}. **No
value beats replay alone.** Full per-task tables and the λ sweep are in
[results/raw_numbers.txt](results/raw_numbers.txt) and
[results/lambda_sweep.txt](results/lambda_sweep.txt).

---

## What's in this repository

```
car_reproduce.py               Full reproduction: lambda sweep + ablation (all tables/figures)
car_debug.py                   Verification: structural checks + checkpoint comparison
recompute_from_checkpoints.py  Regenerates the aggregate numbers from saved checkpoints
results/
  raw_numbers.txt              All aggregate values (paper Tables I-III)
  lambda_sweep.txt             Lambda sweep summary (paper Table II)
requirements.txt
CITATION.cff
LICENSE
```

---

## How to reproduce the results

```bash
pip install -r requirements.txt

python car_reproduce.py                # main run (~6-8 h on a single T4 GPU)
python car_debug.py                    # sanity/structural checks (~1 h)
python recompute_from_checkpoints.py   # regenerate numbers from saved checkpoints
```

[car_reproduce.py](car_reproduce.py) saves a checkpoint (a `.npz` file) after
every (variant, seed) run under `car_results/checkpoints/`. If a run is
interrupted, just run the script again — it picks up where it left off and can
rebuild all aggregate numbers without retraining.

---

## Experimental setup (the details)

- **Backbone:** ResNet-18 trained from scratch, adapted for 32×32 images
  (first conv is 3×3 stride-1, and the initial max-pool is removed).
- **Optimiser:** Adam, learning rate 1e-3, batch size 32, 20 epochs per task.
- **Protocol:** Split CIFAR-10, 5 tasks of 2 classes each, evaluated in the
  class-incremental setting over a single 10-way classifier. Seeds {42, 123, 456}.
- **Replay memory:** 20 randomly chosen examples per class.
- **Repulsion (ICF) term:** after each task, one centroid per newly seen class
  is computed from that class's full current-task training data, normalised to
  unit length, stored, and **never refreshed**. The loss maximises the average
  distance from every (current + replayed) example's features to the stored
  old-class centroids.
- **Statistics:** every "±" is the population standard deviation (ddof = 0)
  across the three per-seed final-average-accuracy scores.

---

## Why the method fails (short version)

Because the centroids are frozen after their task, the network's features keep
changing during later training while the centroids do not. The old centroids
quickly become **stale** and stop describing where those classes actually live
in feature space. Pushing away from stale points adds noise rather than useful
structure, so the repulsion term ends up doing nothing helpful. Section VI of
the paper discusses this in detail.

---

## Known limitations

This is a deliberately small, focused study: one dataset, one backbone, one
class ordering, a small replay memory, and a low-accuracy regime. The claims
should be read only within that scope — this work shows that *this particular
always-on formulation* does not help, not that inter-cluster repulsion can
never help in any form.

---

## Citation

```bibtex
@misc{amin2026failuremodes,
  title  = {Failure Modes of Always-On Inter-Cluster Repulsion in
            Replay-Based Continual Learning},
  author = {Amin, Md Hasibul and Alam, Tamzid Tanvi},
  year   = {2026},
  eprint = {2510.07648},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

MIT — see [LICENSE](LICENSE).
