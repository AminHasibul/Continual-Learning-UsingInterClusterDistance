# Failure Modes of Always-On Inter-Cluster Repulsion in Replay-Based Continual Learning

Reproduction code for the paper:

> **Failure Modes of Always-On Inter-Cluster Repulsion in Replay-Based Continual Learning**
> Md Hasibul Amin, Tamzid Tanvi Alam
> arXiv:2510.07648

This is a **negative-results study**. We evaluate a preliminary Cluster-Aware
Replay (CAR) formulation that adds an always-active inter-cluster repulsion
term (ICF) to a small replay memory on five-task Split CIFAR-10, and show that
the repulsion term provides no measured benefit over replay alone in the
configuration studied. All numbers reported in the paper are produced by the
scripts in this repository.

## Results

Final average accuracy on Split CIFAR-10 (five tasks, ResNet-18 from scratch,
three seeds; mean ± population std of per-seed final scalars):

| Configuration | Accuracy (%) |
|---|---|
| Fine-tuning (no replay, λ = 0) | 19.0 ± 0.1 |
| Replay only (λ = 0) | 23.1 ± 2.5 |
| ICF only (no replay, λ = 0.1) | 19.2 ± 0.1 |
| Replay + ICF (λ = 0.1) | 22.5 ± 1.4 |

No tested repulsion weight (λ ∈ {0.1, 1, 5, 10, 20, 50}) exceeds the
replay-only baseline. Full per-task matrices and the λ sweep are in
[`results/raw_numbers.txt`](results/raw_numbers.txt), which matches the
paper's Tables I–III exactly.

## Repository contents

```
├── car_reproduce.py              # Full reproduction: λ sweep + ablation (all tables/figures)
├── car_debug.py                  # Verification: structural checks + checkpoint comparison
├── recompute_from_checkpoints.py # Regenerates aggregates from saved per-seed checkpoints
├── results/
│   ├── raw_numbers.txt           # All aggregate values (= paper Tables I–III)
│   └── lambda_sweep.txt          # λ sweep summary (= paper Table II)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Reproducing the results

```bash
pip install -r requirements.txt
python car_reproduce.py                # ~6–8 h on a single T4 GPU
python car_debug.py                    # structural verification (~1 h)
python recompute_from_checkpoints.py   # regenerate aggregates from checkpoints
```

`car_reproduce.py` checkpoints each (variant, seed) run as a `.npz` under
`car_results/checkpoints/`, so interrupted runs resume and all aggregates can
be recomputed without retraining.

## Method summary

- **Backbone:** ResNet-18 trained from scratch (3×3 stride-1 first conv, no
  initial max-pool), Adam (lr 1e-3), batch size 32, 20 epochs per task.
- **Protocol:** Split CIFAR-10, five tasks of two classes, class-incremental
  evaluation over a 10-way head, seeds {42, 123, 456}.
- **Replay:** 20 randomly selected exemplars per class.
- **ICF term:** after each task, a centroid per newly observed class is
  computed from the full current-task training subset of that class,
  L2-normalized, stored, and not refreshed thereafter. The loss maximizes the
  mean distance of all (current + replayed) batch features to previous-class
  centroids. See Sec. III of the paper for the exact objective and Sec. VI for
  the limitations of this design, including centroid staleness.
- **Statistics:** every "±" is the population standard deviation (ddof = 0)
  across the three per-seed final-average-accuracy scalars.

## Known limitations

This is a preliminary study: one dataset, one backbone, one class order, a
small memory, and a low absolute accuracy regime. The paper's Limitations
section states the scope of the claims precisely; nothing beyond that scope
should be inferred from this code.

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
