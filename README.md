# Continual Learning with Inter-Cluster Fitness (CAR)

Code and reproduction scripts for the paper **"Continual Learning for
Adaptive AI Systems"** (arXiv:2510.07648).

> **Note on scope.** This is a **preliminary study with a negative
> result.** We introduce Cluster-Aware Replay (CAR) — a small replay
> buffer combined with an Inter-Cluster Fitness (ICF) feature-space
> regularizer — and find that the ICF term *in its current unbounded
> form* does **not** improve over replay alone. The main contribution is
> the identification and analysis of a loss-collapse failure mode. See
> the paper for full discussion.

Paper: https://arxiv.org/abs/2510.07648

---

## What this is

Cluster-Aware Replay (CAR) combines:
- a small, class-balanced **replay buffer** (20 exemplars/class), and
- an **Inter-Cluster Fitness (ICF)** loss that pushes current-task
  features away from the stored centroids of previously seen classes.

The intent was that geometric separation between task representations
would reduce interference. The experiments show the opposite: the
unbounded ICF term overwhelms the cross-entropy loss in later training
epochs, driving the network to prioritize centroid repulsion over
classification, which collapses retention of earlier tasks.

---

## Key result (Split CIFAR-10, ResNet-18, 3 seeds)

| Configuration | Final Avg. Accuracy (%) |
|---|---|
| Fine-tuning (no replay, λ=0) | 19.0 ± 0.1 |
| Replay only (λ=0) | 23.1 ± 3.1 |
| ICF only (no replay, λ=0.1) | 19.2 ± 0.1 |
| **CAR (replay + ICF, λ=0.1)** | **22.5 ± 2.1** |

Full CAR (22.5%) does **not** beat replay alone (23.1%): the unbounded
ICF term adds noise, not signal. A λ sweep over {0.1, 1, 5, 10, 20, 50}
confirms accuracy only decreases as ICF weight increases.

**Takeaway:** unbounded inter-cluster repulsion is the wrong formulation;
bounded feature-space regularization is the needed fix (future work).

---

## Repository contents

- `Solving_Catastrophic_fogetting_for_continual_learning_using_custom_ResNET_and_ICF.ipynb`
  — main experiment notebook (CAR, ablations, λ sweep on Split CIFAR-10)
- `Preventing_Forgetting_Continual_Learning.ipynb` — earlier exploratory notebook
- `losses/` — ICF loss implementation
- `models/` — ResNet-18 backbone (CIFAR-adapted: 3×3 stride-1 conv1, no max-pool)
- `experiments/` — experiment / reproduction scripts
- `example_cluster_aware_cl.py`, `test_cluster_aware_cl.py`, `test_framework.py`
  — usage example and tests


## Reproducing the results

Requirements: Python 3.8+, PyTorch, torchvision, numpy, matplotlib.

```bash
git clone https://github.com/AminHasibul/Continual-Learning-UsingInterClusterDistance.git
cd Continual-Learning-UsingInterClusterDistance
pip install torch torchvision numpy matplotlib
```

Open the main notebook and run top to bottom; it downloads CIFAR-10,
runs the 5-task split, the ablation, and the λ sweep, reporting per-task
accuracy and final average accuracy over seeds {42, 123, 456}.

---

## Setup details (from the paper)

- **Dataset:** Split CIFAR-10, 5 tasks, 2 classes/task, fixed order
  [0,1],[2,3],[4,5],[6,7],[8,9]
- **Backbone:** ResNet-18; conv1 = 3×3 stride-1, max-pool
  removed (for 32×32 input)
- **Training:** Adam, lr=1e-3, 20 epochs/task, batch size 32
- **Buffer:** 20 exemplars/class
- **Seeds:** {42, 123, 456}; results reported as mean ± std

---

## Citation

```bibtex
@misc{amin2025car,
  title         = {Continual Learning for Adaptive AI Systems},
  author        = {Amin, Md Hasibul and Alam, Tamzid Tanvi},
  year          = {2025},
  eprint        = {2510.07648},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.07648}
}
```

## License

MIT — see [LICENSE](LICENSE).
