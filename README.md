# Continual Learning with Inter-Cluster Fitness (ICF)

Code for **"Continual Learning for Adaptive AI Systems"** (Cluster-Aware
Replay / CAR with an Inter-Cluster Fitness regularizer), a preliminary study
on mitigating catastrophic forgetting via feature-space cluster separation.

Paper (arXiv): https://arxiv.org/abs/2510.07648

> Status: preliminary research code. Results here are single-run and
> exploratory; treat them as such.

---

## Idea

Cluster-Aware Replay (CAR) combines a small class-balanced replay buffer with
an **Inter-Cluster Fitness (ICF)** loss that encourages feature-space
separation between the current task's representations and the centroids of
previously seen classes, to reduce inter-task interference.

[FILL IN: 2-3 sentences describing, in your own words, exactly what the
notebooks implement — the ICF loss form, the ResNet backbone you use, the
replay setup. Only state what the code actually does.]

---

## Repository contents

- `Preventing_Forgetting_Continual_Learning.ipynb` — [FILL IN: what this notebook does]
- `Solving_Catastrophic_fogetting_..._ResNET_and_ICF.ipynb` — [FILL IN]
- `losses/` — [FILL IN: ICF loss implementation]
- `models/` — [FILL IN: backbone]
- `experiments/` — [FILL IN]
- `example_cluster_aware_cl.py`, `test_*.py` — [FILL IN]

[Remove any bullet above that doesn't correspond to a real file.]

---

## Setup

```bash
git clone https://github.com/AminHasibul/Continual-Learning-UsingInterClusterDistance.git
cd Continual-Learning-UsingInterClusterDistance
pip install -r requirements.txt   # [create this file listing ONLY what you import: torch, torchvision, numpy, matplotlib, ...]
```

Then open the notebook(s) in Jupyter / Colab and run top to bottom.
[Adjust to match how the code is ACTUALLY run. Remove any commands that
reference files that don't exist.]

---

## Experiments

- Benchmark: Split-CIFAR-10 (5 tasks, 2 classes each)
- Backbone: ResNet-18 [state your exact variant: CIFAR-adapted stem, etc.]
- [FILL IN: optimizer, epochs/task, buffer size — match the paper's setup]

---

## Results

[ONLY include real numbers your code produces. If your notebook reproduces
the paper's Table I, you can show it here, clearly labeled single-run.
DELETE this section entirely rather than show numbers you didn't measure.]

Split-CIFAR-10, single run, average accuracy after all tasks: [your real number]

> Single-run preliminary result. See the paper for details.

---

## Limitations

This is preliminary, single-seed, CIFAR-10-only exploratory work. The ICF
formulation here is an early version; see the paper for discussion.

---

## Citation

```bibtex
@misc{amin2025car,
  title  = {Continual Learning for Adaptive AI Systems},
  author = {Amin, Md Hasibul and Alam, Tamzid Tanvi},
  year   = {2025},
  eprint = {2510.07648},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url    = {https://arxiv.org/abs/2510.07648}
}
```

## License

MIT — see [LICENSE](LICENSE).
