# Self-Pruning Neural Network — Tredence Case Study

A neural network that learns to **prune itself during training** using learnable gate parameters. Built on CIFAR-10 image classification with PyTorch.

## What This Does

Instead of pruning weights *after* training, this network associates each weight with a learnable "gate" (via sigmoid). An L1 sparsity penalty pushes gates toward 0 during training, effectively removing unimportant connections on the fly.

## Project Structure

```
self_pruning_nn/
├── train.py          # All code: PrunableLinear, Network, Training, Evaluation
├── report.md         # Written report with results and analysis
├── requirements.txt  # Python dependencies
├── plots/            # Gate distribution plots (generated after running)
└── results.txt       # Final accuracy & sparsity table (generated after running)
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (downloads CIFAR-10 automatically)
python train.py
```

Training runs 3 experiments (λ = 0.0001, 0.001, 0.01) × 15 epochs each.
Plots and results are saved automatically.

## Key Concepts

| Component | What it does |
|-----------|-------------|
| `PrunableLinear` | Custom linear layer with learnable gate per weight |
| `sigmoid(gate_score)` | Keeps gate values in (0,1) — differentiable |
| L1 Sparsity Loss | Pushes gates toward 0, pruning weak connections |
| λ (lambda) | Controls sparsity vs accuracy trade-off |

## Results Summary

See `report.md` for full analysis and plots.

## Tech Stack
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
