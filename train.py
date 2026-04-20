"""
Self-Pruning Neural Network on CIFAR-10
Tredence AI Engineering Internship - Case Study

To implement a neural network that learns to prune itself
during training using learnable gate parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable gate parameters.

    Each weight has a corresponding gate_score. After passing through
    sigmoid, it becomes a gate in (0, 1):
      - gate near 1  -> weight is active
      - gate near 0  -> weight is effectively pruned

    The L1 sparsity loss pushes gates toward 0 during training.

    KEY: gate_scores are initialized to +3.0 so sigmoid(3) = 0.95
    meaning all gates start nearly open. The optimizer then closes
    unimportant ones as training progresses.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), 3.0)
        )

        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)

        pruned_weights = self.weight * gates

        return x @ pruned_weights.t() + self.bias

    def get_gates(self):
        """Return current gate values detached from graph (for analysis)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)


class SelfPruningNet(nn.Module):
    """
    Simple feedforward network for CIFAR-10.
    Input: 32x32x3 = 3072 features  |  Output: 10 classes
    Uses PrunableLinear instead of nn.Linear throughout.
    """
    def __init__(self):
        super(SelfPruningNet, self).__init__()

        self.network = nn.Sequential(
            PrunableLinear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            PrunableLinear(128, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

    def get_all_gates(self):
        """Concatenate all gate tensors from every PrunableLinear layer."""
        gates = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates.append(m.get_gates().flatten())
        return torch.cat(gates)

    def get_prunable_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]


def sparsity_loss(model):
    """
    L1 norm of all gates = sum of sigmoid(gate_scores) across all layers.

    Why L1 encourages sparsity:
      The gradient of sum(gates) w.r.t. gate_score is constant and does
      not shrink as gate -> 0. This keeps pushing gates all the way to 0.
      L2 would have a diminishing gradient near 0 and leave small values alive.
    """
    total = 0.0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            total = total + gates.sum()
    return total


def get_data_loaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_epoch(model, loader, optimizer, lam, device, epoch):
    """One training epoch. Returns (train_accuracy, avg_total_loss)."""
    model.train()
    ce_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        cls_loss = ce_fn(outputs, targets)
        sp_loss  = sparsity_loss(model)
        loss     = cls_loss + lam * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += outputs.max(1)[1].eq(targets).sum().item()
        total      += targets.size(0)

    return 100.0 * correct / total, total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            correct += model(inputs).max(1)[1].eq(targets).sum().item()
            total   += targets.size(0)
    return 100.0 * correct / total


def compute_sparsity(model, threshold=0.1):
    """
    Sparsity = percentage of gates below threshold.

    We use 0.1 (not 0.01) because sigmoid never reaches exactly 0.
    A gate < 0.1 means the weight contributes less than 10% of its value
    and is effectively pruned.
    """
    all_gates = model.get_all_gates().cpu()
    pruned    = (all_gates < threshold).sum().item()
    sparsity  = 100.0 * pruned / all_gates.numel()
    return sparsity, all_gates.numpy()


def plot_gate_distribution(gate_values, lam, test_acc, sparsity, save_path):
    """
    Histogram of final gate values.
    Success = large spike near 0 (pruned) + cluster near 0.5-1.0 (active).
    """
    plt.figure(figsize=(8, 5))
    plt.hist(gate_values, bins=80, color='steelblue', edgecolor='none', alpha=0.85)
    plt.axvline(x=0.1, color='red', linestyle='--', linewidth=1.2,
                label='Prune threshold (0.1)')
    plt.xlabel('Gate Value', fontsize=12)
    plt.ylabel('Number of Weights', fontsize=12)
    plt.title(
        f'Gate Value Distribution  |  lambda = {lam}\n'
        f'Test Accuracy: {test_acc:.2f}%   |   Sparsity: {sparsity:.1f}%',
        fontsize=12
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {save_path}")


def run_experiment(lam, train_loader, test_loader, device, epochs=20):
    print(f"\n{'='*52}")
    print(f"  Training with lambda = {lam}")
    print(f"{'='*52}")

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        train_acc, avg_loss = train_epoch(
            model, train_loader, optimizer, lam, device, epoch)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            all_gates    = model.get_all_gates().cpu()
            pct_below_01 = (all_gates < 0.1).float().mean().item() * 100
            print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Gates<0.1: {pct_below_01:.1f}%")

    test_acc        = evaluate(model, test_loader, device)
    sparsity, gates = compute_sparsity(model, threshold=0.1)

    print(f"\n  Final Test Accuracy : {test_acc:.2f}%")
    print(f"  Final Sparsity Level: {sparsity:.1f}%  (gates < 0.1)")

    return test_acc, sparsity, gates, model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    train_loader, test_loader = get_data_loaders(batch_size=128)

    lambdas = [0.0001, 0.001, 0.01]
    results = []

    os.makedirs("plots", exist_ok=True)

    for lam in lambdas:
        test_acc, sparsity, gate_values, model = run_experiment(
            lam, train_loader, test_loader, device, epochs=20
        )
        results.append((lam, test_acc, sparsity))

        plot_path = f"plots/gate_distribution_lambda_{lam}.png"
        plot_gate_distribution(gate_values, lam, test_acc, sparsity, plot_path)

    print("\n\n" + "="*58)
    print(f"  {'Lambda':<10} {'Test Accuracy':>15} {'Sparsity (%)':>14}")
    print("="*58)
    for lam, acc, spar in results:
        print(f"  {lam:<10} {acc:>14.2f}%  {spar:>13.1f}%")
    print("="*58)

    with open("results.txt", "w") as f:
        f.write("Lambda,TestAccuracy,SparsityLevel\n")
        for lam, acc, spar in results:
            f.write(f"{lam},{acc:.2f},{spar:.1f}\n")

    print("\nDone!  Results -> results.txt  |  Plots -> plots/")


if __name__ == "__main__":
    main()