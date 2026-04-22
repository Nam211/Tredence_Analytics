from __future__ import annotations

import math
import os
import random
import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


RANDOM_SEED = 42
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "results"
LAMBDA_VALUES = (1e-5, 5e-5, 2e-4)
NUM_EPOCHS = 10
SUBSET_SIZE = 5_000
GATE_THRESHOLD = 1e-2
INITIAL_GATE = 2.0


class GatedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), INITIAL_GATE)
        )
        self._init_params()

    def _init_params(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def compute_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.compute_gates(), self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class SparseNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dims: tuple[int, ...] = (512, 256),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        dim_seq = [input_dim, *hidden_dims, num_classes]
        self.layers = nn.ModuleList(
            GatedLinear(dim_seq[i], dim_seq[i + 1]) for i in range(len(dim_seq) - 1)
        )

    def _gated_layers(self):
        return [m for m in self.modules() if isinstance(m, GatedLinear)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def gate_penalty(self) -> torch.Tensor:
        return torch.stack(
            [layer.compute_gates().sum() for layer in self._gated_layers()]
        ).sum()

    def compute_sparsity(self, threshold: float = GATE_THRESHOLD) -> float:
        with torch.no_grad():
            flat = self.collect_gate_values()
            return 100.0 * (flat < threshold).float().mean().item()

    def collect_gate_values(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.cat(
                [layer.compute_gates().flatten() for layer in self._gated_layers()]
            )


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def balanced_sample_indices(labels: list[int], per_class: int, seed: int = RANDOM_SEED) -> list[int]:
    rng = random.Random(seed)
    class_map: dict[int, list[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        class_map[y].append(idx)
    chosen: list[int] = []
    for cls in sorted(class_map):
        pool = class_map[cls]
        rng.shuffle(pool)
        chosen.extend(pool[:per_class])
    return chosen


def build_dataloaders(
    train_subset_total: int = SUBSET_SIZE,
    batch_size: int = 128,
) -> Tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    full_train = datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=tfm)
    test = datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=tfm)

    per_class = train_subset_total // 10
    idx = balanced_sample_indices(full_train.targets, per_class=per_class)

    train_loader = DataLoader(Subset(full_train, idx), batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader


def run_evaluation(model: SparseNet, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
            total += yb.size(0)
    return 100.0 * correct / total


def partition_params(model: SparseNet):
    weights, gates = [], []
    for name, p in model.named_parameters():
        (gates if name.endswith("gate_scores") else weights).append(p)
    return weights, gates


def run_experiment(
    lam: float,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr_weights: float = 1e-3,
    lr_gates: float = 5e-2,
    grad_clip: float = 5.0,
    tag: str = "",
) -> dict:
    seed_everything(RANDOM_SEED)
    model = SparseNet().to(device)
    weight_params, gate_params = partition_params(model)
    optimizer = torch.optim.Adam([
        {"params": weight_params, "lr": lr_weights},
        {"params": gate_params, "lr": lr_gates},
    ])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"ce": 0.0, "sp": 0.0, "n": 0}
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            ce = loss_fn(logits, yb)
            sp = model.gate_penalty()
            loss = ce + lam * sp

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            bs = yb.size(0)
            totals["ce"] += ce.item() * bs
            totals["sp"] += sp.item() * bs
            totals["n"] += bs

        print(
            f"{tag}epoch {epoch}/{epochs}: "
            f"train_ce={totals['ce']/totals['n']:.4f} "
            f"sparsity_sum={totals['sp']/totals['n']:.1f} "
            f"sparsity_pct={model.compute_sparsity():.2f}%"
        )

    return {
        "lambda": lam,
        "test_accuracy": run_evaluation(model, test_loader, device),
        "sparsity_pct": model.compute_sparsity(),
        "gate_values_flat": model.collect_gate_values().cpu().numpy(),
    }


def render_gate_histograms(runs: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(13, 4), sharey=True)
    for ax, r in zip(axes, runs):
        ax.hist(
            r["gate_values_flat"],
            bins=60,
            range=(0.0, 1.0),
            color="#3366cc",
            edgecolor="black",
            linewidth=0.2,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Gate value (sigmoid of gate_scores)")
        ax.set_title(
            f"λ={r['lambda']:.0e}\n"
            f"sparsity={r['sparsity_pct']:.1f}%  "
            f"acc={r['test_accuracy']:.1f}%"
        )
        ax.axvline(GATE_THRESHOLD, color="red", linestyle="--", linewidth=1, label="sparsity threshold (1e-2)")
        ax.legend(loc="upper center", fontsize=8)
    axes[0].set_ylabel("Count (log scale)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_dataloaders(train_subset_total=SUBSET_SIZE, batch_size=128)
    print(f"Train samples: {len(train_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    runs: list[dict] = []
    for lam in LAMBDA_VALUES:
        print(f"\n{'='*60}\nTraining with λ = {lam}\n{'='*60}")
        runs.append(run_experiment(
            lam=lam, epochs=NUM_EPOCHS,
            train_loader=train_loader, test_loader=test_loader,
            device=device, tag=f"[λ={lam}] ",
        ))

    print("\n=== Results ===")
    print(f"{'Lambda':>10} | {'Test Acc (%)':>13} | {'Sparsity (%)':>13}")
    print("-" * 44)
    for r in runs:
        print(f"{r['lambda']:>10.0e} | {r['test_accuracy']:>13.2f} | {r['sparsity_pct']:>13.2f}")

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "config": {
                "train_subset_total": SUBSET_SIZE,
                "epochs": NUM_EPOCHS,
                "lambdas": list(LAMBDA_VALUES),
                "seed": RANDOM_SEED,
                "device": str(device),
            },
            "runs": [
                {"lambda": r["lambda"], "test_accuracy": r["test_accuracy"], "sparsity_pct": r["sparsity_pct"]}
                for r in runs
            ],
        }, f, indent=2)
    print(f"Wrote {metrics_path}")

    plot_path = OUTPUT_DIR / "gate_distribution.png"
    render_gate_histograms(runs, plot_path)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()