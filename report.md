# Self-Pruning Neural Network — Experiment Report

AI Engineer Case Study Implementation

---

# 1. Overview

This project demonstrates a **self-pruning neural network** that learns to automatically deactivate unnecessary weights during training.

Instead of manually pruning parameters after training, the model learns **which connections are important** through a differentiable gating mechanism.

The experiment is performed on the **CIFAR-10** using the **PyTorch**.

The goal of the study is to analyze the **trade-off between predictive accuracy and network sparsity**.

---

# 2. Model Architecture

The implemented model is a **fully-connected feed-forward neural network** where each linear layer is replaced with a custom **PrunableLinear** module.

Each weight in the network is associated with a learnable gate that controls whether the connection remains active.

### Network Structure

```
Input: CIFAR-10 image (3 × 32 × 32)

Flatten → 3072 features

PrunableLinear (3072 → 512)
ReLU

PrunableLinear (512 → 256)
ReLU

PrunableLinear (256 → 10)

Output: Class logits
```

Total trainable gated parameters:

```
≈ 1.7 million weights
```

No convolutional layers, dropout, or batch normalization were used.
The objective was to **clearly demonstrate the pruning mechanism**, not maximize CIFAR-10 performance.

---

# 3. Prunable Linear Layer

Each `PrunableLinear` layer contains three learnable tensors:

```
weight        → normal linear weights
gate_scores   → controls whether a weight is active
bias          → output bias
```

The forward pass is computed as:

```
output = F.linear(x, weight * sigmoid(gate_scores), bias)
```

The sigmoid function converts the gate score into a value between **0 and 1**.

Meaning:

| Gate Value | Interpretation               |
| ---------- | ---------------------------- |
| close to 1 | weight is active             |
| close to 0 | weight is effectively pruned |

Because the forward pass uses differentiable operations, gradients automatically flow through both **weights and gate parameters**.

---

# 4. Sparsity Regularization

To encourage pruning, an **L1 penalty** is applied to the gate values.

Total loss:

```
Loss = CrossEntropy + λ * Σ sigmoid(gate_scores)
```

Where:

* CrossEntropy measures classification error
* λ controls pruning strength
* The second term penalizes active gates

### Why L1 Encourages Sparsity

L1 regularization applies a constant penalty to every gate value.

If a gate contributes little to classification performance, the optimizer reduces its value toward zero.

Once a gate becomes very small:

```
sigmoid(score) ≈ 0
```

The corresponding weight effectively stops contributing to the network output.

This produces a **bimodal distribution of gates**:

* one cluster near **0 (pruned connections)**
* one cluster near **1 (active connections)**

---

# 5. Experimental Setup

Dataset: CIFAR-10
Framework: PyTorch

Training configuration:

```
Training samples: 5000
Test samples: 10000
Batch size: 128
Epochs per experiment: 10
Optimizer: Adam
Random seed: 42
```

Three different regularization strengths were tested.

---

# 6. Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
| ---------- | ----------------- | ------------ |
| 1 × 10⁻⁵   | 43.7              | 64.9         |
| 5 × 10⁻⁵   | 40.7              | 94.4         |
| 2 × 10⁻⁴   | 33.0              | 99.1         |

Sparsity is defined as:

```
percentage of gates with value < 0.01
```

Higher sparsity means more connections are removed.

---

# 7. Gate Distribution

The gate distribution plot illustrates how the model separates weights into two groups.

```
Active weights  → gate ≈ 1
Pruned weights  → gate ≈ 0
```

As λ increases:

* the number of pruned weights grows
* the model becomes smaller
* accuracy gradually decreases

This confirms the expected **sparsity vs performance trade-off**.

---

# 8. Discussion

The regularization parameter λ acts as a **direct control knob for model compression**.

### Small λ (1e-5)

Weak pruning pressure.

The model keeps many connections active and achieves the best accuracy.

### Medium λ (5e-5)

Large fraction of weights are removed while performance remains relatively stable.

This region provides a **good balance between sparsity and accuracy**.

### Large λ (2e-4)

Very strong pruning.

Almost all weights are removed, significantly reducing model capacity.

Accuracy drops noticeably but still remains above random guessing.

---

# 9. Implementation Notes

Several practical considerations were important during implementation.

### Gradient Flow

The design relies entirely on automatic differentiation.

Because the gating function uses standard operations, **no custom backward function is required**.

### Optimizer Behaviour

Using separate learning rates for gates and weights improves pruning behaviour.

```
weights lr → 1e-3
gates lr → 5e-2
```

This allows gates to move quickly toward zero while weights learn normally.

### Numerical Stability

Gradient clipping was applied to prevent rare gradient spikes during training.

```
clip_grad_norm_(model.parameters(), 5.0)
```

This had no visible effect on final accuracy but improved training stability.

---

# 10. Limitations

Some constraints of the current experiment include:

**Limited training data**

Only 5000 CIFAR-10 samples were used to ensure the experiment runs quickly.

**Short training time**

Each configuration was trained for only 10 epochs.

**Simple architecture**

A fully-connected model is not ideal for image classification.

The architecture was intentionally simplified to focus on pruning behaviour.

**Approximate sparsity**

Sigmoid gates approach zero but do not become exactly zero.
A small threshold is used to measure effective pruning.

---

# 11. Potential Improvements

Future improvements could significantly increase performance.

### Use Convolutional Features

Adding convolutional layers before the prunable MLP would improve image feature extraction.

### Data Augmentation

Applying standard augmentations such as random cropping and flipping would improve generalization.

### Longer Training

Training for more epochs would allow the network to reach a higher accuracy baseline.

---

# 12. Reproducibility

Clone the repository:

```
git clone <(https://github.com/Nam211/Tredence_Analytics.git)>
cd project
```

Create environment:

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run experiment:

```
python self_pruning_nn.py
```

This generates:

```
results/
    metrics.json
    gate_distribution.png
```

Run tests:

```
pytest tests/ -v
```

All tests should pass successfully.

---

# 13. Conclusion

This project demonstrates how **learnable gating mechanisms** can be used to create **self-pruning neural networks**.

By introducing a sparsity penalty on gate activations, the model learns which connections are necessary for prediction and which can be safely removed.

The experiments confirm the expected behaviour:

* increasing λ increases sparsity
* higher sparsity reduces accuracy
* a middle region provides a good compression-performance trade-off

Self-pruning methods are promising for building **efficient neural networks suitable for real-world deployment**.
