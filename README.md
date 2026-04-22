# Self-Pruning Neural Network using L0 Regularization

This project implements a **self-pruning neural network** that automatically removes unnecessary neurons during training using **L0 regularization with hard-concrete gates**.

The goal is to demonstrate how neural networks can learn **sparse representations** while maintaining predictive performance.

---

## Dataset

CIFAR-10

- 10 image classes
- 60,000 images
- 32x32 RGB images

Classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## Objective

Implement a neural network that:

1. Learns to classify CIFAR-10 images
2. Uses L0 regularization to prune neurons
3. Demonstrates the trade-off between **accuracy and sparsity**

---

## Model Architecture

CNN Backbone:

Conv(3→32) → ReLU → MaxPool  
Conv(32→64) → ReLU → MaxPool  
Conv(64→128) → ReLU → MaxPool  

Flatten → L0 Gate → Linear → Softmax

The **L0 gate learns which neurons to deactivate**, enabling automatic pruning.

---

## Results

| Lambda | Test Accuracy | Sparsity |
|------|------|------|
| 1e-05 | 43.70% | 64.95% |
| 5e-05 | 40.69% | 94.45% |
| 2e-04 | 33.01% | 99.18% |

### Observation

Increasing λ increases sparsity but reduces accuracy.

This demonstrates the **compression vs performance trade-off** in neural networks.

---

## Output Files
