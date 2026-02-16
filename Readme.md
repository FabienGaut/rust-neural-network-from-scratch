# Rust Neural Network from Scratch

A lightweight, dependency-minimal implementation of a Multi-Layer Perceptron (MLP) in Rust. This project demonstrates the fundamental mathematics of Deep Learning, including vector-based operations, activation functions, and the backpropagation algorithm.

## Features

- **Layer-based architecture** — layers as vectors, weights as flattened matrices for memory locality
- **Multiple activation functions** — Sigmoid, ReLU, Leaky ReLU, Tanh, Identity
- **Mini-batch gradient descent** — gradients accumulated over batches, averaged before update
- **CSV data loading** — generic loader with optional column skipping and header support
- **Min-Max normalization** — inputs and outputs scaled to [0, 1]
- **Train/Test split** — randomized dataset split to evaluate generalization

## Network Architecture

The default configuration uses synthetic data with 4 input features to predict 1 output:

```
Input (4) → Hidden (16, Leaky ReLU) → Hidden (10, Leaky ReLU) → Output (1, Identity)
```

```mermaid
graph LR
    subgraph Input Layer - 4 neurons
        x1["x1"]
        x2["x2"]
        x3["x3"]
        x4["x4"]
    end

    subgraph Hidden Layer 1 - 16 neurons
        h1["Leaky ReLU"]
        h2["Leaky ReLU"]
        h3["..."]
    end

    subgraph Hidden Layer 2 - 10 neurons
        h4["Leaky ReLU"]
        h5["Leaky ReLU"]
        h6["..."]
    end

    subgraph Output Layer - 1 neuron
        o1["Identity → Output"]
    end

    x1 & x2 & x3 & x4 --> h1 & h2 & h3
    h1 & h2 & h3 --> h4 & h5 & h6
    h4 & h5 & h6 --> o1
```

## How It Works

### Forward Propagation

For every neuron in a layer, we calculate the **Weighted Sum** (s) then pass it through an activation function:

```
s = sum(w_i * x_i) + b
a = activation(s)
```

The `forward` method iterates through each neuron, computes the dot product of weights and inputs, adds the bias, caches the result in `last_sums`, and returns activated outputs for the next layer.

### Backpropagation

The network learns using the **Chain Rule** to compute the gradient of the cost function with respect to every weight and bias.

```mermaid
graph TD
    subgraph Forward Pass
        direction TB
        A["Input (x)"] -->|"x"| B["Weighted Sum: s = W*x + b"]
        B -->|"s (cached)"| C["Activation: a = f(s)"]
        C -->|"a"| D["Output / Next Layer Input"]
    end

    subgraph Backward Pass
        direction BT
        E["Output Error: dC/da = a - y"] -->|"dC/da"| F["Local Delta: delta = dC/da * f'(s)"]
        F -->|"delta"| G["Weight Gradient: delta * x"]
        F -->|"delta"| H["Bias Gradient: delta"]
        F -->|"delta * W"| I["Error for Previous Layer"]
    end

    subgraph Update
        J["w_new = w_old - lr * avg(grad_w)"]
        K["b_new = b_old - lr * avg(grad_b)"]
    end

    D -.->|"compare with target"| E
    G --> J
    H --> K
```

**Step A — Output Error:** `Error = prediction - target`

**Step B — Local Delta:** `delta = dC/da * f'(s)`

**Step C — Parameter Gradients:**
- Bias gradient: `delta`
- Weight gradient: `delta * input`
- Error relay to previous layer: `sum(delta * w)`

### Mini-Batch Gradient Descent

Gradients are accumulated across all samples in a batch, then averaged before updating weights and biases:

```
w_new = w_old - learning_rate * (accumulated_gradient / batch_size)
```

### Data Pipeline

1. **CSV Loading** — Parses CSV with header skipping and optional first-column skip (for date fields)
2. **Normalization** — Min-Max scaling to [0, 1] for both inputs and outputs
3. **Shuffled Split** — Random 80/20 train/test split to avoid ordering bias

## Sample Output

```
Loaded 1500 samples from data/synthetic_data.csv
Training on 1200 samples, Testing on 300 samples...
Epoch 0 | Train MSE: 0.034394
Epoch 100 | Train MSE: 0.000844
...
Epoch 1000 | Train MSE: 0.000554

Final Test MSE: 0.000599

Sample Predictions:
Target: 0.2432 | Predicted: 0.2396
Target: 0.4294 | Predicted: 0.4523
Target: 0.9433 | Predicted: 0.9182
```

## How to Run

1. Ensure you have Rust and Cargo installed.
2. Place a CSV file (e.g. `synthetic_data.csv`) in the `data/` directory.
3. Run:

```bash
cargo run
```

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `rand` | 0.8 | Weight initialization & dataset shuffling |
| `csv` | 1.1 | CSV file parsing |

## Key Implementation Notes

- **Weight Initialization:** Random uniform values in [-1, 1]. Never initialize to zero — it prevents symmetry breaking.
- **Biases** start at 0, which is safe since weights already break symmetry.
- **Activation choice matters:** Leaky ReLU for hidden layers (fast, avoids vanishing gradients and dead neurons), Identity for output (unbounded, suitable for regression).
