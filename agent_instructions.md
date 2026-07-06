# Coding Agent Instructions: Cora Dataset Benchmark Implementation

## Goal
Implement four different machine learning approaches to solve the Node Classification task on the Cora dataset. The target system is a laptop with a Ryzen 5500U (CPU execution) and 8GB of RAM (3200MHz). 

## Memory Constraints (CRITICAL)
Due to the strict 8GB RAM limit, do **NOT** load the dataset multiple times in a single process, and do **NOT** run multiple models in the same Python process. The overarching orchestrator script handles process isolation to avoid Out-Of-Memory (OOM) crashes. Your implementations must focus on single-run efficiency.

## Dependencies
* `torch`
* `torch_geometric`
* `scikit-learn`
* `networkx`

## Task 1: Dataset Loader
Create a shared data loading utility to download/load the Cora dataset via `torch_geometric.datasets.Planetoid`. Ensure it only downloads once and caches locally.

## Task 2: Implement the Four Models

### 1. Feature-Only: Multi-Layer Perceptron (MLP)
* **Input:** `data.x` (1433-dim bag-of-words features). Ignore `data.edge_index`.
* **Architecture:** 2 or 3 linear layers with ReLU activations and Dropout.
* **Loss:** Negative Log-Likelihood (`nll_loss`) or CrossEntropy.
* **Goal:** Establish the baseline of using only text features.

### 2. Topology-Only: Label Propagation Algorithm (LPA)
* **Input:** `data.edge_index` and `data.y[data.train_mask]`. Ignore `data.x`.
* **Implementation:** Use PyTorch Geometric's built-in `torch_geometric.nn.models.LabelPropagation`.
* **Hyperparameters:** `num_layers` (e.g., 50), `alpha` (e.g., 0.9).
* **Goal:** Establish the baseline of relying purely on graph homophily (connected nodes share labels).

### 3. Topology-Only: Node2Vec + Logistic Regression
* **Input:** `data.edge_index`. Ignore `data.x`.
* **Implementation:** 1. Train `torch_geometric.nn.models.Node2Vec` to generate 128-dim node embeddings using random walks.
  2. Extract the embeddings for the train/test nodes.
  3. Train a `sklearn.linear_model.LogisticRegression` classifier on the training embeddings.
* **Goal:** Prove that structural embeddings capture graph topology but fail to leverage text features.

### 4. Hybrid: Graph Convolutional Network (GCN)
* **Input:** `data.x` AND `data.edge_index`.
* **Architecture:** 2 `GCNConv` layers. Hidden dimension: 16.
* **Loss:** `nll_loss`.
* **Goal:** Demonstrate that combining features and structure yields the highest accuracy (~81.5%).

## Task 3: Output Formatting
Each model's run must calculate the accuracy strictly on `data.test_mask` and return/print it as a float so the orchestrator script can log it to the benchmark file.
