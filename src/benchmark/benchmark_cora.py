import argparse
import subprocess
import sys
import os
import time

# ==============================================================================
# BENCHMARK ORCHESTRATOR
# ==============================================================================
# This script is designed for low-RAM environments (8GB).
# Instead of running all models in one script (which leaves tensors in memory),
# this script calls itself via subprocess for each model.
# When the subprocess finishes, the OS entirely reclaims the RAM.
# ==============================================================================

def orchestrate_benchmark():
    models = ['MLP', 'LPA', 'Node2Vec', 'GCN']
    output_file = 'benchmark_results.txt'
    
    with open(output_file, 'w') as f:
        f.write("Cora Dataset Node Classification Benchmark\n")
        f.write("="*45 + "\n")
        f.write("System: AMD Ryzen 5500U | 8GB RAM\n")
        f.write("Execution Mode: Isolated Subprocesses\n")
        f.write("="*45 + "\n\n")

    print(f"Starting isolated benchmark sequence. Logs will be saved to {output_file}")

    for model in models:
        print(f"\n[{model}] Launching isolated process...")
        start_time = time.time()
        
        # Call this same file, but pass the --model argument
        result = subprocess.run(
            [sys.executable, __file__, '--model', model],
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # Parse the last line of standard output to get the accuracy
            output_lines = result.stdout.strip().split('\n')
            accuracy_line = output_lines[-1]
            
            print(f"[{model}] Finished in {elapsed:.1f}s. Result: {accuracy_line}")
            
            with open(output_file, 'a') as f:
                f.write(f"Model: {model.ljust(10)} | {accuracy_line} | Time: {elapsed:.1f}s\n")
        else:
            print(f"[{model}] FAILED.")
            print("Error log:", result.stderr)
            with open(output_file, 'a') as f:
                f.write(f"Model: {model.ljust(10)} | FAILED | Time: {elapsed:.1f}s\n")

    print(f"\nBenchmark complete. View results in {output_file}")

# ==============================================================================
# MODEL IMPLEMENTATION SKELETONS (For the Coding Agent to fill/verify)
# ==============================================================================

def load_data():
    from torch_geometric.datasets import Planetoid
    return Planetoid(root='./data/Cora', name='Cora')[0]

def run_mlp():
    import torch
    import torch.nn.functional as F
    data = load_data()
    
    class MLP(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
            super().__init__()
            self.lin1 = torch.nn.Linear(in_dim, hidden_dim)
            self.lin2 = torch.nn.Linear(hidden_dim, out_dim)
            self.dropout = dropout

        def forward(self, x):
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)

    model = MLP(data.num_node_features, 64, int(data.y.max()) + 1, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = 0.0
    best_test_acc = 0.0
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x)
            pred = out.argmax(dim=1)
            
            # val acc
            val_correct = pred[data.val_mask] == data.y[data.val_mask]
            val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
            
            # test acc
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                
    return best_test_acc

def run_lpa():
    import torch
    from torch_geometric.nn.models import LabelPropagation
    data = load_data()
    
    model = LabelPropagation(num_layers=50, alpha=0.9)
    out = model(data.y, data.edge_index, mask=data.train_mask)
    pred = out.argmax(dim=-1)
    
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

def run_node2vec():
    import torch
    import random
    import numpy as np
    from torch_geometric.utils import sort_edge_index
    from torch_geometric.index import index2ptr
    from torch_geometric.utils.num_nodes import maybe_num_nodes
    from sklearn.linear_model import LogisticRegression
    data = load_data()
    
    class CustomNode2Vec(torch.nn.Module):
        def __init__(self, edge_index, embedding_dim, walk_length, context_size, walks_per_node=1, num_negative_samples=1, num_nodes=None, sparse=False):
            super().__init__()
            self.num_nodes = maybe_num_nodes(edge_index, num_nodes)
            row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
            self.rowptr, self.col = index2ptr(row, self.num_nodes), col
            self.EPS = 1e-15
            assert walk_length >= context_size
            self.embedding_dim = embedding_dim
            self.walk_length = walk_length - 1
            self.context_size = context_size
            self.walks_per_node = walks_per_node
            self.num_negative_samples = num_negative_samples
            self.embedding = torch.nn.Embedding(self.num_nodes, embedding_dim, sparse=sparse)
            self.reset_parameters()

        def reset_parameters(self):
            self.embedding.reset_parameters()

        def forward(self, batch=None):
            emb = self.embedding.weight
            return emb if batch is None else emb[batch]

        def loader(self, **kwargs):
            from torch.utils.data import DataLoader
            return DataLoader(range(self.num_nodes), collate_fn=self.sample, **kwargs)

        def pos_sample(self, batch):
            batch = batch.repeat(self.walks_per_node)
            rowptr_np = self.rowptr.numpy()
            col_np = self.col.numpy()
            start_nodes_np = batch.numpy()
            num_walks = len(start_nodes_np)
            walks = np.zeros((num_walks, self.walk_length + 1), dtype=np.int64)
            walks[:, 0] = start_nodes_np
            for i in range(num_walks):
                curr = start_nodes_np[i]
                for step in range(1, self.walk_length + 1):
                    start_idx = rowptr_np[curr]
                    end_idx = rowptr_np[curr + 1]
                    if start_idx == end_idx:
                        pass
                    else:
                        curr = col_np[random.randint(start_idx, end_idx - 1)]
                    walks[i, step] = curr
            rw = torch.from_numpy(walks).to(batch.device)
            walks_list = []
            num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
            for j in range(num_walks_per_rw):
                walks_list.append(rw[:, j:j + self.context_size])
            return torch.cat(walks_list, dim=0)

        def neg_sample(self, batch):
            batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
            rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length), dtype=batch.dtype, device=batch.device)
            rw = torch.cat([batch.view(-1, 1), rw], dim=-1)
            walks_list = []
            num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
            for j in range(num_walks_per_rw):
                walks_list.append(rw[:, j:j + self.context_size])
            return torch.cat(walks_list, dim=0)

        def sample(self, batch):
            if not isinstance(batch, torch.Tensor):
                batch = torch.tensor(batch)
            return self.pos_sample(batch), self.neg_sample(batch)

        def loss(self, pos_rw, neg_rw):
            start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
            h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
            h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1, self.embedding_dim)
            out = (h_start * h_rest).sum(dim=-1).view(-1)
            pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()
            start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
            h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
            h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1, self.embedding_dim)
            out = (h_start * h_rest).sum(dim=-1).view(-1)
            neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()
            return pos_loss + neg_loss

    model = CustomNode2Vec(data.edge_index, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10, num_nodes=data.num_nodes, sparse=True)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    for epoch in range(1, 11): # 10 epochs
        model.train()
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model()
        train_z, train_y = z[data.train_mask], data.y[data.train_mask]
        test_z, test_y = z[data.test_mask], data.y[data.test_mask]
        
        clf = LogisticRegression(solver='lbfgs', max_iter=500).fit(train_z.cpu().numpy(), train_y.cpu().numpy())
        acc = clf.score(test_z.cpu().numpy(), test_y.cpu().numpy())
    return acc

def run_gcn():
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    data = load_data()
    
    class SimpleGCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    model = SimpleGCN(data.num_node_features, 16, int(data.y.max()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            # val acc
            val_correct = pred[data.val_mask] == data.y[data.val_mask]
            val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
            
            # test acc
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

    return best_test_acc

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['MLP', 'LPA', 'Node2Vec', 'GCN'], help="Run specific model")
    args = parser.parse_args()

    if not args.model:
        # No model specified -> Run the orchestrator
        orchestrate_benchmark()
    else:
        # Run the specific model requested by the subprocess
        accuracy = 0.0
        try:
            if args.model == 'MLP':
                accuracy = run_mlp()
            elif args.model == 'LPA':
                accuracy = run_lpa()
            elif args.model == 'Node2Vec':
                accuracy = run_node2vec()
            elif args.model == 'GCN':
                accuracy = run_gcn()
                
            # Print strictly the accuracy at the end so the orchestrator can parse it
            print(f"Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Error running {args.model}: {str(e)}", file=sys.stderr)
            sys.exit(1)

