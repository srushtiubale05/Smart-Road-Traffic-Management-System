"""
rf_gwn_pipeline.py

Practical RF -> Graph + Temporal model pipeline inspired by RF-GWN paper.

Assumptions:
- Input CSV has columns: DateTime, Junction, Vehicles
- DateTime is parseable by pandas.to_datetime
- One junction per row per timestamp (if missing timestamps, resample/forward fill)

Outputs:
- adjacency matrix saved as 'rf_adjacency.npy'
- trained random forest models saved as 'rf_models/' (one per node)
- trained PyTorch model saved as 'gwn_st_model.pt'
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm

# ---------------------------
# 0. Config
# ---------------------------
DATA_CSV = "data/traffic.csv"   # your uploaded file path
LAGS = 6                        # number of lag hours to use per node
RF_N_ESTIMATORS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RF_MODELS_DIR = "rf_models"
os.makedirs(RF_MODELS_DIR, exist_ok=True)

# ---------------------------
# 1. Load and pivot dataset
# ---------------------------
df = pd.read_csv(DATA_CSV)
df['DateTime'] = pd.to_datetime(df['DateTime'])
# ensure sorted
df = df.sort_values(['DateTime', 'Junction']).reset_index(drop=True)

# pivot so columns = junctions, rows = time index
pivot = df.pivot(index='DateTime', columns='Junction', values='Vehicles')
# forward/backfill missing timestamps if any
pivot = pivot.resample('H').mean()            # assuming hourly; change as needed
pivot = pivot.ffill().bfill()
print("Pivot shape (time x nodes):", pivot.shape)

junctions = pivot.columns.tolist()
n_nodes = len(junctions)
print(f"Detected {n_nodes} junctions.")

# ---------------------------
# 2. Build supervised dataset using lag features
# Each sample will include lag features for all nodes:
# For time t we use t-1..t-LAGS as features and predict t for each node
# ---------------------------
def build_supervised_matrix(series_df, lags):
    # series_df: DataFrame time x nodes
    X_list = []
    y_list = []
    time_idx = []
    arr = series_df.values  # shape T x N
    T, N = arr.shape
    for t in range(lags, T):
        # features: flatten of (lags x N) -> length lags*N
        window = arr[t-lags:t, :]  # shape lags x N
        X_list.append(window.flatten())
        y_list.append(arr[t, :])   # predict all N nodes at time t
        time_idx.append(series_df.index[t])
    X = np.stack(X_list)  # (samples, lags*N)
    y = np.stack(y_list)  # (samples, N)
    return X, y, time_idx

X_all, y_all, times = build_supervised_matrix(pivot, LAGS)
print("Supervised dataset shapes:", X_all.shape, y_all.shape)

# ---------------------------
# 3. Random Forest per-node to compute Variable Importance Matrix
# For each target node j, train RF to predict node j using ALL features (lags for all nodes).
# Then extract feature importances, sum importances per-source-node across the lag block to get node->node importance
# ---------------------------
node_importance_matrix = np.zeros((n_nodes, n_nodes))  # source_node x target_node

print("\nTraining Random Forests per node to compute VIM-based adjacency...")

for target_idx in tqdm(range(n_nodes), desc="RF per-node"):
    y_target = y_all[:, target_idx]
    # For quickness, use a single RF and small subsample; you can tune later
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=42)
    rf.fit(X_all, y_target)
    # save model
    joblib.dump(rf, os.path.join(RF_MODELS_DIR, f"rf_target_{target_idx}.pkl"))
    fi = rf.feature_importances_  # length = lags * n_nodes
    # sum importances for each source node across lag positions
    fi_nodes = fi.reshape(LAGS, n_nodes).sum(axis=0)  # length n_nodes
    node_importance_matrix[:, target_idx] = fi_nodes

# Normalize the matrix (column-wise or row-wise). Here we normalize columns (influence to each target)
col_sums = node_importance_matrix.sum(axis=0, keepdims=True)
# avoid division by zero
col_sums[col_sums == 0] = 1.0
adj = node_importance_matrix / col_sums
print("Adjacency (normalized) shape:", adj.shape)

# optionally symmetrize if desired
adj_sym = (adj + adj.T) / 2.0

# save adjacency
np.save("rf_adjacency.npy", adj_sym)
print("Saved RF-derived adjacency to rf_adjacency.npy")

# ---------------------------
# 4. Prepare temporal graph dataset for PyTorch Geometric (simple splitting)
# We'll create input sequences of length 'in_len' to predict out_len future steps for all nodes.
# ---------------------------
IN_LEN = LAGS
OUT_LEN = 1

# Using X_all & y_all: X_all[t] corresponds to features for predicting y_all[t]
# But for the GNN we want sequential windows: we prefer to build sequences of shape (samples, in_len, n_nodes)
T_effective = X_all.shape[0]
sequences = []
targets = []
for i in range(T_effective - IN_LEN + 1):
    seq_window = X_all[i:i+IN_LEN, :].reshape(IN_LEN, n_nodes)  # careful: X_all rows already are flattened windows for times t
    # simpler approach: directly use pivot raw series to build windows
# Let's rebuild sequences directly from pivot values (safer)

arr = pivot.values  # T x N (original T)
T_full = arr.shape[0]
seqs = []
tars = []
for t in range(LAGS, T_full - OUT_LEN + 1):
    seqs.append(arr[t-LAGS:t, :])  # LAGS x N
    tars.append(arr[t:t+OUT_LEN, :])  # OUT_LEN x N
seqs = np.stack(seqs)   # samples x LAGS x N
tars = np.stack(tars)   # samples x OUT_LEN x N
print("Seqs/tars shapes:", seqs.shape, tars.shape)

# train/test split (time-ordered)
train_ratio = 0.8
train_samples = int(len(seqs) * train_ratio)
X_train = seqs[:train_samples]  # (S, LAGS, N)
y_train = tars[:train_samples]   # (S, OUT_LEN, N)
X_val = seqs[train_samples:train_samples+int(len(seqs)*0.1)]
y_val = tars[train_samples:train_samples+int(len(seqs)*0.1)]
X_test = seqs[train_samples+int(len(seqs)*0.1):]
y_test = tars[train_samples+int(len(seqs)*0.1):]

print("Train/Val/Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

# ---------------------------
# 5. PyTorch model: Spatial (GCN) + Temporal (1D conv / TCN)
# Minimal implementation: apply GCN to every timestep, then a temporal Conv1d across time steps for each node's features.
# ---------------------------
class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # single GCN layer per time step
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, x_time, edge_index, edge_weight):
        # x_time: (batch, N, in_channels)
        batch, N, C = x_time.shape
        out = []
        for b in range(batch):
            xb = x_time[b]  # N x C
            # torch_geometric expects x as [N, C] and edge_index e.g. 2 x E
            xb = self.gcn(xb, edge_index, edge_weight)
            out.append(xb)
        out = torch.stack(out, dim=0)  # batch x N x out_channels
        return out

class STModel(nn.Module):
    def __init__(self, node_count, in_channels=1, gcn_hidden=32, tcn_hidden=32, out_len=1):
        super().__init__()
        self.node_count = node_count
        self.in_channels = in_channels
        self.spatial = SpatialBlock(in_channels, gcn_hidden)
        # after spatial: batch x N x gcn_hidden
        # temporal conv: treat each node independently, conv over sequence length
        self.temporal = nn.Conv1d(in_channels=gcn_hidden, out_channels=tcn_hidden, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.readout = nn.Linear(tcn_hidden, out_len)  # map per-node features to OUT_LEN values

    def forward(self, x_seq, edge_index, edge_weight):
        # x_seq: batch x T x N  (raw values); we reshape to batch x N x in_channels and run GCN for each timestep
        batch, T, N = x_seq.shape
        # apply spatial GCN on each time step
        x_t_list = []
        for t in range(T):
            xt = x_seq[:, t, :].unsqueeze(-1)  # batch x N x 1
            out_sp = self.spatial(xt, edge_index, edge_weight)  # batch x N x gcn_hidden
            x_t_list.append(out_sp)
        # stack time dimension: batch x T x N x gcn_hidden
        x_time = torch.stack(x_t_list, dim=1)
        # permute to: batch*N x gcn_hidden x T for conv1d
        bNT = x_time.permute(0,2,3,1).contiguous()  # batch x N x gcn_hidden x T
        b, n, c, t = bNT.shape
        bNT = bNT.view(b*n, c, t)
        temp = self.temporal(bNT)  # (b*n, tcn_hidden, T)
        temp = self.relu(temp)
        # global pooling along time to produce per-node feature
        pooled = temp.mean(dim=-1)  # (b*n, tcn_hidden)
        out = self.readout(pooled)  # (b*n, out_len)
        out = out.view(b, n, -1)  # batch x N x out_len
        # return shape: batch x out_len x N (match y format)
        out = out.permute(0,2,1)
        return out

# ---------------------------
# 6. Build graph edge_index + edge_weight from adj_sym
# PyG expects edge_index (2 x E) and edge_weight (E)
# We'll include edges where weight > small threshold
# ---------------------------
adj_threshold = 1e-4
src, dst, w = [], [], []
for i in range(n_nodes):
    for j in range(n_nodes):
        val = adj_sym[i, j]
        if val > adj_threshold:
            src.append(i); dst.append(j); w.append(float(val))
# convert to tensors
edge_index = torch.tensor([src, dst], dtype=torch.long).to(DEVICE)
edge_weight = torch.tensor(w, dtype=torch.float).to(DEVICE)
print("Graph edges:", edge_index.shape[1])

# ---------------------------
# 7. Training utilities
# ---------------------------
def batches_from_numpy(X, y, batch_size=32):
    idxs = np.arange(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = idxs[start:start+batch_size]
        yield torch.tensor(X[batch_idx], dtype=torch.float32, device=DEVICE), torch.tensor(y[batch_idx], dtype=torch.float32, device=DEVICE)

# Instantiate model
model = STModel(node_count=n_nodes, in_channels=1, gcn_hidden=64, tcn_hidden=64, out_len=OUT_LEN).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ---------------------------
# 8. Training loop
# ---------------------------
EPOCHS = 20
BATCH_SIZE = 64

# Convert numpy arrays to training shape: model expects (batch, T, N)
# Currently X_train is (S, LAGS, N) OK. y_train is (S, OUT_LEN, N)
Xtr = X_train.astype(np.float32)
Ytr = y_train.astype(np.float32)
Xval = X_val.astype(np.float32)
Yval = y_val.astype(np.float32)
Xtest = X_test.astype(np.float32)
Ytest = y_test.astype(np.float32)

best_val_loss = np.inf
for epoch in range(1, EPOCHS+1):
    model.train()
    train_losses = []
    for xb, yb in batches_from_numpy(Xtr, Ytr, BATCH_SIZE):
        # xb: batch x T x N
        pred = model(xb, edge_index, edge_weight)  # batch x out_len x N
        loss = criterion(pred, yb.permute(0,2,1))   # yb -> batch x out_len x N
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    # validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in batches_from_numpy(Xval, Yval, BATCH_SIZE):
            pred = model(xb, edge_index, edge_weight)
            loss = criterion(pred, yb.permute(0,2,1))
            val_losses.append(loss.item())
    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses) if val_losses else 0.0
    print(f"Epoch {epoch}/{EPOCHS}  TrainLoss={avg_train:.6f}  ValLoss={avg_val:.6f}")
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "gwn_st_model.pt")
        print("Saved best model.")

# ---------------------------
# 9. Evaluate on test set
# ---------------------------
model.load_state_dict(torch.load("gwn_st_model.pt", map_location=DEVICE))
model.eval()
preds = []
gts = []
with torch.no_grad():
    for xb, yb in batches_from_numpy(Xtest, Ytest, BATCH_SIZE):
        pred = model(xb, edge_index, edge_weight)  # batch x out_len x N
        preds.append(pred.cpu().numpy())
        gts.append(yb.cpu().numpy())
preds = np.concatenate(preds, axis=0)  # samples x out_len x N
gts = np.concatenate(gts, axis=0)

# flatten to compare (samples * out_len * N)
pred_flat = preds.reshape(-1, n_nodes)
gt_flat = gts.reshape(-1, n_nodes)
mae = np.mean(np.abs(pred_flat - gt_flat))
rmse = np.sqrt(np.mean((pred_flat - gt_flat)**2))
print(f"Test MAE: {mae:.4f}  RMSE: {rmse:.4f}")

# Save adjacency and summary
np.save("rf_adjacency.npy", adj_sym)
print("Pipeline finished. Saved models and adjacency.")
