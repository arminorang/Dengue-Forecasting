# gnn.py

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import GCNConv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Utility: encode categorical region codes as embeddings
# ---------------------------------------------------------
def build_region_embeddings(df, cols):
    """
    Takes columns like ['CD_UF', 'CD_RGI', ...] and returns:
      - embedding lookup tables
      - encoded integer arrays per node
    """
    encoders = {}
    encoded = {}

    for col in cols:
        uniques = sorted(df[col].unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        encoders[col] = mapping
        encoded[col] = df[col].map(mapping).values

    return encoders, encoded


# ---------------------------------------------------------
# Spatiotemporal dataset
# ---------------------------------------------------------
class DengueSpatioTemporalDataset(Dataset):
    """
    Builds sequences of T weeks to predict week T+1 incidence.
    Includes:
      - dynamic feature: p_inc100k
      - static features: AREA_KM2 + region embeddings
    """

    def __init__(
        self,
        df: pd.DataFrame,
        edge_index: torch.Tensor,
        T: int = 4,
        train: bool = True,
        train_frac: float = 0.8,
    ):
        self.T = T
        self.edge_index = edge_index

        # Sort by time
        df = df.sort_values(["year", "epiweek", "CD_MUN"]).copy()

        # Create time index
        df["time_id"] = df["year"].astype(int) * 100 + df["epiweek"].astype(int)
        time_ids = sorted(df["time_id"].unique())

        # Train/test split
        split_idx = int(len(time_ids) * train_frac)
        self.time_ids = time_ids[:split_idx] if train else time_ids[split_idx:]

        # Node ordering
        mun_ids = sorted(df["CD_MUN"].unique())
        self.mun_to_idx = {m: i for i, m in enumerate(mun_ids)}
        self.num_nodes = len(mun_ids)

        # Static features
        static_df = df.drop_duplicates("CD_MUN").sort_values("CD_MUN")

        region_cols = ["CD_UF", "CD_RGI", "CD_RGINT", "CD_REGIAO", "CD_CONCURB"]
        _, encoded_regions = build_region_embeddings(static_df, region_cols)

        static_feats = [
            static_df["AREA_KM2"].values[:, None].astype(np.float32)
        ] + [
            encoded_regions[col][:, None].astype(np.int64)
            for col in region_cols
        ]

        # Concatenate static features
        self.static_X = np.concatenate(static_feats, axis=1)  # [N, F_static]

        # Dynamic feature: p_inc100k
        self.time_to_dynamic = {}
        for t in time_ids:
            df_t = df[df["time_id"] == t]
            x = np.zeros(self.num_nodes, dtype=np.float32)
            for _, row in df_t.iterrows():
                idx = self.mun_to_idx[row["CD_MUN"]]
                x[idx] = row["p_inc100k"]
            self.time_to_dynamic[t] = x

        # Build samples
        self.samples = []
        for i in range(len(self.time_ids) - T):
            hist = self.time_ids[i : i + T]
            target = self.time_ids[i + T]
            self.samples.append((hist, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist_times, target_time = self.samples[idx]

        # Dynamic sequence: [T, N, 1]
        dyn_seq = []
        for t in hist_times:
            x = self.time_to_dynamic[t]
            dyn_seq.append(x[:, None])
        dyn_seq = np.stack(dyn_seq, axis=0)

        # Static features: [N, F_static]
        static = self.static_X

        # Target: [N]
        y = self.time_to_dynamic[target_time]

        return (
            torch.from_numpy(dyn_seq).float(),     # [T, N, 1]
            torch.from_numpy(static).float(),      # [N, F_static]
            torch.from_numpy(y).float(),           # [N]
            self.edge_index.long(),
        )


# ---------------------------------------------------------
# Spatiotemporal GNN model
# ---------------------------------------------------------
class SpatioTemporalGNN(nn.Module):
    """
    GCN applied to dynamic + static features, then GRU over time.
    """

    def __init__(
        self,
        dyn_in=1,
        static_in=6,   # AREA_KM2 + 5 region codes
        gcn_hidden=32,
        gcn_layers=2,
        rnn_hidden=64,
    ):
        super().__init__()

        self.input_dim = dyn_in + static_in

        # GCN stack
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(self.input_dim, gcn_hidden))
        for _ in range(gcn_layers - 1):
            self.gcn_layers.append(GCNConv(gcn_hidden, gcn_hidden))

        # GRU over time
        self.rnn = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=rnn_hidden,
            batch_first=True,
        )

        # Final prediction
        self.mlp = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden),
            nn.ReLU(),
            nn.Linear(rnn_hidden, 1),
        )

    def forward(self, dyn_seq, static, edge_index):
        """
        dyn_seq: [T, N, 1]
        static:  [N, F_static]
        edge_index: [2, E]
        """
        T, N, _ = dyn_seq.shape

        gcn_outputs = []

        for t in range(T):
            x_t = dyn_seq[t]  # [N, 1]
            x = torch.cat([x_t, static], dim=1)  # [N, dyn+static]

            h = x
            for gcn in self.gcn_layers:
                h = torch.relu(gcn(h, edge_index))
            gcn_outputs.append(h)

        # [T, N, H] → [N, T, H]
        H = torch.stack(gcn_outputs, dim=0).permute(1, 0, 2)

        out, _ = self.rnn(H)
        last = out[:, -1, :]  # [N, rnn_hidden]

        return self.mlp(last).squeeze(-1)  # [N]


# ---------------------------------------------------------
# Training utilities
# ---------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total = 0

    for dyn_seq, static, y, edge_index in loader:
        dyn_seq = dyn_seq.squeeze(0).to(DEVICE)
        static = static.to(DEVICE)
        y = y.squeeze(0).to(DEVICE)
        edge_index = edge_index.to(DEVICE)

        optimizer.zero_grad()
        y_hat = model(dyn_seq, static, edge_index)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total = 0

    for dyn_seq, static, y, edge_index in loader:
        dyn_seq = dyn_seq.squeeze(0).to(DEVICE)
        static = static.to(DEVICE)
        y = y.squeeze(0).to(DEVICE)
        edge_index = edge_index.to(DEVICE)

        y_hat = model(dyn_seq, static, edge_index)
        loss = loss_fn(y_hat, y)
        total += loss.item()

    return total / len(loader)