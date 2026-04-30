import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np


def build_graph_snapshot(df_window: pd.DataFrame):
    unique_ips = pd.concat([df_window["src_ip"], df_window["dst_ip"]]).unique()
    ip_to_idx = {ip: i for i, ip in enumerate(unique_ips)}

    src_idx = df_window["src_ip"].map(ip_to_idx).values
    dst_idx = df_window["dst_ip"].map(ip_to_idx).values

    # FIX: Convert to numpy array first to avoid slow tensor warning
    edge_index = torch.tensor(np.array([src_idx, dst_idx]), dtype=torch.long)

    edge_attr = torch.tensor(
        df_window[["flow_duration", "total_packets", "total_bytes", "protocol_encoded"]].values,
        dtype=torch.float
    )

    num_nodes = len(unique_ips)

    out_degree = torch.zeros(num_nodes)
    in_degree = torch.zeros(num_nodes)

    for s, d in zip(src_idx, dst_idx):
        out_degree[s] += 1
        in_degree[d] += 1

    x = torch.stack([out_degree, in_degree], dim=1)

    graph_label = 1 if (df_window["label"] == 1).any() else 0
    y = torch.tensor([graph_label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.ip_mapping = unique_ips.tolist()

    return data


def build_snapshots(df: pd.DataFrame, window_size: int = 20):
    snapshots = []

    df = df.sort_values("timestamp")

    max_time = df["timestamp"].max()
    start = df["timestamp"].min()

    t = start
    while t <= max_time:
        window_df = df[(df["timestamp"] >= t) & (df["timestamp"] < t + window_size)]

        if len(window_df) > 0:
            graph = build_graph_snapshot(window_df)
            graph.snapshot_start = int(t)
            graph.snapshot_end = int(t + window_size)
            snapshots.append(graph)

        t += window_size

    return snapshots