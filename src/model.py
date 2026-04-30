import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGE_IDS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.lin1 = torch.nn.Linear(hidden_channels, 32)
        self.lin2 = torch.nn.Linear(32, 2)

        self.dropout = 0.3

    def forward(self, x=None, edge_index=None, batch=None, data=None):
        """
        Supports:
        - model(data=batch_obj)      (training + streamlit prediction)
        - model(x, edge_index, batch=batch_tensor) (torch-geometric explainer)
        """

        if data is not None:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = F.relu(x)

        out = self.lin2(x)
        return out