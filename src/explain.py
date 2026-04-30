import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig


def explain_graph(model, graph, device="cpu"):
    model.eval()
    graph = graph.to(device)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=80),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="multiclass_classification",  # FIXED
            task_level="graph",
            return_type="log_probs"
        )
    )

    batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)

    explanation = explainer(
        graph.x,
        graph.edge_index,
        batch=batch
    )

    return explanation