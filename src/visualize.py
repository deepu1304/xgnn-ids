import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(graph, explanation=None):
    edge_index = graph.edge_index.cpu().numpy()
    ip_map = graph.ip_mapping

    G = nx.DiGraph()

    for i, ip in enumerate(ip_map):
        G.add_node(i, label=ip)

    for s, d in zip(edge_index[0], edge_index[1]):
        G.add_edge(int(s), int(d))

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 6))

    if explanation is not None and explanation.edge_mask is not None:
        edge_mask = explanation.edge_mask.detach().cpu().numpy()
        edges = list(G.edges())

        edge_colors = []
        for i in range(len(edges)):
            edge_colors.append(edge_mask[i] if i < len(edge_mask) else 0.1)

        nx.draw(G, pos, with_labels=False, node_size=600, edge_color=edge_colors, width=2.5)
    else:
        nx.draw(G, pos, with_labels=False, node_size=600)

    labels = {i: ip_map[i] for i in range(len(ip_map))}
    nx.draw_networkx_labels(G, pos, labels, font_size=7)

    return plt