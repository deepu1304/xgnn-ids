import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch_geometric.loader import DataLoader

from src.preprocess import preprocess_flows
from src.graph_builder import build_snapshots
from src.model import GraphSAGE_IDS
from src.explain import explain_graph
from src.visualize import visualize_graph
from src.train_quick import quick_train


st.set_page_config(page_title="XGNN-IDS")
st.title("🔐 XGNN-IDS (Explainable Graph Neural Network IDS)")
st.write("This demo builds graph snapshots from network flow data and performs intrusion detection with explanations.")


def generate_demo_dataset(n_flows=500):
    np.random.seed(42)

    ips = [f"192.168.1.{i}" for i in range(2, 50)]
    protocols = ["TCP", "UDP"]

    data = []

    for t in range(1, n_flows + 1):
        src = np.random.choice(ips)
        dst = np.random.choice(ips)

        flow_duration = np.random.randint(1, 100)
        total_packets = np.random.randint(1, 600)
        total_bytes = total_packets * np.random.randint(40, 200)

        protocol = np.random.choice(protocols)

        # Attack simulation logic (heavy UDP floods)
        label = 1 if (total_packets > 400 and protocol == "UDP") else 0

        data.append([t, src, dst, flow_duration, total_packets, total_bytes, protocol, label])

    df = pd.DataFrame(data, columns=[
        "timestamp", "src_ip", "dst_ip",
        "flow_duration", "total_packets", "total_bytes",
        "protocol", "label"
    ])

    return df


st.sidebar.header("⚙️ Dataset Options")

use_demo = st.sidebar.checkbox("Use Auto Demo Dataset", value=True)

if use_demo:
    n_flows = st.sidebar.slider("Number of Demo Flows", 200, 2000, 500)
    df = generate_demo_dataset(n_flows=n_flows)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is None:
        st.warning("Upload a CSV file or enable Auto Demo Dataset.")
        st.stop()

    df = pd.read_csv(uploaded_file)


st.subheader("📌 Dataset Preview")
st.dataframe(df.head(15))


# Preprocess
df, _ = preprocess_flows(df)

window_size = st.sidebar.slider("Snapshot Window Size", min_value=5, max_value=100, value=20)

graphs = build_snapshots(df, window_size=window_size)

if len(graphs) == 0:
    st.error("No graph snapshots created. Dataset might be invalid.")
    st.stop()

st.sidebar.success(f"Snapshots Created: {len(graphs)}")

snapshot_id = st.sidebar.selectbox("Select Snapshot Graph", list(range(len(graphs))))
graph = graphs[snapshot_id]

st.sidebar.write(f"Time Window: {graph.snapshot_start} → {graph.snapshot_end}")
st.sidebar.write(f"Nodes: {graph.num_nodes}")
st.sidebar.write(f"Edges: {graph.num_edges}")


# Load model (or auto train if missing)
MODEL_PATH = "models/graphsage_model.pt"

if not os.path.exists(MODEL_PATH):
    st.warning("No trained model found. Training quick demo model now...")
    os.makedirs("models", exist_ok=True)
    model = quick_train(graphs, save_path=MODEL_PATH)
else:
    model = GraphSAGE_IDS(in_channels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()


# Predict
loader = DataLoader([graph], batch_size=1)
batch = next(iter(loader))

with torch.no_grad():
    output = model(data=batch)   # FIXED
    probs = torch.softmax(output, dim=1)[0]
    pred = torch.argmax(probs).item()

label_name = "ATTACK 🚨" if pred == 1 else "BENIGN ✅"


st.subheader("📌 Prediction Result")
st.write(f"### Prediction: **{label_name}**")
st.write(f"Benign Probability: **{probs[0].item():.4f}**")
st.write(f"Attack Probability: **{probs[1].item():.4f}**")


col1, col2 = st.columns(2)

with col1:
    st.subheader("🌐 Full Communication Graph")
    fig = visualize_graph(graph)
    st.pyplot(fig)

with col2:
    if pred == 1:
        st.subheader("🧠 Explanation Graph (GNNExplainer)")

        with st.spinner("Generating explanation..."):
            explanation = explain_graph(model, graph)

        fig2 = visualize_graph(graph, explanation=explanation)
        st.pyplot(fig2)

        st.subheader("📊 Node Feature Importance")
        if explanation.node_mask is not None:
            feature_scores = explanation.node_mask.mean(dim=0).detach().cpu().numpy()
            feature_names = ["out_degree", "in_degree"]

            for name, score in zip(feature_names, feature_scores):
                st.write(f"- {name}: {score:.4f}")

        st.subheader("📊 Top Suspicious Edges")
        if explanation.edge_mask is not None:
            edge_scores = explanation.edge_mask.detach().cpu().numpy()
            top_edges = edge_scores.argsort()[-5:][::-1]

            edge_index = graph.edge_index.cpu().numpy()

            for idx in top_edges:
                src = graph.ip_mapping[edge_index[0][idx]]
                dst = graph.ip_mapping[edge_index[1][idx]]
                st.write(f"Edge {src} → {dst} | Score: {edge_scores[idx]:.4f}")

    else:
        st.info("Graph predicted BENIGN. Explanation is generated only for ATTACK graphs.")