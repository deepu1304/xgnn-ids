# XGNN-IDS Demo (Explainable Graph Neural Network IDS)

This project is a **Streamlit-based demo** for an Explainable Intrusion Detection System (IDS) using **Graph Neural Networks (GraphSAGE)** and **GNNExplainer**.

It converts network flow data into **graph snapshots**, performs attack/benign classification, and provides graph-based explainability.

---

## Features
- Upload CSV dataset directly in Streamlit
- Auto-generate demo dataset if no CSV is provided
- Builds graph snapshots from network flows
- GraphSAGE-based IDS model prediction
- Explainable AI using GNNExplainer
- Visualization of communication graph and suspicious edges

---

## Project Structure
xgnn_ids_demo/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│ └── graphsage_model.pt
│
├── src/
│ ├── preprocess.py
│ ├── graph_builder.py
│ ├── model.py
│ ├── explain.py
│ ├── visualize.py
│ └── train_quick.py
│

---

## Dataset Format (CSV)

Your uploaded CSV must contain these columns:

- timestamp
- src_ip
- dst_ip
- flow_duration
- total_packets
- total_bytes
- protocol
- label

Where:
- label = 0 → BENIGN
- label = 1 → ATTACK

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/xgnn-ids-demo.git
cd xgnn-ids-demo

2. Create Virtual Environment
Windows:
python -m venv .venv
.venv\Scripts\activate
Linux/Mac:
python3 -m venv .venv
source .venv/bin/activate
3. Install Requirements
pip install -r requirements.txt

Install PyTorch Geometric (IMPORTANT)

If you face error:
ModuleNotFoundError: No module named torch_geometric

Install using:

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install torch-geometric

Run the Streamlit App
streamlit run app.py

Open in browser:

http://localhost:8501

Model Behavior
If no trained model is found inside models/, the app automatically performs quick training.
The trained model will be saved as:

models/graphsage_model.pt

Output

The Streamlit app displays:

Graph snapshot visualization
Prediction (Benign/Attack)
Probabilities
Explainability graph with suspicious edges
Feature importance
Tech Stack
Python
Streamlit
PyTorch
PyTorch Geometric
NetworkX
Scikit-learn