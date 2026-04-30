import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from src.model import GraphSAGE_IDS


def quick_train(graphs, save_path="models/graphsage_model.pt"):
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)

    model = GraphSAGE_IDS(in_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 21):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            out = model(data=batch)  # FIXED
            loss = criterion(out, batch.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Quick Train Epoch {epoch} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print("Saved model:", save_path)

    return model