"""
PyTorch Tabular ResNet & Deep MLP Architecture for Tabular Classification.
Combines entity embeddings for categorical features, continuous feature normalization,
residual dense blocks with Batch Normalization and SiLU activations, and Cosine Annealing.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler


class TabularDataset(Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray = None):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_num[idx], self.X_cat[idx], self.y[idx]
        return self.X_num[idx], self.X_cat[idx]


class ResNetBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: list[int],
        embedding_dims: list[int] = None,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        if embedding_dims is None:
            embedding_dims = [max(4, min(32, (card + 1) // 2)) for card in cat_cardinalities]

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim)
            for card, emb_dim in zip(cat_cardinalities, embedding_dims)
        ])
        total_emb_dim = sum(embedding_dims)

        self.input_layer = nn.Sequential(
            nn.Linear(num_continuous + total_emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )

        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_num, x_cat):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        if emb_outs:
            x_all = torch.cat([x_num] + emb_outs, dim=1)
        else:
            x_all = x_num

        h = self.input_layer(x_all)
        for block in self.blocks:
            h = block(h)
        logits = self.head(h)
        return logits


def train_tabular_nn(
    X_tr_num, X_tr_cat, y_tr,
    X_va_num, X_va_cat, y_va,
    X_te_num=None, X_te_cat=None,
    cat_cardinalities=None,
    epochs=12,
    batch_size=2048,
    lr=1e-3,
    device='cpu'
):
    """
    Trains a TabularResNet model and returns validation probabilities and test predictions.
    """
    scaler = RobustScaler()
    X_tr_num_scaled = scaler.fit_transform(np.nan_to_num(X_tr_num, nan=0.0))
    X_va_num_scaled = scaler.transform(np.nan_to_num(X_va_num, nan=0.0))
    X_te_num_scaled = scaler.transform(np.nan_to_num(X_te_num, nan=0.0)) if X_te_num is not None else None

    # Ensure categorical features are strictly non-negative (shift -1 to 0)
    X_tr_cat = np.where(X_tr_cat < 0, 0, X_tr_cat + 1)
    X_va_cat = np.where(X_va_cat < 0, 0, X_va_cat + 1)
    if X_te_cat is not None:
        X_te_cat = np.where(X_te_cat < 0, 0, X_te_cat + 1)

    if cat_cardinalities is None:
        cat_cardinalities = [int(max(X_tr_cat[:, i].max(), X_va_cat[:, i].max()) + 2) for i in range(X_tr_cat.shape[1])]

    train_ds = TabularDataset(X_tr_num_scaled, X_tr_cat, y_tr)
    val_ds = TabularDataset(X_va_num_scaled, X_va_cat, y_va)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    num_continuous = X_tr_num_scaled.shape[1]
    model = TabularResNet(
        num_continuous=num_continuous,
        cat_cardinalities=cat_cardinalities,
        hidden_dim=256,
        num_blocks=2,
        dropout=0.15
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for batch_num, batch_cat, batch_y in train_loader:
            batch_num, batch_cat, batch_y = batch_num.to(device), batch_cat.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_num, batch_cat)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for batch_num, batch_cat, batch_y in val_loader:
                batch_num, batch_cat, batch_y = batch_num.to(device), batch_cat.to(device), batch_y.to(device)
                logits = model(batch_num, batch_cat)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())

        val_loss /= len(val_ds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()

    # Get final val predictions
    final_val_preds = []
    with torch.no_grad():
        for batch_num, batch_cat, _ in val_loader:
            batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
            final_val_preds.extend(torch.sigmoid(model(batch_num, batch_cat)).cpu().numpy().flatten())

    # Get test predictions if test set provided
    test_preds = None
    if X_te_num_scaled is not None and X_te_cat is not None:
        test_ds = TabularDataset(X_te_num_scaled, X_te_cat)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)
        test_preds = []
        with torch.no_grad():
            for batch_num, batch_cat in test_loader:
                batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
                test_preds.extend(torch.sigmoid(model(batch_num, batch_cat)).cpu().numpy().flatten())
        test_preds = np.array(test_preds)

    return np.array(final_val_preds), test_preds, model
