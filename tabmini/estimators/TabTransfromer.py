from itertools import product
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.validation import check_is_fitted, check_array

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, depth, num_classes=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=0.1),
            num_layers=depth
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_layers(x.unsqueeze(1)).squeeze(1)
        x = self.fc(x)
        return x  # KHÔNG DÙNG SOFTMAX


class TabTransformer(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=100, batch_size=32, learning_rate=0.0005, embed_dim=32, num_heads=4, depth=2, device="cpu"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, y):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X, y = check_X_y(X, y, accept_sparse=False)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = TransformerClassifier(X.shape[1], self.embed_dim, self.num_heads, self.depth).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()  # Cập nhật learning rate
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

    def predict(self, X):
        check_is_fitted(self, "model")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.transform(X)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        return outputs.argmax(dim=1).cpu().numpy()

    def train_and_evaluate(self, x_train, y_train, x_test, y_test, save_dir):
        x_train_check, y_train_check = check_X_y(x_train, y_train, accept_sparse=False)
        x_test_check = check_array(x_test, accept_sparse=False)

        results = []
        embedding_dims = [32, 64, 128]
        attn_heads = [4, 6, 8]
        depth_values = [2, 4]

        for embed_dim, num_heads, depth in product(embedding_dims, attn_heads, depth_values):
            print(f"Training with embed_dim: {embed_dim}, attn_heads: {num_heads}, depth: {depth}")
            
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.depth = depth
            
            self.fit(x_train_check, y_train_check)
            y_pred = self.predict(x_test_check)
            
            f1 = f1_score(y_test, y_pred, average="binary")
            acc = accuracy_score(y_test, y_pred)
            results.append({"params": f'{embed_dim}-{num_heads}-{depth}', "accuracy": acc, "f1_score": f1})
        
        result_df = pd.DataFrame(results)
        result_df.to_csv(save_dir, index=False)
