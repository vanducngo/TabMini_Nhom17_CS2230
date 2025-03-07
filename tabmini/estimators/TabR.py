import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_is_fitted, check_array
from itertools import product
from torch.utils.data import DataLoader, TensorDataset

class TabRClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, context_size=64, n_neighbors=10):
        super(TabRClassifier, self).__init__()
        self.context_size = context_size
        self.n_neighbors = n_neighbors

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class TabR(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible estimator that uses a PyTorch-based TabR model to fit and predict data."""

    def __init__(
        self,
        time_limit: int = 3600,
        device: str = "cpu",
        seed: int = 42,
        hidden_dim: int = 64,
        context_size: int = 64,
        n_neighbors: int = 10,
        learning_rate: float = 0.01,
        epochs: int = 10,
        batch_size: int = 32,
    ):
        self.time_limit = time_limit
        self.device = device
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        self.n_neighbors = n_neighbors
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes_ = 2  # Binary classification
        self.classes_ = [0, 1]
        self.model = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X.shape[1]
        self.model = TabRClassifier(input_dim, self.hidden_dim, self.context_size, self.n_neighbors).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        check_is_fitted(self, "model")
        X = check_array(X, accept_sparse=True)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return (outputs.cpu().numpy() > 0.5).astype(int)

    def train_and_evaluate(self, x_train, y_train, x_test, y_test, save_dir):
        x_train_check, y_train_check = check_X_y(x_train, y_train, accept_sparse=True)

        results = []
        context_size_sample = [32, 64, 128]
        n_neighbors_sample = [5, 10, 20]
        combinations = list(product(n_neighbors_sample, context_size_sample))
        
        for combo in combinations:
            print(f"Training with parameters: n_neighbors={combo[0]}, context_size={combo[1]}")
            
            self.context_size = combo[1]
            self.n_neighbors = combo[0]
            self.fit(x_train_check, y_train_check)
            y_inference = self.predict(x_test)

            f1 = f1_score(y_test, y_inference, average="binary")
            acc = accuracy_score(y_test, y_inference)
            results.append({"params": f'{combo[0]}-{combo[1]}', "accuracy": acc, "f1_score": f1})
        
        result_df = pd.DataFrame(results)
        result_df.to_csv(save_dir, index=False)
