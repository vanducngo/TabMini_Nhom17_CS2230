from itertools import product
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array

def build_mlp(input_dim, hidden_layers, activation=nn.ReLU):
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, 1))  # Output layer (binary classification)
    return nn.Sequential(*layers)

class MLP_PLR(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 hidden_layers=(64, 32), 
                 learning_rate=0.001, 
                 epochs=10, 
                 batch_size=32, 
                 device='cpu', 
                 seed=0):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        torch.manual_seed(self.seed)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        y = y.unsqueeze(1)
        
        self.input_dim = X.shape[1]
        self.model = build_mlp(self.input_dim, self.hidden_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)  # Không bị lỗi shape
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'model')
        X = check_array(X, accept_sparse=True)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = torch.sigmoid(self.model(X)).squeeze()
        return (outputs.cpu().numpy() > 0.5).astype(int)
    
    def train_and_evaluate(self, x_train, y_train, x_test, y_test, save_dir):
        x_train, y_train = check_X_y(x_train, y_train, accept_sparse=True)
        x_test, y_test = check_X_y(x_test, y_test, accept_sparse=True)
        
        results = []
        hidden_layers_options = [(64, 32), (128, 64), (256, 128)]
        learning_rates = [0.001, 0.0005]
        epochs_options = [300, 500]
        
        combinations = list(product(hidden_layers_options, learning_rates, epochs_options))
        
        for combo in combinations:
            hidden_layers, lr, epochs = combo
            print(f"Training with hidden_layers={hidden_layers}, lr={lr}, epochs={epochs}")
            
            model = MLP_PLR(hidden_layers=hidden_layers, learning_rate=lr, epochs=epochs, device=self.device, seed=self.seed)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            
            f1 = f1_score(y_test, y_pred, average="binary")
            acc = accuracy_score(y_test, y_pred)
            results.append({"params": f"{hidden_layers}-{lr}-{epochs}", "accuracy": acc, "f1_score": f1})
        
        result_df = pd.DataFrame(results)
        result_df.to_csv(save_dir, index=False)
