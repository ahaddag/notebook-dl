import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/moons.csv")
X = df[["X1", "X2"]].values
y = df["y"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Conversion en tenseurs
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        layers = []
        input_dim = 2
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 2))  # Deux classes dans notres cas, on ne s'intéresse qu'à des problèmes de classification binaire
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Définition de l'entraînement du modèle
def train_model(model, X_train, y_train, epochs=500, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = [] # Pour suivre l'évolution de la loss et comparer les deux architectures
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

models = {
    "Underfitting": MLP([1]),
    "Architecture adaptée": MLP([4, 4]),
}

losses = {}
for modelname, model in models.items():
    losses[modelname] = train_model(model, X_train, y_train)