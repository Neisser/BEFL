import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class FLClient:
    def __init__(self, client_id: str, model: nn.Module, train_loader: DataLoader):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.local_model: Optional[Dict[str, Any]] = None

    def train_local_model(self, epochs: int = 1) -> None:
        """Train the local model on the client's data."""
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

        # Store the local model state
        self.local_model = {
            name: param.data.clone() 
            for name, param in self.model.state_dict().items()
        }

    def evaluate_model(self, test_loader: DataLoader) -> float:
        """Evaluate the model on test data and return accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def get_model_update(self) -> Dict[str, Any]:
        """Get the model update to be sent to the blockchain."""
        if self.local_model is None:
            raise ValueError("Local model has not been trained yet")
        return self.local_model

    def update_global_model(self, global_model: Dict[str, Any]) -> None:
        """Update the local model with the global model parameters."""
        self.model.load_state_dict(global_model)
        self.local_model = None  # Reset local model after global update

    def get_client_id(self) -> str:
        """Get the client's ID."""
        return self.client_id 