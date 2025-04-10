"""Run federated learning experiments as described in the BEFL paper."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
from typing import List, Dict, Tuple
import numpy as np
import random
from datetime import datetime

from .compression.powersgd import PowerSGDCompressor
from .validation.mutual_info import MutualInformationValidator
from .utils.logger import ExperimentLogger

class FEMNISTModel(nn.Module):
    """CNN model for FEMNIST dataset as described in the paper."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 62)  # FEMNIST has 62 classes
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BasicBlock(nn.Module):
    """Basic block for ResNet14."""
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet14(nn.Module):
    """ResNet14 model for CIFAR10 as described in the paper."""
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def load_cifar10(root='./data'):
    """Load and preprocess CIFAR10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    
    return trainset, testset

def create_non_iid_data(dataset, num_clients=50, alpha=0.75):
    """Create non-IID data distribution as described in the paper."""
    # Sort data by label
    labels = torch.tensor(dataset.targets)
    sorted_indices = torch.argsort(labels)
    
    # Create non-IID splits using Dirichlet distribution
    client_data_indices = []
    label_indices = [[] for _ in range(len(dataset.classes))]
    
    # Group indices by label
    for idx in sorted_indices:
        label = labels[idx]
        label_indices[label].append(idx)
    
    # Convert to numpy arrays
    label_indices = [np.array(indices) for indices in label_indices]
    
    # Use Dirichlet distribution to allocate data
    for _ in range(num_clients):
        client_indices = []
        # For each class
        for label_idx in label_indices:
            # Draw proportion from Dirichlet distribution
            props = np.random.dirichlet(np.repeat(alpha, num_clients))
            # Allocate samples according to proportion
            n_samples = int(len(label_idx) * props[_])
            # Take samples without replacement
            if len(label_idx) > 0:
                selected = np.random.choice(label_idx, n_samples, replace=False)
                client_indices.extend(selected)
                # Remove selected indices
                label_idx = np.setdiff1d(label_idx, selected)
        client_data_indices.append(client_indices)
    
    return client_data_indices

def train_local_model(model: nn.Module, 
                     train_loader: DataLoader,
                     epochs: int = 5,
                     device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> Dict[str, torch.Tensor]:
    """Train a local model and return the updates."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Store initial parameters
    initial_params = {name: p.clone().detach() for name, p in model.named_parameters()}
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Calculate updates
    updates = {}
    for name, p_final in model.named_parameters():
        updates[name] = p_final.detach() - initial_params[name]
    
    return updates

def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader,
                  device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> float:
    """Evaluate model accuracy."""
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_experiment(dataset: str = 'cifar10',
                  num_clients: int = 5,
                  clients_per_round: int = 3,
                  local_epochs: int = 5,
                  global_rounds: int = 3,
                  batch_size: int = 64,
                  experiment_name: str = None):
    """Run federated learning experiment as described in the paper."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize logger
    logger = ExperimentLogger(experiment_name)
    
    # Initialize components
    compressor = PowerSGDCompressor(rank=4 if dataset == 'cifar10' else 2)
    validator = MutualInformationValidator(threshold=0.01)
    
    # Load dataset
    if dataset == 'cifar10':
        trainset, testset = load_cifar10()
        model = ResNet14().to(device)
    else:
        raise NotImplementedError("FEMNIST dataset not implemented yet")
    
    # Create non-IID data distribution
    client_indices = create_non_iid_data(trainset, num_clients=num_clients)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # Initialize momentum
    momentum = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    
    # Training loop
    best_acc = 0.0
    for round in range(global_rounds):
        print(f"\nGlobal Round {round + 1}/{global_rounds}")
        
        # Select random subset of clients
        selected_clients = random.sample(range(num_clients), clients_per_round)
        
        # Simulate clients
        all_updates = []
        client_models = []
        
        for client_idx in selected_clients:
            print(f"\nTraining client {client_idx + 1}/{num_clients}")
            
            # Create client's data loader
            client_data = torch.utils.data.Subset(trainset, client_indices[client_idx])
            client_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
            
            # Create local model copy
            local_model = type(model)().to(device)
            local_model.load_state_dict(model.state_dict())
            
            # Train local model
            updates = train_local_model(local_model, client_loader, local_epochs, device)
            
            # Compress updates
            compressed_updates = compressor.compress_model(updates)
            all_updates.append(compressed_updates)
            
            # Store client model for validation
            client_models.append(local_model.state_dict())
        
        # Validate updates using mutual information
        valid_indices = validator.validate_model_updates(
            client_models,
            model,
            input_shape=(batch_size, 3, 32, 32) if dataset == 'cifar10' else (batch_size, 1, 28, 28)
        )
        
        # Select valid updates
        valid_updates = [u for i, u in enumerate(all_updates) if valid_indices[i]]
        print(f"Selected {len(valid_updates)} valid updates out of {len(all_updates)}")
        
        # Aggregate valid updates
        if valid_updates:
            # Decompress all valid updates
            decompressed_updates = []
            for update in valid_updates:
                params = compressor.decompress_model(update)
                decompressed_updates.append(params)
            
            # Average the updates
            aggregated = {}
            for name, param in model.named_parameters():
                if name in decompressed_updates[0]:
                    # Average the updates
                    update_tensor = torch.mean(torch.stack([
                        update[name] for update in decompressed_updates
                    ]), dim=0)
                    
                    # Apply Nesterov momentum
                    momentum[name] = 0.9 * momentum[name] - 0.01 * update_tensor
                    aggregated[name] = param - 0.9 * momentum[name] + 1.9 * momentum[name]
            
            # Update global model
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in aggregated:
                        param.copy_(aggregated[name])
            
            # Evaluate model
            accuracy = evaluate_model(model, test_loader, device)
            print(f"\nAccuracy after round {round + 1}: {accuracy:.2f}%")
            
            # Log round metrics
            metrics = {
                "accuracy": accuracy,
                "num_valid_updates": len(valid_updates),
                "num_total_updates": len(all_updates)
            }
            logger.log_round(round + 1, metrics)
            
            # Save model if it's the best so far
            if accuracy > best_acc:
                best_acc = accuracy
                print(f"New best accuracy: {best_acc:.2f}%")
                logger.save_model(model, round + 1, accuracy)
        
        print(f"Round {round + 1} completed with {len(valid_updates)} valid updates")
    
    # Print path to best model
    best_model_path = logger.get_best_model_path()
    if best_model_path:
        print(f"\nBest model saved at: {best_model_path}")
    else:
        print("\nNo models were saved during the experiment.")

if __name__ == "__main__":
    run_experiment() 