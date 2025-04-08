import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from blockchain.blockchain import Blockchain
from fl.blockchain_fl import BlockchainFL
from fl.client import FLClient

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    return train_dataset, test_dataset

def create_clients(train_dataset, num_clients=3):
    # Split the training data among clients
    data_per_client = len(train_dataset) // num_clients
    splits = [data_per_client] * num_clients
    splits[-1] += len(train_dataset) - sum(splits)  # Add remainder to last client
    
    train_splits = random_split(train_dataset, splits)
    
    clients = []
    for i in range(num_clients):
        train_loader = DataLoader(train_splits[i], batch_size=32, shuffle=True)
        client = FLClient(f"client_{i}", SimpleCNN(), train_loader)
        clients.append(client)
    
    return clients

def main():
    # Initialize blockchain and FL system
    blockchain = Blockchain(difficulty=2)
    fl_system = BlockchainFL(blockchain, min_clients=3)
    
    # Load data and create clients
    train_dataset, test_dataset = load_mnist_data()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    clients = create_clients(train_dataset)
    
    # Training loop
    num_rounds = 5
    for round in range(num_rounds):
        print(f"\nRound {round + 1}/{num_rounds}")
        
        # Train local models
        for client in clients:
            print(f"Training {client.get_client_id()}...")
            client.train_local_model(epochs=1)
            accuracy = client.evaluate_model(test_loader)
            print(f"{client.get_client_id()} accuracy: {accuracy:.4f}")
            
            # Submit model update to blockchain
            model_update = client.get_model_update()
            fl_system.submit_model_update(client.get_client_id(), model_update, accuracy)
        
        # Aggregate updates and update global model
        print("Aggregating updates...")
        fl_system.update_global_model("miner_1")
        
        # Update clients with new global model
        global_model = fl_system.get_global_model()
        if global_model:
            for client in clients:
                client.update_global_model(global_model)
        
        # Save system state
        fl_system.save_state(f"fl_state_round_{round + 1}.json")
        
        # Evaluate global model
        global_accuracy = clients[0].evaluate_model(test_loader)
        print(f"Global model accuracy: {global_accuracy:.4f}")

if __name__ == "__main__":
    main() 