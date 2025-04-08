import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fl.interact import init, honest_run, test
from fl.client import Worker
import torch

def main():
    # Initialize with CIFAR dataset
    print("Initializing FL system...")
    data_feeders, test_data, unlabeled_data, model, optimizer, shape, comp_shape, param, momentum = init("cifar", lr=0.01, rank=0)
    
    # Run one round of honest training
    print("Running one round of honest training...")
    round_model = param.copy()  # Start with initial model
    K = 10  # Number of local steps
    B = 64  # Batch size
    
    # Create a worker with the first data feeder
    honest_client = Worker(data_feeders[0], model, optimizer)
    client_grads = honest_run(honest_client, round_model, shape, 0, K, B)
    
    # Test the model
    print("Testing model...")
    test_acc = test(model, shape, round_model, test_data)
    print(f"Test accuracy: {test_acc}")

if __name__ == "__main__":
    main() 