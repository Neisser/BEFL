import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.fl.experiment import run_experiment

def main():
    print("Starting Federated Learning experiment...")
    
    # Generate experiment name with timestamp
    experiment_name = f"cifar10_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run experiment with parameters matching run.py
    run_experiment(
        dataset='cifar10',
        num_clients=5,
        clients_per_round=3,
        local_epochs=5,
        global_rounds=3,
        batch_size=64,
        experiment_name=experiment_name
    )

if __name__ == "__main__":
    main() 