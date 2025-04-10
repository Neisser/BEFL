import unittest
import sys
from datetime import datetime
from src.fl.experiment import run_experiment

def main():
    # Run unit tests
    # print("Running unit tests...")
    # test_loader = unittest.TestLoader()
    # test_suite = test_loader.discover('tests', pattern='test_*.py')
    # test_runner = unittest.TextTestRunner(verbosity=2)
    # test_result = test_runner.run(test_suite)
    
    # if not test_result.wasSuccessful():
    #     print("\nTests failed. Fix the issues before running the experiment.")
    #     sys.exit(1)
    
    # Generate experiment name with timestamp
    experiment_name = f"cifar10_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run experiment
    print("\nRunning federated learning experiment...")
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