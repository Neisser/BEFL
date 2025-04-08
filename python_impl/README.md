# Blockchain-Based Federated Learning

This project implements a blockchain-based federated learning system that combines the security and transparency of blockchain with the privacy-preserving benefits of federated learning.

## Features

- Blockchain-based model update tracking
- Weighted model aggregation based on client accuracy
- Secure and transparent model update process
- MNIST dataset support with CNN model
- State persistence and recovery
- Multiple client support

## Project Structure

```
python_impl/
├── blockchain/
│   ├── block.py
│   └── blockchain.py
├── fl/
│   ├── blockchain_fl.py
│   └── client.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python main.py
```

The script will:
- Initialize the blockchain and FL system
- Load and split the MNIST dataset among clients
- Perform multiple rounds of federated learning
- Save the system state after each round
- Display training progress and accuracy metrics

## Components

### Blockchain
- `Block`: Represents a single block in the blockchain
- `Blockchain`: Manages the chain of blocks and transactions

### Federated Learning
- `BlockchainFL`: Integrates blockchain with federated learning
- `FLClient`: Handles local training and model updates

### Main Script
- Implements a simple CNN for MNIST classification
- Manages the training process and client coordination
- Handles data loading and splitting

## Configuration

You can modify the following parameters in `main.py`:
- `num_clients`: Number of federated learning clients
- `num_rounds`: Number of training rounds
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for the optimizer
- `difficulty`: Blockchain mining difficulty

## State Management

The system state is saved after each round in JSON format. You can load a previous state using:
```python
fl_system = BlockchainFL.load_state("fl_state_round_X.json")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 