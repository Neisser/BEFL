# Blockchain-Empowered Federated Learning (BEFL)

A lightweight, secure, and efficient federated learning system that integrates blockchain technology for decentralized model aggregation and distribution.

## Features

- **Secure Aggregation**: Byzantine-robust model update aggregation using mutual information
- **Communication Efficiency**: PowerSGD compression for reduced communication overhead
- **Decentralized Architecture**: Blockchain-based consensus for model aggregation
- **IPFS Integration**: Optional scalable storage for model parameters

## Prerequisites

- Python 3.7+
- CUDA-capable GPU (optional, for faster training)
- IPFS daemon (optional, only needed if using IPFS storage)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/befl.git
cd befl
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Start IPFS daemon if you plan to use IPFS storage:
```bash
ipfs daemon
```

## Project Structure

```
befl/
├── src/
│   ├── fl/
│   │   ├── aggregation/     # Secure aggregation protocol
│   │   ├── compression/     # PowerSGD compression
│   │   └── validation/      # Mutual information validation
│   └── blockchain/
│       ├── consensus/       # VRF-based consensus
│       └── storage/         # IPFS storage (optional)
├── tests/                   # Test suite
└── requirements.txt         # Python dependencies
```

## Quick Start

To run the federated learning experiment:

```bash
python main.py
```

This will start a federated learning experiment with:
- 5 clients
- 5 local epochs per client
- 3 global rounds
- PowerSGD compression
- Mutual information-based validation

## Advanced Usage

### 1. Initialize Components

```python
from src.fl.aggregation.secure_aggregation import SecureAggregator
from src.fl.compression.powersgd import PowerSGDCompressor
from src.fl.validation.mutual_info import MutualInformationValidator

# Initialize components
compressor = PowerSGDCompressor(rank=2)
validator = MutualInformationValidator()
aggregator = SecureAggregator(
    compressor=compressor,
    validator=validator,
    learning_rate=0.01
)
```

### 2. Train a Model

```python
import torch
import torch.nn as nn

# Define your model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Initialize global model
global_model = model.state_dict()
```

### 3. Run Federated Learning

```python
from src.fl.experiment import run_experiment

# Run experiment with custom parameters
run_experiment(
    num_clients=5,
    local_epochs=5,
    global_rounds=3
)
```

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python -m unittest tests.test_secure_aggregation
python -m unittest tests.test_ipfs_storage
```

## Configuration

The system can be configured through the following parameters:

- **PowerSGD Compression**:
  - `rank`: Rank of low-rank approximation (default: 2)
  - `num_power_iterations`: Number of power iterations (default: 1)

- **Mutual Information Validation**:
  - `threshold`: MI threshold for update selection (default: 0.1)

- **Secure Aggregation**:
  - `learning_rate`: Learning rate for model updates (default: 0.01)

## Performance Considerations

- Use CUDA if available for faster computation
- Adjust PowerSGD rank based on model size and available bandwidth
- Tune MI threshold based on expected update distribution
- If using IPFS storage, monitor IPFS daemon performance for large models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{jin2023lightweight,
  title={Lightweight Blockchain-Empowered Secure and Efficient Federated Edge Learning},
  author={Jin, Rui and Hu, Jia and Min, Geyong and Mills, Jed},
  journal={IEEE Transactions on Computers},
  volume={72},
  number={11},
  pages={3314--3325},
  year={2023},
  publisher={IEEE}
}
```