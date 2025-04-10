# Blockchain-Empowered Federated Learning (BEFL)

A lightweight, secure, and efficient federated learning system that integrates blockchain technology for decentralized model aggregation and distribution.

## Features

- **Secure Aggregation**: Byzantine-robust model update aggregation using mutual information
- **Communication Efficiency**: PowerSGD compression for reduced communication overhead
- **Decentralized Architecture**: Blockchain-based consensus for model aggregation
- **IPFS Integration**: Scalable storage for model parameters
- **VRF-based Consensus**: Energy-efficient committee selection

## Prerequisites

- Python 3.7+
- Go 1.16+
- IPFS daemon
- CUDA-capable GPU (optional, for faster training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/befl.git
cd befl
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Go dependencies:
```bash
go mod download
```

4. Start IPFS daemon:
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
│       └── storage/         # IPFS storage
├── tests/                   # Test suite
└── requirements.txt         # Python dependencies
```

## Usage

### 1. Initialize the System

```python
from src.fl.aggregation.secure_aggregation import SecureAggregator
from src.fl.compression.powersgd import PowerSGDCompressor
from src.fl.validation.mutual_info import MutualInformationValidator

# Initialize components
compressor = PowerSGDCompressor(rank=2)
validator = MutualInformationValidator(threshold=0.1)
aggregator = SecureAggregator(
    compressor=compressor,
    validator=validator
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

# Initialize global model and momentum
global_model = {name: param.clone() for name, param in model.named_parameters()}
momentum = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
```

### 3. Aggregate Updates

```python
# Prepare updates and unlabeled data
updates = [...]  # List of compressed model updates
unlabeled_data = torch.randn(100, 10)  # Example unlabeled data

# Aggregate updates
result = aggregator.aggregate(
    global_model=global_model,
    momentum=momentum,
    updates=updates,
    unlabeled_data=unlabeled_data
)
```

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python -m unittest tests.test_secure_aggregation
```

## Configuration

The system can be configured through the following parameters:

- **PowerSGD Compression**:
  - `rank`: Rank of low-rank approximation (default: 2)
  - `num_power_iterations`: Number of power iterations (default: 1)

- **Mutual Information Validation**:
  - `threshold`: MI threshold for update selection (default: 0.1)

- **Secure Aggregation**:
  - `beta`: Momentum parameter (default: 0.9)
  - `eta`: Learning rate (default: 1.0)

## Performance Considerations

- Use CUDA if available for faster computation
- Adjust PowerSGD rank based on model size and available bandwidth
- Tune MI threshold based on expected update distribution
- Monitor IPFS daemon performance for large models

## Security Considerations

- Keep private keys secure
- Monitor for suspicious update patterns
- Regularly update dependencies
- Use secure communication channels

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