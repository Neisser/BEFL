"""Tests for secure aggregation protocol."""
import unittest
import torch
from torch import nn
from src.fl.compression import PowerSGDCompressor, CompressedUpdate
from src.fl.validation import MutualInformationValidator
from src.fl.aggregation import SecureAggregator

class TestSecureAggregation(unittest.TestCase):
    def setUp(self):
        # Create a simple model for testing
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
        # Initialize components
        self.compressor = PowerSGDCompressor(rank=2)
        self.validator = MutualInformationValidator(
            threshold=0.1,
            sample_size=100
        )
        self.aggregator = SecureAggregator(
            compressor=self.compressor,
            validator=self.validator,
            momentum=0.9,
            learning_rate=0.01
        )
        
        # Initialize momentum
        self.momentum = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }
        
        # Create test data shape
        self.input_shape = (32, 10)  # batch_size=32, features=10
        
    def test_aggregation(self):
        # Create random updates
        updates = []
        for _ in range(5):
            params = [
                param + 0.1 * torch.randn_like(param)
                for name, param in self.model.named_parameters()
            ]
            compressed = self.compressor.compress(params)
            updates.append(compressed)
        
        # Aggregate updates
        result = self.aggregator.aggregate(
            updates=updates,
            global_model=self.model,
            current_momentum=self.momentum,
            input_shape=self.input_shape
        )
        
        # Check result structure
        self.assertIsNotNone(result.global_model)
        self.assertIsNotNone(result.momentum)
        self.assertIsInstance(result.selected_updates, list)
        
        # Check model state dict keys match
        self.assertEqual(
            set(result.global_model.keys()),
            set(self.model.state_dict().keys())
        )
        
        # Check momentum keys match
        self.assertEqual(
            set(result.momentum.keys()),
            set(self.momentum.keys())
        )
        
        # Check for NaN or Inf values
        for tensor in result.global_model.values():
            self.assertFalse(torch.isnan(tensor).any())
            self.assertFalse(torch.isinf(tensor).any())
    
    def test_empty_updates(self):
        # Test aggregation with no updates
        result = self.aggregator.aggregate(
            updates=[],
            global_model=self.model,
            current_momentum=self.momentum,
            input_shape=self.input_shape
        )
        
        # Should return current model and momentum
        self.assertEqual(
            set(result.global_model.keys()),
            set(self.model.state_dict().keys())
        )
        self.assertEqual(
            set(result.momentum.keys()),
            set(self.momentum.keys())
        )
        self.assertEqual(result.selected_updates, [])
    
    def test_malicious_updates(self):
        # Create normal updates
        updates = []
        for _ in range(3):
            params = [
                param + 0.1 * torch.randn_like(param)
                for name, param in self.model.named_parameters()
            ]
            compressed = self.compressor.compress(params)
            updates.append(compressed)
        
        # Create malicious updates (large values)
        for _ in range(2):
            params = [
                param + 100.0 * torch.randn_like(param)
                for name, param in self.model.named_parameters()
            ]
            compressed = self.compressor.compress(params)
            updates.append(compressed)
        
        # Aggregate updates
        result = self.aggregator.aggregate(
            updates=updates,
            global_model=self.model,
            current_momentum=self.momentum,
            input_shape=self.input_shape
        )
        
        # Should select fewer updates than provided
        self.assertLess(len(result.selected_updates), len(updates))
        
        # Check for NaN or Inf values
        for tensor in result.global_model.values():
            self.assertFalse(torch.isnan(tensor).any())
            self.assertFalse(torch.isinf(tensor).any())

if __name__ == '__main__':
    unittest.main() 