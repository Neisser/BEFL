import unittest
import torch
import torch.nn as nn
import numpy as np
from src.fl.compression.powersgd import PowerSGDCompressor, CompressedUpdate
from src.fl.validation.mutual_info import MutualInformationValidator

class TestComponents(unittest.TestCase):
    def setUp(self):
        # Create a simple model for testing
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Initialize components
        self.compressor = PowerSGDCompressor(rank=2)
        self.validator = MutualInformationValidator(threshold=0.1)
        
    def test_powersgd_compression(self):
        """Test PowerSGD compression and decompression"""
        # Get model parameters
        params = list(self.model.parameters())
        
        # Compress parameters
        compressed = self.compressor.compress_model(params)
        
        # Decompress parameters
        decompressed = self.compressor.decompress_model(compressed)
        
        # Verify shapes match
        for orig, decomp in zip(params, decompressed):
            self.assertEqual(orig.shape, decomp.shape)
            
    def test_mutual_info_validation(self):
        """Test Mutual Information validation"""
        # Create some test updates
        updates = []
        for _ in range(5):
            update = torch.randn(10, 5)  # Random update
            updates.append(update)
            
        # Validate updates
        results = self.validator.validate_model_updates(updates, self.model)
        
        # Check results format
        self.assertEqual(len(results), len(updates))
        for is_valid, mi_score in results:
            self.assertIsInstance(is_valid, bool)
            self.assertIsInstance(mi_score, float)
            
if __name__ == '__main__':
    unittest.main() 