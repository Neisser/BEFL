import unittest
import torch
import torch.nn as nn
import numpy as np
from src.fl.validation.mutual_info import MutualInformationValidator

class TestMutualInformationValidator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple model
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
        # Initialize validator
        self.validator = MutualInformationValidator(threshold=0.1, num_bins=10)
        
        # Create some test data
        self.test_data = torch.randn(100, 10)
        
    def test_validate_update(self):
        """Test validation of a single update."""
        # Create a valid update
        valid_update = torch.randn_like(self.model[0].weight)
        
        # Create reference updates
        ref_updates = [p.data for p in self.model.parameters()]
        
        # Validate update
        is_valid, mi_score = self.validator.validate_update(valid_update, ref_updates)
        
        # Check results
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(mi_score, float)
        self.assertGreaterEqual(mi_score, 0.0)
        self.assertLessEqual(mi_score, 1.0)
        
    def test_validate_model_updates(self):
        """Test validation of multiple updates."""
        # Create some updates
        updates = [
            torch.randn_like(p.data) for p in self.model.parameters()
        ]
        
        # Validate updates
        results = self.validator.validate_model_updates(updates, self.model)
        
        # Check results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(updates))
        self.assertTrue(all(isinstance(r, bool) for r in results))
        
    def test_compute_histogram(self):
        """Test histogram computation."""
        # Create test tensor
        tensor = torch.randn(100, 10)
        
        # Compute histogram
        hist = self.validator._compute_histogram(tensor)
        
        # Check results
        self.assertIsInstance(hist, np.ndarray)
        self.assertEqual(hist.shape, (self.validator.num_bins,))
        self.assertTrue(np.all(hist >= 0))
        self.assertAlmostEqual(hist.sum(), 1.0, places=6)
        
    def test_compute_mi(self):
        """Test mutual information computation."""
        # Create test histograms
        hist1 = np.random.rand(self.validator.num_bins)
        hist1 = hist1 / hist1.sum()
        
        hist2 = np.random.rand(self.validator.num_bins)
        hist2 = hist2 / hist2.sum()
        
        # Compute mutual information
        mi = self.validator._compute_mi(hist1, hist2)
        
        # Check results
        self.assertIsInstance(mi, float)
        self.assertGreaterEqual(mi, 0.0)
        self.assertLessEqual(mi, 1.0)
        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with zero tensor
        zero_tensor = torch.zeros(100, 10)
        hist = self.validator._compute_histogram(zero_tensor)
        self.assertTrue(np.all(hist > 0))
        
        # Test with identical histograms
        hist1 = np.ones(self.validator.num_bins) / self.validator.num_bins
        hist2 = hist1.copy()
        mi = self.validator._compute_mi(hist1, hist2)
        self.assertAlmostEqual(mi, 1.0, places=6)
        
        # Test with orthogonal histograms
        hist1 = np.zeros(self.validator.num_bins)
        hist1[0] = 1.0
        hist2 = np.zeros(self.validator.num_bins)
        hist2[-1] = 1.0
        mi = self.validator._compute_mi(hist1, hist2)
        self.assertAlmostEqual(mi, 0.0, places=6)

if __name__ == '__main__':
    unittest.main() 