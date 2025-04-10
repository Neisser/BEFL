import torch
import numpy as np
from typing import List, Dict, Tuple
from torch import nn

class MutualInformationValidator:
    """Validates model updates using mutual information between model outputs."""
    
    def __init__(self, threshold: float = 0.01, sample_size: int = 1000):
        """
        Initialize the validator.
        
        Args:
            threshold: Minimum similarity score for valid updates
            sample_size: Number of random samples to use for validation
        """
        self.threshold = threshold
        self.sample_size = sample_size
        self.eps = 1e-300  # Small constant for numerical stability
        
    def validate_model_updates(
        self, 
        updates: List[Dict[str, torch.Tensor]], 
        model: nn.Module,
        input_shape: Tuple[int, ...]
    ) -> List[bool]:
        """
        Validate multiple model updates.
        
        Args:
            updates: List of model parameter updates to validate
            model: The current model state
            input_shape: Shape of input tensors (batch_size, *features)
            
        Returns:
            List of boolean values indicating valid updates
        """
        # Generate random input data for validation
        device = next(model.parameters()).device
        x = torch.randn(self.sample_size, *input_shape[1:], device=device)
        
        # Get outputs from all models
        outputs = []
        temp_model = type(model)().to(device)  # Create new instance of same model type
        
        for update in updates:
            # Load update into temporary model
            temp_model.load_state_dict(update)
            
            # Get model outputs
            with torch.no_grad():
                out = temp_model(x)
                outputs.append(out)
        
        # Calculate mutual information matrix
        n_updates = len(updates)
        mutual_mi = np.zeros((n_updates, n_updates))
        
        for i in range(n_updates):
            out_i = outputs[i]
            exp_i = out_i.mean(dim=1, keepdim=True)
            
            for j in range(i+1, n_updates):
                out_j = outputs[j]
                exp_j = out_j.mean(dim=1, keepdim=True)
                
                # Calculate correlation coefficient
                rho = torch.sum((out_i - exp_i) * (out_j - exp_j), dim=1) / \
                      torch.sqrt(torch.sum(torch.square(out_i - exp_i), dim=1) * \
                               torch.sum(torch.square(out_j - exp_j), dim=1))
                
                # Calculate mutual information
                intermediate = 1 - pow(rho.mean().item(), 2)
                intermediate = max(intermediate, self.eps)  # Ensure numerical stability
                mi = -np.log(intermediate) / 2
                
                mutual_mi[i, j] = mi
                mutual_mi[j, i] = mi
        
        # Calculate average MI for each update
        avg_mi = np.mean(mutual_mi, axis=1)
        
        # Use MAD/MADN for robust outlier detection
        mad = np.median(np.abs(avg_mi - np.median(avg_mi)))
        madn = mad / 0.6745  # 0.6745 is the MAD of standard normal distribution
        
        # Apply two-sigma edit rule
        if madn < self.eps:
            # If MADN is too small, accept all updates
            return [True] * n_updates
            
        ts = (avg_mi - np.median(avg_mi)) / madn
        return [abs(t) < 2 for t in ts]  # Two-sigma edit rule
        
    def _compute_mi(self, out1: torch.Tensor, out2: torch.Tensor) -> float:
        """
        Compute mutual information between two model outputs.
        
        Args:
            out1: First model output
            out2: Second model output
            
        Returns:
            Mutual information score
        """
        # Calculate means
        exp1 = out1.mean(dim=1, keepdim=True)
        exp2 = out2.mean(dim=1, keepdim=True)
        
        # Calculate correlation coefficient
        rho = torch.sum((out1 - exp1) * (out2 - exp2), dim=1) / \
              torch.sqrt(torch.sum(torch.square(out1 - exp1), dim=1) * \
                        torch.sum(torch.square(out2 - exp2), dim=1))
        
        # Calculate mutual information
        intermediate = 1 - pow(rho.mean().item(), 2)
        intermediate = max(intermediate, self.eps)  # Ensure numerical stability
        return -np.log(intermediate) / 2 