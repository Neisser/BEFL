"""PowerSGD compression for efficient model updates."""
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class CompressedUpdate:
    """Represents a compressed model update using PowerSGD."""
    p_matrix: torch.Tensor
    q_matrix: torch.Tensor
    original_shape: torch.Size
    error_feedback: Optional[torch.Tensor] = None

class PowerSGDCompressor:
    """Implements PowerSGD compression for model updates."""
    
    def __init__(self, rank: int = 2, num_power_iterations: int = 1, reuse_query: bool = True):
        """
        Initialize the PowerSGD compressor.
        
        Args:
            rank: Rank of the low-rank approximation
            num_power_iterations: Number of power iterations for better approximation
            reuse_query: Whether to reuse Q matrices from previous iterations
        """
        self.rank = rank
        self.num_power_iterations = num_power_iterations
        self.reuse_query = reuse_query
        self.previous_q = {}  # Store Q matrices for reuse
        
    def compress(self, tensor: torch.Tensor, key: str = "") -> CompressedUpdate:
        """
        Compress a tensor using PowerSGD.
        
        Args:
            tensor: Input tensor to compress
            key: Unique identifier for the tensor (for Q matrix reuse)
            
        Returns:
            CompressedUpdate containing the low-rank approximation
        """
        original_shape = tensor.shape
        
        # For small tensors or 1D tensors, store directly
        if tensor.numel() < 100 or tensor.dim() == 1:
            return CompressedUpdate(
                p_matrix=tensor,
                q_matrix=torch.eye(1, device=tensor.device),
                original_shape=original_shape
            )
            
        # Reshape tensor to 2D if needed
        if tensor.dim() > 2:
            tensor = tensor.reshape(tensor.size(0), -1)
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
            
        matrix = tensor
        n, m = matrix.shape
        rank = min(n, m, self.rank)
        
        # Initialize or reuse Q
        if not self.reuse_query or key not in self.previous_q:
            q = torch.randn(m, rank, device=tensor.device)
        else:
            q = self.previous_q[key]
            
        # Ensure Q is orthogonal
        q = torch.linalg.qr(q)[0]
        
        # Power iterations for better approximation
        p = None
        for _ in range(self.num_power_iterations):
            # Compute P = M @ Q
            p = torch.matmul(matrix, q)
            # Orthogonalize P
            p = torch.linalg.qr(p)[0]
            # Compute Q = M.T @ P
            q = torch.matmul(matrix.t(), p)
            # Orthogonalize Q
            q = torch.linalg.qr(q)[0]
            
        if self.reuse_query:
            self.previous_q[key] = q.detach()
            
        # Calculate error feedback
        reconstructed = torch.matmul(p, q.t()).reshape(original_shape)
        error = tensor.reshape(original_shape) - reconstructed
            
        return CompressedUpdate(
            p_matrix=p,
            q_matrix=q,
            original_shape=original_shape,
            error_feedback=error
        )
        
    def decompress(self, compressed: CompressedUpdate) -> torch.Tensor:
        """
        Decompress a CompressedUpdate back into a tensor.
        
        Args:
            compressed: CompressedUpdate to decompress
            
        Returns:
            Decompressed tensor
        """
        # Handle small tensors stored directly
        if compressed.q_matrix.size() == (1, 1):
            return compressed.p_matrix.reshape(compressed.original_shape)
            
        # Reconstruct the tensor
        tensor = torch.matmul(compressed.p_matrix, compressed.q_matrix.t())
        tensor = tensor.reshape(compressed.original_shape)
        
        # Add error feedback if available
        if compressed.error_feedback is not None:
            tensor = tensor + compressed.error_feedback
            
        return tensor
        
    def compress_model(self, model_params: Dict[str, torch.Tensor]) -> Dict[str, CompressedUpdate]:
        """
        Compress a model's parameters.
        
        Args:
            model_params: Dictionary of model parameters
            
        Returns:
            Dictionary of compressed updates
        """
        compressed_updates = {}
        for name, param in model_params.items():
            compressed_updates[name] = self.compress(param, key=name)
        return compressed_updates
        
    def decompress_model(self, compressed_updates: Dict[str, CompressedUpdate]) -> Dict[str, torch.Tensor]:
        """
        Decompress a dictionary of compressed updates back into model parameters.
        
        Args:
            compressed_updates: Dictionary of compressed updates
            
        Returns:
            Dictionary of decompressed parameters
        """
        params = {}
        for name, compressed in compressed_updates.items():
            params[name] = self.decompress(compressed)
        return params 