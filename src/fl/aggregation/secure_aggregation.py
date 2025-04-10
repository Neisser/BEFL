"""Secure aggregation protocol for BEFL."""
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from dataclasses import dataclass
from ..compression import PowerSGDCompressor, CompressedUpdate
from ..validation import MutualInformationValidator

@dataclass
class AggregatedUpdate:
    """Result of secure aggregation."""
    global_model: Dict[str, torch.Tensor]
    momentum: Dict[str, torch.Tensor]
    selected_updates: List[int]

class SecureAggregator:
    """Securely aggregates model updates using mutual information validation."""
    
    def __init__(
        self,
        compressor: PowerSGDCompressor,
        validator: MutualInformationValidator,
        momentum: float = 0.9,
        learning_rate: float = 0.01,
    ):
        """
        Initialize the secure aggregator.
        
        Args:
            compressor: PowerSGD compressor for efficient updates
            validator: Mutual information validator for update validation
            momentum: Momentum coefficient for Nesterov momentum
            learning_rate: Learning rate for model updates
        """
        self.compressor = compressor
        self.validator = validator
        self.beta = momentum  # Using beta to match paper notation
        self.lr = learning_rate
        
    def aggregate(
        self,
        updates: List[CompressedUpdate],
        global_model: torch.nn.Module,
        current_momentum: Dict[str, torch.Tensor],
        input_shape: Tuple[int, ...],
    ) -> AggregatedUpdate:
        """
        Securely aggregate model updates.
        
        Args:
            updates: List of compressed model updates
            global_model: Current global model
            current_momentum: Current momentum state
            input_shape: Shape of input tensors (batch_size, *features)
            
        Returns:
            AggregatedUpdate with new global model, momentum, and selected indices
        """
        try:
            # Decompress updates and create client models
            client_models = []
            for update in updates:
                try:
                    # Decompress update
                    params = self.compressor.decompress(update)
                    
                    # Create client model by applying update to global model
                    client_state = {}
                    for name, param in global_model.state_dict().items():
                        if name in params:
                            client_state[name] = param - params[name]  # Subtract update to get client model
                        else:
                            client_state[name] = param.clone()
                    
                    client_models.append(client_state)
                except Exception as e:
                    print(f"Failed to process update: {e}")
                    continue
            
            if not client_models:
                return AggregatedUpdate(
                    global_model=global_model.state_dict(),
                    momentum=current_momentum,
                    selected_updates=[]
                )
            
            # Validate updates using mutual information
            valid_indices = self.validator.validate_model_updates(
                client_models,
                global_model,
                input_shape
            )
            
            # Select valid updates
            selected_updates = [u for i, u in enumerate(updates) if valid_indices[i]]
            selected_indices = [i for i, v in enumerate(valid_indices) if v]
            
            if not selected_updates:
                return AggregatedUpdate(
                    global_model=global_model.state_dict(),
                    momentum=current_momentum,
                    selected_updates=[]
                )
            
            # Aggregate valid updates
            aggregated_update = {}
            new_momentum = {}
            N = len(selected_updates)
            
            # First decompress all selected updates
            decompressed_updates = []
            for update in selected_updates:
                params = self.compressor.decompress(update)
                decompressed_updates.append(params)
            
            # Average the updates
            for name, param in global_model.state_dict().items():
                if name in decompressed_updates[0]:
                    # Average the updates
                    update_tensor = torch.mean(torch.stack([
                        update[name] for update in decompressed_updates
                    ]), dim=0)
                    
                    # Apply Nesterov momentum
                    new_momentum[name] = self.beta * current_momentum[name] - self.lr * update_tensor
                    
                    # Update global model
                    aggregated_update[name] = param - self.beta * current_momentum[name] + \
                                           (1 + self.beta) * new_momentum[name]
                else:
                    new_momentum[name] = current_momentum[name]
                    aggregated_update[name] = param
            
            return AggregatedUpdate(
                global_model=aggregated_update,
                momentum=new_momentum,
                selected_updates=selected_indices
            )
            
        except Exception as e:
            print(f"Error during aggregation: {e}")
            return AggregatedUpdate(
                global_model=global_model.state_dict(),
                momentum=current_momentum,
                selected_updates=[]
            ) 