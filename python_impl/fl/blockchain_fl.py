import json
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from blockchain.blockchain import Blockchain
from blockchain.block import TensorEncoder

class BlockchainFL:
    def __init__(self, blockchain: Blockchain, min_clients: int = 3):
        self.blockchain = blockchain
        self.min_clients = min_clients
        self.model_updates: Dict[str, Any] = {}
        self.global_model: Optional[Dict[str, Any]] = None

    def submit_model_update(self, client_id: str, model_update: Dict[str, Any], accuracy: float) -> None:
        """Submit a model update from a client to the blockchain."""
        transaction = {
            "from": client_id,
            "to": "global_model",
            "amount": 1.0,  # Using amount as a placeholder for model updates
            "type": "model_update",
            "model_update": model_update,
            "accuracy": accuracy,
            "timestamp": time.time()
        }
        
        self.blockchain.add_transaction(transaction)
        self.model_updates[client_id] = {
            "update": model_update,
            "accuracy": accuracy,
            "timestamp": transaction["timestamp"]
        }

    def aggregate_updates(self) -> Dict[str, Any]:
        """Aggregate model updates from all clients."""
        if len(self.model_updates) < self.min_clients:
            raise ValueError(f"Not enough clients. Need at least {self.min_clients}")

        # Sort updates by accuracy
        sorted_updates = sorted(
            self.model_updates.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True
        )

        # Weight updates by accuracy
        total_accuracy = sum(update[1]["accuracy"] for update in sorted_updates)
        weights = [update[1]["accuracy"] / total_accuracy for update in sorted_updates]

        # Aggregate weighted updates
        aggregated_update = {}
        for key in sorted_updates[0][1]["update"].keys():
            weighted_sum = None
            for (_, update), weight in zip(sorted_updates, weights):
                if weighted_sum is None:
                    weighted_sum = update["update"][key].clone() * weight
                else:
                    weighted_sum += update["update"][key].clone() * weight
            aggregated_update[key] = weighted_sum

        return aggregated_update

    def update_global_model(self, miner_address: str) -> None:
        """Update the global model with aggregated updates and mine a new block."""
        try:
            aggregated_update = self.aggregate_updates()
            
            # Create a transaction for the global model update
            transaction = {
                "from": "global_model",
                "to": "all_clients",
                "amount": 1.0,
                "type": "global_update",
                "model_update": aggregated_update,
                "timestamp": time.time()
            }
            
            self.blockchain.add_transaction(transaction)
            self.blockchain.mine_pending_transactions(miner_address)
            
            # Update the global model
            self.global_model = aggregated_update
            
            # Clear the model updates after successful aggregation
            self.model_updates.clear()
            
        except ValueError as e:
            print(f"Failed to update global model: {str(e)}")

    def get_global_model(self) -> Optional[Dict[str, Any]]:
        """Get the current global model."""
        return self.global_model

    def get_pending_updates_count(self) -> int:
        """Get the number of pending model updates."""
        return len(self.model_updates)

    def save_state(self, filename: str) -> None:
        """Save the current state of the FL system."""
        state = {
            "blockchain": self.blockchain.to_dict(),
            "model_updates": self.model_updates,
            "global_model": self.global_model,
            "min_clients": self.min_clients
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=4, cls=TensorEncoder)

    @classmethod
    def load_state(cls, filename: str) -> 'BlockchainFL':
        """Load the state of the FL system from a file."""
        with open(filename, 'r') as f:
            state = json.load(f)
        
        blockchain = Blockchain.from_dict(state["blockchain"])
        fl_system = cls(blockchain, state["min_clients"])
        fl_system.model_updates = state["model_updates"]
        fl_system.global_model = state["global_model"]
        return fl_system 