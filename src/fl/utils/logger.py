import os
import json
import torch
from datetime import datetime
from typing import Dict, Any

class ExperimentLogger:
    def __init__(self, experiment_name: str = None):
        """Initialize the experiment logger."""
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.log_dir = os.path.join("experiments", experiment_name)
        self.model_dir = os.path.join(self.log_dir, "models")
        
        # Create directories if they don't exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(self.log_dir, "experiment_log.json")
        self.current_log = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "rounds": []
        }
    
    def log_round(self, round_num: int, metrics: Dict[str, Any]):
        """Log metrics for a round."""
        round_log = {
            "round": round_num,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "metrics": metrics
        }
        self.current_log["rounds"].append(round_log)
        
        # Save log after each round
        with open(self.log_file, 'w') as f:
            json.dump(self.current_log, f, indent=4)
    
    def save_model(self, model: torch.nn.Module, round_num: int, accuracy: float):
        """Save model state."""
        model_path = os.path.join(self.model_dir, f"model_round_{round_num}_acc_{accuracy:.2f}.pt")
        torch.save({
            'round': round_num,
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy
        }, model_path)
    
    def get_best_model_path(self) -> str:
        """Get the path to the best model based on accuracy."""
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pt')]
        if not model_files:
            return None
        
        # Extract accuracy from filename
        accuracies = []
        for f in model_files:
            try:
                acc = float(f.split('acc_')[1].split('.pt')[0])
                accuracies.append((f, acc))
            except:
                continue
        
        if not accuracies:
            return None
            
        # Return path to model with highest accuracy
        best_model = max(accuracies, key=lambda x: x[1])
        return os.path.join(self.model_dir, best_model[0]) 