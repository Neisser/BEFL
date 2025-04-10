import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import shutil
import requests
import tarfile
from tqdm import tqdm

class DatasetDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.cifar_dir = self.data_dir / "CIFAR10"
        self.femnist_dir = self.data_dir / "FEMNIST"
        
    def setup_directories(self):
        """Create necessary directories for datasets"""
        self.cifar_dir.mkdir(parents=True, exist_ok=True)
        self.femnist_dir.mkdir(parents=True, exist_ok=True)
        
    def download_cifar10(self):
        """Download and prepare CIFAR-10 dataset"""
        print("Downloading CIFAR-10 dataset...")
        
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Download training set
        trainset = torchvision.datasets.CIFAR10(
            root=self.cifar_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        # Download test set
        testset = torchvision.datasets.CIFAR10(
            root=self.cifar_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        print("CIFAR-10 dataset downloaded successfully!")
        return trainset, testset
    
    def download_femnist(self):
        """Download and prepare FEMNIST dataset"""
        print("Downloading FEMNIST dataset...")
        
        # FEMNIST download URL
        femnist_url = "https://raw.githubusercontent.com/TalwalkarLab/leaf/master/data/femnist/data/train/all_data_niid_0_keep_0_train_9.json"
        
        # Create FEMNIST directory structure
        train_dir = self.femnist_dir / "train"
        test_dir = self.femnist_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Download FEMNIST data
        try:
            response = requests.get(femnist_url, stream=True)
            response.raise_for_status()
            
            # Save the downloaded file
            with open(train_dir / "all_data_niid_0_keep_0_train_9.json", 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print("FEMNIST dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading FEMNIST dataset: {e}")
            return None
    
    def prepare_datasets(self):
        """Prepare all required datasets"""
        self.setup_directories()
        
        # Download CIFAR-10
        cifar_trainset, cifar_testset = self.download_cifar10()
        
        # Download FEMNIST
        self.download_femnist()
        
        print("\nDataset preparation complete!")
        print(f"CIFAR-10 data location: {self.cifar_dir}")
        print(f"FEMNIST data location: {self.femnist_dir}")

def main():
    downloader = DatasetDownloader()
    downloader.prepare_datasets()

if __name__ == "__main__":
    main() 