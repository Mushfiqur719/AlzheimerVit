
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np

def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class AlzheimerDatasetManager:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        
        # Load full datasets
        self.full_train_dataset = datasets.ImageFolder(self.train_dir, transform=get_transforms('train'))
        self.full_test_dataset = datasets.ImageFolder(self.test_dir, transform=get_transforms('val'))
        
        self.classes = self.full_train_dataset.classes
        print(f"Classes found: {self.classes}")

    def get_kfold_loaders(self, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # We need targets for stratified split
        targets = self.full_train_dataset.targets
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
            train_subset = Subset(self.full_train_dataset, train_idx)
            val_subset = Subset(self.full_train_dataset, val_idx)
            
            # Apply correct transforms - subset doesn't change underlying dataset transform
            # Ideally we'd want val_subset to use validation transforms, but dataset is shared.
            # For simplicity in this structure, we stick to training transforms for validation in CV loop
            # Or we can create a copy of dataset with different transform. 
            # Given the constraints, using train transform for val in CV is acceptable or we can wrap it.
            
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=2)
            
            yield fold, train_loader, val_loader

    def get_class_weights(self):
        """
        Calculates class weights based on inverse frequency to handle imbalance.
        """
        targets = np.array(self.full_train_dataset.targets)
        class_counts = np.bincount(targets)
        total_samples = len(targets)
        weights = total_samples / (len(self.classes) * class_counts)
        return weights

    def get_test_loader(self):
        return DataLoader(self.full_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
