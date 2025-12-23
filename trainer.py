
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, device='cuda', class_weights=None):
        self.model = model.to(device)
        self.device = device
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader, optimizer):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            curr_acc = accuracy_score(all_labels, all_preds)
            curr_f1 = f1_score(all_labels, all_preds, average='macro')
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{curr_acc:.4f}", 'f1': f"{curr_f1:.4f}"})
            
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        return epoch_loss, epoch_acc, epoch_f1

    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(loader, desc="Evaluating")
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                curr_acc = accuracy_score(all_labels, all_preds)
                curr_f1 = f1_score(all_labels, all_preds, average='macro')
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{curr_acc:.4f}", 'f1': f"{curr_f1:.4f}"})

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return epoch_loss, epoch_acc, epoch_f1, np.array(all_labels), np.array(all_preds), np.array(all_probs)

def train_model(model, train_loader, val_loader, params, epochs=5, device='cuda'):
    # Extract params
    lr = params.get('lr', 1e-4)
    weight_decay = params.get('weight_decay', 1e-4)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    trainer = Trainer(model, device)
    
    best_f1 = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc, train_f1 = trainer.train_epoch(train_loader, optimizer)
        val_loss, val_acc, val_f1, _, _, _ = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            print(f"Validation F1 improved from {best_f1:.4f} to {val_f1:.4f}")
            best_f1 = val_f1
        else:
            print(f"Validation F1 did not improve from {best_f1:.4f}")
            
    return best_f1
