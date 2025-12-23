
import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from dataset_loader import AlzheimerDatasetManager
from model_builder import build_model, freeze_layers
from pso_optimizer import PSOOptimizer
from trainer import train_model, Trainer
from visualization import plot_confusion_matrix
from tqdm import tqdm

def main():
    # Configuration
    DATA_DIR = r"c:/Users/Mushfiqur Rahman/Documents/Projects/AlzheimerVit/Alzheimer_s Dataset"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    dm = AlzheimerDatasetManager(DATA_DIR, batch_size=32)
    
    # Calculate class weights for imbalance
    class_weights = dm.get_class_weights()
    print(f"Calculated class weights: {class_weights}")

    # 2. PSO Optimization
    # Fitness function for PSO
    def fitness_function(params):
        # Unpack params
        lr = params.get('lr', 1e-4)
        weight_decay = params.get('weight_decay', 1e-4)
        dropout = params.get('dropout', 0.0)
        
        # Build model with these params
        model = build_model(num_classes=4, dropout_rate=dropout)
        model = model.to(DEVICE)
        
        # Get just one fold for optimization
        fold_gen = dm.get_kfold_loaders(n_splits=3)
        _, train_loader, val_loader = next(fold_gen)
        
        # Train for few epochs (using weighted loss)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        trainer = Trainer(model, device=DEVICE, class_weights=class_weights)
        
        # PSO search uses a shorter training cycle
        best_f1 = 0.0
        for _ in range(2):
            trainer.train_epoch(train_loader, optimizer)
            _, _, f1, _, _, _ = trainer.evaluate(val_loader)
            best_f1 = max(best_f1, f1)
            
        return best_f1

    print("--- Starting Hybrid Swin Transformer + PSO Optimization ---")
    
    # Define bounds for PSO: LR (more conservative), WeightDecay, Dropout
    bounds = {
        'lr': (1e-5, 1e-4),
        'weight_decay': (1e-5, 1e-3),
        'dropout': (0.0, 0.4)
    }
    
    # Run PSO
    pso = PSOOptimizer(bounds, num_particles=3, iterations=2)
    best_params, best_f1_score = pso.optimize(fitness_function)
    
    print(f"Best Parameters found: {best_params}")
    print(f"Best Macro F1 during search: {best_f1_score:.4f}")
    
    # 3. Full Training with Best Params using 5-Fold CV
    print("\n--- Starting Full 5-Fold Cross Validation with Optimized Parameters ---")
    
    fold_gen = dm.get_kfold_loaders(n_splits=5)
    fold_results = {'f1': [], 'acc': []}
    best_model_state = None
    best_overall_f1 = 0.0
    
    fold_pbar = tqdm(fold_gen, total=5, desc="K-Fold Cross Validation")
    for fold, train_loader, val_loader in fold_pbar:
        tqdm.write(f"\n--- Fold {fold+1}/5 ---")
        model = build_model(num_classes=4, dropout_rate=best_params['dropout'])
        model = model.to(DEVICE)
        
        # Stage 1: Train only the head (Backbone Frozen)
        tqdm.write("Stage 1: Fine-tuning Head (Backbone Frozen)")
        model = freeze_layers(model, freeze_backbone=True)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=best_params['lr'] * 10, weight_decay=best_params['weight_decay'])
        trainer = Trainer(model, device=DEVICE, class_weights=class_weights)
        
        for _ in range(5):
            trainer.train_epoch(train_loader, optimizer)
            
        # Stage 2: Full Fine-tuning
        tqdm.write("Stage 2: Full Fine-tuning")
        model = freeze_layers(model, freeze_backbone=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        
        best_fold_f1 = 0.0
        patience = 5
        trigger_times = 0
        num_epochs = 25
        
        for epoch in range(num_epochs):
            t_loss, t_acc, t_f1 = trainer.train_epoch(train_loader, optimizer)
            v_loss, v_acc, v_f1, _, _, _ = trainer.evaluate(val_loader)
            
            tqdm.write(f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} - Val F1: {v_f1:.4f} (Best: {max(best_fold_f1, v_f1):.4f})")
            
            if v_f1 > best_fold_f1:
                tqdm.write(f"  [Improvement] Val F1 increased by {v_f1 - best_fold_f1:.4f}")
                best_fold_f1 = v_f1
                trigger_times = 0
                if best_fold_f1 > best_overall_f1:
                    best_overall_f1 = best_fold_f1
                    best_model_state = model.state_dict()
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    tqdm.write(f"  [Early Stopping] No improvement for {patience} epochs.")
                    break
                    
        fold_results['f1'].append(best_fold_f1)
        tqdm.write(f"Fold {fold+1} Final Best Val F1: {best_fold_f1:.4f}")
    
    print(f"Average CV Macro F1: {np.mean(fold_results['f1']):.4f}")
    
    # 4. Final Evaluation on Test Set
    print("\n--- Final Evaluation on Test Set ---")
    final_model = build_model(num_classes=4, dropout_rate=best_params['dropout'])
    final_model.load_state_dict(best_model_state)
    final_model = final_model.to(DEVICE)
    
    test_loader = dm.get_test_loader()
    trainer = Trainer(final_model, device=DEVICE)
    loss, acc, f1, labels, preds, probs = trainer.evaluate(test_loader)
    
    print(f"Test Set Accuracy: {acc:.4f}")
    print(f"Test Set Macro F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=dm.classes))
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, dm.classes, filename='final_confusion_matrix.png')
    
    print("Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()
