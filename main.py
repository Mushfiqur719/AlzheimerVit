
import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from dataset_loader import AlzheimerDatasetManager
from model_builder import build_model, freeze_layers
from pso_optimizer import PSOOptimizer
from trainer import train_model, Trainer
from visualization import plot_confusion_matrix, plot_ablation_results
from tqdm import tqdm

import platform
import subprocess
import argparse

def load_config(config_path='config.txt'):
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                    config[key] = value
    return config

def parse_args(config):
    parser = argparse.ArgumentParser(description="Alzheimer's ViT Training Pipeline")
    parser.add_argument('--data_dir', type=str, default=config.get('data_dir', 'Alzheimer_s Dataset'))
    parser.add_argument('--epochs', type=int, default=config.get('num_epochs', 25))
    parser.add_argument('--folds', type=int, default=config.get('num_folds', 5))
    parser.add_argument('--particles', type=int, default=config.get('pso_particles', 3))
    parser.add_argument('--pso_iters', type=int, default=config.get('pso_iterations', 2))
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 32))
    parser.add_argument('--lr', type=float, default=config.get('learning_rate', 1e-4))
    parser.add_argument('--weight_decay', type=float, default=config.get('weight_decay', 1e-4))
    parser.add_argument('--dropout', type=float, default=config.get('dropout', 0.1))
    return parser.parse_args()

def run_training_variant(variant_name, dm, device, class_weights, params, epochs=10, folds=3):
    """
    Helper function to run a specific training variant for ablation study.
    """
    print(f"\n>>> Running Variant: {variant_name}")
    fold_gen = dm.get_kfold_loaders(n_splits=folds)
    f1_scores = []
    
    for fold, train_loader, val_loader in fold_gen:
        print(f"  Fold {fold+1}/{folds}")
        model = build_model(num_classes=4, dropout_rate=params.get('dropout', 0.0))
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.get('lr', 1e-4), weight_decay=params.get('weight_decay', 1e-4))
        trainer = Trainer(model, device=device, class_weights=class_weights)
        
        best_fold_f1 = 0.0
        for _ in range(epochs):
            trainer.train_epoch(train_loader, optimizer)
            _, _, f1, _, _, _ = trainer.evaluate(val_loader)
            best_fold_f1 = max(best_fold_f1, f1)
        
        f1_scores.append(best_fold_f1)
    
    avg_f1 = np.mean(f1_scores)
    print(f">>> {variant_name} Average F1: {avg_f1:.4f}")
    return avg_f1

def main():
    # 0. Load Configuration
    config = load_config()
    args = parse_args(config)
    
    # Configuration
    DATA_DIR = args.data_dir
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    print(f"Active Configuration: {vars(args)}")
    
    # Create results directory
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load Data
    dm = AlzheimerDatasetManager(DATA_DIR, batch_size=args.batch_size)
    
    # Calculate class weights for imbalance
    class_weights = dm.get_class_weights()
    print(f"Calculated class weights: {class_weights}")

    # --- Ablation Study ---
    print("\n--- Starting Ablation Study ---")
    ablation_results = {}
    
    # 1. Baseline: No advanced augmentation, default parameters
    dm_baseline = AlzheimerDatasetManager(DATA_DIR, batch_size=args.batch_size, augment=False)
    ablation_results['Baseline'] = run_training_variant("Baseline", dm_baseline, DEVICE, class_weights, {'lr': args.lr, 'weight_decay': args.weight_decay, 'dropout': 0.0}, epochs=min(5, args.epochs), folds=min(3, args.folds))
    
    # 2. Enhanced Augmentation: Added augmentations, default parameters
    ablation_results['Enhanced Aug'] = run_training_variant("Enhanced Aug", dm, DEVICE, class_weights, {'lr': args.lr, 'weight_decay': args.weight_decay, 'dropout': 0.0}, epochs=min(5, args.epochs), folds=min(3, args.folds))
    
    # --- PSO Optimization (Proposed Method) ---
    print("\n--- Starting Hybrid Swin Transformer + PSO Optimization ---")
    
    # Fitness function for PSO (uses enhanced dm)
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
    pso = PSOOptimizer(bounds, num_particles=args.particles, iterations=args.pso_iters)
    best_params, best_f1_score = pso.optimize(fitness_function)
    
    ablation_results['Proposed (PSO)'] = best_f1_score
    
    # Plot Ablation Chart
    plot_ablation_results(ablation_results, filename=os.path.join(RESULTS_DIR, 'ablation_study_chart.png'))
    
    print(f"Best Parameters found: {best_params}")
    print(f"Best Macro F1 during search: {best_f1_score:.4f}")
    
    # 3. Full Training with Best Params using 5-Fold CV
    print(f"\n--- Starting Full {args.folds}-Fold Cross Validation with Optimized Parameters ---")
    
    fold_gen = dm.get_kfold_loaders(n_splits=args.folds)
    fold_results = {'f1': [], 'acc': []}
    best_model_state = None
    best_overall_f1 = 0.0
    
    fold_pbar = tqdm(fold_gen, total=args.folds, desc="K-Fold Cross Validation")
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
        
        best_fold_f1 = 0.0
        patience = 5
        trigger_times = 0
        num_epochs = args.epochs
        
        for epoch in range(num_epochs):
            t_loss, t_acc, t_f1 = trainer.train_epoch(train_loader, optimizer)
            v_loss, v_acc, v_f1, _, _, _ = trainer.evaluate(val_loader)
            
            # Step the scheduler
            scheduler.step()
            
            tqdm.write(f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} - Val F1: {v_f1:.4f} (Best: {max(best_fold_f1, v_f1):.4f})")
            
            if v_f1 > best_fold_f1:
                tqdm.write(f"  [Improvement] Val F1 increased by {v_f1 - best_fold_f1:.4f}")
                best_fold_f1 = v_f1
                trigger_times = 0
                if best_fold_f1 > best_overall_f1:
                    best_overall_f1 = best_fold_f1
                    best_model_state = model.state_dict()
                    # Save the best model so far
                    torch.save(best_model_state, os.path.join(RESULTS_DIR, "best_model.pth"))
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
    
    report_text = f"""--- Alzheimer's ViT Training Report ---
Date: {np.datetime64('now')}
Ablation Study Results: {ablation_results}

PSO Best Parameters: {best_params}
PSO Best Search F1: {best_f1_score:.4f}

K-Fold Results (Macro F1): {fold_results['f1']}
Average CV Macro F1: {np.mean(fold_results['f1']):.4f}

Final Test Results:
Accuracy: {acc:.4f}
Macro F1: {f1:.4f}

Classification Report:
{classification_report(labels, preds, target_names=dm.classes)}
"""
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
        f.write(report_text)
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=dm.classes))
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, dm.classes, filename=os.path.join(RESULTS_DIR, 'final_confusion_matrix.png'))
    
    print(f"Results saved to '{RESULTS_DIR}' directory.")
    print("Pipeline Completed Successfully.")
    
    # Auto-open results folder
    try:
        if platform.system() == "Windows":
            os.startfile(os.path.abspath(RESULTS_DIR))
        elif platform.system() == "Darwin": # macOS
            subprocess.Popen(["open", RESULTS_DIR])
        else: # Linux
            subprocess.Popen(["xdg-open", RESULTS_DIR])
    except Exception as e:
        print(f"Could not open results folder: {e}")

if __name__ == "__main__":
    main()
