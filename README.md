# Alzheimer's Classification with Swin Transformer & PSO

This project implements a state-of-the-art Alzheimer's disease classification pipeline using a Swin Transformer backbone optimized by Particle Swarm Optimization (PSO).

## 🚀 Getting Started

### 1. Dataset Setup
The model is designed to work with the **Alzheimer's Dataset (4 classes)**.
- **Source**: [Kaggle - Alzheimer's Dataset](https://www.kaggle.com/datasets/yasserhessein/dataset-alzheimer)
- **Preparation**:
  1. Download and extract the dataset.
  2. Ensure the following structure:
     ```text
     Alzheimer_s Dataset/
     ├── train/
     │   ├── MildDemented/
     │   ├── ModerateDemented/
     │   ├── NonDemented/
     │   └── VeryMildDemented/
     └── test/
         ├── MildDemented/
         ├── ModerateDemented/
         ├── NonDemented/
         └── VeryMildDemented/
     ```

### 2. Environment Setup
Clone the repository and install the dependencies:
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configuration
Open `main.py` and update the `DATA_DIR` variable to point to your local dataset folder:
```python
# main.py line 15
DATA_DIR = r"path/to/your/Alzheimer_s Dataset"
```

If using the Jupyter Notebook (`Alzheimer_ViT_Complete.ipynb`), update the `DATA_DIR` in the config cell.

## 🛠️ Configuration & Tuning

You can now easily adjust training metrics to check for the best performance using two methods:

### 1. Using `config.txt`
Edit the `config.txt` file in the root directory. This is the easiest way to manage persistent settings:
```text
num_epochs=25        # Increase for better convergence
num_folds=5          # Higher folds = more robust evaluation
pso_particles=3      # More particles = broader search
pso_iterations=2     # More iterations = deeper search
learning_rate=1e-4   # Base learning rate
```

### 2. Using Command Line (CLI)
You can override any setting directly when running the program:
```bash
# Example: Run with 20 epochs and 10 PSO particles
python main.py --epochs 20 --particles 10 --folds 3
```

**Available Flags:**
- `--epochs`: Number of training epochs per fold.
- `--folds`: Number of CV folds.
- `--particles`: Number of PSO particles for optimization.
- `--pso_iters`: Number of PSO search iterations.
- `--lr`: Initial learning rate.
- `--batch_size`: Training batch size.
- `--data_dir`: Override the path to your dataset.

## 📈 Ablation Study & Results
After training, a folder named `results/` will be created (and will automatically open) containing:
- `ablation_study_chart.png`: Visual comparison of Baseline vs. Enhanced vs. PSO methods.
- `report.txt`: Complete summary of the run and best hyperparameters found.
- `best_model.pth`: The highest-performing model weights.

## 📈 Improvement Features
- **PSO Hyperparameter Tuning**: Automatically searches for the best Learning Rate, Weight Decay, and Dropout.
- **Advanced Augmentation**: Includes ColorJitter, RandomAffine, and RandomResizedCrop for better generalization.
- **Learning Rate Scheduling**: Uses Cosine Annealing to refine convergence.