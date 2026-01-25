import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DynamicsDataset
from src.xresnet1d import xresnet1d50 
from hparams import DATA_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_bootstrap_auroc(y_true, y_pred, n_bootstraps=1000, seed=42):
    """
    Computes 95% Confidence Interval using bootstrap resampling.
    """
    print(f"\n>>> Running Bootstrap Analysis ({n_bootstraps} iterations)...")
    rng = np.random.RandomState(seed)
    boot_scores = []
    n_samples = y_true.shape[0]
    indices = np.arange(n_samples)
    
    # Pre-calculate which columns were valid in the full set (optimization)
    # We only care about columns that *could* theoretically exist
    globally_valid_cols = [i for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1]
    
    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        # Sample indices with replacement
        boot_idx = resample(indices, replace=True, n_samples=n_samples, random_state=rng)
        
        y_t_boot = y_true[boot_idx]
        y_p_boot = y_pred[boot_idx]
        
        # Check for valid classes in THIS specific sample
        # (Rare classes might disappear in a random sample)
        iteration_valid_cols = []
        for i in globally_valid_cols:
            if len(np.unique(y_t_boot[:, i])) > 1:
                iteration_valid_cols.append(i)
        
        if len(iteration_valid_cols) == 0:
            continue # Should not happen with reasonable batch size
            
        # Calculate Macro AUC for this iteration
        score = roc_auc_score(
            y_t_boot[:, iteration_valid_cols], 
            y_p_boot[:, iteration_valid_cols], 
            average='macro'
        )
        boot_scores.append(score)
    
    boot_scores = np.array(boot_scores)
    
    # Calculate Percentiles
    lower = np.percentile(boot_scores, 2.5)
    upper = np.percentile(boot_scores, 97.5)
    mean_score = np.mean(boot_scores)
    
    return mean_score, lower, upper

def evaluate(args):
    print(f"--- EVALUATION ON TEST SET (Fold 19) ---")
    print(f"Checkpoint: {args.checkpoint_path}")
    
    # 1. Dataset Setup
    print("Loading Test Dataset...")
    test_ds = DynamicsDataset(
        split='test',
        return_pairs=args.return_pairs 
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )

    # 2. Model Setup
    print("Initializing Model...")
    model = xresnet1d50(input_channels=12, num_classes=76)
    
    # Load Weights
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
        
    msg = model.load_state_dict(state_dict, strict=True)
    print(f"Weights Loaded. {msg}")
    
    model = model.to(DEVICE)
    model.eval()

    # 3. Inference Loop
    all_preds = []
    all_targets = []
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x = batch['waveform'].to(DEVICE).float()
            y = batch['icd'].numpy()
            
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.append(probs)
            all_targets.append(y)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    print(f"Total Samples Evaluated: {len(all_targets)}")

    # 4. Point Estimate (Full Set)
    print("Calculating Point Estimate...")
    valid_class_indices = []
    dropped_classes = []
    
    for i in range(all_targets.shape[1]):
        if len(np.unique(all_targets[:, i])) > 1:
            valid_class_indices.append(i)
        else:
            dropped_classes.append(i)
            
    if len(valid_class_indices) == 0:
        print("CRITICAL ERROR: No valid classes found in Test Set.")
        return

    y_true_filtered = all_targets[:, valid_class_indices]
    y_pred_filtered = all_preds[:, valid_class_indices]
    
    point_auc = roc_auc_score(y_true_filtered, y_pred_filtered, average='macro')
    
    # 5. Bootstrap Interval
    mean_boot, lower, upper = compute_bootstrap_auroc(all_targets, all_preds, n_bootstraps=args.n_bootstraps)

    print("-" * 60)
    print(f"TEST RESULT checkpoint {os.path.basename(args.checkpoint_path)}")
    print(f"Split: Fold 19 | Return Pairs (All ECGs): {args.return_pairs}")
    print("-" * 60)
    print(f"Classes Evaluated: {len(valid_class_indices)} / 76")
    print(f"Classes Dropped: {len(dropped_classes)}")
    print("-" * 60)
    print(f"Macro AUROC (Point):     {point_auc:.4f}")
    print(f"Macro AUROC (Bootstrap): {mean_boot:.4f} [{lower:.4f} - {upper:.4f}]")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help="Path to .pth file")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--return_pairs', action='store_true', 
                        help="If set, uses ALL ECGs in test set. If not, uses only first ECG per stay.")
    parser.add_argument('--n_bootstraps', type=int, default=1000, 
                        help="Number of bootstrap iterations for CI")
    
    args = parser.parse_args()
    evaluate(args)