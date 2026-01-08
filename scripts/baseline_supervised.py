import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DynamicsDataset
from src.xresnet1d import xresnet1d50 
from hparams import DATA_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- REUSING ASYMMETRIC LOSS (Copying for standalone execution) ---
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # --- FIX: Force inputs to Float32 to prevent AMP Underflow/NaN ---
        x = x.float() 
        y = y.float()
        
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross Entropy
        # Use a slightly larger epsilon for safety
        los_pos = y * torch.log(xs_pos.clamp(min=1e-6))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-6))
        
        # Asymmetric Focusing
        loss = -1 * (los_pos * torch.pow(1 - xs_pos, self.gamma_pos) + \
                     los_neg * torch.pow(1 - xs_neg, self.gamma_neg))
        
        return loss.sum()

def train(args):
    wandb.init(project="ecg-supervised-baseline", config=args)
    config = wandb.config

    # 1. Dataset (Classification Mode)
    print("Loading Dataset...")
    train_ds = DynamicsDataset(
        split='train',
        return_pairs=False
    )
    val_ds = DynamicsDataset(
        split='val',
        return_pairs=False
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    # 2. Model (Standard ResNet50)
    # We use xresnet1d50 directly with num_classes=76
    print("Initializing ResNet1d50 (Random Weights)...")
    model = xresnet1d50(input_channels=12, num_classes=76).to(DEVICE)
    
    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, steps_per_epoch=len(train_loader), epochs=config.epochs)
    criterion = AsymmetricLoss()
    scaler = GradScaler('cuda')

    # 4. Loop
    best_val_auc = 0.0
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            x = batch['waveform'].to(DEVICE).float()
            y = batch['icd'].to(DEVICE)
            
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(x) # Direct Classification
                loss = criterion(logits, y)
                
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if torch.isnan(loss):
                print("WARNING: NaN loss detected, skipping step")
                continue
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        
        loop = tqdm(val_loader, desc=f"validation")
        with torch.no_grad():
            for batch in loop:
                x = batch['waveform'].to(DEVICE).float()
                y = batch['icd'].cpu().numpy()
                with autocast('cuda'):
                    logits = model(x)
                
                probs = torch.sigmoid(logits).float().cpu().numpy()
                all_preds.append(probs)
                all_targets.append(y)
                
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        try:
            val_auc = roc_auc_score(all_targets, all_preds, average='macro')
        except:
            val_auc = 0.0
            
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val AUROC: {val_auc:.4f}")
        wandb.log({"train/loss": train_loss/len(train_loader), "val/auc": val_auc, "epoch": epoch+1})
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "checkpoints/supervised_best.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--waveform_h5_path', type=str, required=True)
    # parser.add_argument('--label_h5_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    train(args)