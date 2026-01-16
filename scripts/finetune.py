import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DynamicsDataset
from src.xresnet1d import xresnet1d50 
from src.utils import save_checkpoint
from hparams import DATA_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- REUSING ROBUST LOSS FROM BASELINE ---
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x = x.float() 
        y = y.float()
        x = torch.clamp(x, min=-20, max=20)
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = -1 * (los_pos * torch.pow(1 - xs_pos, self.gamma_pos) + \
                     los_neg * torch.pow(1 - xs_neg, self.gamma_neg))
        return loss.sum()

def load_pretrained_weights(model, checkpoint_path):
    """
    Loads weights from the Latent Dynamics checkpoint into the Classification ResNet.
    Handles the prefix mismatch ('encoder_backbone.' -> '').
    """
    print(f"Loading pre-trained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # The pre-trained model has keys like 'encoder_backbone.body.0...'
        # The classifier model expects 'body.0...'
        if k.startswith('encoder_backbone.'):
            new_key = k.replace('encoder_backbone.', '')
            new_state_dict[new_key] = v
            
    # Load into model
    # strict=False is CRITICAL here because:
    # 1. We are missing the 'head' weights (classification layer) -> Randomly initialized
    # 2. We have extra keys in state_dict (predictor/projector) -> Ignored due to filtering above
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"   > Weights loaded. Missing keys (expected for head): {len(msg.missing_keys)}")
    print(f"   > Unexpected keys (should be 0): {len(msg.unexpected_keys)}")
    
    return model

def train(args):
    wandb.init(project="ecg-finetuning", config=args, resume="allow", id=args.run_id if args.run_id else None)
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

    # 2. Model Setup
    print("Initializing ResNet1d50...")
    # Initialize standard classifier architecture
    model = xresnet1d50(input_channels=12, num_classes=76)
    
    # Load the Pre-Trained Encoder
    model = load_pretrained_weights(model, args.checkpoint_path)
    model = model.to(DEVICE)

    # 3. Optimization
    # We use the same optimizer settings as baseline to keep comparison fair,
    # or slightly lower LR if we want to preserve features better.
    # Defaulting to 1e-4 as established in baseline stabilization.
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, steps_per_epoch=len(train_loader), epochs=config.epochs)
    criterion = AsymmetricLoss()
    scaler = GradScaler('cuda')

    start_epoch = 0
    best_val_auc = 0.0

    # 4. Training Loop
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            x = batch['waveform'].to(DEVICE).float()
            y = batch['icd'].to(DEVICE)
            
            if torch.isnan(x).any(): continue

            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(x) 
                loss = criterion(logits, y)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if not torch.isnan(loss):
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                x = batch['waveform'].to(DEVICE).float()
                y = batch['icd'].cpu().numpy()
                with autocast('cuda'):
                    logits = model(x)
                
                probs = torch.sigmoid(logits).float().cpu().numpy()
                all_preds.append(probs)
                all_targets.append(y)
                
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Robust Metrics
        valid_class_indices = []
        for i in range(all_targets.shape[1]):
            if len(np.unique(all_targets[:, i])) > 1:
                valid_class_indices.append(i)
        
        if len(valid_class_indices) > 0:
            val_auc = roc_auc_score(
                all_targets[:, valid_class_indices], 
                all_preds[:, valid_class_indices], 
                average='macro'
            )
        else:
            val_auc = 0.0

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val AUROC: {val_auc:.4f}")
        wandb.log({"train/loss": avg_train_loss, "val/auc": val_auc, "epoch": epoch+1})
        
        # Save Best
        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, "checkpoints/finetuned_latest.pth")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # Save as 'finetuned_best' to distinguish from baseline
            torch.save(model.state_dict(), "checkpoints/finetuned_best.pth")
            print(">>> Saved Best Finetuned Model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path to the .pth file from the Dynamics Training (e.g. checkpoints/best_model_online.pth)
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to pre-trained dynamics model")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--run_id', type=str, default=None)
    
    args = parser.parse_args()
    train(args)