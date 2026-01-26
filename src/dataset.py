import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

from hparams import DATA_ROOT

class DynamicsDataset(Dataset):
    def __init__(self, 
                 waveform_h5_path = os.path.join(DATA_ROOT, 'mimic_iv_ecg_waveforms.h5'),
                 label_h5_path = os.path.join(DATA_ROOT, 'mimic_iv_ecg_icd.h5'),
                 metadata_csv_path=os.path.join(DATA_ROOT, 'metadata.csv'),
                 split='train',
                 return_pairs=True,
                 data_fraction=1.0, # NEW: Percentage of patients to keep
                 in_memory=False,   # NEW: Load selected data into RAM
                 seed=42):          # NEW: Deterministic splitting
        """
        Args:
            data_fraction (float): 0.0 < x <= 1.0. Fraction of PATIENTS to use.
            in_memory (bool): If True, loads the filtered subset into RAM.
        """
        self.wave_path = waveform_h5_path
        self.label_path = label_h5_path
        self.return_pairs = return_pairs
        self.in_memory = in_memory
        
        # File handles (initialized lazily if not in_memory)
        self.f_wave = None
        self.f_label = None

        # 1. Load and Filter Metadata
        df = pd.read_csv(metadata_csv_path)

        # Define Splits
        if split == 'train':
            target_folds = list(range(0, 18))
        elif split == 'val':
            target_folds = [18]
        elif split == 'test':
            target_folds = [19]
        elif split == 'all':
            target_folds = list(range(0, 20))
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Filter by Fold
        df_subset = df[df['fold'].isin(target_folds)].copy()
        
        # --- NEW: Data Fraction Logic (Patient-wise) ---
        if data_fraction < 1.0 and split == 'train':
            print(f"   > Subsampling {data_fraction*100}% of patients...")
            unique_subjects = df_subset['subject_id'].unique()
            n_keep = int(len(unique_subjects) * data_fraction)
            
            # Deterministic Shuffle
            rng = np.random.default_rng(seed)
            keep_subjects = rng.choice(unique_subjects, size=n_keep, replace=False)
            
            # Filter DataFrame
            df_subset = df_subset[df_subset['subject_id'].isin(keep_subjects)]
            print(f"   > Reduced from {len(unique_subjects)} to {len(keep_subjects)} patients.")
            
        # --- Baseline Filter for Val/Test ---
        if not return_pairs and split in ['val', 'test']:
            print(f"   > Applying Baseline Filter: Keeping only first ECG per stay")
            df_subset = df_subset[df_subset['ecg_no_within_stay'] == 0]

        # The absolute indices in the HDF5 file
        self.indices = df_subset['h5_index'].values
        total_records = len(self.indices)
        print(f"   > Total Records in {split} split: {total_records}")

        # 2. In-Memory Loading (Optional)
        if self.in_memory:
            print(f"   > Loading {total_records} records into RAM...")
            with h5py.File(self.wave_path, 'r') as fw, h5py.File(self.label_path, 'r') as fl:
                # Optimized read: Sort indices to minimize disk seeks
                sorted_indices = np.sort(self.indices)
                
                # Load Waveforms (Float32 or Float16 depending on file, converted to Float32 tensor)
                # Note: h5py supports list indexing. 
                self.ram_wave = torch.from_numpy(fw['waveforms'][sorted_indices]).float()
                self.ram_label = torch.from_numpy(fl['icd'][sorted_indices]).float()
                
                # Map original H5 index -> Index in RAM tensor
                # This handles the case where indices might not be contiguous
                self.h5_to_ram = {h5_idx: ram_idx for ram_idx, h5_idx in enumerate(sorted_indices)}
                
            print(f"   > RAM Load Complete. Shape: {self.ram_wave.shape}")
            
        # 3. Calculate Valid Indices (Pairs or Singles)
        if self.return_pairs:
            # We need to find pairs (i, i+1) that are:
            # A) Both present in our df_subset (handled by filtering)
            # B) From the same patient
            
            # Since df_subset is sorted by (Subject, Time), we can check subject continuity
            # We work with the numpy arrays from the dataframe for speed
            subjs = df_subset['subject_id'].values
            h5_idxs = df_subset['h5_index'].values
            
            # Check: Subject[i] == Subject[i+1]
            # AND: H5_Index[i+1] == H5_Index[i] + 1 (Ensures they are physically adjacent in file)
            # The second check is implicitly true if sorted, but crucial if we filtered weirdly.
            # Actually, we rely on physical adjacency for the optimized slice read [i:i+2]
            
            same_patient = (subjs[:-1] == subjs[1:])
            contiguous = (h5_idxs[1:] == h5_idxs[:-1] + 1)
            
            valid_mask = same_patient & contiguous
            
            # These are indices relative to the DataFrame (0..len(df))
            self.valid_df_indices = np.where(valid_mask)[0]
            
            # Balancing Logic (Stable vs Changed)
            # We need to peek at labels. 
            # If in_memory, use RAM. If not, use file.
            if self.in_memory:
                # These are Tensors because ram_label is a Tensor
                curr_labels = self.ram_label[self.valid_df_indices]
                next_labels = self.ram_label[self.valid_df_indices + 1]
                
                # FIX: Convert to numpy for the check to avoid "axis" vs "dim" error
                if isinstance(curr_labels, torch.Tensor):
                    curr_labels = curr_labels.numpy()
                    next_labels = next_labels.numpy()
            else:
                print("   > Scanning labels from disk for balancing...")
                with h5py.File(self.label_path, 'r') as fl:
                    # Bulk read filtered labels to minimize disk seeks
                    # Note: We must be careful not to blow up RAM if indices are scattered
                    # But for 10% data, it's fine. 
                    # For safety, let's read only what we need if possible, 
                    # but h5py doesn't support fancy indexing efficiently. 
                    # We will rely on the fact that self.indices is sorted-ish.
                    
                    # Workaround: Read the full label set for the subset if it fits
                    # or read in a loop if memory is tight. 
                    # Given 'icd' is int8 and subset is small, bulk read is fine.
                    all_labels_subset = fl['icd'][self.indices]
                
                curr_labels = all_labels_subset[self.valid_df_indices]
                next_labels = all_labels_subset[self.valid_df_indices + 1]

            is_stable = np.all(curr_labels == next_labels, axis=1)
            
            # Map back to valid_df_indices
            self.stable_indices = self.valid_df_indices[is_stable]
            self.changed_indices = self.valid_df_indices[~is_stable]
            
            # The main list of indices we will iterate over
            self.valid_iterable = self.valid_df_indices
            
            print(f"   > Valid Pairs: {len(self.valid_iterable)}")
            print(f"   > Stable: {len(self.stable_indices)} | Changed: {len(self.changed_indices)}")
            
        else:
            # Classification Mode
            # We iterate over all records in the subset
            self.valid_iterable = np.arange(len(self.indices))
            print(f"   > Total Samples: {len(self.valid_iterable)}")

    def _open_files(self):
        if self.f_wave is None:
            self.f_wave = h5py.File(self.wave_path, 'r', rdcc_nbytes=1024*1024*4)
        if self.f_label is None:
            self.f_label = h5py.File(self.label_path, 'r')

    def __len__(self):
        return len(self.valid_iterable)

    def __getitem__(self, idx):
        # Index relative to the DataFrame subset
        df_idx = self.valid_iterable[idx]
        
        # 1. Pre-training Mode (Pairs)
        if self.return_pairs:
            if self.in_memory:
                # Direct RAM access
                x_t = self.ram_wave[df_idx]
                x_next = self.ram_wave[df_idx + 1]
                y_t = self.ram_label[df_idx].long()
                y_next = self.ram_label[df_idx + 1].long()
            else:
                self._open_files()
                # Get absolute H5 index
                real_h5_idx = self.indices[df_idx]
                
                # Optimized Slice Read [i : i+2]
                # We validated they are contiguous in __init__
                x_pair = torch.from_numpy(self.f_wave['waveforms'][real_h5_idx : real_h5_idx + 2])
                y_pair = torch.from_numpy(self.f_label['icd'][real_h5_idx : real_h5_idx + 2]).long()
                
                x_t, x_next = x_pair[0], x_pair[1]
                y_t, y_next = y_pair[0], y_pair[1]

            # Process
            x_t = x_t.transpose(0, 1)
            x_next = x_next.transpose(0, 1)
            action = (y_next - y_t).float()
            
            return {
                'waveform': x_t, 
                'waveform_next': x_next, 
                'action': action, 
                'icd': y_t.float() # For online probing
            }

        # 2. Classification Mode (Single)
        else:
            if self.in_memory:
                x_t = self.ram_wave[df_idx]
                y_t = self.ram_label[df_idx].long()
            else:
                self._open_files()
                real_h5_idx = self.indices[df_idx]
                x_t = torch.from_numpy(self.f_wave['waveforms'][real_h5_idx])
                y_t = torch.from_numpy(self.f_label['icd'][real_h5_idx]).long()
            
            x_t = x_t.transpose(0, 1)
            
            return {'waveform': x_t, 'icd': y_t.float()}

    def get_weights_for_balanced_sampling(self):
        if not self.return_pairs: return None
        weights = np.zeros(len(self.valid_iterable))
        n_stable, n_changed = len(self.stable_indices), len(self.changed_indices)
        if n_stable == 0 or n_changed == 0: return None
        
        w_stable, w_changed = 1.0 / n_stable, 1.0 / n_changed
        
        # valid_iterable contains the df_indices. 
        # stable_indices contains specific df_indices.
        # We need to map df_index -> index in weights array
        df_idx_to_weight_idx = {val: i for i, val in enumerate(self.valid_iterable)}
        
        for idx in self.stable_indices: 
            if idx in df_idx_to_weight_idx:
                weights[df_idx_to_weight_idx[idx]] = w_stable
        for idx in self.changed_indices:
            if idx in df_idx_to_weight_idx:
                weights[df_idx_to_weight_idx[idx]] = w_changed
                
        return torch.DoubleTensor(weights)