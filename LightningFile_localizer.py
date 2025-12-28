import os
import torch
# torch.set_float32_matmul_precision('medium') 
# from Loss import Loss
import torch.nn as nn
import torchaudio
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningModule
from Datasets import TrajDatasets as Datasets
# from AudioFeature_TC import InputFeature
from AudioFeature_Modular import InputFeature
from collections import OrderedDict, defaultdict
import pandas as pd
import scipy
import numpy as np
from asteroid.losses import pairwise_neg_sisdr,PITLossWrapper,PairwiseNegSDR,pairwise_neg_snr
from torchmetrics import ScaleInvariantSignalNoiseRatio
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio,signal_distortion_ratio,scale_invariant_signal_noise_ratio,signal_noise_ratio
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torch.optim.lr_scheduler import ExponentialLR

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import DataLoader,DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.loss_utils import permutation_loss, snr_loss
from models import SeparationNet
from scipy.io.wavfile import write
import wandb
from torch.utils.data import Subset
import h5py
from models import EnhancementNet
from separation_handler import SeparationHandler

class Lightning(LightningModule):
    def __init__(self, config, TrajectoryNet,
                feature_dim = 64,
                hidden_dim = 256, 
                num_block = 5,
                num_layer = 7,
                kernel_size = 3,
                stft_win = 512,
                stft_hop = 32,
                num_cls = 37,
                 lr=1e-3):
        super(Lightning, self).__init__()
        # ------------------Dataset&DataLoader Parameter-----------------

        self.batch_size = config["batch_size"]
        self.learning_rate = lr
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.features = InputFeature() 
        self.num_cls=num_cls

        # -----------------------model-----------------------
        #enhancement_model
        self.model = TrajectoryNet(
            feature_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"],
            num_block=config["num_block"],
            num_layer=config["num_layer"],
            kernel_size=config["kernel_size"],
            num_cls=num_cls
        )

    def forward(self, x):  # forward(self, x,spatial)
        return self.model(x)




    # ---------------------
    # TRAINING STEP
    # ---------------------

    # def training_step(self, batch, batch_idx):
    #     enh_sound = batch["enh_sound"]          # (B, 2, L)
    #     ref_traj = batch["trajectory"]        # (B, L_traj)
    #     ref_traj = ref_traj.float()

    #     # Forward pass → model should return (B, T, num_cls)
    #     batch_output = self.forward(enh_sound)  # (B, T_out, num_cls)

    #     # Prepare reference trajectory in shape (B, L_traj)
    #     # ref_traj =self.features.prepare_traj_reference(trajectory, batch_output.size(-2))  # Ensure type: torch.Tensor

    #     # Downsample trajectory to match T_out
    #     down_sampler = nn.Upsample(scale_factor=batch_output.size(-2) / ref_traj.size(-1), mode='nearest')
    #     batch_traj_down = down_sampler(ref_traj.unsqueeze(1)).squeeze(1)  # shape: (B, T_out)

    #     # Flatten for cross-entropy: inputs: (B*T, num_cls), targets: (B*T,)
    #     est_label = batch_output.reshape(-1, batch_output.size(-1))        # (B*T, num_cls)
    #     tgt_label = batch_traj_down.reshape(-1).long()                     # (B*T,)

    #     # Loss and error
    #     loss = self.criterion(est_label, tgt_label)                        # CrossEntropyLoss
    #     pred = torch.argmax(est_label, dim=1)
    #     err = torch.mean((pred != tgt_label).float())

    #     self.log("train_loss", loss, on_step=False, on_epoch=True,sync_dist=True)
    #     self.log("train_error", err, on_step=False, on_epoch=True,sync_dist=True)

    #     return loss

    def training_step(self, batch, batch_idx):
        enh_sound = batch["enh_sound"]      # (B, 2, L)
        ref_traj   = batch["trajectory"]    # (B, T_out)

        # forward
        out = self(enh_sound)               # (B, T_out, num_cls)
        logits = out.permute(0, 2, 1)       # (B, num_cls, T_out)

        # loss (with ignore_index already set on self.criterion)
        loss = self.criterion(logits, ref_traj)

        # error
        pred = logits.argmax(dim=1)         # (B, T_out)
        mask = ref_traj != self.criterion.ignore_index
        err  = (pred[mask] != ref_traj[mask]).float().mean()

        # log
        self.log("train_loss",  loss, on_step=False,on_epoch=True, sync_dist=True)
        self.log("train_error", err, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    # ---------------------
    # VALIDATION SETUP
    # ---------------------


    def validation_step(self, batch, batch_idx):
        enh_sound = batch["enh_sound"]    # (B, 2, L_audio)
        ref_traj  = batch["trajectory"]   # (B, T_out)

        # 1) forward
        out    = self(enh_sound)             # (B, T_out, num_cls)
        logits = out.permute(0, 2, 1)        # (B, num_cls, T_out)

        # 2) loss
        loss = self.criterion(logits, ref_traj)

        # 3) frame-wise error rate
        pred_idx = logits.argmax(dim=1)      # (B, T_out)
        mask     = ref_traj != self.criterion.ignore_index
        err      = (pred_idx[mask] != ref_traj[mask]).float().mean()

        # 4) mean absolute DOA error (in degrees)
        # MAE
        pred_ang = pred_idx.float() * 5.0 - 90.0 + 2.5
        true_ang = ref_traj.float()  * 5.0 - 90.0 + 2.5
        mae_deg = (pred_ang[mask] - true_ang[mask]).abs().mean()

        # 5) logging
        self.log("val_loss",    loss,    on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_error",   err,     on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_mae_deg", mae_deg, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        enh_sound   = batch["enh_sound"]      # (B, 2, L_audio)
        ref_sound   = batch["ref_sound"]      # (B, 2, L_audio)
        ref_traj    = batch["trajectory"]     # (B, T_out)
        h5_paths    = batch["h5_path"]        # list of str, len B
        # mix_keys    = batch["mix_key"]        # list of str, len B
        ref_keys    = batch["ref_key"]        # list of str, len B
        conditions  = batch["condition"]      # list of str, len B
        time_points = batch["time_points"]    # list of lists, len B
        sdr_enh     = batch["sdr_enh"]        # tensor or list, len B

        # 1) forward through localizer for enhanced and clean
        out         = self(enh_sound)         # (B, T_out, num_cls) The predicted trajectory representing the 37 classes
        out_clean   = self(ref_sound)         # (B, T_out, num_cls)
        logits      = out.permute(0, 2, 1)    # (B, num_cls, T_out)
        logits_clean= out_clean.permute(0, 2, 1)

        enhancer_rows = []
        B = ref_traj.size(0)
        all_true = []
        all_pred = []
        for i in range(B):
            # 2) exact‐match error on enhanced
            pred_idx      = logits[i].argmax(dim=0)       # (T_out,)
            mask          = ref_traj[i] != self.criterion.ignore_index
            err           = (pred_idx[mask] != ref_traj[i][mask]).float().mean()

            # 3) MAE on enhanced
            pred_ang      = pred_idx.float() * 5.0 - 90.0 + 2.5
            true_ang      = ref_traj[i].float() * 5.0 - 90.0 + 2.5
            mask = ref_traj[i] != self.criterion.ignore_index
            true_ang_masked = true_ang[mask]
            doa_clean = true_ang_masked.mean()             # Mean DOA (clean/reference) across time
            mae_deg       = (pred_ang[mask] - true_ang[mask]).abs().mean()

            # 4) error & MAE on clean reference
            pred_idx_c    = logits_clean[i].argmax(dim=0)
            mask_c        = mask  # same mask
            err_c         = (pred_idx_c[mask_c] != ref_traj[i][mask_c]).float().mean()
            pred_ang_c    = pred_idx_c.float() * 5.0 - 90.0 + 2.5
            mae_deg_c     = (pred_ang_c[mask_c] - true_ang[mask_c]).abs().mean()

            #For confusion Matrix
            valid_indices = mask.nonzero(as_tuple=True)[0]
            all_true.extend(ref_traj[i][valid_indices].cpu().tolist())
            all_pred.extend(pred_idx[valid_indices].cpu().tolist())

            enhancer_rows.append({
                "h5_path":      h5_paths[i],
                # "mix_key":      mix_keys[i],
                "ref_key":      ref_keys[i],
                "condition":    conditions[i],
                "trajectory_ref":   ref_traj[i].tolist(),
                "trajectory_est": logits[i].argmax(dim=0).tolist(),
                "time_points":  time_points[i],
                "confusion_true": all_true,
                "confusion_pred": all_pred,
                "sdr_enh":      sdr_enh[i],
                "doa":          float(mae_deg),
                "doa_clean":    float(mae_deg_c),
                "true_doa": float(doa_clean),
                "error":        float(err),
                "error_clean":  float(err_c),
            })

        # 5) Save metadata CSV for this batch
        df_enhancer = pd.DataFrame(enhancer_rows)
        csv_dir     = "Data/enhancementCSV_class_train"
        os.makedirs(csv_dir, exist_ok=True)
        csv_path    = os.path.join(csv_dir, f"localizer_test_batch{batch_idx}.csv")
        df_enhancer.to_csv(csv_path, index=False)

        # 6) Log the batch‐level averages if desired
        print(f"test_mae_deg: {sum(r['doa'] for r in enhancer_rows)/B}")
        print(f"test_mae_deg_c: {sum(r['doa_clean'] for r in enhancer_rows)/B}")
        print(f"test_error: {sum(r['error'] for r in enhancer_rows)/B,}")
        print(f"test_error_c: {sum(r['error_clean'] for r in enhancer_rows)/B}")


            


    # def train_collate_fn(self,batch):
    #     # 1) pad audio
    #     sounds   = [b["enh_sound"] for b in batch]  # each: (2, L_i)

    #     lengths  = [s.shape[-1]       for s in sounds]
    #     L_max    = max(lengths)
    #     padded_s = torch.stack([
    #         torch.nn.functional.pad(s, (0, L_max - s.shape[-1]))
    #         for s in sounds
    #     ], dim=0)                                  # (B, 2, L_max)

    #     # 2) pad trajectories
    #     trajs = [b["traj"] for b in batch]         # each: (T_i,)
    #     padded_t = torch.nn.utils.rnn.pad_sequence(
    #         trajs, 
    #         batch_first=True, 
    #         padding_value=-100
    #     )                                          # (B, T_max)

    #     return {
    #         "enh_sound": padded_s,
    #         "trajectory": padded_t
    #     }

    def train_collate_fn(self,batch):
        sounds     = [b["enh_sound"] for b in batch]
        ref_sounds = [b["ref_sound"] for b in batch]
        Ls         = [s.shape[-1] for s in sounds]
        L_max      = max(Ls)

        padded_s    = torch.stack([ torch.nn.functional.pad(s,    (0, L_max-s.shape[-1])) 
                                    for s in sounds ], dim=0)
        padded_ref  = torch.stack([ torch.nn.functional.pad(r,    (0, L_max-r.shape[-1])) 
                                    for r in ref_sounds ], dim=0)

        trajs       = [b["trajectory"] for b in batch]
        padded_traj = torch.stack(trajs, dim=0)
        # padded_traj = torch.nn.utils.rnn.pad_sequence(
        #                   trajs, batch_first=True, padding_value=-100
        #               )

        return {
            "enh_sound": padded_s,
            "ref_sound": padded_ref,
            "trajectory": padded_traj,
            "h5_path":   [b["h5_path"]   for b in batch],
            # "mix_key":   [b["mix_key"]   for b in batch],
            "ref_key":   [b["ref_key"]   for b in batch],
            "condition": [b["condition"] for b in batch],
            "time_points":[b["time_points"] for b in batch],
            "sdr_enh":   [b["sdr_enh"]   for b in batch],
        }

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    # def setup(self, stage):
    #     torch.manual_seed(20230222)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["lr"], weight_decay=1e-5)#, weight_decay=1e-5
        scheduler = {
            'scheduler': ExponentialLR(optimizer, gamma=0.98),
            'interval': 'epoch',
            'frequency': 5# Update LR every 2 epochs for regulra, 5 for finetuning
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"} #"lr_scheduler": scheduler,


    def train_dataloader(self):
        df_train = pd.read_csv(self.config["training_file_path"])
        train_ds  = Datasets(df_train)
        sampler   = DistributedSampler(train_ds, shuffle=True, drop_last=True)

        return torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            num_workers=8,
            sampler=sampler,
            pin_memory=False,
            persistent_workers=True,
            drop_last=True,
            collate_fn=self.train_collate_fn
        )
    def val_dataloader(self):
        df_val = pd.read_csv(self.config["validation_file_path"])#Val-10k.csv ../Data/WSJ2Mix-2spk/
        # df_val = df_val[:4]
        val_ds = Datasets(df_val)
        sampler = DistributedSampler(val_ds,shuffle=False, drop_last=True)

        return torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.batch_size,
            num_workers=8,
            sampler=sampler,
            shuffle=False,
            pin_memory=False,
            persistent_workers=True,
            drop_last=True,
            collate_fn=self.train_collate_fn
        )

    def test_dataloader(self):
        df_test = pd.read_csv(self.config["test_file_path"])
        test_ds = Datasets(df_test)
        # sampler = DistributedSampler(test_ds, shuffle=False, drop_last=False)

        return DataLoader(
            test_ds,
            batch_size=self.batch_size,
            # sampler=sampler,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            # persistent_workers=True,
            collate_fn= self.train_collate_fn
        )
