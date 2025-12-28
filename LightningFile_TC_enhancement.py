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
from Datasets import EnhDatasets
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
import time
from torch.utils.data import DataLoader,DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.loss_utils import permutation_loss, snr_loss
from scipy.io.wavfile import write
import wandb
from torch.utils.data import Subset
import h5py
import tracemalloc
import torch.nn.utils.prune as prune



class Lightning(LightningModule):
    def __init__(self, config, EnhancementNet,
                enc_dim = 64,
                feature_dim = 64,
                hidden_dim = 256, 
                enc_win = 64,
                enc_stride = 32,
                num_block = 5,
                num_layer = 7,
                kernel_size = 3,
                num_spk = 1,
                batch_size=6,
                lr=1e-3):
        super(Lightning, self).__init__()
        # ------------------Dataset&DataLoader Parameter-----------------

        self.batch_size = config["batch_size"]
        self.learning_rate = lr
        self.config = config
        self.num_spk = num_spk


        # -----------------------model-----------------------
        self.model = EnhancementNet(
            enc_dim=config['enc_dim'],
            feature_dim=config['feature_dim'],
            hidden_dim=config['hidden_dim'],
            enc_win=config['enc_win'],
            enc_stride=config['enc_stride'],
            num_block=config['num_block'],
            num_layer=config['num_layer'],
            kernel_size=config['kernel_size'],
            num_spk=self.num_spk,
        )

    def forward(self, x,sep):  # forward(self, x,spatial)
        return self.model(x,sep)

    # ---------------------
    # TRAINING STEP
    # ---------------------
    def training_step(self, batch, batch_idx):
        # df = batch['df']
        sep_sound = batch['sep_spk'].cuda()
        mix_sound = batch['mix_spk'].cuda()
        ref_sound  = batch['ref_spk'].cuda()

        batch_output = self.forward(mix_sound,sep_sound)

        loss = torch.mean(snr_loss(ref_sound.view(-1, ref_sound.size(-1)), batch_output.view(-1, ref_sound.size(-1))))
 
        self.log("train_loss", loss, on_step=False, on_epoch=True,sync_dist=True)

        return loss

    # ---------------------
    # VALIDATION SETUP
    # ---------------------
    def validation_step(self, batch, batch_idx):
        sep_sound = batch['sep_spk'].cuda()
        mix_sound = batch['mix_spk'].cuda()
        ref_sound  = batch['ref_spk'].cuda()

        batch_output = self.forward(mix_sound,sep_sound)

        loss = torch.mean(snr_loss( ref_sound.view(-1, ref_sound.size(-1)), batch_output.view(-1, ref_sound.size(-1))))

        self.log("val_loss", loss, on_step=False, on_epoch=True,sync_dist=True)

        return {'val_loss': loss}


    def test_step(self, batch, batch_idx):
        """
        To get data to train localizer, 
        """
        def normalize_audio(signal):
            return signal / (torch.norm(signal) + 1e-8)

        sr = 16000
        sep_sound = batch['sep_spk'].cuda()
        mix_sound = batch['mix_spk'].cuda()
        ref_sound = batch['ref_spk'].cuda()
        trajectory = batch['trajectory']
        trajectory_processed = batch['trajectory_processed']
        time_points = batch['time_points']
        condition = batch['condition']
        h5_path = batch['h5_path']
        snr = batch['snr']
        snr_up = batch['snr_up']

        est_sound = self.model(mix_sound, sep_sound)

        # HDF5 and metadata output setup
        h5_path = f"Data/enhanced_Adult-noBab-Val/hdf5_audio_batch{batch_idx}.h5"
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        h5f = h5py.File(h5_path, 'w')
        enhancer_rows = []

        counter = batch_idx * len(est_sound)

        for i in range(len(est_sound)):
            ref = normalize_audio(ref_sound[i])
            est = normalize_audio(est_sound[i])
            sep = normalize_audio(sep_sound[i])
            mix = normalize_audio(mix_sound[i])

            sdr_enh = torch.mean(signal_noise_ratio(est, ref, sr)).item()
            sdr_sep = torch.mean(signal_noise_ratio(sep, ref, sr)).item()
            sdr_up = torch.mean(signal_noise_ratio(mix, ref, sr)).item()

            print(f"sdr unprocessed: {sdr_up:.2f}, sdr separation: {sdr_sep:.2f}, sdr enhancement: {sdr_enh:.2f} \n")

            est_np = est_sound[i].detach().cpu().numpy()
            ref_np = ref_sound[i].detach().cpu().numpy()

            ref_key = f"ref{i}/{counter}"
            est_key = f"est{i}/{counter}"

            h5f.create_dataset(ref_key, data=ref_np, compression="gzip")
            h5f.create_dataset(est_key, data=est_np, compression="gzip")

            enhancer_rows.append({
                "h5_path": h5_path,
                "ref_key": ref_key,
                "est_key": est_key,
                "spk_index": i,
                "condition": condition[i],
                "trajectory": trajectory[i],
                "trajectory_processed": trajectory_processed[i],
                "time_points": time_points[i],
                "snr_up": sdr_up,
                "snr_sep": sdr_sep,
                "sdr_enh": sdr_enh
            })

            counter += 1

        h5f.close()

        # Save metadata CSV
        df_enhancer = pd.DataFrame(enhancer_rows)
        csv_path = f"Data/enhancementCSV_class_train/enhancement_test_batch{batch_idx}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_enhancer.to_csv(csv_path, index=False)



    # def test_step(self, batch, batch_idx):
    #     import psutil
    #     process = psutil.Process()

    #     # Timing and memory before inference
    #     torch.cuda.empty_cache()
        
        

    #     mix_sound = batch["mix"]
    #     spk1_sound = batch["spk1"]
    #     spk2_sound = batch["spk2"]

    #     start_time = time.perf_counter()
    #     mem_before = process.memory_info().rss
    #     # Forward pass
    #     left_output, right_output = self.forward(mix_sound)

    #     # Timing and memory after inference
    #     latency = time.perf_counter() - start_time
    #     mem_after = process.memory_info().rss
    #     memory_used = (mem_after - mem_before) / 1e6  # Convert to MB

    #     latency_per_sample = latency / len(left_output)
    #     self.inference_times.append(latency_per_sample)
    #     self.memory_usages.append(memory_used/ len(left_output))

    #     self.log("test/latency", latency, on_step=True, on_epoch=True, prog_bar=False)
    #     self.log("test/memory", memory_used, on_step=True, on_epoch=True, prog_bar=False)
    #     counter = batch_idx * len(left_output) 
    #     sr = 16000
    #     enhancer_rows = []

    #     df_test = pd.DataFrame(columns = ["ref path","est path","mix path","trajectory 1","trajectory 2","time_points","snr","snr_up","condition","starting distance","distance_from_listener"])

    #     loss_sdr_ =  PITLossWrapper(PairwiseNegSDR('snr'), pit_from='pw_mtx')
    #     total_avg_unp = 0
    #     total_avg_p = 0
    #     mix = mix_sound
    #     for i in range(len(left_output)):
    #         est_1 = torch.vstack((left_output[i][0],right_output[i][0]))
    #         est_2 = torch.vstack((left_output[i][1],right_output[i][1]))
            
    #         ests = [est_1,est_2]
    #         refs = [spk1_sound[i],spk2_sound[i]]

    #         # print(f"Shapes are: before reshaping, len of list: {len(ests),len(refs)}, shape of each: {est_1.shape,spk1_sound[i].shape }")

    #         ests_tensor = torch.stack(ests).unsqueeze(0)  
    #         ref_signals_tensor = torch.stack(refs).unsqueeze(0)

    #         # print(f"\n after unsqueeze: {ests_tensor.shape, ref_signals_tensor.shape}")

    #         ests_tensor_flat = ests_tensor.reshape(1, 2, -1) 
    #         ref_signals_tensor_flat = ref_signals_tensor.reshape(1, 2, -1)

    #         # print(f"\n after flat: {ests_tensor_flat.shape, ref_signals_tensor_flat.shape}")

    #         loss_sisdr, reordered_sources_flat = loss_sdr_(ests_tensor_flat, ref_signals_tensor_flat, return_est=True)

    #         # Reshape reordered sources back to [n_sources=2, n_channels=2, T]
    #         reordered_sources = reordered_sources_flat.reshape(1, 2, 2, -1)  # shape: [1, 2, 2, T]

    #         est_1_ = reordered_sources[:, 0, :, :]
    #         est_2_ = reordered_sources[:, 1, :, :]
      

    #         sdr_s1_up = signal_noise_ratio(mix[i], refs[0],sr)
    #         sdr_s2_up = signal_noise_ratio(mix[i], refs[1],sr)
    #         avg_sdr_up = np.mean([torch.mean(sdr_s1_up).item(),torch.mean(sdr_s2_up).item()])

    #         total_avg_unp +=avg_sdr_up

    #         sdr_s1 = signal_noise_ratio(est_1_[0], refs[0],sr)
    #         sdr_s2 = signal_noise_ratio(est_2_[0], refs[1],sr)
    #         avg_sdr = np.mean([torch.mean(sdr_s1).item(),torch.mean(sdr_s2).item()]) 

    #         total_avg_p += avg_sdr

    #         print(f"unprocessed: {total_avg_unp}, processed: {avg_sdr}")


    #     print(f"Average un processed is: {total_avg_unp/len(left_output)}, processed is : {total_avg_p/len(left_output)}\n")
    #     self.sdr_unp.append(total_avg_unp/len(left_output))
    #     self.sdr_p.append(total_avg_p/len(left_output))
    #     # print(f"latency used: {latency}, memory used : {memory_used}\n")


    # def on_test_end(self):
    #     avg_latency = sum(self.inference_times) / len(self.inference_times)
    #     avg_memory = sum(self.memory_usages) / len(self.memory_usages)

    #     print(f"\n✅ Average Latency: {avg_latency:.4f} seconds per sample")
    #     print(f"✅ Average Peak Memory Usage: {avg_memory:.2f} MB")
    #     print(f"✅ sdr unprocessed: {sum(self.sdr_unp) / len(self.sdr_unp):.2f} DB,processed: {sum(self.sdr_p) / len(self.sdr_p):.2f} ")
 


    # ---------------------
    # TRAINING SETUP
    # ---------------------
    # def setup(self, stage):
    #     torch.manual_seed(20230222)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])#, weight_decay=1e-5
        scheduler = {
            'scheduler': ExponentialLR(optimizer, gamma=0.98),
            'interval': 'epoch',
            'frequency': 5# Update LR every 2 epochs for regulra, 5 for finetuning
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}



    def train_dataloader(self):
        df_train = pd.read_csv(self.config["training_file_path"])#Train-20k.csvTrain-30sec Train-8k-posSNR.csv
        # df_test = df_test[:8]
        train_ds = EnhDatasets(df_train)
        sampler = DistributedSampler(train_ds,shuffle=True, drop_last=True)#sampler=sampler, 
        train_dl = DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            num_workers=8, 
            drop_last=True,
            pin_memory=True, 
            sampler=sampler,
            collate_fn=lambda batch: {  # Fixed syntax error here
                "mix_spk": torch.stack([b["mix_spk"] for b in batch], dim=0),
                "sep_spk": torch.stack([b["sep_spk"] for b in batch], dim=0).squeeze(1),
                "ref_spk": torch.stack([b["ref_spk"] for b in batch], dim=0).squeeze(1)   # Keep metadata in list
            }
        )
        return train_dl
    def val_dataloader(self):
        df_val = pd.read_csv(self.config["validation_file_path"])#Val-10k.csv ../Data/WSJ2Mix-2spk/
        # df_val = df_val[:4]
        val_ds = EnhDatasets(df_val)
        sampler = DistributedSampler(val_ds,shuffle=False, drop_last=True)
        val_dl = torch.utils.data.DataLoader(val_ds,batch_size=self.batch_size, sampler=sampler, shuffle=False,num_workers=8,drop_last=True,pin_memory=True,
            collate_fn=lambda batch: { 
                "mix_spk": torch.stack([b["mix_spk"] for b in batch], dim=0),
                "sep_spk": torch.stack([b["sep_spk"] for b in batch], dim=0).squeeze(1) ,
                "ref_spk": torch.stack([b["ref_spk"] for b in batch], dim=0).squeeze(1)
            })#sampler=sampler,
        return val_dl
    # def test_dataloader(self):
    #     df_test = pd.read_csv(self.config["test_file_path"])#Test-posSNR
    #     test_ds = Datasets(df_test)
    #     print(f"\nlen of test df: {len(df_test)}")
    #     test_dl = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=False,num_workers=16,drop_last=True)
    #     return test_dl
    def test_dataloader(self):
        df_test = pd.read_csv(self.config["test_file_path"])  # Load test dataset
        test_ds = EnhDatasets(df_test)  # Create dataset instance
        print(f"\nlen of test df: {len(df_test)}")

        # Define DataLoader
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True,
            collate_fn=lambda batch: { 
                "mix_spk": torch.stack([b["mix_spk"] for b in batch], dim=0),
                "sep_spk": torch.stack([b["sep_spk"] for b in batch], dim=0).squeeze(1) ,
                "ref_spk": torch.stack([b["ref_spk"] for b in batch], dim=0).squeeze(1),
                "h5_path": [b["h5_path"] for b in batch],
                "trajectory": [b["trajectory"] for b in batch],
                "trajectory_processed": [b["trajectory_processed"] for b in batch],
                "time_points": [b["time_points"] for b in batch],
                "condition": [b["condition"] for b in batch],
                "snr_up": [b["snr_up"] for b in batch],
                "snr": [b["snr"] for b in batch],
            })

        return test_dl  
