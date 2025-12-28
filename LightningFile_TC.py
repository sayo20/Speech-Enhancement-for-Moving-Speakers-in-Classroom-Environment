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
from Datasets import Datasets
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
from models import SeparationNet
from scipy.io.wavfile import write
import wandb
from torch.utils.data import Subset
import h5py
import tracemalloc
from models import EnhancementNet
from separation_handler import SeparationHandler
import torch.nn.utils.prune as prune


class Lightning(LightningModule):
    def __init__(self, config, SeparationNet,
                 enc_dim=64,
                 feature_dim=64,
                 hidden_dim=256, 
                 enc_win=64,
                 enc_stride=32,
                 num_block=5,
                 num_layer=7,
                 kernel_size=3,
                 stft_win=512,
                 stft_hop=32,
                 batch_size=128,
                 lr=1e-3):
        super(Lightning, self).__init__()
        # ------------------Dataset&DataLoader Parameter-----------------

        self.batch_size = config["batch_size"]
        self.learning_rate = lr
        self.config = config
        self.inference_times = []
        self.memory_usages = []
        self.sdr_unp = []
        self.sdr_p = []
        self.features = InputFeature() 
        # self.prune_amount = prune_amount
        # -----------------------model-----------------------
        # self.model = SeparationNet(
        #     enc_dim=config['enc_dim'],
        #     feature_dim=config['feature_dim'],
        #     hidden_dim=config['hidden_dim'],
        #     enc_win=config['enc_win'],
        #     enc_stride=config['enc_stride'],
        #     num_block=config['num_block'],
        #     num_layer=config['num_layer'],
        #     kernel_size=config['kernel_size'],
        #     stft_win=config['stft_win'],
        #     stft_hop=config['stft_hop'],
        #     num_spk=config['num_spk'],
        # )

        # #enhancement_model ...you need this part to get test data for the localizer model, cause you first separate and then enhance
        self.model = EnhancementNet(
            enc_dim=config["enc_dim"],
            feature_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"],
            enc_win=config["enc_win"],
            enc_stride=config["enc_stride"],
            num_block=config["num_block"],
            num_layer=config["num_layer"],
            kernel_size=config["kernel_size"],
            num_spk=config["num_spk"]
        )
        checkpoint = torch.load("Data/checkPoints_newTraj_childBRIR/mimo-enhancer-FinetuneClass-bab-ChildBRIR-finalepoch=08-val_loss=-10.44.ckpt", map_location="cuda")

        # Fix key mismatch
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}

        # Load the updated state_dict into the model
        self.model.load_state_dict(state_dict)

        self.model.eval()
        self.model.requires_grad_(False)

    # def forward(self, x):  # forward(self, x,spatial)
    #     return self.model(x)

    def forward(self, y, est):
        return self.model(y, est)  

    def setup(self, stage=None):
        SeparationHandler(self.config)
        # Initialize separation handler per process
        self.separator = SeparationHandler.get_instance()
    # ---------------------
    # TRAINING STEP
    # ---------------------

    def training_step(self, batch, batch_idx):
        mix_sound = batch["mix"].cuda()  # Preloaded mix
        spk1_sound = batch["spk1"].cuda()  # Preloaded speaker 1
        spk2_sound = batch["spk2"].cuda()  # Preloaded speaker 2

        batch_left_clean = torch.stack([spk1_sound[:, 0, :], spk2_sound[:, 0, :]], dim=1)
        batch_right_clean = torch.stack([spk1_sound[:, 1, :], spk2_sound[:, 1, :]], dim=1)

        left_output, right_output = self.forward(mix_sound)

        loss = torch.mean(permutation_loss(batch_left_clean.unbind(dim=1), batch_right_clean.unbind(dim=1),
                                           left_output.unbind(dim=1), right_output.unbind(dim=1),
                                           snr_loss))

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    # ---------------------
    # VALIDATION SETUP
    # ---------------------


    def validation_step(self, batch, batch_idx):
        mix_sound = batch["mix"].cuda()  # Preloaded mix
        spk1_sound = batch["spk1"].cuda()  # Preloaded speaker 1
        spk2_sound = batch["spk2"].cuda()  # Preloaded speaker 2

        batch_left_clean = torch.stack([spk1_sound[:, 0, :], spk2_sound[:, 0, :]], dim=1)
        batch_right_clean = torch.stack([spk1_sound[:, 1, :], spk2_sound[:, 1, :]], dim=1)

        left_output, right_output = self.forward(mix_sound)

        loss = torch.mean(permutation_loss(batch_left_clean.unbind(dim=1), batch_right_clean.unbind(dim=1),
                                           left_output.unbind(dim=1), right_output.unbind(dim=1),
                                           snr_loss))
        # print("loss in val: ",loss)
        self.log("val_loss", loss, on_step=False, on_epoch=True,sync_dist=True)

        return {'val_loss': loss}

    # def test_step(self, batch, batch_idx):
    #     """
    #     evaluate: Regular
    #     """
    #     mix_sound = batch["mix"].cuda()  # Preloaded mix
    #     spk1_sound = batch["spk1"].cuda()  # Preloaded speaker 1
    #     spk2_sound = batch["spk2"].cuda()  # Preloaded speaker 2
    #     condition = batch["condition"]
    #     starting_distance = batch["starting distance"]
    #     distance_from_listener = batch["distance_from_listener"]
    #     trajectory_1 = batch["trajectory 1"]
    #     trajectory_2 = batch["trajectory 2"]
    #     time_points_1=batch["time_points_1"]
    #     time_points_2=batch["time_points_2"]
    #     # time_points = batch["time points"]
    #     # bab_snr = batch["bab_snr"]
    #     # bab_traj = batch["bab_traj"]

    #     batch_left_clean = torch.stack([spk1_sound[:, 0, :], spk2_sound[:, 0, :]], dim=1)
    #     batch_right_clean = torch.stack([spk1_sound[:, 1, :], spk2_sound[:, 1, :]], dim=1)

    #     targ = spk1_sound
    #     interf = spk2_sound
    #     mix = mix_sound
        
    #     # print(f"target shape: {targ.shape}, interf shape {interf.shape}")

    #     left_output, right_output = self.forward(mix_sound)


    #     counter = batch_idx * len(left_output) 
    #     base_path = "Data/separated_Adult-noBab-Train/"
    #     sr = 16000
    #     enhancer_rows = []

    #     df_test = pd.DataFrame(columns = ["ref path","est path","mix path","trajectory","time_points","snr","snr_up","condition","starting distance","distance_from_listener"])

    #     loss_sdr_ =  PITLossWrapper(PairwiseNegSDR('snr'), pit_from='pw_mtx')
    
    #     for i in range(len(left_output)):
    #         est_1 = torch.vstack((left_output[i][0],right_output[i][0]))
    #         est_2 = torch.vstack((left_output[i][1],right_output[i][1]))
            
    #         ests = [est_1,est_2]
    #         refs = [spk1_sound[i],spk2_sound[i]]

    #         # print(f"Shapes are: before reshaping, len of list: {len(ests),len(refs)}, shape of each: {est_1.shape,spk1_sound[i].shape }")

    #         ests_tensor = torch.stack(ests).unsqueeze(0)  
    #         ref_signals_tensor = torch.stack(refs).unsqueeze(0).cuda() 

    #         # print(f"\n after unsqueeze: {ests_tensor.shape, ref_signals_tensor.shape}")

    #         ests_tensor_flat = ests_tensor.reshape(1, 2, -1) 
    #         ref_signals_tensor_flat = ref_signals_tensor.reshape(1, 2, -1)

    #         # print(f"\n after flat: {ests_tensor_flat.shape, ref_signals_tensor_flat.shape}")

    #         loss_sisdr, reordered_sources_flat = loss_sdr_(ests_tensor_flat, ref_signals_tensor_flat, return_est=True)

    #         # Reshape reordered sources back to [n_sources=2, n_channels=2, T]
    #         reordered_sources = reordered_sources_flat.reshape(1, 2, 2, -1)  # shape: [1, 2, 2, T]

    #         est_1_ = reordered_sources[:, 0, :, :]
    #         est_2_ = reordered_sources[:, 1, :, :]
      

    #         sdr_s1_up = signal_noise_ratio(mix[i], refs[0].cuda(),sr)
    #         sdr_s2_up = signal_noise_ratio(mix[i], refs[1].cuda(),sr)
    #         avg_sdr_up = np.mean([torch.mean(sdr_s1_up).item(),torch.mean(sdr_s2_up).item()])

    #         sdr_s1 = signal_noise_ratio(est_1_[0], refs[0].cuda(),sr)
    #         sdr_s2 = signal_noise_ratio(est_2_[0], refs[1].cuda(),sr)
    #         avg_sdr = np.mean([torch.mean(sdr_s1).item(),torch.mean(sdr_s2).item()])    

    #         print(f"metrics; sdr: {avg_sdr}, unprocessed sdr: {avg_sdr_up}, sdrI: {avg_sdr - avg_sdr_up} \n") 
    #         print(f"metric swapped: {torch.mean(permutation_loss(batch_left_clean.unbind(dim=1), batch_right_clean.unbind(dim=1),left_output.unbind(dim=1), right_output.unbind(dim=1),snr_loss))}\n")
    #         print(f"dist: {distance_from_listener[i]}, {refs[0].shape,refs[0][0].shape}\n")

    #         df_test.loc[i,"snr"] = avg_sdr
    #         df_test.loc[i,"snr_up"] = avg_sdr_up

    #         df_test.loc[i,"condition"] = condition[i]
    #         df_test.loc[i,"starting distance"] = str(starting_distance[i].item())
    #         df_test.loc[i,"starting distance_from_listener"] = distance_from_listener[i].item()
    #         # df_test.loc[i,"trajectory 1"] = trajectory_1[i]
    #         # df_test.loc[i,"trajectory 2"] = trajectory_2[i]
    #         # df_test.loc[i,"time points"] = time_points[i]


    #         mix_path = base_path + f"batch{batch_idx}_mix_index{i}_counter{counter}.flac"


    #         write(mix_path, sr, mix[i].T.cpu().numpy())

    #         # Loop over each estimated source (2 in this case)
    #         est = [est_1_,est_2_]
    #         for j in range(2):

    #             ref_path = base_path + f"batch{batch_idx}_Refsource{j}_index{i}_counter{counter}.flac"
    #             est_path = base_path + f"batch{batch_idx}_Estsource{j}_index{i}_{j}_counter{counter}.flac"

    #             #here we will need pass the est sound through the enhancement model and then save the enhanced sound
                
    #             write(ref_path, sr, refs[j].T.cpu().numpy())  # Save audio file
    #             write(est_path, sr, est[j].T.cpu().numpy())  # Save audio file

    #             # Append row data
    #             enhancer_rows.append({
    #                 "ref_path": ref_path,
    #                 "est_path":est_path,
    #                 "mix_path": mix_path,
    #                 "condition": condition[i],
    #                 # "starting distance": starting_distance[i],
    #                 # "distance_from_listener": distance_from_listener[i],
    #                 "trajectory 1": trajectory_1[i],
    #                 "trajectory 2": trajectory_2[i],
    #                 # "time_points": time_points[i],
    #                 "snr":avg_sdr,
    #                 "snr_up":avg_sdr_up,
    #                 # "bab_snr":bab_snr[i],
    #                 # "bab_traj":bab_traj[i]

    #             })
    #         counter = counter + 1


    #     # Create DataFrame at the end and save to CSV
    #     df_enhancer = pd.DataFrame(enhancer_rows)
    #     save_path = f"batch{batch_idx}_source{j}_index{i}_counter{counter}"
    #     df_enhancer.to_csv("Data/enhancementCSV_class/enhancement_test"+save_path+".csv", index=False)




    # def test_step(self, batch, batch_idx):

    #     """
    #     TO get data to train localizer, uses both separation and enhancement model

    #     """

    #     sr = 16000
    #     mix_sound = batch["mix"].cuda()
    #     spk1_sound = batch["spk1"].cuda()
    #     spk2_sound = batch["spk2"].cuda()
    #     condition = batch["condition"]
    #     starting_distance = batch["starting distance"]
    #     distance_from_listener = batch["distance_from_listener"]
    #     trajectory_1 = batch["trajectory 1"]
    #     trajectory_2 = batch["trajectory 2"]
    #     time_points_1 = batch["time_points_1"]
    #     time_points_2 = batch["time_points_2"]

    #     # bab_snr = batch["bab_snr"]
    #     # bab_traj = batch["bab_traj"]

    #     mix = mix_sound
    #     left_output, right_output = self.forward(mix_sound)

    #     # left_output, right_output  = self.separator.separate(mix_sound)

    #     # HDF5 and metadata output setup
    #     h5_path = f"Data/separated_FinetuneClass-noBab-Train/hdf5_audio_batch{batch_idx}.h5"
    #     os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    #     h5f = h5py.File(h5_path, 'w')
    #     enhancer_rows = []

    #     # df_test = pd.DataFrame(columns = ["ref path","est path","mix path","trajectory 1","trajectory 2","time_points","snr","snr_up","condition","starting distance","distance_from_listener"])

    #     loss_sdr_ = PITLossWrapper(PairwiseNegSDR('snr'), pit_from='pw_mtx')
    #     counter = batch_idx * len(left_output)

    #     for i in range(len(left_output)):
    #         est_1 = torch.vstack((left_output[i][0], right_output[i][0]))
    #         est_2 = torch.vstack((left_output[i][1], right_output[i][1]))
    #         ests = [est_1, est_2]
    #         refs = [spk1_sound[i], spk2_sound[i]]

    #         ests_tensor = torch.stack(ests).unsqueeze(0)
    #         ref_signals_tensor = torch.stack(refs).unsqueeze(0).cuda()
    #         ests_tensor_flat = ests_tensor.reshape(1, 2, -1)
    #         ref_signals_tensor_flat = ref_signals_tensor.reshape(1, 2, -1)

    #         loss_sisdr, reordered_sources_flat = loss_sdr_(ests_tensor_flat, ref_signals_tensor_flat, return_est=True)
    #         reordered_sources = reordered_sources_flat.reshape(1, 2, 2, -1)

    #         est_1_ = reordered_sources[:, 0, :, :][0]
    #         est_2_ = reordered_sources[:, 1, :, :][0]
    #         reordered_ests = [est_1_, est_2_]


    #         mix_np = mix[i].cpu().numpy()
    #         h5f.create_dataset(f"mix/{counter}", data=mix_np, compression="gzip")

    #         traj = [trajectory_1[i],trajectory_2[i]]
    #         time_point = [time_points_1[i],time_points_2[i]]
    #         for j in range(2):
    #             ref_np = refs[j].cpu().numpy()

    #             #enhance the estimated sound
    #             # enh_est= self.enhancement_model(mix[i],reordered_ests[j])
    #             # mix_single = mix[i].unsqueeze(0)  # [1, 2, L] if mix[i] is [2, L]
    #             est_single = reordered_ests[j].unsqueeze(0)  # [1, 2, L]

     

    #             ####enhanced snr
    #             sdr_up  = signal_noise_ratio(mix[i], refs[j].cuda(),sr)
    #             sdr_up = torch.mean(sdr_up).item()

    #             sdr_sep = signal_noise_ratio(est_single[0,:], refs[j].cuda(),sr)
    #             sdr_sep = torch.mean(sdr_sep).item()

    #             # sdr_enh = signal_noise_ratio(enh_est, refs[j].cuda(),sr)
    #             # sdr_enh = torch.mean(sdr_enh).item()

    #             # print(f"sdr unprocessed: {sdr_up}, sdr seperation: {sdr_sep}, sdr enhancement: {sdr_enh} \n")

    #             # enh_est = self.model(mix[i],reordered_ests[j])

    #             # est_np = enh_est.cpu().numpy()

    #             # Save each reference and estimate individually
    #             ref_key = f"ref{j}/{counter}"
    #             est_key = f"est{j}/{counter}"

    #             h5f.create_dataset(ref_key, data=ref_np, compression="gzip")
    #             # h5f.create_dataset(est_key, data=est_np, compression="gzip")
    #             h5f.create_dataset(est_key, data=est_single.cpu().numpy(), compression="gzip")

    #             ###i want to get the snrs from enhanceent

    #             enhancer_rows.append({
    #                 "h5_path": h5_path,
    #                 "mix_key": f"mix/{counter}",
    #                 "ref_key": ref_key,
    #                 "est_key": est_key,
    #                 "spk_index": j,
    #                 "condition": condition[i],
    #                 "starting_distance": starting_distance[i].item(),
    #                 "distance_from_listener": distance_from_listener[i].item(),
    #                 "trajectory": traj[j],
    #                 "time_points": time_point[j],
    #                 "snr_up":sdr_up,
    #                 # "snr_sep":sdr_sep,
    #                 "snr":sdr_sep,
    #                 # "sdr_enh":sdr_enh
    #                 # "bab_snr": bab_snr[i],
    #                 # "bab_traj": bab_traj[i]
    #             })

    #         counter += 1

    #     h5f.close()

    #     # Save metadata CSV
    #     df_enhancer = pd.DataFrame(enhancer_rows)
    #     csv_path = f"Data/enhancementCSV_class/enhancement_test_batch{batch_idx}.csv"
    #     os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    #     df_enhancer.to_csv(csv_path, index=False)

    def test_step(self, batch, batch_idx):

        """
        TO get data, test set for localizer, uses both separation and enhancement model

        """
        # def normalize_audio(signal):
        #     return signal / (torch.norm(signal) + 1e-8)
        # def scale_match_binaural(est_lr, ref_lr, eps=1e-8):
        #     """
        #     est_lr, ref_lr: [2, L] tensors
        #     Returns est_scaled with a single scalar gain applied to both channels.
        #     """
        #     num = (ref_lr * est_lr).sum()
        #     den = (est_lr ** 2).sum() + eps
        #     alpha = num / den
        #     return alpha * est_lr  # same alpha to L and R
        # resume_from_batch =2369
        # if batch_idx < resume_from_batch:
        #     if batch_idx % 500 == 0:
        #         print(f"Skipping batch {batch_idx}")
        #     return
        sr = 16000
        mix_sound = batch["mix"].cuda()
        spk1_sound = batch["spk1"].cuda()
        spk2_sound = batch["spk2"].cuda()
        condition = batch["condition"]
        starting_distance = batch["starting distance"]
        distance_from_listener = batch["distance_from_listener"]
        trajectory_1 = batch["trajectory 1"]
        trajectory_2 = batch["trajectory 2"]
        time_points_1 = batch["time_points_1"]
        time_points_2 = batch["time_points_2"]

        bab_snr = batch["bab_snr"]
        bab_traj = batch["bab_traj"]

        mix = mix_sound
        # left_output, right_output = self.forward(mix_sound)

        left_output, right_output  = self.separator.separate(mix_sound)

        # HDF5 and metadata output setup
        h5_path = f"Data/enhanced_FinetunedClass-Bab-ClassBab-1.5-Test/hdf5_audio_batch{batch_idx}.h5"
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        h5f = h5py.File(h5_path, 'w')
        enhancer_rows = []

        # df_test = pd.DataFrame(columns = ["ref path","est path","mix path","trajectory 1","trajectory 2","time_points","snr","snr_up","condition","starting distance","distance_from_listener"])

        loss_sdr_ = PITLossWrapper(PairwiseNegSDR('snr'), pit_from='pw_mtx')
        counter = batch_idx * len(left_output)

        for i in range(len(left_output)):
            est_1 = torch.vstack((left_output[i][0], right_output[i][0]))
            est_2 = torch.vstack((left_output[i][1], right_output[i][1]))
            ests = [est_1, est_2]
            refs = [spk1_sound[i], spk2_sound[i]]

            ests_tensor = torch.stack(ests).unsqueeze(0)
            ref_signals_tensor = torch.stack(refs).unsqueeze(0).cuda()
            ests_tensor_flat = ests_tensor.reshape(1, 2, -1)
            ref_signals_tensor_flat = ref_signals_tensor.reshape(1, 2, -1)

            loss_sisdr, reordered_sources_flat = loss_sdr_(ests_tensor_flat, ref_signals_tensor_flat, return_est=True)
            reordered_sources = reordered_sources_flat.reshape(1, 2, 2, -1)

            est_1_ = reordered_sources[:, 0, :, :][0]
            est_2_ = reordered_sources[:, 1, :, :][0]
            reordered_ests = [est_1_, est_2_]


            mix_np = mix[i].cpu().numpy()
            # h5f.create_dataset(f"mix/{counter}", data=mix_np, compression="gzip")

            traj = [trajectory_1[i],trajectory_2[i]]
            time_point = [time_points_1[i],time_points_2[i]]

                ###i want to get the snrs from enhanceent
            for j in range(2):
                ref_np = refs[j].detach().cpu().numpy()         # [2, L]

                # enhance — add batch dim for the model, then squeeze back
                mix_single = mix[i].unsqueeze(0)                # [1, 2, L]
                sep_single = reordered_ests[j].unsqueeze(0)     # [1, 2, L]
                enh_est = self.model(mix_single, sep_single).squeeze(0)  # [2, L]

                # metrics: use 2D tensors [2, L]
                # sdr_up  = signal_noise_ratio(scale_match_binaural(mix[i]), scale_match_binaural(refs[j]), sr).mean().item()
                # sdr_sep = signal_noise_ratio(scale_match_binaural(reordered_ests[j]), scale_match_binaural(refs[j]), sr).mean().item()
                # sdr_enh = signal_noise_ratio(scale_match_binaural(enh_est), scale_match_binaural(refs[j]), sr).mean().item()
                
                # mix_sm  = scale_match_binaural(mix[i],          refs[j])
                # sep_sm  = scale_match_binaural(reordered_ests[j], refs[j])
                # enh_sm  = scale_match_binaural(enh_est,         refs[j])

                # sdr_up  = signal_noise_ratio(mix_sm, refs[j], sr).mean().item()
                # sdr_sep = signal_noise_ratio(sep_sm,  refs[j], sr).mean().item()
                # sdr_enh = signal_noise_ratio(enh_sm,  refs[j], sr).mean().item()

                sdr_up  = signal_noise_ratio(mix[i], refs[j], sr).mean().item()
                sdr_sep = signal_noise_ratio(reordered_ests[j],  refs[j], sr).mean().item()
                sdr_enh = signal_noise_ratio(enh_est,  refs[j], sr).mean().item()
                
                print(f"sdr unprocessed: {sdr_up:.2f}, sdr separation: {sdr_sep:.2f}, sdr enhancement: {sdr_enh:.2f}, enh_improvement: {sdr_enh - sdr_up:.2f}, condition: {condition[i]}\n")

                est_np = enh_est.detach().cpu().numpy()         # [2, L]

                ref_key = f"ref{j}/{counter}"
                est_key = f"est{j}/{counter}"
                h5f.create_dataset(ref_key, data=ref_np, compression="gzip")
                h5f.create_dataset(est_key, data=est_np, compression="gzip")
                enhancer_rows.append({
                    "h5_path": h5_path,
                    "mix_key": f"mix/{counter}",
                    "ref_key": ref_key,
                    "est_key": est_key,
                    "spk_index": j,
                    "condition": condition[i],
                    "starting_distance": starting_distance[i].item(),
                    "distance_from_listener": distance_from_listener[i].item(),
                    "trajectory": traj[j],
                    "trajectory_processed" : self.features.process_row(traj[j]),
                    "time_points": time_point[j],
                    "snr_up":sdr_up,
                    "snr_sep":sdr_sep,
                    # "snr":sdr_sep,
                    "sdr_enh":sdr_enh,
                    "bab_snr": bab_snr[i],
                    "bab_traj": bab_traj[i]
                })

            counter += 1

        # h5f.close()

        # Save metadata CSV
        df_enhancer = pd.DataFrame(enhancer_rows)
        csv_path = f"Data/enhancementCSV_class_dev/enhancement_test_batch{batch_idx}.csv"
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
        train_ds = Datasets(df_train)
        sampler = DistributedSampler(train_ds,shuffle=True, drop_last=True)#sampler=sampler, 
        train_dl = DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            num_workers=8, 
            drop_last=True,
            pin_memory=True, 
            sampler=sampler,
            collate_fn=lambda batch: {  # Fixed syntax error here
                "mix": torch.cat([b["mix"] for b in batch], dim=0),
                "spk1": torch.cat([b["spk1"] for b in batch], dim=0),
                "spk2": torch.cat([b["spk2"] for b in batch], dim=0)  # Keep metadata in list
            }
        )
        return train_dl
    def val_dataloader(self):
        df_val = pd.read_csv(self.config["validation_file_path"])#Val-10k.csv ../Data/WSJ2Mix-2spk/
        # df_val = df_val[:4]
        val_ds = Datasets(df_val)
        sampler = DistributedSampler(val_ds,shuffle=False, drop_last=True)
        val_dl = torch.utils.data.DataLoader(val_ds,batch_size=self.batch_size,  sampler=sampler, shuffle=False,num_workers=8,drop_last=True,pin_memory=True,
            collate_fn=lambda batch: { 
                "mix": torch.cat([b["mix"] for b in batch], dim=0),
                "spk1": torch.cat([b["spk1"] for b in batch], dim=0),
                "spk2": torch.cat([b["spk2"] for b in batch], dim=0),
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
        test_ds = Datasets(df_test)  # Create dataset instance
        print(f"\nlen of test df: {len(df_test)}")

        # Define DataLoader
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True,
            collate_fn=lambda batch: { 
                "mix": torch.cat([b["mix"] for b in batch], dim=0).squeeze(1),
                "spk1": torch.cat([b["spk1"] for b in batch], dim=0).squeeze(1),
                "spk2": torch.cat([b["spk2"] for b in batch], dim=0).squeeze(1),
                "condition": [b["condition"] for b in batch],  # No need to wrap in extra ()
                "starting distance": [b["starting distance"] for b in batch],
                "distance_from_listener": [b["distance_from_listener"] for b in batch],
                "trajectory 1": [b["trajectory 1"] for b in batch],
                "trajectory 2": [b["trajectory 2"] for b in batch],
                "time_points_1": [b["time points 1"] for b in batch],
                "time_points_2": [b["time points 2"] for b in batch],
                "bab_snr": [b["bab_snr"] for b in batch],
                "bab_traj": [b["bab_traj"] for b in batch],
            }
        )

        return test_dl  
