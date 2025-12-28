

import torch
import torchaudio
import pandas as pd
import numpy as np
import random
import ast
from torch.utils.data import Dataset
# from AudioFeature_TC import InputFeature
from AudioFeature_Modular import InputFeature


import torch.nn.functional as F


import ast
import h5py
import torch
from torch.utils.data import Dataset
from functools import lru_cache
import threading

###NEW TRAJ..easier

class Datasets(Dataset):
    def __init__(self, df):
        self.df = df
        self.features = InputFeature()  # Initialize feature processor

        # Convert string lists to actual lists for relevant columns
        list_cols = ["trajectory 1", "trajectory 2", "time_points_1","time_points_2","babble_traj"]  # Add "babble_traj" during testing,"babble_traj" 
        for col in list_cols:
            self.df[col] = self.df[col].apply(ast.literal_eval)

        # Store trajectory data in a dictionary for quick lookup
        self.trajectory_dict = {
            idx: (row["trajectory 1"], row["trajectory 2"], row["time_points_1"],row["time_points_2"])
            for idx, row in self.df.iterrows()
        }

        # Config flags (adjust as needed)
        self.babble = True
        self.test = True
        self.enhance = False

    def __len__(self):
        return len(self.df)

    def get_random_trajectory(self):
        """Randomly selects a trajectory from the stored dictionary."""
        return self.trajectory_dict[random.choice(list(self.trajectory_dict.keys()))]

    def _generate_audio(self, row, trajectory=None):
        """Helper method to generate moving audio (reused across conditions)."""
        if trajectory is None:  # Use row's trajectory if test=True
            trajectory = (row["trajectory 1"], row["trajectory 2"], row["time_points_1"],row["time_points_2"])
        
        return self.features.generate_moving_audio_TS_each(
            row["Speaker_1 path"], 
            row["Speaker_2 path"], 
            row["config"], 
            row["snr"], 
            row["pad_length"],
            *trajectory,
            row["distance_from_listener"], 
            row["room"],
            self.test
        )

    def _get_base_return_dict(self, row, trajectory):
        """Shared structure for all return dictionaries."""
        return {
            "trajectory 1": trajectory[0],
            "trajectory 2": trajectory[1],
            "time points 1": trajectory[2],
            "time points 2": trajectory[3],
            "condition": row["config"],#config
            "starting distance": row["starting distance"],
            "distance_from_listener": row["distance_from_listener"],
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        trajectory = (
            (row["trajectory 1"], row["trajectory 2"], row["time_points_1"],row["time_points_2"]) 
            if self.test 
            else self.get_random_trajectory()
        )

        # Generate speaker and mixed audio
        spk1, spk2, mix = self._generate_audio(row, trajectory)
        spk1_s = spk1.to(torch.float32)
        spk2_s = spk2.to(torch.float32)

        # Handle babble/enhance cases
        if self.babble:
            # test_ = True
            mix_bab,snr_bab,traj = self.features.diffuseBabble(
                row["Babble path"], 
                mix, 
                row["config"], 
                row["room"],
                row["distance_from_listener"],
                row["bab_snr"],
                row["babble_traj"],
                self.test,
            )
            mix_bab = self.features.mean_normalize_per_channel(mix_bab).to(torch.float32)
            
            return_dict = self._get_base_return_dict(row, trajectory)
            return_dict.update({
                "mix": mix_bab.unsqueeze(0),
                "spk1": spk1_s.unsqueeze(0),
                "spk2": spk2_s.unsqueeze(0),
                "bab_traj":traj,
                "bab_snr": snr_bab,
            })
            return return_dict

        elif self.enhance:
            mix, sr = torchaudio.load(row["mix_path"])
            ref, sr = torchaudio.load(row["ref_path"])
            est, sr = torchaudio.load(row["est_path"])

            return_dict = self._get_base_return_dict(row, trajectory)
            return_dict.update({
                "mix_sound": mix.unsqueeze(0),
                "ref_sound": ref.unsqueeze(0),
                "est_sound": est.unsqueeze(0),
                "mix path": row["mix_path"],
                "ref path": row["ref_path"],
                "est path": row["est_path"],
            })
            return return_dict

        else:
            # mix = torch.clamp(mix, -0.95, 0.95).to(torch.float32)
            mix = self.features.mean_normalize_per_channel(mix).to(torch.float32)
            
            return_dict = self._get_base_return_dict(row, trajectory)
            return_dict.update({
                "mix": mix.unsqueeze(0),
                "spk1": spk1_s.unsqueeze(0),
                "spk2": spk2_s.unsqueeze(0),
                "bab_snr":row["bab_snr"],
                "bab_traj":row["babble_traj"]
            })
            return return_dict




def normalize_traj_indices(x):
    # if it’s a string, parse it first
    if isinstance(x, str):
        x = ast.literal_eval(x)
    # now x should be a list of things; convert tensors→ints
    return [int(v.item()) if isinstance(v, torch.Tensor) else int(v) for v in x]


class TrajDatasets(Dataset):
    def __init__(self, df, sample_rate=16000, max_cache_size=100):
        def maybe_parse(x):
            return ast.literal_eval(x) if isinstance(x, str) else x

        self.trajectories = [maybe_parse(x) for x in df["trajectory_processed"]]
        # self.trajectories = [maybe_parse(x) for x in df["trajectory_processed"]]
        self.time_points = [maybe_parse(x) for x in df["time_points"]]
        self.h5_paths = df["h5_path"].tolist()
        # self.mix_keys = df["mix_key"].tolist()
        self.ref_keys = df["ref_key"].tolist()
        self.est_keys = df["est_key"].tolist()
        self.conditions = df["condition"].tolist()
        self.sdr_enh = df["sdr_enh"].tolist()

        del df
        self.features = InputFeature()
        self.sample_rate = sample_rate
        self.lock = threading.Lock()
        
        # Cache for normalized audio tensors: (h5_path, key) -> tensor
        self.audio_cache = {}
        self.cache_order = []  # For LRU eviction
        self.max_cache_size = max_cache_size

    def _load_audio(self, h5_path, key):
        """Load and normalize audio with LRU caching"""
        cache_key = (h5_path, key)
        
        with self.lock:
            # Return cached tensor if available
            if cache_key in self.audio_cache:
                self.cache_order.remove(cache_key)
                self.cache_order.append(cache_key)
                return self.audio_cache[cache_key]

            # Load from disk
            with h5py.File(h5_path, "r") as f:
                audio = torch.tensor(f[key][:], dtype=torch.float32)
            
            # Normalize and cache
            normalized = self.features.mean_normalize_per_channel(audio)
            
            # Manage cache size
            if len(self.audio_cache) >= self.max_cache_size:
                oldest_key = self.cache_order.pop(0)
                del self.audio_cache[oldest_key]
            
            self.audio_cache[cache_key] = normalized
            self.cache_order.append(cache_key)
            return normalized

    def __len__(self):
        return len(self.h5_paths)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        tp = self.time_points[idx]
        h5_path = self.h5_paths[idx]
        # mix_key = self.mix_keys[idx]
        est_key = self.est_keys[idx]
        ref_key = self.ref_keys[idx]
        print("\nh5_path:", h5_path, "est_key:", est_key, "ref_key:", ref_key)
        # Load audio through cache
        enh = self._load_audio(h5_path, est_key)
        ref_sound = self._load_audio(h5_path, ref_key)

        # Process trajectory
        ref_traj = self.features.prepare_traj_reference(traj, tp, self.sample_rate)
        # ref_traj = torch.tensor(ref_traj, dtype=torch.long)
        ref_traj = ref_traj.clone().detach().long()

        return {
            "enh_sound": enh,
            "ref_sound": ref_sound,
            "trajectory": ref_traj,
            "h5_path": h5_path,
            # "mix_key": mix_key,
            "ref_key": ref_key,
            "condition": self.conditions[idx],
            "time_points": tp,
            "sdr_enh": self.sdr_enh[idx],
        }

class EnhDatasets(Dataset):
    def __init__(self, df):
        self.df = df
        # self.df["trajectory"] = self.df["trajectory"].apply(ast.literal_eval)
        self.features = InputFeature() 

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        h5_path = row["h5_path"]
        est_path =  row["est_key"]
        ref_path = row["ref_key"]
        mix_path = row["mix_key"]

        h5pyLoader = h5py.File(h5_path, 'r')
        mix_sound = h5pyLoader[mix_path][:]
        sep_sound = h5pyLoader[est_path][:]
        ref_sound = h5pyLoader[ref_path][:]

        return{
        "mix_spk": torch.tensor(mix_sound),
        "sep_spk": torch.tensor(sep_sound),
        "ref_spk": torch.tensor(ref_sound),
        "h5_path": h5_path,
        "condition": row["condition"],
        "snr_up": row["snr_up"],
        "snr": row["snr"],
        "trajectory": row["trajectory"],
        "trajectory_processed":self.features.process_row( row["trajectory"]),
        "time_points": row["time_points"],
        }



