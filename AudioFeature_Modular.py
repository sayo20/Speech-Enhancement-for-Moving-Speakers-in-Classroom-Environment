import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.fftpack import fft, ifft
from scipy.signal import correlate
import torchaudio
import torch
import ast
import torchaudio.transforms as T
from TrajectoryGenerator import TrajectoryGenerator
import h5py
import torch.nn.functional as F
import os
import random
from torch.autograd import Variable
from scipy.signal import fftconvolve

# Set random seeds at the beginning of your functions
# np.random.seed(42)
# random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# features = InputFeature()

class InputFeature():

    def get_allBRIRs(self, base_path, trajectories, target_sample_rate=16000):
        all_brir = []
        for traj in trajectories: 
            # Load trajectories, resample them to 16k, save them in a list
            path = base_path + str(traj) + ".flac"#.item()
            brir_waveform, sr = torchaudio.load(path)

            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)

            # Cutting the tail at 2500 samples
            lh = resampler(brir_waveform[0,:])
            lr = resampler(brir_waveform[1,:])

            all_brir.append(torch.tensor(np.vstack((lh, lr))))

        return all_brir


    # def get_allBRIRs(self, base_path, trajectories, target_sample_rate=16000):
    #     """
    #     Load, resample, and return BRIRs for given trajectories.

    #     Args:
    #         base_path (str): Base directory of BRIRs.
    #         trajectories (list): List of angles to load BRIRs for.
    #         target_sample_rate (int): Desired sample rate for resampling.

    #     Returns:
    #         torch.Tensor: Batched BRIRs of shape (num_trajs, 2, max_length).
    #     """
    #     all_brir = []
    #     resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=target_sample_rate)

    #     max_len = 0  # Track the longest BRIR length

    #     for traj in trajectories:
    #         path = f"{base_path}{traj}.flac"
    #         brir_waveform, sr = torchaudio.load(path)

    #         # Resample both channels
    #         lh = resampler(brir_waveform[0, :])
    #         lr = resampler(brir_waveform[1, :])

    #         # Update max length
    #         brir_length = max(lh.shape[0], lr.shape[0])
    #         max_len = max(max_len, brir_length)

    #         # Stack channels
    #         all_brir.append((lh, lr))

    #     # Pad all BRIRs to the max length
    #     padded_brir = []
    #     for lh, lr in all_brir:
    #         pad_len = max_len - lh.shape[0]  # Calculate padding amount
    #         lh = torch.nn.functional.pad(lh, (0, pad_len))  # Pad at the end
    #         lr = torch.nn.functional.pad(lr, (0, pad_len))  # Pad at the end
    #         padded_brir.append(torch.vstack((lh, lr)))  # Stack as (2, max_len)

    #     return torch.stack(padded_brir)  # Shape: (num_trajs, 2, max_len)


    def apply_fade(self, signal, fade_len, fade_in, fade_out):
        sig_len = len(signal)

        # Limit fade_len to half the signal length
        max_fade_len = sig_len // 2
        if fade_len > max_fade_len:
            fade_len = max_fade_len
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)

        if fade_len > 0:
            signal[:fade_len] *= fade_in
            signal[-fade_len:] *= fade_out

        return signal

    def fft_convolve(self, signal, kernel):
        """ Perform convolution using FFT (fast convolution) """
        # Zero-padding to avoid circular convolution
        n = len(signal) + len(kernel) - 1
        signal_fft = fft(signal, n=n)
        kernel_fft = fft(kernel, n=n)

        # Element-wise multiplication in the frequency domain
        result_fft = signal_fft * kernel_fft

        # Convert back to the time domain
        result = np.real(ifft(result_fft))

        # Return the first part of the result (valid length)
        return result[:len(signal)]

    def simulate_moving_source_fft(self, sound, sample_rate, trajectory, tp, brirs, duration=2.4, fade_len_ms=5):
        Lh = brirs[0].shape[1]  # Length of BRIR filter
        n_samples = sound.shape[1]
        sound = sound.numpy()
        # Initialize output signals
        sL = np.zeros(n_samples)
        sR = np.zeros(n_samples)

        fade_len = int((fade_len_ms / 1000) * sample_rate)
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)

        for i in range(len(tp)-1):
            # print(f"Convolving for angle {trajectory[i]}")

            # Determine the start and end sample for this segment
            start_sample = int(tp[i] * sample_rate)
            end_sample = int(tp[i + 1] * sample_rate)
            # print("brir shape: ",brirs[i].shape)
            # Retrieve the corresponding BRIR
            brir_L = brirs[i][0].numpy()
            brir_R = brirs[i][1].numpy()

            # # Temporary segment output
            # seg_sL = np.zeros(end_sample - start_sample)
            # seg_sR = np.zeros(end_sample - start_sample)

            # Convolve the segment with the BRIR using FFT convolution
            seg_sL = self.fft_convolve(sound[0, start_sample:end_sample], brir_L)
            seg_sR = self.fft_convolve(sound[0, start_sample:end_sample], brir_R)

            seg_sL = self.apply_fade(seg_sL, fade_len,fade_in,fade_out)
            seg_sR = self.apply_fade(seg_sR,fade_len, fade_in,fade_out)

            # Add the segment result to the overall output
            sL[start_sample:end_sample] += seg_sL
            sR[start_sample:end_sample] += seg_sR

        # Trim the final signal to match the desired duration
        n_samples = int(duration * sample_rate)
        sL = sL[:n_samples]
        sR = sR[:n_samples]

        # Combine left and right channels into a stereo waveform
        final_waveform = np.vstack((sL, sR))

        return torch.tensor(final_waveform)

    def audio_length_check(self, waveform, sample_rate):
        # Define the required minimum length (in seconds)
        min_duration = 2.4

        # Calculate the number of samples needed for 2.4 seconds
        min_samples = int(min_duration * sample_rate)

        c = random.choice(["beginning", "end"])

        # Check if the audio is shorter than 2.4 seconds
        if waveform.shape[1] < min_samples:
            # Calculate the padding length
            padding_length = min_samples - waveform.shape[1]

            # Randomly decide to pad at the beginning or the end
            if c == "beginning":
                # Pad at the beginning
                padding = torch.zeros((waveform.shape[0], padding_length))
                waveform = torch.cat((padding, waveform), dim=1)
            else:
                # Pad at the end
                padding = torch.zeros((waveform.shape[0], padding_length))
                waveform = torch.cat((waveform, padding), dim=1)
        return waveform
    def audio_length_check_test(self, waveform,sample_rate, aud_length):
        # Define the required minimum length (in seconds)
        min_duration = 2.4

        # Calculate the number of samples needed for 2.4 seconds
        min_samples = int(min_duration * sample_rate)

        # Check if the audio is shorter than 2.4 seconds
        if waveform.shape[1] < min_samples:
            # Calculate the padding length
            padding_length = min_samples - waveform.shape[1]

            # Randomly decide to pad at the beginning or the end
            if aud_length == 0:
                # Pad at the beginning
                padding = torch.zeros((waveform.shape[0], padding_length))
                waveform = torch.cat((padding, waveform), dim=1)
            else:
                # Pad at the end
                padding = torch.zeros((waveform.shape[0], padding_length))
                waveform = torch.cat((waveform, padding), dim=1)
        return waveform

    def set_snr_norm(self, signal, noise, target_snr, eps=1e-8):
        # Calculate power of the signal and noise
        signal_power = torch.sum(signal**2)
        noise_power = torch.sum(noise**2) + eps

        # Calculate the scaling factor for the noise to achieve the target SNR
        scaling_factor = torch.sqrt(signal_power / (10**(target_snr / 10) * noise_power))

        # Scale the noise signal
        scaled_noise = noise* scaling_factor

        # Adjust the amplitudes of the signals to achieve the target SNR
        adjusted_signal = signal
        adjusted_noise = scaled_noise

        return adjusted_signal, adjusted_noise

    def mean_variance_normalize(self, audio):
        """
        Perform global mean-variance normalization on the multi-channel audio tensor.
        """
        mean = audio.mean()  # Global mean across all channels and samples
        std = audio.std()    # Global standard deviation across all channels and samples

        # Normalize the entire audio using the global mean and std
        audio_normalized = (audio - mean) / std
        return audio_normalized
    def mean_normalize_per_channel(self, audio):
        """
        Perform mean normalization per channel on an audio tensor.
        
        Args:
            audio (Tensor): The audio tensor with shape (channels, samples).
        
        Returns:
            Tensor: The mean-normalized audio tensor, preserving channel-specific means.
        """
        # Calculate the mean along the sample dimension (dim=1) for each channel
        mean_per_channel = audio.mean(dim=1, keepdim=True)
        
        # Subtract the mean from each channel independently
        audio_normalized = audio - mean_per_channel
        return audio_normalized
    def convolveBrir(self,sound, sample_rate, brirs, duration=2.4):
        """
        brir_path: path to the brir you want to convolve
        sound: the sound you wan to convolve with the brir (convert from tensor to numpy)
        threshold: to prevent the audio from truncating at certain frequenct
        uni_bi: 0 means unilateral so we only one ear of our microphones(so 3ch) and 1: means bilateral, we use all 6
        """
        # start_time = timer()

        Lh = brirs[0].shape[1]  # Length of BRIR filter
        n_samples = sound.shape[1]
        sound = sound.numpy()
        # Initialize output signals
        sL = np.zeros(n_samples)
        sR = np.zeros(n_samples)

        for i in range(len(brirs)):

            # Retrieve the corresponding BRIR
            brir_L = brirs[i][0].numpy()
            brir_R = brirs[i][1].numpy()

            # Temporary segment output
            seg_sL = np.zeros(n_samples)
            seg_sR = np.zeros(n_samples)

            # Convolve the segment with the BRIR using FFT convolution
            seg_sL = self.fft_convolve(sound[0, :], brir_L)
            seg_sR = self.fft_convolve(sound[0, :], brir_R)


        # Trim the final signal to match the desired duration
        n_samples = int(duration * sample_rate)
        sL = seg_sL[:n_samples]
        sR = seg_sR[:n_samples]

        # Combine left and right channels into a stereo waveform
        final_waveform = torch.vstack((torch.tensor(sL), torch.tensor(sR)))
        # print(f"Brir size before resampling is: {brir_size}, after resampling is {brir_0.shape}, sound has size :{sound.shape}, after convolution sound is: {conv.shape}")
        return final_waveform

    # def convolveBrir(self,sound, sample_rate, brirs):
    #     """
    #     Convolve input mono sound with multiple BRIRs (stereo) and sum them to create a diffuse soundfield.

    #     Args:
    #         sound (torch.Tensor): Mono input sound tensor of shape (1, T).
    #         sample_rate (int): Sampling rate of the sound.
    #         brirs (torch.Tensor): Preloaded BRIRs tensor of shape (num_trajs, 2, 2500).

    #     Returns:
    #         torch.Tensor: Diffuse stereo sound tensor of shape (2, T).
    #     """
    #     device = sound.device  

    #     # Ensure sound is mono (shape: (1, T))
    #     if sound.shape[0] != 1:
    #         raise ValueError(f"Expected mono sound with shape (1, T), got {sound.shape}")

    #     # Get lengths
    #     sound_len = sound.shape[1]   # Number of samples in sound
    #     brir_len = brirs.shape[2]    # Number of samples in BRIR (e.g., 2500)

    #     # **Corrected Padding: Use the correct FFT size**
    #     fft_size = max(sound_len, brir_len) + brir_len - 1  # Ensure full convolution

    #     # Compute FFT of input mono sound (zero-padding to fft_size)
    #     sound_padded = torch.nn.functional.pad(sound, (0, fft_size - sound_len))  # Correctly pad
    #     sound_fft = torch.fft.rfft(sound_padded, n=fft_size, dim=1, norm="ortho")  # (1, F)

    #     # Compute FFT of all BRIRs with the same size
    #     brir_padded = torch.nn.functional.pad(brirs, (0, fft_size - brir_len))  # Correctly pad BRIR
    #     brir_fft = torch.fft.rfft(brir_padded, n=fft_size, dim=2, norm="ortho")  # (num_trajs, 2, F)

    #     # Multiply in the frequency domain
    #     conv_left_fft = sound_fft * brir_fft[:, 0, :]  # (num_trajs, F)
    #     conv_right_fft = sound_fft * brir_fft[:, 1, :]  # (num_trajs, F)

    #     # Convert back to time domain
    #     convolved_left = torch.fft.irfft(conv_left_fft, n=fft_size, dim=1)  
    #     convolved_right = torch.fft.irfft(conv_right_fft, n=fft_size, dim=1)  

    #     # Sum across different BRIR trajectories
    #     diffuse_left = convolved_left.sum(dim=0) / torch.sqrt(torch.tensor(brirs.shape[0], dtype=torch.float32))  
    #     diffuse_right = convolved_right.sum(dim=0) / torch.sqrt(torch.tensor(brirs.shape[0], dtype=torch.float32))  

    #     # Stack left and right channels
    #     diffuse_sound = torch.stack([diffuse_left, diffuse_right])  # Shape: (2, T)

    #     # Trim final result to 2.4 seconds
    #     target_samples = int(sample_rate * 2.4)
    #     diffuse_sound = diffuse_sound[:, :target_samples]

    #     # Normalize to prevent clipping (but not too aggressively)
    #     diffuse_sound = diffuse_sound / torch.max(torch.abs(diffuse_sound)) * 0.9  # Keep it in [-0.9, 0.9] range

    #     return diffuse_sound.to(device)


    # def diffuseBabble(self, babble_path, mixed_sound, config, room_num):
    #     # Initialize babble sum as a tensor
    #     babble, sr = torchaudio.load(babble_path)
    #     total_bab = torch.zeros((2, babble.shape[1]), dtype=babble.dtype, device=babble.device)

    #     # Generate trajectory angles
    #     values = np.arange(0, 360, 5)  # Angles from 0° to 360° in 5° steps
    #     traj_num = random.choice(range(3, 9))  # Pick 3-8 angles
    #     trajs = np.random.choice(values, size=traj_num, replace=False).tolist()

    #     # Determine BRIR base path
    #     if config in ["SSM", "TSM"]:
    #         brir_basepath = f"Data/Viking_ours/Chasar_BRIR_classRoom_children_1.0/Room{room_num}_"
    #     elif config == "TTM":
    #         brir_basepath = f"/gpfs/home1/folalere/Year 3/ClassroomSeparation/Data/Viking_ours/Viking_BRIR_classRoom_teacher_1.0/Room{room_num}_"
    #     else:
    #         raise ValueError(f"Unknown config: {config}")

    #     # Get BRIRs for the selected trajectories
    #     trajs_br = self.get_allBRIRs(brir_basepath, trajs)
    #     print("Selected trajectories:", trajs)

    #     # Convolve babble with each BRIR and sum it
    #     for each in trajs_br:
    #         babble_sound = self.convolveBrir(babble, sr, [each]) # Ensure tensor
    #         total_bab += babble_sound

    #     # Normalize babble to prevent clipping
    #     total_bab = total_bab / len(trajs)
    #     total_bab = self.audio_length_check(total_bab,  sr)

    #     # Choose SNR for mixing (-2.5 to 15 dB)
    #     snr_bab = np.random.choice(np.arange(-2.5, 15.5, 0.5))  # Ensures 15 is included

    #     # Adjust SNR and mix
    #     mix_sound_, total_bab_ = self.set_snr_norm(mixed_sound, total_bab, snr_bab)
    #     mix_sound_bab = mix_sound_ + total_bab_

    #     return mix_sound_bab

    def diffuseBabble(self, babble_path, mixed_sound, config, room_num,distance_from_listener,snr_bab=None, trajs=None, test=False):
        """
        Generate diffuse babble noise by convolving babble with multiple BRIRs.

        Args:
            babble_path (str): Path to babble noise file.
            mixed_sound (torch.Tensor): Mixed signal tensor of shape (2, T).
            config (str): Configuration name (e.g., 'SSM', 'TSM', 'TTM').
            room_num (int): Room number identifier.

        Returns:
            torch.Tensor: Mixed sound with diffuse babble noise.
        """
        # Load babble noise
        babble, sr = torchaudio.load(babble_path)

        if test ==False:
            # Generate random trajectory angles
            values = np.arange(0, 360, 5)  # Angles from 0° to 360° in 5° steps
            traj_num = random.choice(range(3, 9))  # Pick 3-8 angles
            trajs = np.random.choice(values, size=traj_num, replace=False).tolist()

            # Choose SNR for mixing (-2.5 to 15 dB)
            snr_bab = np.random.choice(np.arange(-2.5, 15.5, 0.5))
        # trajs=bab_traj

        # Determine BRIR base path
        if config in ["SSM", "TSM"]:
            brir_basepath = f"Data/Viking_ours/Chasar_BRIR_classRoom_children_1.5/Room{room_num}_"
            # brir_basepath="Data/Viking_ours/Chasar_BRIR_classRoom_children_"+str(float(distance_from_listener.item())) +"/Room" + room_num + "_"
        elif config == "TTM":
            # brir_basepath = f"/gpfs/home1/folalere/Year 3/ClassroomSeparation/Data/Viking_ours/Viking_BRIR_classRoom_teacher_1.0/Room{room_num}_"
            #with child brir
            brir_basepath = f"Data/Viking_ours/Chasar_BRIR_classRoom_children_1.5/Room{room_num}_"
        else:
            raise ValueError(f"Unknown config: {config}")

        # Load BRIRs for selected trajectories (returns stacked tensor of shape (num_trajs, 2, 2500))
        trajs_br = self.get_allBRIRs(brir_basepath, trajs)

        # print("Selected trajectories:", trajs)

        # Convolve babble with all BRIRs at once
        total_bab = self.convolveBrir(babble, sr, trajs_br)

        # Normalize babble to prevent clipping
        total_bab = self.audio_length_check(total_bab / len(trajs), sr)

        # Adjust SNR and mix with original sound
        mix_sound_, total_bab_ = self.set_snr_norm(mixed_sound, total_bab, snr_bab)
        mix_sound_bab = mix_sound_ + total_bab_

        return mix_sound_bab, snr_bab, trajs

    def check_path(self,spk1,spk2,path1,path2,base_path_lib):
        if not os.path.exists(path1):
            # path1 = base_path_lib + "Data/LibriSpeech/" + spk1
            path1 = base_path_lib + path1
        if not os.path.exists(path2):
            # path2 = base_path_lib+ "Data/LibriSpeech/" + spk2
            path2 = base_path_lib + path2
        return path1,path2
        
    def getMovingSound(self, spk1_path,spk2_path,traj1,traj2,timepoint_1,timepoint_2,spk_snr,brir_base_path_1,brir_base_path_2,test,pad):

        spk1_, sample_rate = torchaudio.load(spk1_path)
        spk2_, _ = torchaudio.load(spk2_path)

        if test == True:
            spk1_ = self.audio_length_check_test(spk1_,  sample_rate,pad)
            spk2_ = self.audio_length_check_test(spk2_, sample_rate,pad)
        else:
            spk1_ = self.audio_length_check(spk1_,  sample_rate)
            spk2_ = self.audio_length_check(spk2_, sample_rate)

        brir_1 = self.get_allBRIRs(brir_base_path_1, traj1)
        brir_2 = self.get_allBRIRs(brir_base_path_2, traj2)

        spk1_sound = self.simulate_moving_source_fft(spk1_, sample_rate, traj1, timepoint_1, brir_1)
        spk2_sound = self.simulate_moving_source_fft(spk2_, sample_rate, traj2, timepoint_2, brir_2)

        spk1_s, spk2_s = self.set_snr_norm(spk1_sound, spk2_sound, spk_snr)

        mix_sound = spk1_s + spk2_s 

        # mix_sound = self.mean_normalize_per_channel(mix_sound).to(torch.float32) removed cause i do it in dataset

        return spk1_s,spk2_s,mix_sound



    def generate_moving_audio_TS_each(self,spk1, spk2, condition, spk_snr,pad,trajectory1, trajectory2, timepoint_1,timepoint_2 ,distance_from_listener,room_num,test=False):
        new_spk1 = []
        new_spk2 = []
        mix = []
        mix_test = []
        

        base_path_lib = "/gpfs/home1/folalere/Year 3/ClassroomSeparation/"

        # print(f"\nHere: {distance_from_listener,room_num} \n")

        if condition == "TSM":

            spk1_path = "Data/" + spk1
            spk2_path =  "Data/" + spk2

            spk1_path,spk2_path=self.check_path(spk1,spk2,spk1_path,spk2_path,base_path_lib)

            # print(f"paths are: p1:{spk1_path}, p2:{spk1_path}\n")
            #from a child listener perspective: we need to convolve with the children BRIRs
            # brir_basePath_teacher = base_path_lib+"Data/Viking_ours/Viking_BRIR_classRoom_teacher_1.0/Room" +room_num + "_"
            
            brir_basePath_teacher = base_path_lib+"Data/Viking_ours/Chasar_BRIR_classRoom_children_1.5/Room"+room_num + "_"
            brir_basePath_children = base_path_lib+"Data/Viking_ours/Chasar_BRIR_classRoom_children_1.5/Room"+ room_num + "_"


            # brir_basePath_children = "Data/Viking_ours/Chasar_BRIR_classRoom_children_"+str(float(distance_from_listener.item())) +"/Room" + room_num + "_"
            # brir_basePath_teacher = "Data/Viking_ours/Viking_BRIR_classRoom_teacher_"+str(float(distance_from_listener.item()))  +"/Room" + room_num + "_"

            spk1_s,spk2_s, mix_sound = self.getMovingSound(spk1_path,spk2_path,trajectory1,trajectory2,timepoint_1,timepoint_2 ,spk_snr,brir_basePath_teacher,brir_basePath_children,test,pad)

            return spk1_s,spk2_s, mix_sound

        elif condition == "TTM":
            spk1_path = base_path_lib+"Data/" + spk1
            spk2_path = base_path_lib+ "Data/" + spk2

            spk1_path,spk2_path= self.check_path(spk1,spk2,spk1_path,spk2_path,base_path_lib)


            # Paths for teacher and children BRIRs
            # brir_basePath_teacher =base_path_lib+ "Data/Viking_ours/Viking_BRIR_classRoom_teacher_1.0/Room" +room_num + "_"
            brir_basePath_teacher = base_path_lib+"Data/Viking_ours/Chasar_BRIR_classRoom_children_1.5/Room"+room_num + "_"

            #to test realroom
            # brir_basePath_teacher ="/gpfs/work5/0/prjs1040/Year 4/ClassroomSeparation/Data/Viking_ours/Real_room/Room" + "_"


            spk1_s,spk2_s, mix_sound = self.getMovingSound(spk1_path,spk2_path,trajectory1,trajectory2,timepoint_1,timepoint_2 ,spk_snr,brir_basePath_teacher,brir_basePath_teacher,test,pad)

            return spk1_s , spk2_s ,mix_sound    

        elif condition == "SSM":
            spk1_path = "Data/" + spk1
            spk2_path =  "Data/" + spk2
            
            brir_basePath_children = base_path_lib+"Data/Viking_ours/Chasar_BRIR_classRoom_children_1.5/Room"+ room_num + "_"


            spk1_s,spk2_s, mix_sound = self.getMovingSound(spk1_path,spk2_path,trajectory1,trajectory2,timepoint_1,timepoint_2 ,spk_snr,brir_basePath_children,brir_basePath_children,test,pad)


            return spk1_s,spk2_s,mix_sound



    # def getMovingSound(self, spk1_path,spk2_path,traj1,traj2,timepoint,spk_snr,brir_base_path_1,brir_base_path_2,test,pad):

    #     spk1_, sample_rate = torchaudio.load(spk1_path)
    #     spk2_, _ = torchaudio.load(spk2_path)

    #     if test == True:
    #         spk1_ = self.audio_length_check_test(spk1_,  sample_rate,pad)
    #         spk2_ = self.audio_length_check_test(spk2_, sample_rate,pad)
    #     else:
    #         spk1_ = self.audio_length_check(spk1_,  sample_rate)
    #         spk2_ = self.audio_length_check(spk2_, sample_rate)

    #     brir_1 = self.get_allBRIRs(brir_base_path_1, traj1)
    #     brir_2 = self.get_allBRIRs(brir_base_path_2, traj2)

    #     spk1_sound = self.simulate_moving_source_fft(spk1_, sample_rate, traj1, timepoint, brir_1)
    #     spk2_sound = self.simulate_moving_source_fft(spk2_, sample_rate, traj2, timepoint, brir_2)

    #     spk1_s, spk2_s = self.set_snr_norm(spk1_sound, spk2_sound, spk_snr)

    #     mix_sound = spk1_s + spk2_s 

    #     # mix_sound = self.mean_normalize_per_channel(mix_sound).to(torch.float32) removed cause i do it in dataset

    #     return spk1_s,spk2_s,mix_sound




    # def generate_moving_audio_TS_each(self,spk1, spk2, condition, spk_snr,pad,trajectory1, trajectory2, timepoint ,distance_from_listener,room_num,test=False):
    #     new_spk1 = []
    #     new_spk2 = []
    #     mix = []
    #     mix_test = []
        

    #     base_path_lib = "/gpfs/home1/folalere/Year 3/ClassroomSeparation/"

    #     # print(f"\nHere: {distance_from_listener,room_num} \n")

    #     if condition == "TSM":

    #         spk1_path = "Data/" + spk1
    #         spk2_path =  "Data/" + spk2

    #         spk1_path,spk2_path=self.check_path(spk1,spk2,spk1_path,spk2_path,base_path_lib)

    #         # print(f"paths are: p1:{spk1_path}, p2:{spk1_path}\n")

    #         brir_basePath_teacher = base_path_lib+"Data/Viking_ours/Viking_BRIR_classRoom_teacher_1.0/Room" +room_num + "_"
    #         brir_basePath_children = base_path_lib+"Data/Viking_ours/Chasar_BRIR_classRoom_children_1.0/Room"+ room_num + "_"

    #         spk1_s,spk2_s, mix_sound = self.getMovingSound(spk1_path,spk2_path,trajectory1,trajectory2,timepoint ,spk_snr,brir_basePath_teacher,brir_basePath_children,test,pad)

    #         return spk1_s,spk2_s, mix_sound

    #     elif condition == "TTM":
    #         spk1_path = base_path_lib+"Data/" + spk1
    #         spk2_path = base_path_lib+ "Data/" + spk2

    #         spk1_path,spk2_path=self.check_path(spk1,spk2,spk1_path,spk2_path,base_path_lib)


    #         # Paths for teacher and children BRIRs
    #         # brir_basePath_teacher = "Data/Viking_ours/Viking_BRIR_classRoom_teacher_"+str(float(distance_from_listener.item()))  +"/Room" + room_num + "_"
    #         brir_basePath_teacher =base_path_lib+ "Data/Viking_ours/Viking_BRIR_classRoom_teacher_1.0/Room" +room_num + "_"

    #         spk1_s,spk2_s, mix_sound = self.getMovingSound(spk1_path,spk2_path,trajectory1,trajectory2,timepoint ,spk_snr,brir_basePath_teacher,brir_basePath_teacher,test,pad)

    #         return spk1_s , spk2_s ,mix_sound    

    #     elif condition == "SSM":
    #         spk1_path = "Data/" + spk1
    #         spk2_path =  "Data/" + spk2

    #         # brir_basePath_children = "Data/Viking_ours/Chasar_BRIR_classRoom_children_"+str(float(distance_from_listener.item())) +"/Room" + room_num + "_"
    #         brir_basePath_children = base_path_lib+"Data/Viking_ours/Chasar_BRIR_classRoom_children_1.0/Room"+ room_num + "_"
    #         # brir_basePath_children = base_path_lib+"Data/Viking_ours/Viking_BRIR_classRoom_teacher_1.0/Room"+ room_num + "_"

    #         spk1_s,spk2_s, mix_sound = self.getMovingSound(spk1_path,spk2_path,trajectory1,trajectory2,timepoint ,spk_snr,brir_basePath_children,brir_basePath_children,test,pad)


    #         return spk1_s,spk2_s,mix_sound

    # # def deg_to_class_index(self,angle):
    # #     return int(angle // 5)


    # # def prepare_traj_reference(self,traj_list_deg, output_len):
    # #     print(f"\n list traj is: {traj_list_deg}, output_len is {output_len}")
    # #     class_indices = [self.deg_to_class_index(a) for a in traj_list_deg]
    # #     traj_tensor = torch.tensor(class_indices).unsqueeze(0).unsqueeze(0).float()
    # #     traj_interp = F.interpolate(traj_tensor, size=output_len, mode='nearest')
    # #     return traj_interp.squeeze(0).squeeze(0).long()





    # def prepare_traj_reference(self, traj_list_deg, output_len):
    #     # print(f"\n list traj is: {traj_list_deg}, output_len is {output_len}")
    #     class_indices = [self.deg_to_class_index(a) for a in traj_list_deg]
    #     traj_tensor = torch.tensor(class_indices).unsqueeze(0).unsqueeze(0).float()  # (1, 1, L)
    #     traj_interp = F.interpolate(traj_tensor, size=output_len, mode='nearest')    # (1, 1, T_out)
    #     return traj_interp.squeeze(0).squeeze(0).long()  # (T_out,)

    # def deg_to_class_index(self, angle): OLD
    #     # Convert to [-180°,180°)
    #     angle = angle % 360
    #     if angle > 180:
    #         angle -= 360
        
    #     # Handle back hemisphere (90-270° in original)
    #     if abs(angle) > 90:
    #         raise ValueError(f"Invalid binaural angle: {angle}°")
        
    #     return int((angle + 90) // 5)  # -90°→0, 0°→18, +90°→36
    def deg_to_class_index(angle: float) -> int:
        # Normalize to [-180, 180)
        angle = angle % 360
        if angle > 180:
            angle -= 360
        # Only front hemisphere valid
        if abs(angle) > 90:
            raise ValueError(f"Invalid binaural angle: {angle}°")
        # Round to nearest 5° bin, then shift to 0..36
        idx = int(round((angle + 90) / 5))
        return max(0, min(idx, 36))

    def calculate_output_len(self, duration: float, sample_rate: int) -> int:
        stft_win = 512
        stft_hop = 32
        audio_samples = int(duration * sample_rate)
        stft_frames = (audio_samples - stft_win) // stft_hop + 1
        frames_per_chunk = int(0.08 * sample_rate / stft_hop)
        # account for padding in AvgPool1d: kernel_size=frames_per_chunk, padding=frames_per_chunk//2
        pooled = (stft_frames + 2 * (frames_per_chunk // 2) - frames_per_chunk) // frames_per_chunk + 1
        return pooled



    # def prepare_traj_reference(self, trajectory, tp, sample_rate):
    #     # 1) determine number of output bins
    #     audio_dur = tp[-1]
    #     T_out = self.calculate_output_len(audio_dur, sample_rate)

    #     # 2) center time bins (80ms chunks)
    #     chunk_dur = audio_dur / T_out
    #     times = (np.arange(T_out) + 0.5) * chunk_dur

    #     # 3) find segment index for each bin
    #     seg_idx = np.searchsorted(tp, times, side='right') - 1
    #     seg_idx = np.clip(seg_idx, 0, len(trajectory) - 1)

    #     # 4) convert to class indices
    #     # class_idx = [deg_to_class_index(trajectory[s]) for s in seg_idx]
    #     # return torch.tensor(class_idx, dtype=torch.long)

    #     # Precompute: for each bin index 0…len(trajectory)-1, map trajectory angle → class.
    #     traj_arr   = np.array(trajectory)            # shape (S,)
    #     # idx_array  = np.round((traj_arr + 90.0)/5.0).astype(int).clip(0,36)
    #     idx_array = np.floor((traj_arr + 90.0) / 5.0).astype(int).clip(0, 36)
    #     class_idx  = idx_array[seg_idx]              # fancy-index all at once
    #     return torch.from_numpy(class_idx).long()

    def wrap_angle(self,angle_deg):
        """Convert 0–360° to range [-180°, +180°], then clip to [-90°, +90°]"""
        angle_wrapped = ((angle_deg + 180) % 360) - 180
        return np.clip(angle_wrapped, -90, 90)

    # def prepare_traj_reference(self, trajectory, tp, sample_rate):
    #     """
    #     Converts raw trajectory angles (possibly in 0–360°) into class indices [0–36] representing -90° to +90°
    #     """
    #     # 1) determine number of output bins
    #     audio_dur = tp[-1]
    #     T_out = self.calculate_output_len(audio_dur, sample_rate)

    #     # 2) center time bins (80ms chunks)
    #     chunk_dur = audio_dur / T_out
    #     times = (np.arange(T_out) + 0.5) * chunk_dur

    #     # 3) find segment index for each bin
    #     seg_idx = np.searchsorted(tp, times, side='right') - 1
    #     seg_idx = np.clip(seg_idx, 0, len(trajectory) - 1)

    #     # 4) Wrap angles to [-90°, +90°] and convert to class indices
    #     traj_arr = np.array(trajectory)
    #     traj_wrapped = self.wrap_angle(traj_arr)  # Now in [-90°, +90°]
    #     idx_array = np.floor((traj_wrapped + 90.0) / 5.0).astype(int)  # Class 0–36

    #     class_idx = idx_array[seg_idx]  # Fancy index per time bin
    #     return torch.from_numpy(class_idx).long()
    def prepare_traj_reference(self, trajectory, tp, sample_rate, max_len=30):
        """
        Converts raw trajectory angles in [-90°, +90°] into class indices [0–36],
        selects the class at each time bin center, and pads the sequence to `max_len` with -100.
        """
        # 1) determine number of output bins
        audio_dur = tp[-1]
        T_out = self.calculate_output_len(audio_dur, sample_rate)

        # 2) center time bins (e.g., 80ms chunks)
        chunk_dur = audio_dur / T_out
        times = (np.arange(T_out) + 0.5) * chunk_dur

        # 3) segment index for each time bin
        seg_idx = np.searchsorted(tp, times, side='right') - 1
        seg_idx = np.clip(seg_idx, 0, len(trajectory) - 1)

        # 4) convert wrapped angles to class indices
        traj_arr = np.array(trajectory)  # assumed already wrapped to [-90, +90]
        idx_array = np.floor((traj_arr + 90.0) / 5.0).astype(int)  # [0–36]
        class_idx = idx_array[seg_idx]

        # 5) pad to `max_len` with -100
        pad_len = max_len - len(class_idx)
        if pad_len > 0:
            class_idx = np.pad(class_idx, (0, pad_len), mode='constant', constant_values=-100)
        else:
            class_idx = class_idx[:max_len]  # in case of slight overflow

        return torch.from_numpy(class_idx).long()

    def wrap_frontal(self,angle_deg):
        """Map 0–360° to frontal [-90°, +90°] azimuths."""
        angle_deg = np.asarray(angle_deg)
        angle_deg = np.where(angle_deg > 180, angle_deg - 360, angle_deg)
        return np.clip(angle_deg, -90, 90)
    def process_row(self,row_str):
        # angles = ast.literal_eval(row_str)
        angles = row_str
        angles = np.array(angles)

        # Identify original angles outside frontal hemifield (>90 and <270)
        # mask_out_of_frontal = (angles > 90) & (angles < 270)
        # has_out_of_frontal = np.any(mask_out_of_frontal)

        # Map to frontal representation
        wrapped = self.wrap_frontal(angles)
        return wrapped.tolist()#, has_out_of_frontal
    

    
    # def process_trajectory_column(self,df, column_name):
    #     """
    #     Converts stringified trajectory angles in df[column_name] to frontal [-90, 90] representation.
    #     Flags angles outside frontal hemifield.
    #     Returns new columns: 'trajectory_processed' and 'has_out_of_frontal'.
    #     """

    #     # Apply processing
    #     processed = df[column_name].apply(process_row)
        
    #     # Unpack into two new columns
    #     df['trajectory_processed'] = processed.apply(lambda x: x[0])
    #     df['has_out_of_frontal'] = processed.apply(lambda x: x[1])

    #     return df
    def padding(self, feature, window):
        self.enc_stride = 32  # stride for the encoder
        batch_size, n_ch, n_sample = feature.shape

        rest = window - (self.enc_stride + n_sample % window) % window
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, n_ch, rest)).type(feature.type())
            feature = torch.cat([feature, pad], 2)

        if window == 512:
            pad_aux = Variable(torch.zeros(batch_size, n_ch, self.enc_stride+120)).type(feature.type())
        else:
            pad_aux = Variable(torch.zeros(batch_size, n_ch, self.enc_stride)).type(feature.type())
            
        feature = torch.cat([pad_aux, feature, pad_aux], 2)
    
        return feature, rest


    def get_frequency_features(self, waveform):
        self.stft_win = 512
        self.stft_hop = 32
        self.eps = 1e-8
        input_fd, _ = self.padding(waveform, self.stft_win)

        phase_left = torch.stft(input_fd[:,0,:], self.stft_win, hop_length=self.stft_hop, window=torch.hann_window(self.stft_win).to(waveform.device), center=False, return_complex=True)
        phase_right = torch.stft(input_fd[:,1,:], self.stft_win, hop_length=self.stft_hop, window=torch.hann_window(self.stft_win).to(waveform.device), center=False, return_complex=True)

        phase_left = torch.view_as_real(phase_left)
        phase_right = torch.view_as_real(phase_right)

        # IPD and IID
        IPD = torch.atan2(phase_left[:,:,:,1],phase_left[:,:,:,0]) - torch.atan2(phase_right[:,:,:,1],phase_right[:,:,:,0])
        IPD_cos = torch.cos(IPD)
        IPD_sin = torch.sin(IPD)
        IPD_feature = torch.cat([IPD_cos, IPD_sin], 1)

        IID = torch.log(phase_left[:,:,:,1]**2 + phase_left[:,:,:,0]**2 + self.eps) - torch.log(phase_right[:,:,:,1]**2 + phase_right[:,:,:,0]**2 + self.eps)

        freq_feature = torch.cat([IPD_feature, IID], 1)

        return freq_feature