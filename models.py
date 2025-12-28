import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.model_utils import cLN, TCN

class SeparationNetONNX(nn.Module):
    """Separation network
    This is a simplified version of the SeparationNet class, designed for ONNX export.
    We dont return complex tensors in the stft computation.
    """
    def __init__(
        self, 
        enc_dim = 64,
        feature_dim = 64,
        hidden_dim = 256, 
        enc_win = 64,
        enc_stride = 32,
        num_block = 5,
        num_layer = 7,
        kernel_size = 3,
        stft_win = 512,
        stft_hop = 32,
        num_spk = 2,
    ):
        super(SeparationNetONNX, self).__init__()
    
        # hyper parameters
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = self.feature_dim*4

        self.enc_win = enc_win
        self.enc_stride = enc_stride

        self.num_layer = num_layer
        self.num_block = num_block
        self.kernel_size = kernel_size

        self.stft_win = stft_win
        self.stft_hop = stft_hop
        self.stft_dim = stft_win//2 + 1

        self.num_spk = num_spk

        # input encoder
        self.encoder1 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        self.encoder2 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        # Causal layer normalization
        self.LN = cLN(self.enc_dim*2, eps=1e-8)
        # Bottom neck layer
        self.BN = nn.Conv1d(self.enc_dim*2+self.stft_dim*3, self.feature_dim, 1, bias=False)

        # TCN encoder
        self.TCN = TCN(self.feature_dim, self.enc_dim*self.num_spk*2, self.num_layer, 
                       self.num_block, self.hidden_dim, self.kernel_size, causal=True)
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.enc_win, stride=self.enc_stride, bias=False)

        self.eps = 1e-12


    def padding(self, feature, window):

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
        """
        Args:
            waveform:  (B, 2, L)  real‐valued float32
        Returns:
            freq_feature:  a purely float32 tensor of shape (B, 3*self.stft_dim, T)
                           (no torch.complex anywhere)
        """
        # 1) pad the input to a multiple of stft_win (just like you did before)
        input_fd, _ = self.padding(waveform, self.stft_win)

        # 2) compute STFT WITHOUT ever returning a true complex tensor
        #    -> return_complex=False yields shape (B, freq_bins, time_frames, 2)
        X_left  = torch.stft(
            input_fd[:, 0, :],
            self.stft_win,
            hop_length=self.stft_hop,
            window=torch.hann_window(self.stft_win).to(waveform.device),
            center=False,
            return_complex=False,  # <— get (real, imag) packed into last dim
        )  # shape = (B, stft_dim, nFrames, 2)
        X_right = torch.stft(
            input_fd[:, 1, :],
            self.stft_win,
            hop_length=self.stft_hop,
            window=torch.hann_window(self.stft_win).to(waveform.device),
            center=False,
            return_complex=False,
        )

        # 3) split real/imag
        real_left  = X_left[..., 0]   # (B, stft_dim, nFrames)
        imag_left  = X_left[..., 1]   # (B, stft_dim, nFrames)
        real_right = X_right[..., 0]  # (B, stft_dim, nFrames)
        imag_right = X_right[..., 1]  # (B, stft_dim, nFrames)

        # 4) compute IPD = angle_diff(left, right):
        #    angle = atan2(imag, real)
        ang_left  = torch.atan2(imag_left, real_left)    # (B, stft_dim, nFrames)
        ang_right = torch.atan2(imag_right, real_right)  # (B, stft_dim, nFrames)
        IPD = ang_left - ang_right                        # (B, stft_dim, nFrames)

        IPD_cos = torch.cos(IPD)  # (B, stft_dim, nFrames)
        IPD_sin = torch.sin(IPD)  # (B, stft_dim, nFrames)

        # 5) compute IID = log(mag²_left + eps) - log(mag²_right + eps)
        mag2_left  = real_left**2  + imag_left**2  + self.eps
        mag2_right = real_right**2 + imag_right**2 + self.eps
        IID = torch.log(mag2_left) - torch.log(mag2_right)  # (B, stft_dim, nFrames)

        # 6) stack them into a single real‐valued feature
        #    IPD_cos and IPD_sin each have “stft_dim” channels,
        #    IID has “stft_dim” channels, so total = 3 * stft_dim
        IPD_feature = torch.cat([IPD_cos, IPD_sin], dim=1)     # (B, 2*stft_dim, T)
        freq_feature = torch.cat([IPD_feature, IID], dim=1)    # (B, 3*stft_dim, T)

        return freq_feature



    def forward(self, input,freq_features):
        """
        Args: 
            input: mixed waveform (B, 2, L)

        Returns:
            separated waveforms in the left channel (B, num_spk, L)
            separated waveforms in the right channel (B, num_spk, L)

        """

        batch_size, n_ch, n_sample = input.shape # B, 2, L

        input_td, rest = self.padding(input, self.enc_win) 

        # encoder
        enc_map_left = self.encoder1(input_td[:,0,:].unsqueeze(1))  # B, N, T
        enc_map_right = self.encoder2(input_td[:,1,:].unsqueeze(1))  # B, N, T

        # Apply layer normalization
        enc_features = self.LN(torch.cat([enc_map_left, enc_map_right], dim=1))

        # Cross domain features
        # freq_features = self.get_frequency_features(input)
        freq_features = freq_features[:, :, :enc_features.shape[-1]]

        print("    enc_features.shape =", enc_features.shape)
        print("    freq_features.shape =", freq_features.shape)

        all_features = torch.cat([enc_features, freq_features], 1)

        # TCN separator
        # generate C feature matrices for the C speakers
        mask = torch.sigmoid(self.TCN(self.BN(all_features))).view(batch_size, self.enc_dim, self.num_spk, 2, -1)  # B, H, 2, T

        # left channel
        output_left = torch.cat([mask[:, :, i, 0, :] * enc_map_left for i in range(self.num_spk)], 0)  # B*num_spk, H, T
        output_left = self.decoder(output_left)  # B*C, 1, L
        output_left = output_left[:,:,self.enc_stride:-(rest+self.enc_stride)].contiguous()  # B*num_spk, 1, L
        output_left = torch.cat([output_left[batch_size*i:batch_size*(i+1), :, :] for i in range(self.num_spk)], 1)  # B, num_spk, L

        # right channel
        output_right = torch.cat([mask[:, :, i, 1, :] * enc_map_right for i in range(self.num_spk)], 0)  # B*num_spk, H, T
        output_right = self.decoder(output_right)  # B*C, 1, L
        output_right = output_right[:,:,self.enc_stride:-(rest+self.enc_stride)].contiguous()  # B*num_spk, 1, L
        output_right = torch.cat([output_right[batch_size*i:batch_size*(i+1), :, :] for i in range(self.num_spk)], 1)  # B, num_spk, L

        return output_left, output_right
class SeparationNet(nn.Module):
    """Separation network"""
    def __init__(
        self, 
        enc_dim = 64,
        feature_dim = 64,
        hidden_dim = 256, 
        enc_win = 64,
        enc_stride = 32,
        num_block = 5,
        num_layer = 7,
        kernel_size = 3,
        stft_win = 512,
        stft_hop = 32,
        num_spk = 2,
    ):
        super(SeparationNet, self).__init__()
    
        # hyper parameters
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = self.feature_dim*4

        self.enc_win = enc_win
        self.enc_stride = enc_stride

        self.num_layer = num_layer
        self.num_block = num_block
        self.kernel_size = kernel_size

        self.stft_win = stft_win
        self.stft_hop = stft_hop
        self.stft_dim = stft_win//2 + 1

        self.num_spk = num_spk

        # input encoder
        self.encoder1 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        self.encoder2 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        # Causal layer normalization
        self.LN = cLN(self.enc_dim*2, eps=1e-8)
        # Bottom neck layer
        self.BN = nn.Conv1d(self.enc_dim*2+self.stft_dim*3, self.feature_dim, 1, bias=False)

        # TCN encoder
        self.TCN = TCN(self.feature_dim, self.enc_dim*self.num_spk*2, self.num_layer, 
                       self.num_block, self.hidden_dim, self.kernel_size, causal=True)
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.enc_win, stride=self.enc_stride, bias=False)

        self.eps = 1e-12


    def padding(self, feature, window):

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


    def forward(self, input):
        """
        Args: 
            input: mixed waveform (B, 2, L)

        Returns:
            separated waveforms in the left channel (B, num_spk, L)
            separated waveforms in the right channel (B, num_spk, L)

        """

        batch_size, n_ch, n_sample = input.shape # B, 2, L

        input_td, rest = self.padding(input, self.enc_win) 

        # encoder
        enc_map_left = self.encoder1(input_td[:,0,:].unsqueeze(1))  # B, N, T
        enc_map_right = self.encoder2(input_td[:,1,:].unsqueeze(1))  # B, N, T

        # Apply layer normalization
        enc_features = self.LN(torch.cat([enc_map_left, enc_map_right], dim=1))

        # Cross domain features
        freq_features = self.get_frequency_features(input)
        freq_features = freq_features[:, :, :enc_features.shape[-1]]

        # print("    enc_features.shape =", enc_features.shape)
        # print("    freq_features.shape =", freq_features.shape)

        all_features = torch.cat([enc_features, freq_features], 1)

        # TCN separator
        # generate C feature matrices for the C speakers
        mask = torch.sigmoid(self.TCN(self.BN(all_features))).view(batch_size, self.enc_dim, self.num_spk, 2, -1)  # B, H, 2, T

        # left channel
        output_left = torch.cat([mask[:, :, i, 0, :] * enc_map_left for i in range(self.num_spk)], 0)  # B*num_spk, H, T
        output_left = self.decoder(output_left)  # B*C, 1, L
        output_left = output_left[:,:,self.enc_stride:-(rest+self.enc_stride)].contiguous()  # B*num_spk, 1, L
        output_left = torch.cat([output_left[batch_size*i:batch_size*(i+1), :, :] for i in range(self.num_spk)], 1)  # B, num_spk, L

        # right channel
        output_right = torch.cat([mask[:, :, i, 1, :] * enc_map_right for i in range(self.num_spk)], 0)  # B*num_spk, H, T
        output_right = self.decoder(output_right)  # B*C, 1, L
        output_right = output_right[:,:,self.enc_stride:-(rest+self.enc_stride)].contiguous()  # B*num_spk, 1, L
        output_right = torch.cat([output_right[batch_size*i:batch_size*(i+1), :, :] for i in range(self.num_spk)], 1)  # B, num_spk, L

        return output_left, output_right
    
    
class EnhancementNet(nn.Module):
    """Post enhancement module"""
    def __init__(
        self, 
        enc_dim = 64,
        feature_dim = 64,
        hidden_dim = 256, 
        enc_win = 64,
        enc_stride = 32,
        num_block = 5,
        num_layer = 7,
        kernel_size = 3,
        num_spk = 1,
    ):
        super(EnhancementNet, self).__init__()
    
        # hyper parameters
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = self.feature_dim*4

        self.enc_win = enc_win
        self.enc_stride = enc_stride

        self.num_layer = num_layer
        self.num_block = num_block
        self.kernel_size = kernel_size

        self.num_spk = num_spk

        # input encoder
        self.encoder1 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        self.encoder2 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        self.encoder3 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        self.encoder4 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        # Causal layer normalization
        self.LN = cLN(self.enc_dim*4, eps=1e-8)
        # Bottom neck layer
        self.BN = nn.Conv1d(self.enc_dim*4, self.feature_dim, 1, bias=False)

        # TCN encoder
        self.TCN = TCN(self.feature_dim, self.enc_dim*4, self.num_layer, 
                       self.num_block, self.hidden_dim, self.kernel_size, causal=True)
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.enc_win, stride=self.enc_stride, bias=False)

        self.eps = 1e-12


    def padding(self, feature, window):

        batch_size, n_ch, n_sample = feature.shape

        rest = window - (self.enc_stride + n_sample % window) % window
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, n_ch, rest)).type(feature.type())
            feature = torch.cat([feature, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, n_ch, self.enc_stride)).type(feature.type())
            
        feature = torch.cat([pad_aux, feature, pad_aux], 2)
    
        return feature, rest

    
    def forward(self, mixture, separated_waveform):
        """
        Args: 
            mixture: mixed waveform (B, 2, L)
            separated_waveform: separated waveform of the target speaker from the separation module (B, 2, L)

        Returns:
            separated waveforms in the left channel (B, num_spk, L)
            separated waveforms in the right channel (B, num_spk, L)

        """
        batch_size, n_ch, n_sample = mixture.shape # B, 2, L

        # print("    mixture.shape =", mixture.shape, separated_waveform.shape)

        mixture_td, rest = self.padding(mixture, self.enc_win) 
        separated_waveform_td, _ = self.padding(separated_waveform, self.enc_win) 

        # encoder
        mix_enc_map_left = self.encoder1(mixture_td[:,0,:].unsqueeze(1))  # B, N, T
        mix_enc_map_right = self.encoder2(mixture_td[:,1,:].unsqueeze(1))  # B, N, T
        sep_enc_map_left = self.encoder3(separated_waveform_td[:,0,:].unsqueeze(1))  # B, N, T
        sep_enc_map_right = self.encoder4(separated_waveform_td[:,1,:].unsqueeze(1))  # B, N, T

        # Apply layer normalization
        enc_features = self.LN(torch.cat([mix_enc_map_left, mix_enc_map_right, sep_enc_map_left, sep_enc_map_right], dim=1))

        # TCN separator
        # generate C feature matrices for the C speakers
        masks = torch.sigmoid(self.TCN(self.BN(enc_features))).view(batch_size, self.enc_dim, 2, 2, -1)  # B, H, 2, T
        # mask and sum
        masked_feature_left =  masks[:,:,0,0,:] * mix_enc_map_left + masks[:,:,0,1,:] * mix_enc_map_right
        masked_feature_right = masks[:,:,1,0,:] * mix_enc_map_right + masks[:,:,1,1,:] * mix_enc_map_left
        masked_features = torch.cat([masked_feature_left, masked_feature_right], dim=0)  # 2*B, H, T
        
        # waveform decoder
        output = self.decoder(masked_features)  # 2*B, 1, L
        output = output[:,:,self.enc_stride:-(rest+self.enc_stride)].contiguous()  # 2*B, 1, L
        output = torch.cat([output[batch_size*i:batch_size*(i+1), :, :] for i in range(2)], 1)  # B, 2, L
        
        return output


class TrajectoryNet(nn.Module):
    """Trajectory network"""
    def __init__(
        self, 
        feature_dim = 64,
        hidden_dim = 256, 
        num_block = 5,
        num_layer = 7,
        kernel_size = 3,
        stft_win = 512,
        stft_hop = 32,
        num_cls = 37,
    ):
        super(TrajectoryNet, self).__init__()
    
        # hyper parameters
        self.feature_dim = feature_dim
        self.hidden_dim = self.feature_dim*4

        self.num_layer = num_layer
        self.num_block = num_block
        self.kernel_size = kernel_size

        self.stft_win = stft_win
        self.stft_hop = stft_hop
        self.stft_dim = stft_win//2 + 1
        
        self.num_cls = num_cls

        # Bottom neck layer
        self.BN = nn.Conv1d(self.stft_dim*4, self.feature_dim, 1, bias=False)

        # TCN encoder
        self.TCN = TCN(self.feature_dim, self.feature_dim, self.num_layer, 
                       self.num_block, self.hidden_dim, self.kernel_size, causal=True)
        
        # self.loc = nn.Sequential(nn.AvgPool1d(10, stride=10, padding=5),
        #                          nn.PReLU(),
        #                          nn.Conv1d(self.feature_dim, self.feature_dim, 1),
        #                          nn.PReLU(),
        #                          nn.Conv1d(self.feature_dim, self.num_cls, 1)
                                # )

        ##we should 80ms chunks
        frames_per_chunk = int(0.08 * 16000 / self.stft_hop)  # = 40
        self.loc = nn.Sequential(
            nn.AvgPool1d(kernel_size=frames_per_chunk, stride=frames_per_chunk, padding=frames_per_chunk // 2),
            nn.PReLU(),
            nn.Conv1d(self.feature_dim, self.feature_dim, 1),
            nn.PReLU(),
            nn.Conv1d(self.feature_dim, self.num_cls, 1)
        )

        self.eps = 1e-12


    def forward(self, input):
        """
        Args: 
            input: mixed waveform (B, 2, L)

        Returns:
            localization logits (B, L, num_cls)

        """
        batch_size, n_ch, n_sample = input.shape # B, 2, L
        
        stft_left = torch.stft(input[:,0,:], self.stft_win, hop_length=self.stft_hop, window=torch.hann_window(self.stft_win).to(input.device), center=False, return_complex=True)
        stft_right = torch.stft(input[:,1,:], self.stft_win, hop_length=self.stft_hop, window=torch.hann_window(self.stft_win).to(input.device), center=False, return_complex=True)

        stft_left = torch.view_as_real(stft_left)
        stft_right = torch.view_as_real(stft_right)
        
        stft_feat = torch.cat([stft_left[:,:,:,0], stft_left[:,:,:,1],
                               stft_right[:,:,:,0], stft_right[:,:,:,1]], dim=1)
        
        # localization on top of tcn features
        tcn_feature = self.TCN(self.BN(stft_feat))
        loc_res = self.loc(tcn_feature) # B, num_cls, T
        loc_res = loc_res.permute(0,2,1) # B, T, num_cls
                
        return loc_res