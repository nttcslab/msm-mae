"""MSM-MAE wrapper for EVAR.
"""

import sys
sys.path.append('..')
from evar.ar_base import BaseAudioRepr, calculate_norm_stats, normalize_spectrogram
import torch
from msm_mae.runtime import RuntimeMAE


class AR_MSM_MAE_BatchNormStats(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.backbone = RuntimeMAE(cfg=cfg, weight_file=cfg.weight_file)
        self.backbone.eval()

    def encode_frames(self, batch_audio):
        with torch.no_grad():
            x = self.backbone.get_timestamp_embeddings(batch_audio)
        return x.transpose(1, 2) # [B, T, D] -> [B, D, T]

    def forward(self, batch_audio):
        with torch.no_grad():
            x = self.backbone.get_scene_embeddings(batch_audio)
        return x


class AR_MSM_MAE(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.runtime = RuntimeMAE(cfg=cfg, weight_file=cfg.weight_file)
        self.runtime.eval()

    def precompute(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, self.runtime.to_feature)

    def encode_frames(self, batch_audio):
        x = self.runtime.to_feature(batch_audio)
        x = normalize_spectrogram(self.norm_stats, x)

        with torch.no_grad():
            x = self.runtime.encode_lms(x)
        return x.transpose(1, 2) # [B, T, D] -> [B, D, T]

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        return x.mean(dim=-1) # [B, D, T] -> [B, D]
