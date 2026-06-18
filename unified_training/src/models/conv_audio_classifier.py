"""
1D convolution-based binary classifier for audio spoofing detection.
Uses Mel spectrogram or LFCC features (standard in antispoofing) instead of raw waveform.
VAD preprocessing is applied in the data collator (train: VAD+crop; test: VAD+pad).
At test time, the test dataloader produces overlapping windows; generate() runs forward on them;
test epoch aggregates window-level probs per sample.
"""

from typing import Dict, Any, List, Union, Optional
import math
import torch
import torch.nn as nn
import torchaudio
from .base import Model
from ..utils.focal_loss import focal_binary_cross_entropy_with_logits


# Default antispoofing frontend settings (16 kHz)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_FFT = 512
DEFAULT_WIN_LENGTH = 400  # 25 ms at 16 kHz
DEFAULT_HOP_LENGTH = 160  # 10 ms
DEFAULT_N_MELS = 80
DEFAULT_N_LFCC = 60


class AudioFeatureExtractor(nn.Module):
    """
    Extracts Mel spectrogram or LFCC from raw waveform. Standard frontend for antispoofing.
    Input: (B, 1, T) waveform; output: (B, C, T_frames) where C is n_mels or n_lfcc.
    """

    def __init__(
        self,
        feature_type: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = DEFAULT_N_FFT,
        win_length: int = DEFAULT_WIN_LENGTH,
        hop_length: int = DEFAULT_HOP_LENGTH,
        n_mels: int = DEFAULT_N_MELS,
        n_lfcc: int = DEFAULT_N_LFCC,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        self.feature_type = feature_type.lower()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self._n_freqs = n_fft // 2 + 1

        if self.feature_type == "mel":
            self._out_channels = n_mels
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max or float(sample_rate // 2),
                power=2.0,
            )
            self.linear_filterbank = None
        elif self.feature_type == "lfcc":
            self._out_channels = n_lfcc
            self.spectrogram = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                power=2.0,
            )
            # Linear filterbank: group freq bins into n_lfcc bands (equal bins per band)
            W = torch.zeros(n_lfcc, self._n_freqs)
            bins_per_band = self._n_freqs / n_lfcc
            for i in range(n_lfcc):
                start = int(i * bins_per_band)
                end = int((i + 1) * bins_per_band)
                end = min(end, self._n_freqs)
                if start < end:
                    W[i, start:end] = 1.0
            self.register_buffer("linear_filterbank", W)
            # DCT-II orthonormal matrix: D[k,n] = c_k * cos(pi * k * (2n+1) / (2*N)), c_0=1/sqrt(N), c_k=sqrt(2/N)
            n = n_lfcc
            dct_mat = torch.zeros(n, n)
            for k in range(n):
                scale = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
                for i in range(n):
                    dct_mat[k, i] = scale * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
            self.register_buffer("_dct_matrix", dct_mat)
            self.mel_spec = None
        else:
            raise ValueError(f"feature_type must be 'mel' or 'lfcc', got {feature_type!r}")

    @property
    def output_channels(self) -> int:
        return self._out_channels

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, 1, T)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        device = waveform.device

        if self.feature_type == "mel":
            # (B, n_mels, T_frames) — squeeze in case transform returns (B, 1, n_mels, T)
            mel = self.mel_spec(waveform)
            if mel.dim() == 4:
                mel = mel.squeeze(1)
            # Log compression (standard for antispoofing)
            features = torch.log(mel.clamp(min=1e-5))
            return features

        # LFCC: power spec -> linear filterbank -> log -> DCT
        power = self.spectrogram(waveform)  # (B, n_freqs, T_frames) or (B, 1, n_freqs, T)
        if power.dim() == 4:
            power = power.squeeze(1)
        # (B, n_freqs, T) @ linear_filterbank.T -> (B, n_lfcc, T)
        band_energy = torch.einsum("bft,lf->blt", power, self.linear_filterbank.to(device))
        log_bands = torch.log(band_energy.clamp(min=1e-5))
        # DCT-II along the coefficient dimension (per frame): (B, n_lfcc, T) via D @ log_bands
        dct_mat = self._dct_matrix.to(device)
        features = torch.einsum("ln,bnt->blt", dct_mat, log_bands)
        return features


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        pool: int = 1,
        padding: Optional[int] = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        return self.pool(x)


class ConvAudioClassifierNet(nn.Module):
    """
    Lightweight 1D CNN over time: (batch, n_mels or n_lfcc, T_frames) -> logits.
    Input is log-Mel or LFCC features; convolutions run along the time axis.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 32, 8, stride=2, pool=2),
            ConvBlock(32, 64, 3, pool=2),
            ConvBlock(64, 128, 3, pool=2),
            ConvBlock(128, 128, 3, pool=1),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T_frames)
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return self.fc(x).squeeze(-1)


class ConvAudioClassifierModel(Model):
    """
    Binary classifier for spoofing detection. Expects raw_wav (VAD + crop by collator).
    Uses Mel spectrogram or LFCC features (standard in antispoofing) before the CNN.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        cc = config.get("additional_kwargs", {}).get("conv_audio_classifier", {})
        feature_type = cc.get("feature_type", "mel")
        sample_rate = cc.get("sample_rate", DEFAULT_SAMPLE_RATE)
        n_fft = cc.get("n_fft", DEFAULT_N_FFT)
        win_length = cc.get("win_length", DEFAULT_WIN_LENGTH)
        hop_length = cc.get("hop_length", DEFAULT_HOP_LENGTH)
        n_mels = cc.get("n_mels", DEFAULT_N_MELS)
        n_lfcc = cc.get("n_lfcc", DEFAULT_N_LFCC)

        self.frontend = AudioFeatureExtractor(
            feature_type=feature_type,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            n_lfcc=n_lfcc,
        )
        self.net = ConvAudioClassifierNet(in_channels=self.frontend.output_channels)

        self.use_focal_loss = cc.get("use_focal_loss", config.get("use_focal_loss", False))
        self.focal_gamma = cc.get("focal_gamma", config.get("focal_gamma", 2.0))
        self.focal_alpha = cc.get("focal_alpha", config.get("focal_alpha", None))

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            self._load_checkpoint(ckpt_path)

    def _load_checkpoint(self, ckpt_path: str):
        import os
        if not os.path.isfile(ckpt_path) or not ckpt_path.endswith(".pt"):
            return
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        # Strip 'module.' prefix if saved under DDP
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k.replace("module.", "", 1) if k.startswith("module.") else k
            new_state[name] = v
        self.load_state_dict(new_state, strict=False)

    def _raw_wav_to_features(self, raw_wav: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(raw_wav, torch.Tensor):
            x = raw_wav.to(device=device, dtype=dtype)
        else:
            x = torch.tensor(raw_wav, device=device, dtype=dtype)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.frontend(x)

    def forward(self, samples: Dict[str, Any], verbose: bool = False) -> Dict[str, torch.Tensor]:
        raw_wav = samples.get("raw_wav")
        if raw_wav is None:
            raise ValueError("conv_audio_classifier expects 'raw_wav' in batch")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        features = self._raw_wav_to_features(raw_wav, device, dtype)
        logits = self.net(features)
        probs = torch.sigmoid(logits)
        is_bonafide = samples["is_bonafide"].to(device=probs.device, dtype=torch.float32)
        if self.use_focal_loss:
            loss = focal_binary_cross_entropy_with_logits(
                logits, is_bonafide, gamma=self.focal_gamma, alpha=self.focal_alpha, reduction="none"
            )
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, is_bonafide, reduction="none")
        out = {"loss": loss}
        if verbose:
            pred = (probs >= 0.5).long()
            gt = (is_bonafide >= 0.5).long()
            correct = (pred == gt).sum().item()
            out["correct"] = correct
            out["total"] = pred.numel()
        return out

    def generate(
        self,
        samples: Dict[str, Any],
        generate_cfg: Dict[str, Any],
        prompts: Optional[List[str]] = None,
        return_outputs: bool = False,
    ) -> Union[List[str], Any]:
        self.eval()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        with torch.no_grad():
            raw_wav = samples.get("raw_wav")
            if raw_wav is None:
                raise ValueError("conv_audio_classifier expects 'raw_wav' in batch")
            features = self._raw_wav_to_features(raw_wav, device, dtype)
            # Batch is already windowed by test dataloader: (num_windows, 1, window_samples)
            logits = self.net(features)
            probs = torch.sigmoid(logits)
            windows_per_sample = samples.get("windows_per_sample")

            if return_outputs and windows_per_sample is not None:
                dummy_ids = torch.zeros(len(windows_per_sample), 1, device=device, dtype=torch.long)
                return (
                    {"window_probs": probs.cpu(), "windows_per_sample": windows_per_sample},
                    dummy_ids,
                    None,
                )
            # Single segment per sample (e.g. train/val or no windowing): one prob per batch item
            if windows_per_sample is None:
                texts = [
                    "Final Answer: Real" if p >= 0.5 else "Final Answer: Fake"
                    for p in probs.cpu().tolist()
                ]
                if return_outputs:
                    batch_size = len(texts)
                    dummy_ids = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
                    return texts, dummy_ids, None
                return texts
            # Backward compat: aggregate window probs when windows_per_sample present but not return_outputs
            texts = []
            offset = 0
            for n_w in windows_per_sample:
                mean_prob = probs[offset : offset + n_w].mean().item()
                texts.append(
                    "Final Answer: Real" if mean_prob >= 0.5 else "Final Answer: Fake"
                )
                offset += n_w
            return texts

    def compute_logits_for_completions(
        self, samples: Dict[str, Any], completion_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return None

    def generate_text_only(
        self, prompt_texts: List[str], generate_cfg: Dict[str, Any]
    ) -> List[str]:
        return []

    def get_tokenizer(self):
        return None
