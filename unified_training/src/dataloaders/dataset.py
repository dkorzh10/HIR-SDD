import json
import os
import random
import re
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

class AudioDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        target_sample_rate: int = 16000,
        task_type: str = "hard_label",
        samples_offset: int = 0,
        grounding_cfg: Optional[Dict[str, Any]] = None,
        shuffle_reasons: bool = False,
        max_audio_duration_s: Optional[float] = 30.0,
        max_len_seconds: Optional[float] = None,
    ):
        self.data_path = data_path
        self.target_sample_rate = target_sample_rate
        self.task_type = task_type
        self.samples: List[Dict[str, Any]] = []
        self.grounding_cfg = grounding_cfg or {}
        self._grounding_phrases = self._load_grounding_phrases()
        
        self.max_audio_samples = (
            int(max_audio_duration_s * target_sample_rate) if max_audio_duration_s is not None else None
        )
        self.max_len_samples = (
            int(max_len_seconds * target_sample_rate) if max_len_seconds is not None else None
        )

        self.shuffle_reasons = shuffle_reasons
        self.samples_offset = samples_offset
        self._num_broken = 0
        self._missing_audio_warns = 0

        self._load_data(max_samples)


    def _load_data(self, max_samples: Optional[int]):
        if self.data_path == "dummy":
            self.samples = [
                {
                    "audio_id": f"dummy_{i}",
                    "original_path": f"/tmp/dummy_audio_{i}.wav",
                    "is_bonafide": i % 2 == 0,
                    "reasons": {"strange_voice": True} if i % 2 != 0 else None,
                    "reasoning": "This sounds fake." if i % 2 != 0 else "This sounds real."
                }
                for i in range(100)
            ]
            # Continue to max_samples slicing
        else:
            try:
                with open(self.data_path, 'r') as f:
                    if self.data_path.endswith('.json'):
                        data = json.load(f)
                        # If list, assign directly
                        if isinstance(data, list):
                            self.samples = data
                        # If dict (e.g. wrapper), might be data['samples'] or similar - assuming list for now based on description
                        else:
                            print(f"Warning: Unexpected JSON format in {self.data_path}. Expecting list.")
                            self.samples = []
                    elif self.data_path.endswith('.jsonl'):
                        self.samples = [json.loads(line) for line in f]
            except FileNotFoundError:
                print(f"Warning: Dataset file {self.data_path} not found. Using empty dataset.")
                self.samples = []

        if max_samples:
            self.samples = self.samples[:max_samples]

    def _load_audio(self, path: str):
        if path.startswith("/tmp/dummy"):
            # Return dummy tensor
            return torch.randn(1, 16000), 16000
        
        if not os.path.exists(path):
            # Missing audio should be skipped, not silently converted to 1s silence.
            if self._missing_audio_warns < 20:
                print(f"Warning: Audio not found, skipping sample: {path}", flush=True)
                self._missing_audio_warns += 1
            return None, None
            
        waveform, sample_rate = torchaudio.load(path)
        # try:
        #     # Try torchaudio first, if it works
        # except (ImportError, RuntimeError, TypeError) as e:
        #     # TypeError can occur when file has no audio stream (default_audio_stream is None)
        #     # Fallback to soundfile directly if torchaudio fails (e.g. missing torchcodec)
        #     try:
        #         waveform_np, sample_rate = sf.read(path)
        #         # Soundfile returns (time, channels) or (time,)
        #         waveform = torch.from_numpy(waveform_np)
        #         if waveform.ndim == 1:
        #             waveform = waveform.unsqueeze(0) # (1, time)
        #         else:
        #             waveform = waveform.transpose(0, 1) # (channels, time)
        #         waveform = waveform.float()
        #     except Exception as sf_e:
        #          # If both fail, raise the original error or a combined one
        #          print(f"Failed to load audio {path}. Torchaudio error: {e}. Soundfile error: {sf_e}")
        #          return None, None
        
        return waveform, sample_rate

    def __len__(self):
        return len(self.samples)

    def _load_grounding_phrases(self) -> List[str]:
        inline = self.grounding_cfg.get("phrase_templates")
        if isinstance(inline, list) and inline:
            return [str(x) for x in inline]

        path = self.grounding_cfg.get("phrases_path")
        if isinstance(path, str) and path:
            try:
                with open(path, "r") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    return [str(x) for x in payload]
                if isinstance(payload, dict):
                    phrases = payload.get("grounding_phrases")
                    if isinstance(phrases, list) and phrases:
                        return [str(x) for x in phrases]
            except Exception as e:
                print(f"Warning: failed to load grounding phrases from {path}: {e}")

        return [
            "I noticed a controlled alteration around {region_desc}: {aug_desc}.",
            "A notable cue appears in {region_desc}, where {aug_desc}.",
            "In {region_desc}, the signal behaves as if {aug_desc}.",
            "One suspicious detail is in {region_desc}: {aug_desc}.",
        ]

    def _pick_weighted_augmentation(self) -> str:
        aug_cfg = self.grounding_cfg.get("augmentations", {})
        if not isinstance(aug_cfg, dict) or not aug_cfg:
            return "time_mask"

        keys, weights = [], []
        for name, cfg in aug_cfg.items():
            keys.append(name)
            weights.append(float((cfg or {}).get("weight", 1.0)))
        if sum(weights) <= 0:
            return random.choice(keys)
        return random.choices(keys, weights=weights, k=1)[0]

    def _pick_detail_mode(self) -> str:
        probs = self.grounding_cfg.get("detail_probs", {})
        exact = float(probs.get("exact", 0.3))
        binned = float(probs.get("binned", 0.4))
        type_only = float(probs.get("type_only", 0.3))
        weights = [max(exact, 0.0), max(binned, 0.0), max(type_only, 0.0)]
        if sum(weights) <= 0:
            return "type_only"
        return random.choices(["exact", "binned", "type_only"], weights=weights, k=1)[0]

    def _pick_region_mode(self) -> str:
        probs = self.grounding_cfg.get("region_format_probs", {})
        absolute = float(probs.get("absolute", 0.5))
        percent = float(probs.get("percent", 0.3))
        relative = float(probs.get("relative", 0.2))
        weights = [max(absolute, 0.0), max(percent, 0.0), max(relative, 0.0)]
        if sum(weights) <= 0:
            return "absolute"
        return random.choices(["absolute", "percent", "relative"], weights=weights, k=1)[0]

    def _relative_region_bucket(self, start_pct: float, end_pct: float) -> str:
        center = 0.5 * (start_pct + end_pct)
        if center < 0.2:
            return "the beginning of the audio"
        if center < 0.4:
            return "the early-middle portion"
        if center < 0.6:
            return "the middle portion"
        if center < 0.8:
            return "the late-middle portion"
        return "the final portion"

    def _describe_region(self, start_sec: float, end_sec: float, start_pct: float, end_pct: float) -> str:
        mode = self._pick_region_mode()
        if mode == "absolute":
            return f"from {start_sec:.2f}s to {end_sec:.2f}s"
        if mode == "percent":
            return f"roughly {start_pct * 100:.1f}% to {end_pct * 100:.1f}% of the clip"
        return self._relative_region_bucket(start_pct, end_pct)

    def _describe_augmentation(self, aug_name: str, params: Dict[str, float], detail_mode: str) -> str:
        if detail_mode == "type_only":
            simple = {
                "gain": "a loudness change was introduced",
                "gaussian_noise": "additional background noise was injected",
                "time_mask": "a short muted gap was inserted",
                "lowpass": "the high-frequency content was reduced",
                "highpass": "low-frequency content was attenuated",
                "bandstop_notch": "a narrow spectral band was suppressed",
                "clipping": "the waveform was locally clipped",
                "dropout_bursts": "brief packet-loss-like dropouts were inserted",
            }
            return simple.get(aug_name, "an acoustic perturbation was applied")

        if aug_name == "gain":
            db = float(params.get("db", 0.0))
            if detail_mode == "exact":
                direction = "increased" if db >= 0 else "decreased"
                return f"local loudness was {direction} by about {abs(db):.1f} dB"
            mag = abs(db)
            bucket = "slightly" if mag < 3 else ("moderately" if mag < 7 else "strongly")
            return f"local loudness was {bucket} changed"

        if aug_name == "gaussian_noise":
            snr = float(params.get("snr_db", 20.0))
            if detail_mode == "exact":
                return f"noise was added at around {snr:.1f} dB SNR"
            bucket = "mild" if snr > 20 else ("moderate" if snr > 12 else "strong")
            return f"{bucket} broadband noise was added"

        if aug_name == "time_mask":
            return "a brief masked/muted segment was inserted"

        if aug_name == "lowpass":
            cutoff = float(params.get("cutoff_hz", 3000.0))
            if detail_mode == "exact":
                return f"a low-pass effect was applied with cutoff near {cutoff:.0f} Hz"
            bucket = "light" if cutoff > 3500 else ("moderate" if cutoff > 2200 else "strong")
            return f"{bucket} low-pass filtering was applied"

        if aug_name == "highpass":
            cutoff = float(params.get("cutoff_hz", 250.0))
            if detail_mode == "exact":
                return f"a high-pass effect was applied with cutoff near {cutoff:.0f} Hz"
            bucket = "light" if cutoff < 200 else ("moderate" if cutoff < 500 else "strong")
            return f"{bucket} high-pass filtering was applied"

        if aug_name == "bandstop_notch":
            center_hz = float(params.get("center_hz", 2000.0))
            q = float(params.get("q", 2.0))
            if detail_mode == "exact":
                return f"a notch filter centered around {center_hz:.0f} Hz (Q≈{q:.2f}) was applied"
            bucket = "narrow" if q > 3.0 else ("medium-width" if q > 1.5 else "wide")
            return f"a {bucket} spectral notch was applied"

        if aug_name == "clipping":
            threshold = float(params.get("threshold", 0.7))
            if detail_mode == "exact":
                return f"the waveform was clipped at amplitude ±{threshold:.2f}"
            bucket = "light" if threshold > 0.8 else ("moderate" if threshold > 0.6 else "strong")
            return f"{bucket} clipping distortion was introduced"

        if aug_name == "dropout_bursts":
            n_bursts = int(params.get("n_bursts", 1))
            if detail_mode == "exact":
                return f"{n_bursts} short dropout burst(s) were inserted"
            bucket = "a few" if n_bursts <= 2 else "multiple"
            return f"{bucket} packet-loss-like dropouts were inserted"

        return "an acoustic perturbation was applied"

    def _insert_sentence_into_reasoning(self, reasoning: str, sentence: str) -> str:
        base = (reasoning or "").strip()
        sentence = sentence.strip()
        if sentence and sentence[-1] not in ".!?":
            sentence = sentence + "."
        if not base:
            return sentence
        chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", base) if c.strip()]
        # If base has no sentence punctuation, append as a separate sentence.
        if len(chunks) <= 1 and not re.search(r"[.!?]", base):
            base_norm = base if base.endswith((".", "!", "?")) else f"{base}."
            return f"{base_norm} {sentence}".strip()
        idx = random.randint(0, len(chunks))
        chunks.insert(idx, sentence)
        return " ".join(chunks).strip()

    def _maybe_apply_grounding(self, waveform: torch.Tensor, reasoning: str):
        enabled = bool(self.grounding_cfg.get("enabled", False))
        apply_prob = float(self.grounding_cfg.get("apply_prob", 0.0))
        if (not enabled) or self.task_type != "reasoning" or random.random() > apply_prob:
            return waveform, reasoning, None

        n_samples = waveform.shape[-1]
        min_span_sec = float(self.grounding_cfg.get("min_span_sec", 0.2))
        max_span_sec = float(self.grounding_cfg.get("max_span_sec", 1.2))
        min_span = max(1, int(min_span_sec * self.target_sample_rate))
        max_span = max(min_span, int(max_span_sec * self.target_sample_rate))
        if n_samples <= min_span:
            return waveform, reasoning, None

        span = random.randint(min_span, min(max_span, n_samples))
        start = random.randint(0, n_samples - span)
        end = start + span

        aug_name = self._pick_weighted_augmentation()
        aug_cfg = (self.grounding_cfg.get("augmentations", {}) or {}).get(aug_name, {}) or {}
        segment = waveform[:, start:end].clone()
        params: Dict[str, float] = {}

        if aug_name == "gain":
            min_db = float(aug_cfg.get("min_db", -8.0))
            max_db = float(aug_cfg.get("max_db", 8.0))
            db = random.uniform(min_db, max_db)
            segment = segment * (10.0 ** (db / 20.0))
            params["db"] = db
        elif aug_name == "gaussian_noise":
            min_snr = float(aug_cfg.get("min_snr_db", 8.0))
            max_snr = float(aug_cfg.get("max_snr_db", 25.0))
            snr_db = random.uniform(min_snr, max_snr)
            sig_pow = torch.mean(segment.pow(2))
            noise_pow = sig_pow / (10.0 ** (snr_db / 10.0) + 1e-12)
            noise = torch.randn_like(segment) * torch.sqrt(noise_pow + 1e-12)
            segment = segment + noise
            params["snr_db"] = snr_db
        elif aug_name == "lowpass":
            min_cutoff = float(aug_cfg.get("min_cutoff_hz", 1200.0))
            max_cutoff = float(aug_cfg.get("max_cutoff_hz", 5000.0))
            cutoff = random.uniform(min_cutoff, max_cutoff)
            segment = torchaudio.functional.lowpass_biquad(segment, self.target_sample_rate, cutoff)
            params["cutoff_hz"] = cutoff
        elif aug_name == "highpass":
            min_cutoff = float(aug_cfg.get("min_cutoff_hz", 60.0))
            max_cutoff = float(aug_cfg.get("max_cutoff_hz", 700.0))
            cutoff = random.uniform(min_cutoff, max_cutoff)
            segment = torchaudio.functional.highpass_biquad(segment, self.target_sample_rate, cutoff)
            params["cutoff_hz"] = cutoff
        elif aug_name == "bandstop_notch":
            min_center = float(aug_cfg.get("min_center_hz", 900.0))
            max_center = float(aug_cfg.get("max_center_hz", 5200.0))
            min_q = float(aug_cfg.get("min_q", 1.0))
            max_q = float(aug_cfg.get("max_q", 6.0))
            center_hz = random.uniform(min_center, max_center)
            q = random.uniform(min_q, max_q)
            if hasattr(torchaudio.functional, "bandreject_biquad"):
                segment = torchaudio.functional.bandreject_biquad(segment, self.target_sample_rate, center_hz, q)
            else:
                # Fallback: approximate with a medium low-pass when bandreject is unavailable.
                segment = torchaudio.functional.lowpass_biquad(segment, self.target_sample_rate, max(800.0, center_hz * 0.6))
            params["center_hz"] = center_hz
            params["q"] = q
        elif aug_name == "clipping":
            min_thr = float(aug_cfg.get("min_threshold", 0.55))
            max_thr = float(aug_cfg.get("max_threshold", 0.9))
            threshold = random.uniform(min_thr, max_thr)
            segment = torch.clamp(segment, -threshold, threshold)
            params["threshold"] = threshold
        elif aug_name == "dropout_bursts":
            min_drop_ms = float(aug_cfg.get("min_dropout_ms", 10.0))
            max_drop_ms = float(aug_cfg.get("max_dropout_ms", 80.0))
            min_bursts = int(aug_cfg.get("min_bursts", 1))
            max_bursts = int(aug_cfg.get("max_bursts", 4))
            n_bursts = random.randint(min_bursts, max(min_bursts, max_bursts))
            seg_len = max(1, segment.shape[-1])
            for _ in range(n_bursts):
                drop_len = int(random.uniform(min_drop_ms, max_drop_ms) * self.target_sample_rate / 1000.0)
                drop_len = max(1, min(drop_len, seg_len))
                drop_start = random.randint(0, max(0, seg_len - drop_len))
                segment[:, drop_start:drop_start + drop_len] = 0.0
            params["n_bursts"] = float(n_bursts)
        else:
            aug_name = "time_mask"
            segment = torch.zeros_like(segment)

        waveform[:, start:end] = torch.clamp(segment, -1.0, 1.0)

        start_sec = start / float(self.target_sample_rate)
        end_sec = end / float(self.target_sample_rate)
        start_pct = start / float(max(1, n_samples))
        end_pct = end / float(max(1, n_samples))
        detail_mode = self._pick_detail_mode()
        region_desc = self._describe_region(start_sec, end_sec, start_pct, end_pct)
        aug_desc = self._describe_augmentation(aug_name, params, detail_mode)

        phrase_template = random.choice(self._grounding_phrases)
        try:
            sentence = phrase_template.format(
                region_desc=region_desc,
                aug_desc=aug_desc,
                start_sec=start_sec,
                end_sec=end_sec,
                start_pct=start_pct * 100.0,
                end_pct=end_pct * 100.0,
                augmentation=aug_name,
                detail_mode=detail_mode,
                **params,
            )
        except KeyError:
            sentence = f"In {region_desc}, {aug_desc}."

        reasoning_aug = self._insert_sentence_into_reasoning(reasoning or "", sentence)
        grounding_info = {
            "enabled": True,
            "augmentation": aug_name,
            "params": params,
            "detail_mode": detail_mode,
            "start_sample": start,
            "end_sample": end,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_pct": start_pct,
            "end_pct": end_pct,
            "sentence": sentence,
        }
        return waveform, reasoning_aug, grounding_info

    def __getitem__(self, idx):
        if len(self) == 0:
            raise RuntimeError(f"Dataset is empty: {self.data_path}")
        if self._num_broken >= len(self):
            raise RuntimeError(
                f"Too many broken/missing audio files in dataset: {self.data_path}. "
                "All sampled items failed to load."
            )
        start = (self.samples_offset + idx) % len(self)
        waveform, sample_rate, item = None, None, None
        for attempt in range(len(self)):
            real_idx = (start + attempt) % len(self)
            item = self.samples[real_idx]
            audio_path = item["original_path"]
            audio_path = audio_path.replace(".wav", ".flac")
            audio_path = audio_path.replace(".mp3", ".flac")
            audio_path = audio_path.replace('/path/to/data/', './artifacts/data/')
            waveform, sample_rate = self._load_audio(audio_path)
            if waveform is not None and sample_rate is not None:
                break
            self._num_broken += 1
        else:
            raise RuntimeError(
                f"Too many broken/missing audio files in dataset: {self.data_path}. "
                "All sampled items failed to load."
            )

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)(waveform)

        # Truncate to max_audio_duration so one long file can't block a distributed barrier
        if self.max_audio_samples is not None and waveform.shape[-1] > self.max_audio_samples:
            waveform = waveform[..., :self.max_audio_samples]

        # max_len_seconds: take first N seconds; if shorter, repeat audio until that length
        if self.max_len_samples is not None:
            length = waveform.shape[-1]
            if length >= self.max_len_samples:
                waveform = waveform[..., :self.max_len_samples]
            else:
                n_repeats = (self.max_len_samples + length - 1) // length
                waveform = waveform.repeat(1, n_repeats)[..., :self.max_len_samples]

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Determine target text based on task_type
        is_bonafide = item.get("is_bonafide")
        reasoning = item.get("reasoning")
        reasons_dict = item.get("reasons", {})
        waveform, reasoning_augmented, grounding_info = self._maybe_apply_grounding(waveform, reasoning)
        
        if self.task_type == "reasoning":
            # Compile reasoning in format: <think></think><reasons></reasons><answer></answer>
            think_content = reasoning_augmented if reasoning_augmented else ""
            
            # Robustly parse reasons (could be dict, list, or str)
            reasons_list = []
            if isinstance(reasons_dict, dict):
                reasons_list = [k.upper() for k, v in reasons_dict.items() if v]
            elif isinstance(reasons_dict, list):
                reasons_list = [str(r).upper() for r in reasons_dict]
            elif isinstance(reasons_dict, str) and reasons_dict:
                reasons_list = [reasons_dict.upper()]

            # shuffle reasons_list
            if self.shuffle_reasons:
                reasons_list = random.shuffle(reasons_list)
            
            answer_content = "Real" if is_bonafide else "Fake"
            
            target_text = f"<think>{think_content}</think><reasons>{reasons_list}</reasons><answer>{answer_content}</answer>"
        else:
            # Default to hard label - using Real/Fake as requested
            target_text = "Final Answer: " + ("Real" if is_bonafide else "Fake")
            
        # Normalize reasoning to str so batch/collate never has None (avoids null in logs/judge)
        reasoning_out = reasoning_augmented if reasoning_augmented is not None else ""

        return {
            "audio_id": item.get("audio_id"),
            "raw_wav": waveform, 
            "original_path": item.get("original_path"),
            "is_bonafide": is_bonafide,
            "reasoning": reasoning_out,
            "task": "antispoofing", 
            "text": target_text,
            "grounding_info": grounding_info,
        }

def _format_prompt_for_model(prompt_template: str, model_name: str) -> str:
    """Format a prompt template for the specific model's chat format."""
    # Replace <Audio> placeholder with model-specific token
    if model_name == "flamingo":
        audio_token = "<|audio_ready|>"
    elif model_name == "qwen_audio":
        audio_token = "<|AUDIO|>"
    else:
        audio_token = ""
    
    # Replace the <Audio> placeholder in the template
    prompt_text = prompt_template.replace("<Audio>", audio_token)
    
    # Wrap in chat format for Qwen/Flamingo
    if model_name in ["qwen_audio", "flamingo"]:
        return f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    
    return prompt_text


def _vad_trim_waveforms(wavs: List[torch.Tensor], sample_rate: int) -> List[torch.Tensor]:
    """Trim leading/trailing silence on each waveform using torchaudio VAD. Returns list of (1, T) tensors."""
    try:
        from torchaudio.transforms import Vad
        vad = Vad(sample_rate=sample_rate)
    except Exception:
        return wavs
    out = []
    for w in wavs:
        if w.dim() == 1:
            w = w.unsqueeze(0)
        try:
            w = vad(w)
            w = vad(torch.flip(w, dims=[-1]))
            w = torch.flip(w, dims=[-1])
        except Exception:
            pass
        if w.dim() == 1:
            w = w.unsqueeze(0)
        out.append(w)
    return out


def _apply_vad_and_crop_to_waveforms(
    wavs: List[torch.Tensor],
    sample_rate: int,
    target_duration_sec: float,
) -> torch.Tensor:
    """Apply VAD then crop/pad each waveform to target_duration_sec. Returns (B, 1, target_samples)."""
    target_samples = int(target_duration_sec * sample_rate)
    try:
        from torchaudio.transforms import Vad
        vad = Vad(sample_rate=sample_rate)
    except Exception:
        vad = None
    out_list = []
    for w in wavs:
        if w.dim() == 1:
            w = w.unsqueeze(0)
        if vad is not None:
            try:
                w = vad(w)
            except Exception:
                pass
        if w.dim() == 1:
            w = w.unsqueeze(0)
        L = w.shape[-1]
        if L >= target_samples:
            w = w[..., :target_samples]
        else:
            pad = torch.zeros((w.shape[0], target_samples - L), dtype=w.dtype)
            w = torch.cat([w, pad], dim=-1)
        out_list.append(w)
    return torch.stack(out_list, dim=0)


def _overlapping_windows(
    wav: torch.Tensor,
    window_samples: int,
    stride_samples: int,
) -> torch.Tensor:
    """Extract overlapping windows from a 1D waveform. Returns (num_windows, 1, window_samples)."""
    while wav.dim() > 1:
        wav = wav.squeeze(0)
    L = wav.shape[-1]
    if L <= 0:
        return torch.zeros(1, 1, window_samples, dtype=wav.dtype)
    if L < window_samples:
        pad = torch.zeros(window_samples - L, device=wav.device, dtype=wav.dtype)
        wav = torch.cat([wav, pad], dim=-1)
        return wav.unsqueeze(0).unsqueeze(0)
    starts = list(range(0, L - window_samples + 1, stride_samples))
    if not starts:
        starts = [0]
    windows = []
    for start in starts:
        chunk = wav[start : start + window_samples]
        windows.append(chunk)
    if starts[-1] + window_samples < L:
        last_start = L - window_samples
        chunk = wav[last_start : last_start + window_samples]
        windows.append(chunk)
    out = torch.stack(windows, dim=0)
    if out.dim() == 2:
        out = out.unsqueeze(1)
    return out


def collate_fn(batch: List[Dict[str, Any]], processor: Any = None, model_name: str = None,
               prompt_templates: Optional[List[str]] = None,
               conv_audio_vad_target_sec: Optional[float] = None,
               conv_audio_sample_rate: Optional[int] = None,
               conv_audio_full_wav_at_test: bool = False,
               conv_audio_overlap_ratio: Optional[float] = None,
               conv_audio_only_first_window: bool = False) -> Dict[str, Any]:
    # We need to pad audio/features and stack
    
    audio_ids = [b["audio_id"] for b in batch]
    raw_wavs = [b["raw_wav"].squeeze().numpy() for b in batch] # List of numpy arrays
    texts = [b["text"] for b in batch]
    original_paths = [b.get("original_path") for b in batch]
    
    if model_name == "conv_audio_classifier":
        wavs = [b["raw_wav"].squeeze(0) for b in batch]
        # At test time with overlapping windows: VAD trim, pad, then slice into windows per sample.
        # Train/val or test without windowing: VAD+crop to target_duration_sec or just pad.
        if conv_audio_full_wav_at_test and conv_audio_sample_rate is not None:
            wavs = _vad_trim_waveforms(wavs, conv_audio_sample_rate)
            # Optionally produce overlapping windows or only first window for test
            if (
                conv_audio_vad_target_sec is not None
                and (conv_audio_overlap_ratio is not None or conv_audio_only_first_window)
            ):
                window_samples = int(conv_audio_vad_target_sec * conv_audio_sample_rate)
                if conv_audio_only_first_window:
                    # Only first window per sample after VAD, no overlapping
                    all_windows = []
                    for w in wavs:
                        while w.dim() > 1:
                            w = w.squeeze(0)
                        L = w.shape[-1]
                        if L <= 0:
                            chunk = torch.zeros(window_samples, device=w.device, dtype=w.dtype)
                        elif L < window_samples:
                            pad = torch.zeros(window_samples - L, device=w.device, dtype=w.dtype)
                            chunk = torch.cat([w, pad], dim=-1)
                        else:
                            chunk = w[..., :window_samples].clone()
                        all_windows.append(chunk.unsqueeze(0).unsqueeze(0))
                    raw_wav_tensor = torch.cat(all_windows, dim=0)
                    windows_per_sample = [1] * len(batch)
                else:
                    stride_samples = max(1, int(window_samples * (1.0 - conv_audio_overlap_ratio)))
                    all_windows = []
                    windows_per_sample = []
                    for w in wavs:
                        win = _overlapping_windows(w, window_samples, stride_samples)
                        n_w = win.shape[0]
                        all_windows.append(win)
                        windows_per_sample.append(n_w)
                    raw_wav_tensor = torch.cat(all_windows, dim=0)
                is_bonafide_per_window = []
                for idx, b in enumerate(batch):
                    is_bonafide_per_window.extend([b["is_bonafide"]] * windows_per_sample[idx])
                return {
                    "audio_ids": audio_ids,
                    "original_path": original_paths,
                    "raw_wav": raw_wav_tensor,
                    "windows_per_sample": windows_per_sample,
                    "is_bonafide": torch.tensor(is_bonafide_per_window),
                    "task": [b["task"] for b in batch],
                    "text": texts,
                    "reasoning": [b["reasoning"] for b in batch],
                    "grounding_info": [b.get("grounding_info") for b in batch],
                }
            # Full-length padded (no windowing): pad per sample length
            wavs_1d = [w.squeeze(0) for w in wavs]
            padded = torch.nn.utils.rnn.pad_sequence(wavs_1d, batch_first=True, padding_value=0.0)
            raw_wav_tensor = padded.unsqueeze(1)
        elif conv_audio_vad_target_sec is not None and conv_audio_sample_rate is not None:
            raw_wav_tensor = _apply_vad_and_crop_to_waveforms(
                wavs, conv_audio_sample_rate, conv_audio_vad_target_sec
            )
        else:
            padded = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0.0)
            raw_wav_tensor = padded.unsqueeze(1)
        return {
            "audio_ids": audio_ids,
            "original_path": original_paths,
            "raw_wav": raw_wav_tensor,
            "is_bonafide": torch.tensor([b["is_bonafide"] for b in batch]),
            "task": [b["task"] for b in batch],
            "text": texts,
            "reasoning": [b["reasoning"] for b in batch],
            "grounding_info": [b.get("grounding_info") for b in batch],
        }

    if (model_name == "qwen_audio" or model_name == "flamingo") and processor:
        # Build prompts from templates - prompts are required for these models
        if not prompt_templates:
            raise ValueError(
                f"prompt_templates is required for {model_name}. "
                "Configure Prompts.reasoning_prompts or Prompts.hard_label_prompts in config.yaml"
            )
        
        prompts = []
        for b in batch:
            # Randomly select a prompt template
            template = random.choice(prompt_templates)
            prompt = _format_prompt_for_model(template, model_name)
            prompts.append(prompt)
        
        # Try 'audios' first, then 'audio' as fallback
        try:
            inputs = processor(text=prompts, audios=raw_wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        except (TypeError, ValueError):
            try:
                inputs = processor(text=prompts, audio=raw_wavs, sampling_rate=16000, return_tensors="pt", padding=True)
            except (TypeError, ValueError):
                inputs = processor(prompts, raw_wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        
        return {
            "audio_ids": audio_ids,
            "original_path": original_paths,
            "input_features": inputs.get("input_features"),
            "input_ids": inputs.get("input_ids"), 
            "attention_mask": inputs.get("attention_mask"),
            "feature_attention_mask": inputs.get("feature_attention_mask"),
            "text": texts, 
            "is_bonafide": torch.tensor([b["is_bonafide"] for b in batch]),
            "task": [b["task"] for b in batch],
            "reasoning": [b["reasoning"] for b in batch],
            "grounding_info": [b.get("grounding_info") for b in batch],
            "raw_wav": raw_wavs, # Flamingo needs raw_wav for its own processing in forward()
            "prompts": prompts  # Pass prompts to model for reference
        }

    if processor: # Fallback for Whisper/SALMON
        # Pad/Truncate to 30s usually for Whisper?
        # Whisper expects 3000 frames (30s * 100 frames/s) usually?
        # `feature_extractor` handles padding/truncation.
        inputs = processor(raw_wavs, sampling_rate=16000, return_tensors="pt")
        spectrograms = inputs.input_features
    else:
        # Dummy or fallback
        spectrograms = torch.randn(len(batch), 80, 3000)
    
    # Build prompts for SALMON - prompts are required for GRPO generation
    prompts = None
    if prompt_templates:
        prompts = []
        for b in batch:
            template = random.choice(prompt_templates)
            # SALMON expects <Audio> placeholder (converted to <SpeechHere> internally)
            prompts.append(template)
    

    return {
        "audio_ids": audio_ids,
        "original_path": original_paths,
        "spectrogram": spectrograms,
        "raw_wav": raw_wavs, 
        "is_bonafide": torch.tensor([b["is_bonafide"] for b in batch]),
        "task": [b["task"] for b in batch],
        "text": texts,
        "reasoning": [b["reasoning"] for b in batch],
        "grounding_info": [b.get("grounding_info") for b in batch],
        "prompts": prompts,
    }
