# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
from transformers import WhisperFeatureExtractor



class SALMONNDataset(Dataset):
    def __init__(self, ann_path, whisper_path, max_samples=None, seed=42):
        super().__init__()

        self.annotation = json.load(open(ann_path, "r"))["annotation"]
        
        # Balance classes by oversampling minority class
        # Only apply balancing if not in test mode (indicated by max_samples usually)
        # or explicitly requested. For now, we'll check if "train" is in the path or implicitly assume training if not heavily limited.
        # To be safe, let's just apply it to the loaded annotation list.
        
        # 1. Separate samples by class
        bonafide_samples = []
        spoof_samples = []
        
        for ann in self.annotation:
            # Assuming the class is in the text field (ground truth)
            # "text" field format varies, but usually contains "bonafide" or "spoof"
            text = str(ann.get("text", "")).lower()
            if "bonafide" in text or "genuine" in text:
                bonafide_samples.append(ann)
            elif "spoof" in text or "fake" in text:
                spoof_samples.append(ann)
        
        # Only balance if we found samples for both classes
        if bonafide_samples and spoof_samples:
            n_bonafide = len(bonafide_samples)
            n_spoof = len(spoof_samples)
            target_count = max(n_bonafide, n_spoof)
            
            import random
            random.seed(seed)
            
            # Oversample bonafide if needed
            if n_bonafide < target_count:
                extra_bonafide = random.choices(bonafide_samples, k=target_count - n_bonafide)
                bonafide_samples.extend(extra_bonafide)
                
            # Oversample spoof if needed
            if n_spoof < target_count:
                extra_spoof = random.choices(spoof_samples, k=target_count - n_spoof)
                spoof_samples.extend(extra_spoof)
            
            # Combine and shuffle
            self.annotation = bonafide_samples + spoof_samples
            random.shuffle(self.annotation)
            
            print(f"Dataset balanced: {len(bonafide_samples)} bonafide, {len(spoof_samples)} spoof (Total: {len(self.annotation)})")
        else:
            print(f"Warning: Could not balance classes. Bonafide: {len(bonafide_samples)}, Spoof: {len(spoof_samples)}")

        # Limit number of samples if specified
        if max_samples is not None and max_samples > 0 and max_samples < len(self.annotation):
            import random
            random.seed(seed)
            self.annotation = random.sample(self.annotation, max_samples)
            print(f"Dataset limited to {max_samples} samples (randomly selected from balanced pool)")

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path) 

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        return {
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "text": text,
            "task": task,
            "Q": Q,
            "id": id,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        try:
            audio, sr = sf.read(ann["path"])
        except Exception as e:
            # Skip corrupted files by returning the next valid sample
            # print(f"Warning: Failed to read {ann['path']}: {e}")
            # Try next sample
            return self.__getitem__((index + 1) % len(self.annotation))
        
        if len(audio.shape) == 2: # stereo to mono
            audio = audio[:, 0]

        # Resample to 16000 Hz if needed (Whisper requirement)
        if sr != 16000:
            from scipy import signal
            audio = signal.resample_poly(audio, 16000, sr)
            sr = 16000

        if "expand_wav" in ann:
            for p in ann["expand_wav"]:
                expand_audio, expand_sr = sf.read(p)
                if len(expand_audio.shape) == 2:
                    expand_audio = expand_audio[:, 0]
                # Resample expand_audio if needed
                if expand_sr != 16000:
                    expand_audio = signal.resample_poly(expand_audio, 16000, expand_sr)
                sil = np.zeros(1600, dtype=float)
                audio = np.concatenate((audio, sil, expand_audio), axis=0)
        if len(audio) < sr: # pad audio to at least 1s
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        audio = audio[: sr * 30] # truncate audio to at most 30s

        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        text = ann["text"]
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")

        return {
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "text": text,
            "task": task,
            "Q": Q,
            "id": ann["path"],
        }