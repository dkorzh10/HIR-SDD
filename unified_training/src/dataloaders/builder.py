import torch
from torch.utils.data import DataLoader
from typing import Optional, Any, List
from functools import partial

from .dataset import AudioDataset, collate_fn
from .samplers import StatefulSampler, StatefulDistributedSampler, LengthGroupedDistributedSampler


def get_dataloader(
    dataset_path: str,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    task_type: str = "hard_label",
    whisper_path: Optional[str] = None,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    distributed: bool = False,
    iters_per_epoch: Optional[int] = None,
    stateful: bool = True,
    prompt_templates: Optional[List[str]] = None,
    samples_offset: int = 0,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    grounding_cfg: Optional[dict] = None,
    use_length_grouped_sampler: bool = True,
    sampler_seed: int = 0,
    shuffle_reasons: bool = False,
    conv_audio_vad_target_sec: Optional[float] = None,
    conv_audio_sample_rate: Optional[int] = None,
    conv_audio_full_wav_at_test: bool = False,
    conv_audio_overlap_ratio: Optional[float] = None,
    conv_audio_only_first_window: bool = False,
    max_audio_duration_s: Optional[float] = 30.0,
    max_len_seconds: Optional[float] = None,
) -> DataLoader:
    dataset = AudioDataset(
        dataset_path,
        max_samples=max_samples,
        task_type=task_type,
        samples_offset=samples_offset,
        grounding_cfg=grounding_cfg,
        shuffle_reasons=shuffle_reasons,
        max_audio_duration_s=max_audio_duration_s,
        max_len_seconds=max_len_seconds,
    )

    processor = None
    if model_name == "qwen_audio" and model_path:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=False)
    elif whisper_path:
        try:
            from transformers import WhisperFeatureExtractor
            processor = WhisperFeatureExtractor.from_pretrained(
                whisper_path,
                local_files_only=False
            )
        except Exception as e:
            print(f"Warning: Could not load WhisperFeatureExtractor from {whisper_path}: {e}", flush=True)
            processor = None

    custom_collate = partial(
        collate_fn,
        processor=processor,
        model_name=model_name,
        prompt_templates=prompt_templates,
        conv_audio_vad_target_sec=conv_audio_vad_target_sec,
        conv_audio_sample_rate=conv_audio_sample_rate,
        conv_audio_full_wav_at_test=conv_audio_full_wav_at_test,
        conv_audio_overlap_ratio=conv_audio_overlap_ratio,
        conv_audio_only_first_window=conv_audio_only_first_window,
    )

    if distributed:
        if use_length_grouped_sampler:
            sampler = LengthGroupedDistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                iters_per_epoch=iters_per_epoch,
                batch_size=batch_size,
                stateful=stateful,
                seed=sampler_seed,
            )
        else:
            sampler = StatefulDistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                iters_per_epoch=iters_per_epoch,
                batch_size=batch_size,
                stateful=stateful,
                seed=sampler_seed,
            )
        shuffle = False
    else:
        sampler = StatefulSampler(
            dataset,
            shuffle=shuffle,
            iters_per_epoch=iters_per_epoch,
            batch_size=batch_size,
            stateful=stateful,
            seed=sampler_seed,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate,
        sampler=sampler
    )
