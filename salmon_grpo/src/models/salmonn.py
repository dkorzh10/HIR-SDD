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

import logging
import json
import contextlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, StoppingCriteriaList
from peft import LoraConfig, TaskType, get_peft_model

from .Qformer import BertConfig, BertLMHeadModel
from .modeling_llama import LlamaForCausalLM
from .modeling_whisper import WhisperModel
from .utils import StoppingCriteriaSub

# BEATs is optional - only needed if beats_path is specified
try:
    from .beats.BEATs import BEATsConfig, BEATs
    BEATS_AVAILABLE = True
except (ImportError, Exception):
    BEATS_AVAILABLE = False
    BEATsConfig = None
    BEATs = None
    logging.warning("BEATs not available - training without BEATs encoder")


class SALMONN(nn.Module):
    @classmethod
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        llama_path="",
        whisper_path="",
        freeze_whisper=True,
        whisper_unfreeze_last_n_layers=0,  # Unfreeze last N layers (0 = all frozen)
        whisper_unfreeze_attention_only=False,  # If True, only unfreeze attention in those layers
        beats_path="",
        freeze_beats=True,

        use_speech_Qformer=True,
        num_speech_query_token=1,
        freeze_speech_QFormer=False,
        window_level_Qformer=True,
        second_per_window=0.333333,
        second_stride=0.333333,
        
        speech_llama_proj_model="",
        freeze_speech_llama_proj=False,

        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,

        multi_prompt=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="</s>",
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        
        # Class weighting for imbalanced datasets
        use_class_weights=True,  # Enable weighted loss by default
        class_weights=None,  # Will be computed from dataset if None
        bonafide_weight_multiplier=1.0,  # Additional multiplier for bonafide class (e.g., 2.0 to double its weight)
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.beats_path = beats_path
        self.use_speech_Qformer = use_speech_Qformer
        self.window_level_Qformer = window_level_Qformer
        self.second_per_window = second_per_window
        self.second_stride = second_stride
        self.lora = lora
        self.multi_prompt = multi_prompt
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.low_resource = low_resource
        self.gradient_checkpointing = gradient_checkpointing
        
        # Store class weighting parameters
        self.use_class_weights = use_class_weights
        self.class_weights = class_weights
        self.bonafide_weight_multiplier = bonafide_weight_multiplier
        self.class_weight_tensor = None  # Will be set later when moved to device

        logging.info('Loading LLaMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"

        logging.info('Loading LLaMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit},
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch.float16,
            )

        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLaMA Done')

        # if self.gradient_checkpointing:
        #     if hasattr(self.llama_model, "gradient_checkpointing_enable"):
        #         self.llama_model.gradient_checkpointing_enable()
        #         logging.info("Gradient checkpointing enabled for LLaMA model")
        #     else:
        #         logging.warning("Gradient checkpointing requested but LLaMA model does not support gradient_checkpointing_enable()")
        
        # Also enable for QFormer if it exists and supports it
        # MOVED: This logic is now after QFormer initialization to avoid AttributeError

        if self.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model.print_trainable_parameters()
            logging.info('LoRA Training')

        assert whisper_path
        logging.info('Loading Whisper Model')
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        
        # Handle Whisper freezing with selective unfreezing
        if freeze_whisper:
            # First, freeze all parameters
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder.eval()
            
            # Selectively unfreeze last N layers if requested
            if whisper_unfreeze_last_n_layers > 0:
                total_layers = len(self.speech_encoder.layers)
                start_layer = max(0, total_layers - whisper_unfreeze_last_n_layers)
                
                trainable_params = 0
                for i in range(start_layer, total_layers):
                    if whisper_unfreeze_attention_only:
                        # Only unfreeze attention layers
                        for param in self.speech_encoder.layers[i].self_attn.parameters():
                            param.requires_grad = True
                            trainable_params += param.numel()
                    else:
                        # Unfreeze entire layer
                        for param in self.speech_encoder.layers[i].parameters():
                            param.requires_grad = True
                            trainable_params += param.numel()
                
                if whisper_unfreeze_attention_only:
                    logging.info(f"Whisper: frozen except attention in last {whisper_unfreeze_last_n_layers} layers ({trainable_params/1e6:.1f}M params)")
                else:
                    logging.info(f"Whisper: frozen except last {whisper_unfreeze_last_n_layers} layers ({trainable_params/1e6:.1f}M params)")
            else:
                logging.info("Whisper: fully frozen")
        else:
            total_params = sum(p.numel() for p in self.speech_encoder.parameters())
            logging.info(f"Whisper: fully trainable ({total_params/1e6:.1f}M params)")
        
        if self.beats_path:
            if not BEATS_AVAILABLE:
                logging.warning("BEATs path specified but BEATs module not available. Skipping BEATs.")
                self.beats_path = ""  # Disable BEATs
            else:
                logging.info("Loading BEATs Model")
                try:
                    beats_ckpt = torch.load(self.beats_path, map_location='cpu')
                    beats_cfg = BEATsConfig(beats_ckpt['cfg'])
                    self.beats = BEATs(beats_cfg)
                    self.beats.load_state_dict(beats_ckpt['model'])
                    self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
                    if freeze_beats:
                        for name, param in self.beats.named_parameters():
                            param.requires_grad = False
                        self.beats.eval()
                        logging.info("freeze BEATs")
                except Exception as e:
                    logging.warning(f"Failed to load BEATs: {e}. Continuing without BEATs.")
                    self.beats_path = ""  # Disable BEATs

        if self.use_speech_Qformer:
            if self.beats_path:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim
                )
            else:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model
                )
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.speech_Qformer.cls = None
            if freeze_speech_QFormer:
                for name, param in self.speech_Qformer.named_parameters():
                    param.requires_grad = False
                self.speech_Qformer.eval()
                self.speech_query_tokens.requires_grad = False
                logging.info("freeze Speech QFormer")

            logging.info('Loading speech LLAMA proj')
            self.speech_llama_proj = nn.Linear(
                self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )
            if speech_llama_proj_model:
                logging.info("Loading speech LLAMA proj from {}".format(speech_llama_proj_model))
                speech_llama_proj_weight = torch.load(speech_llama_proj_model, map_location="cpu")
                self.load_state_dict(speech_llama_proj_weight['model'], strict=False)
            if freeze_speech_llama_proj:
                for name, param in self.speech_llama_proj.named_parameters():
                    param.requires_grad = False
                self.speech_llama_proj.eval()
                logging.info("freeze speech LLAMA proj")
        else:
            # feel free to add other aligners here
            raise NotImplementedError

        # prepare prompts
        self.prompt_dict = {}
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                self.prompt_dict[task] = [prompt_template.format(p) for p in filted_prompts]
            print("Loading training prompts done!")

        # Also enable for QFormer if it exists and supports it
        if self.gradient_checkpointing and self.use_speech_Qformer and hasattr(self.speech_Qformer, "gradient_checkpointing_enable"):
            self.speech_Qformer.gradient_checkpointing_enable()
            # Ensure config is also updated as BertEncoder checks config
            if hasattr(self.speech_Qformer, "config"):
                self.speech_Qformer.config.gradient_checkpointing = True
            logging.info("Gradient checkpointing enabled for Speech QFormer")

    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None):
        with self.maybe_autocast():
            if self.use_speech_Qformer:
                speech_embeds = self.ln_speech(speech_embeds)
                if audio_embeds is not None:
                    audio_embeds = self.ln_audio(audio_embeds)
                    if audio_embeds.size(1) < speech_embeds.size(1):
                        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                    elif audio_embeds.size(1) > speech_embeds.size(1):
                        speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                    speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

                if self.window_level_Qformer:
                    B, T, C = speech_embeds.shape
                    kernel = round(1500 * self.second_per_window / 30.0)
                    stride = round(1500 * self.second_stride / 30.0)
                    kernel = (1, kernel)
                    stride = (1, stride)
                    speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
                    speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
                    _, _, L = speech_embeds_overlap.shape
                    speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
                    speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
                    speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
                    speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

                query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
                query_output = self.speech_Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=speech_embeds,
                    encoder_attention_mask=speech_atts,
                    return_dict=True,
                )
                speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)

                if self.window_level_Qformer:
                    speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()

                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
            else:
                raise NotImplementedError

        return speech_embeds, speech_atts

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        with self.maybe_autocast():
            speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state

            if self.beats_path and raw_wav is not None:
                audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
            else:
                audio_embeds = None

        return self._encode_auditory_feature(speech_embeds, audio_embeds=audio_embeds)

    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids.long()) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids.long())

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(embeds.device)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids.long()) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids.long())

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids.long()).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids.long()).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids.long()).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids.long()).expand(batch_size, -1, -1)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts

    def set_class_weights_from_dataset(self, dataset):
        """
        Compute class weights from dataset to handle class imbalance.
        Uses inverse frequency weighting: weight = n_samples / (n_classes * n_samples_per_class)
        
        Args:
            dataset: SALMONNDataset instance with annotation field
        """
        if not self.use_class_weights:
            logging.info("Class weighting disabled")
            return
            
        # Count occurrences of each target text
        from collections import Counter
        target_counts = Counter()
        
        for item in dataset.annotation:
            target_counts[item['text']] += 1
        
        total_samples = len(dataset.annotation)
        n_classes = len(target_counts)
        
        logging.info(f"Computing class weights from {total_samples} samples, {n_classes} classes:")
        for text, count in target_counts.items():
            logging.info(f"  '{text}': {count} samples ({count/total_samples*100:.1f}%)")
        
        # Tokenize each target to get token IDs
        # We'll weight tokens based on which answer they belong to
        token_weights = {}
        
        for text, count in target_counts.items():
            # Calculate weight for this class
            weight = total_samples / (n_classes * count)
            
            # Apply multiplier for bonafide class if configured
            if 'bonafide' in text.lower() and hasattr(self, 'bonafide_weight_multiplier'):
                weight *= self.bonafide_weight_multiplier
                logging.info(f"  Applying bonafide weight multiplier {self.bonafide_weight_multiplier}x to '{text}'")
            
            # Tokenize to get the token IDs for this answer
            tokens = self.llama_tokenizer(
                text + self.end_sym,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids[0]
            
            for token_id in tokens.tolist():
                if token_id not in token_weights:
                    token_weights[token_id] = []
                token_weights[token_id].append(weight)
        
        # Average weights for tokens that appear in multiple classes
        for token_id in token_weights:
            token_weights[token_id] = sum(token_weights[token_id]) / len(token_weights[token_id])
        
        # Create weight tensor for all vocab (default weight = 1.0)
        vocab_size = len(self.llama_tokenizer)
        class_weight_tensor = torch.ones(vocab_size)
        
        for token_id, weight in token_weights.items():
            class_weight_tensor[token_id] = weight
        
        # Move to the same device as the model
        device = next(self.parameters()).device
        class_weight_tensor = class_weight_tensor.to(device)
        
        self.class_weight_tensor = class_weight_tensor
        
        # Store in LLaMA model config so it's accessible during forward pass
        if self.lora:
            self.llama_model.base_model.model.config.class_weight_tensor = class_weight_tensor
        else:
            self.llama_model.config.class_weight_tensor = class_weight_tensor
        
        # Log the weights for the answer tokens
        logging.info("Token weights for answer classes:")
        for text in target_counts.keys():
            tokens = self.llama_tokenizer(text + self.end_sym, add_special_tokens=False).input_ids
            weights = [class_weight_tensor[tid].item() for tid in tokens]
            logging.info(f"  '{text}': tokens {tokens[:5]}... weights {[f'{w:.3f}' for w in weights[:5]]}...")

    def forward(self, samples, verbose=False):
        # detect whether there are multi tasks in this batch
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM
        text = [t + self.end_sym for t in samples["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids.long()) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids.long())
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long
            ).to(spectrogram.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos.long()) if not self.lora else self.llama_model.model.model.embed_tokens(bos.long())
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)

        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

        if verbose:
            nvocab = self.llama_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])

        if verbose:
            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}

    def generate(self, samples, generate_cfg, prompts=None, return_outputs=False):
        batch_size = samples["spectrogram"].shape[0]

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        if prompts is not None:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=True)

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos.long()) if not self.lora else self.llama_model.model.model.embed_tokens(bos.long())
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)

        stop_words_ids = [torch.tensor([2]).to(self.device)]  
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        # Workaround for newer transformers where generate method is not directly available
        # We need to access the generation method through the model hierarchy
        try:
            # Try direct generate (works with older transformers)
            if hasattr(self.llama_model, 'generate'):
                generation_model = self.llama_model
            # For LoRA models, try to access through base_model
            elif hasattr(self.llama_model, 'base_model'):
                # Import GenerationMixin and bind the method
                from transformers import GenerationMixin
                generation_model = self.llama_model.base_model.model
                # Dynamically add generate method if missing
                if not hasattr(generation_model, 'generate'):
                    # Bind GenerationMixin methods to the model instance
                    for method_name in ['generate', '_prepare_attention_mask_for_generation', 
                                       '_prepare_encoder_decoder_kwargs_for_generation',
                                       '_expand_inputs_for_generation']:
                        if hasattr(GenerationMixin, method_name):
                            method = getattr(GenerationMixin, method_name)
                            setattr(generation_model, method_name, method.__get__(generation_model))
            else:
                generation_model = self.llama_model
            
            outputs = generation_model.generate(
                inputs_embeds=embeds,
                max_new_tokens=generate_cfg.get("max_new_tokens", 200),
                stopping_criteria=stopping_criteria,
                num_beams=generate_cfg.get("num_beams", 4),
                do_sample=generate_cfg.get("do_sample", False),
                min_length=generate_cfg.get("min_length", 1),
                temperature=generate_cfg.get("temperature", 1.0),
                top_p=generate_cfg.get("top_p", 0.9),
                repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
                length_penalty=generate_cfg.get("length_penalty", 1.0),
                attention_mask=attns,
            )
        except AttributeError as e:
            logging.error(f"Generate method not available: {e}")
            logging.error("This is likely due to transformers version incompatibility")
            raise RuntimeError(
                "The model's generate() method is not available. "
                "This may be due to transformers>=4.50 removing generate from PreTrainedModel. "
                "Please use transformers<4.50 or update the model code."
            )
        
        text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False)

        if return_outputs:
            # Need to compute logits via forward pass
            # Reconstruct the input embeddings
            with torch.no_grad():
                # outputs contains only the generated tokens (prompt was given as embeddings)
                # So completion_ids is just outputs itself
                completion_ids = outputs
                
                # Get embeddings for completions
                if self.lora:
                    completion_embeds = self.llama_model.model.model.embed_tokens(completion_ids.long())
                else:
                    completion_embeds = self.llama_model.model.embed_tokens(completion_ids.long())
                
                # Concatenate: [bos, audio, completion]
                full_embeds = torch.cat([embeds, completion_embeds], dim=1)
                full_atts = torch.cat([attns, torch.ones_like(completion_ids)], dim=1)
                
            # Forward pass to get logits
            llama_outputs = self.llama_model(
                inputs_embeds=full_embeds,
                attention_mask=full_atts,
                return_dict=True
            )
            
            # Extract logits for completion tokens only
            # For causal LM, logit at position i predicts token at position i+1
            # We have: [bos, audio, completion_tokens]
            # To predict completion_tokens, we need logits from positions [prompt_len-1 : prompt_len+completion_len-1]
            prompt_length = embeds.size(1)  # bos + audio
            completion_length = completion_ids.size(1)
            logits = llama_outputs.logits[:, prompt_length-1:prompt_length-1+completion_length, :]
            
            return text, completion_ids, logits

        return text
    
    def generate_text_only(self, prompt_texts, generate_cfg):
        """
        Generate text from text-only prompts (no audio input).
        Uses base LLaMA model WITHOUT LoRA adapters for unbiased judge evaluation.
        
        Args:
            prompt_texts: List of prompt strings
            generate_cfg: Generation configuration dict
        
        Returns:
            generated_texts: List of generated strings (prompt removed)
        """
        batch_size = len(prompt_texts)
        
        # Critical fix: Use left padding for batched generation
        # This ensures the prompt tokens end right before the generation starts
        original_padding_side = self.llama_tokenizer.padding_side
        self.llama_tokenizer.padding_side = "left"
        
        try:
            inputs = self.llama_tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            input_ids = inputs['input_ids'].to(self.llama_model.device)
            attention_mask = inputs['attention_mask'].to(self.llama_model.device)
            
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
            
            gen_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": generate_cfg.get("max_new_tokens", 50),
                "stopping_criteria": stopping_criteria,
                "num_beams": 1,
                "do_sample": False,
                "attention_mask": attention_mask,
                "pad_token_id": self.llama_tokenizer.pad_token_id
            }
            
            if self.lora:
                with self.llama_model.disable_adapter():
                    outputs = self.llama_model.generate(**gen_kwargs)
            else:
                outputs = self.llama_model.generate(**gen_kwargs)
                
        finally:
            # Restore padding side
            self.llama_tokenizer.padding_side = original_padding_side
        
        # Extract only the newly generated tokens (after the prompt)
        # With left padding, input_ids are [pad, pad, prompt]. 
        # outputs will be [pad, pad, prompt, generated]
        input_lengths = [len(ids) for ids in input_ids]
        result_texts = []
        
        for i, output_ids in enumerate(outputs):
            # Get only the generated part (skip the prompt tokens)
            generated_ids = output_ids[input_lengths[i]:]
            generated_text = self.llama_tokenizer.decode(generated_ids, skip_special_tokens=True)
            result_texts.append(generated_text.strip())
        
        return result_texts

    def compute_logits_for_completions(self, samples, completion_ids):
        """
        Recompute logits for generated completions with gradients enabled.
        Used for GRPO training.
        
        Args:
            samples: Input samples with audio
            completion_ids: Generated token IDs (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size) with gradients
        """
        # Debug: Check completion_ids shape
        if completion_ids.size(1) == 0:
            import logging
            logging.warning(f"compute_logits_for_completions: completion_ids has zero length! Shape: {completion_ids.shape}")
        
        batch_size = samples["spectrogram"].shape[0]
        
        # Encode audio (with gradients)
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)
        
        speech_embeds, speech_atts = self.encode_speech(
            spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
        )
        
        # BOS token
        bos = torch.ones([batch_size, 1], dtype=torch.int32, device=speech_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos.long()) if not self.lora else self.llama_model.model.model.embed_tokens(bos.long())
        atts_bos = speech_atts[:, :1]
        
        # Get completion embeddings
        if self.lora:
            completion_embeds = self.llama_model.model.model.embed_tokens(completion_ids.long())
        else:
            completion_embeds = self.llama_model.model.embed_tokens(completion_ids.long())
        
        # Concatenate: [bos, audio, completion]
        full_embeds = torch.cat([bos_embeds, speech_embeds, completion_embeds], dim=1)
        full_atts = torch.cat([atts_bos, speech_atts, torch.ones_like(completion_ids)], dim=1)
        
        # Forward pass WITH gradients
        llama_outputs = self.llama_model(
            inputs_embeds=full_embeds,
            attention_mask=full_atts,
            return_dict=True
        )
        
        # Extract logits for completion tokens
        # For causal LM, logit at position i predicts token at position i+1
        # We have: [bos, audio, completion_tokens]
        # To predict completion_tokens, we need logits from positions [prompt_len-1 : prompt_len+completion_len-1]
        prompt_length = 1 + speech_embeds.size(1)  # bos + audio
        completion_length = completion_ids.size(1)
        logits = llama_outputs.logits[:, prompt_length-1:prompt_length-1+completion_length, :]
        
        return logits

    @classmethod
    def from_config(cls, config):
        llama_path = config.get("llama_path")
        whisper_path = config.get("whisper_path")
        freeze_whisper = config.get("freeze_whisper", True)
        whisper_unfreeze_last_n_layers = config.get("whisper_unfreeze_last_n_layers", 0)
        whisper_unfreeze_attention_only = config.get("whisper_unfreeze_attention_only", False)
        beats_path = config.get("beats_path", "")
        freeze_beats = config.get("freeze_beats", True)

        use_speech_Qformer = config.get("use_speech_Qformer", True)
        num_speech_query_token = config.get("num_speech_query_token", 1)
        freeze_speech_QFormer = config.get("freeze_speech_QFormer", False)
        window_level_Qformer = config.get("window_level_Qformer", True)
        second_per_window = config.get("second_per_window", 0.333333)
        second_stride = config.get("second_stride", 0.333333)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)
        
        # Class weighting parameters
        use_class_weights = config.get("use_class_weights", True)
        class_weights = config.get("class_weights", None)
        bonafide_weight_multiplier = config.get("bonafide_weight_multiplier", 1.0)
        gradient_checkpointing = config.get("gradient_checkpointing", False)

        model = cls(
            llama_path=llama_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            whisper_unfreeze_last_n_layers=whisper_unfreeze_last_n_layers,
            whisper_unfreeze_attention_only=whisper_unfreeze_attention_only,
            beats_path=beats_path,
            freeze_beats=freeze_beats,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            use_class_weights=use_class_weights,
            class_weights=class_weights,
            bonafide_weight_multiplier=bonafide_weight_multiplier,
            gradient_checkpointing=gradient_checkpointing,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load SALMONN ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)

        return model
