# Copyright 2023-2026 SGLang Team
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
# ==============================================================================
"""SGLang GRT (Geometric Rotation Transformer) model wrappers.

Reuses all GRT model architectures and rotational building blocks directly from
`transformers.models.grt.modeling_grt` without duplicating any layer logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.grt.modeling_grt import (
    GRTASRForConditionalGeneration as HfGRTASRForConditionalGeneration,
    GRTAttention,
    GRTAudioModel,
    GRTCosineClassificationHead,
    GRTCosineHead,
    GRTDecoderLayer,
    GRTEmbedding,
    GRTFeedForward,
    GRTForCausalLM as HfGRTForCausalLM,
    GRTForSequenceClassification as HfGRTForSequenceClassification,
    GRTMixer,
    GRTModel as HfGRTModel,
    GRTOmniForConditionalGeneration as HfGRTOmniForConditionalGeneration,
    GRTPearlNorm,
    GRTPolarAudioStem,
    GRTPolarPatchEmbedding,
    GRTPolarVocoderHead,
    GRTRadialGELU,
    GRTRotaryEmbedding,
    GRTRotaryEmbedding2D,
    GRTSpatialRotationalMerger,
    GRTTemporalRotationalMerger,
    GRTTTSForConditionalGeneration as HfGRTTTSForConditionalGeneration,
    GRTVisionModel,
    GRTVLForConditionalGeneration as HfGRTVLForConditionalGeneration,
)

from sglang.srt.configs.grt import (
    GRTASRConfig,
    GRTAudioConfig,
    GRTConfig,
    GRTOmniConfig,
    GRTTTSConfig,
    GRTVisionConfig,
    GRTVLConfig,
)
from sglang.srt.layers.pooler import (
    EmbeddingPoolerOutput,
    Pooler,
    PoolingType,
    pool_hidden_states,
)

try:
    from sglang.srt.distributed import get_tensor_model_parallel_world_size
except Exception:
    def get_tensor_model_parallel_world_size() -> int:
        return 1

try:
    from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
except Exception:
    class LogitsProcessorOutput:  # type: ignore[no-redef]
        def __init__(self, next_token_logits: torch.Tensor):
            self.next_token_logits = next_token_logits

    class LogitsProcessor(nn.Module):  # type: ignore[no-redef]
        def __init__(self, config: Any, *args: Any, **kwargs: Any):
            super().__init__()
            self.config = config

        def forward(
            self,
            input_ids: Any,
            hidden_states: torch.Tensor,
            lm_head: Any,
            forward_batch: Any = None,
        ) -> LogitsProcessorOutput:
            logits = lm_head(hidden_states) if callable(lm_head) else hidden_states @ lm_head.weight.T
            return LogitsProcessorOutput(next_token_logits=logits)

try:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
except Exception:
    QuantizationConfig = Any  # type: ignore[misc,assignment]

try:
    from sglang.srt.managers.req_types import MultimodalDataItem
except Exception:
    MultimodalDataItem = Any  # type: ignore[misc,assignment]

try:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
except Exception:
    ForwardBatch = Any  # type: ignore[misc,assignment]


def _load_hf_weights(model: nn.Module, weights: Iterable[tuple[str, torch.Tensor]], strip_model_prefix: bool = False) -> set[str]:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded: set[str] = set()
    for name, tensor in weights:
        key = name.removeprefix("model.") if strip_model_prefix else name
        if key in params and params[key].shape == tensor.shape:
            params[key].data.copy_(tensor)
            loaded.add(key)
        elif key in buffers and buffers[key].shape == tensor.shape:
            buffers[key].data.copy_(tensor)
            loaded.add(key)
    if getattr(getattr(model, "config", None), "tie_word_embeddings", True):
        if hasattr(model, "lm_head") and hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "phase_angles") and hasattr(model.lm_head, "phase_angles"):
                model.lm_head.phase_angles = emb.phase_angles
    return loaded


def _ensure_2d(input_ids: Optional[torch.Tensor], positions: Optional[torch.Tensor]):
    was_1d = input_ids is not None and input_ids.ndim == 1
    if input_ids is not None and input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if positions is not None and positions.ndim == 1:
        positions = positions.unsqueeze(0)
    return input_ids, positions, was_1d


class GRTModel(HfGRTModel):
    """SGLang `GRTModel` subclassing `transformers.models.grt.GRTModel`."""

    def __init__(
        self,
        config: GRTConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        self.quant_config = quant_config
        self.pooler = Pooler(pooling_type=PoolingType.CIRCULAR, normalize=True)

    def process_weights_after_loading(self) -> None:
        pass

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        filtered = ((k, v) for k, v in weights if not k.startswith("lm_head."))
        loaded = _load_hf_weights(self, filtered, strip_model_prefix=True)
        self.process_weights_after_loading()
        return loaded

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = True,
        **kwargs: Any,
    ) -> Union[EmbeddingPoolerOutput, torch.Tensor]:
        position_ids = kwargs.pop("position_ids", positions)
        inputs_embeds = kwargs.pop("inputs_embeds", input_embeds)
        ids_2d, pos_2d, was_1d = _ensure_2d(input_ids, position_ids)
        if inputs_embeds is not None and inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
            was_1d = True
        out = super().forward(
            input_ids=ids_2d,
            position_ids=pos_2d,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            **kwargs,
        )
        hidden_states = out.last_hidden_state
        if was_1d:
            hidden_states = hidden_states.squeeze(0)
        if get_embedding and forward_batch is not None:
            return self.pooler(hidden_states, forward_batch)
        return hidden_states


class GRTForCausalLM(HfGRTForCausalLM):
    """SGLang `GRTForCausalLM` subclassing `transformers.models.grt.GRTForCausalLM`."""

    def __init__(
        self,
        config: GRTConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        self.quant_config = quant_config
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.CIRCULAR, normalize=True)

    def process_weights_after_loading(self) -> None:
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.phase_angles = self.model.embedding.phase_angles

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = _load_hf_weights(self, weights, strip_model_prefix=False)
        self.process_weights_after_loading()
        return loaded

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        **kwargs: Any,
    ) -> Union[LogitsProcessorOutput, EmbeddingPoolerOutput, torch.Tensor]:
        position_ids = kwargs.pop("position_ids", positions)
        inputs_embeds = kwargs.pop("inputs_embeds", input_embeds)
        ids_2d, pos_2d, was_1d = _ensure_2d(input_ids, position_ids)
        if inputs_embeds is not None and inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
            was_1d = True
        out = self.model(
            input_ids=ids_2d,
            position_ids=pos_2d,
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        hidden_states = out.last_hidden_state
        if was_1d:
            hidden_states = hidden_states.squeeze(0)
        if get_embedding:
            if forward_batch is not None:
                return self.pooler(hidden_states, forward_batch)
            return hidden_states
        logits = self.lm_head(hidden_states)
        if forward_batch is None:
            return logits
        return LogitsProcessorOutput(next_token_logits=logits)


class GRTForSequenceClassification(HfGRTForSequenceClassification):
    """SGLang `GRTForSequenceClassification` subclassing `transformers.models.grt.GRTForSequenceClassification`."""

    def __init__(
        self,
        config: GRTConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        self.quant_config = quant_config
        self.pooler = Pooler(pooling_type=PoolingType.CIRCULAR, normalize=True)

    @staticmethod
    def _circular_pool_seq(seq_hidden: torch.Tensor) -> torch.Tensor:
        return HfGRTForSequenceClassification._circular_pool(seq_hidden.unsqueeze(0))

    def process_weights_after_loading(self) -> None:
        pass

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = _load_hf_weights(self, weights, strip_model_prefix=False)
        self.process_weights_after_loading()
        return loaded

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = True,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[EmbeddingPoolerOutput, torch.Tensor]:
        if token_type_ids is None and forward_batch is not None:
            token_type_ids = getattr(forward_batch, "token_type_ids", None)

        if forward_batch is not None and getattr(forward_batch, "extend_seq_lens", None) is not None:
            seq_lens = list(forward_batch.extend_seq_lens.tolist())
            flat_ids = input_ids.reshape(-1) if input_ids is not None else None
            flat_tt = token_type_ids.reshape(-1) if token_type_ids is not None else None
            scores_list = []
            offset = 0
            for slen in seq_lens:
                sub_ids = flat_ids[offset : offset + slen].unsqueeze(0) if flat_ids is not None else None
                sub_tt = flat_tt[offset : offset + slen].unsqueeze(0) if flat_tt is not None else None
                sub_pos = torch.arange(slen, device=sub_ids.device).unsqueeze(0)
                out = super().forward(input_ids=sub_ids, position_ids=sub_pos, token_type_ids=sub_tt)
                scores_list.append(out.logits)
                offset += slen
            all_scores = torch.cat(scores_list, dim=0)
            return EmbeddingPoolerOutput(embeddings=all_scores)

        position_ids = kwargs.pop("position_ids", positions)
        ids_2d, pos_2d, _ = _ensure_2d(input_ids, position_ids)
        tt_2d = token_type_ids.unsqueeze(0) if token_type_ids is not None and token_type_ids.ndim == 1 else token_type_ids
        out = super().forward(input_ids=ids_2d, position_ids=pos_2d, token_type_ids=tt_2d, **kwargs)
        return out.logits


class GRTVLForConditionalGeneration(HfGRTVLForConditionalGeneration):
    """SGLang `GRTVLForConditionalGeneration` subclassing `transformers.models.grt.GRTVLForConditionalGeneration`."""

    def __init__(
        self,
        config: GRTVLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        self.quant_config = quant_config
        self.logits_processor = LogitsProcessor(config.text_config)

    def process_weights_after_loading(self) -> None:
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.phase_angles = self.get_input_embeddings().phase_angles

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = _load_hf_weights(self, weights, strip_model_prefix=False)
        self.process_weights_after_loading()
        return loaded

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
        input_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[LogitsProcessorOutput, torch.Tensor]:
        position_ids = kwargs.pop("position_ids", positions)
        ids_2d, pos_2d, was_1d = _ensure_2d(input_ids, position_ids)
        out = super().forward(
            input_ids=ids_2d,
            position_ids=pos_2d,
            inputs_embeds=input_embeds,
            pixel_values=pixel_values,
            use_cache=False,
            **kwargs,
        )
        logits = out.logits.squeeze(0) if was_1d else out.logits
        if forward_batch is None:
            return logits
        return LogitsProcessorOutput(next_token_logits=logits)


class GRTASRForConditionalGeneration(HfGRTASRForConditionalGeneration):
    """SGLang `GRTASRForConditionalGeneration` subclassing `transformers.models.grt.GRTASRForConditionalGeneration`."""

    def __init__(
        self,
        config: GRTASRConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        self.quant_config = quant_config
        self.logits_processor = LogitsProcessor(config.text_config)

    def process_weights_after_loading(self) -> None:
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.phase_angles = self.get_input_embeddings().phase_angles

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = _load_hf_weights(self, weights, strip_model_prefix=False)
        self.process_weights_after_loading()
        return loaded

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
        input_embeds: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[LogitsProcessorOutput, torch.Tensor]:
        position_ids = kwargs.pop("position_ids", positions)
        ids_2d, pos_2d, was_1d = _ensure_2d(input_ids, position_ids)
        out = super().forward(
            input_ids=ids_2d,
            position_ids=pos_2d,
            inputs_embeds=input_embeds,
            audio_values=audio_values,
            use_cache=False,
            **kwargs,
        )
        logits = out.logits.squeeze(0) if was_1d else out.logits
        if forward_batch is None:
            return logits
        return LogitsProcessorOutput(next_token_logits=logits)


class GRTTTSForConditionalGeneration(HfGRTTTSForConditionalGeneration):
    """SGLang `GRTTTSForConditionalGeneration` subclassing `transformers.models.grt.GRTTTSForConditionalGeneration`."""

    def __init__(
        self,
        config: GRTTTSConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        self.quant_config = quant_config
        self.logits_processor = LogitsProcessor(config.text_config)

    def process_weights_after_loading(self) -> None:
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.phase_angles = self.get_input_embeddings().phase_angles

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = _load_hf_weights(self, weights, strip_model_prefix=False)
        self.process_weights_after_loading()
        return loaded

    @torch.no_grad()
    def synthesize_speech(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        return_waveform: bool = True,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        ids_2d, pos_2d, _ = _ensure_2d(input_ids, positions)
        out = super().forward(
            input_ids=ids_2d,
            position_ids=pos_2d,
            return_speech=True,
            return_waveform=return_waveform,
            use_cache=False,
        )
        return out.speech_waveform, out.speech_spec

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[LogitsProcessorOutput, torch.Tensor]:
        position_ids = kwargs.pop("position_ids", positions)
        ids_2d, pos_2d, was_1d = _ensure_2d(input_ids, position_ids)
        out = super().forward(
            input_ids=ids_2d,
            position_ids=pos_2d,
            inputs_embeds=input_embeds,
            return_waveform=False,
            use_cache=False,
            **kwargs,
        )
        logits = out.logits.squeeze(0) if was_1d else out.logits
        if forward_batch is None:
            return logits
        return LogitsProcessorOutput(next_token_logits=logits)


class GRTOmniForConditionalGeneration(HfGRTOmniForConditionalGeneration):
    """SGLang `GRTOmniForConditionalGeneration` subclassing `transformers.models.grt.GRTOmniForConditionalGeneration`."""

    def __init__(
        self,
        config: GRTOmniConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        self.quant_config = quant_config
        self.logits_processor = LogitsProcessor(config.text_config)

    def process_weights_after_loading(self) -> None:
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.phase_angles = self.get_input_embeddings().phase_angles

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = _load_hf_weights(self, weights, strip_model_prefix=False)
        self.process_weights_after_loading()
        return loaded

    @torch.no_grad()
    def synthesize_speech(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        return_waveform: bool = True,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        ids_2d, pos_2d, _ = _ensure_2d(input_ids, positions)
        out = super().forward(
            input_ids=ids_2d,
            position_ids=pos_2d,
            pixel_values=pixel_values,
            audio_values=audio_values,
            return_speech=True,
            return_waveform=return_waveform,
            use_cache=False,
        )
        return out.speech_waveform, out.speech_spec

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
        input_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[LogitsProcessorOutput, torch.Tensor]:
        position_ids = kwargs.pop("position_ids", positions)
        ids_2d, pos_2d, was_1d = _ensure_2d(input_ids, position_ids)
        out = super().forward(
            input_ids=ids_2d,
            position_ids=pos_2d,
            inputs_embeds=input_embeds,
            pixel_values=pixel_values,
            audio_values=audio_values,
            use_cache=False,
            **kwargs,
        )
        logits = out.logits.squeeze(0) if was_1d else out.logits
        if forward_batch is None:
            return logits
        return LogitsProcessorOutput(next_token_logits=logits)


EntryClass = [
    GRTForCausalLM,
    GRTModel,
    GRTForSequenceClassification,
    GRTVLForConditionalGeneration,
    GRTASRForConditionalGeneration,
    GRTTTSForConditionalGeneration,
    GRTOmniForConditionalGeneration,
]
