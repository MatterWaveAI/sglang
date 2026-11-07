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
"""GRT (Geometric Rotation Transformer) model configurations and Toroidal LoRA utilities.

Imports all GRT configuration classes directly from `transformers.models.grt.configuration_grt`.
"""

from __future__ import annotations

import json
import math
import os

import torch
from transformers.models.grt.configuration_grt import (
    GRTASRConfig,
    GRTAudioConfig,
    GRTConfig,
    GRTOmniConfig,
    GRTTTSConfig,
    GRTVisionConfig,
    GRTVLConfig,
)


def merge_grt_lora_weights(
    base_state_dict: dict,
    lora_state_dict: dict,
    r: int = 8,
    lora_alpha: float = 16.0,
    wrap_mod_2pi: bool = True,
) -> dict:
    """Pre-merge toroidal GRTRotationalLoRA weights (`lora_A`, `lora_B`) into `base_state_dict`.

    Computes `theta = (base_angles + (lora_alpha / r) * (lora_A @ lora_B)) mod 2*pi`
    with zero inference overhead.
    """
    merged = dict(base_state_dict)
    lora_a_keys: dict[str, str] = {}
    lora_b_keys: dict[str, str] = {}
    for k in lora_state_dict:
        if k.endswith(".lora_A") or k.endswith(".lora_a"):
            lora_a_keys[k[:-7]] = k
        elif k.endswith(".lora_B") or k.endswith(".lora_b"):
            lora_b_keys[k[:-7]] = k
        elif k in merged:
            merged[k] = lora_state_dict[k].to(
                device=merged[k].device, dtype=merged[k].dtype
            )

    for prefix, key_a in lora_a_keys.items():
        if prefix not in lora_b_keys:
            continue
        lora_a = lora_state_dict[key_a].float()
        lora_b = lora_state_dict[lora_b_keys[prefix]].float()
        inferred_r = int(lora_a.shape[-1]) if lora_a.ndim >= 2 else r
        scale = float(lora_alpha) / float(max(1, inferred_r))
        delta = torch.matmul(lora_a, lora_b) * scale

        candidates = [prefix]
        if not prefix.endswith(".angles"):
            candidates.append(f"{prefix}.angles")
        if prefix.startswith("base_model.model."):
            stripped = prefix[len("base_model.model.") :]
            candidates.extend([stripped, f"{stripped}.angles"])

        target_key = next((c for c in candidates if c in merged), None)
        if target_key is None:
            continue
        base_angles = merged[target_key]
        delta_dev = delta.to(device=base_angles.device, dtype=torch.float32)
        if base_angles.ndim == 3 and delta_dev.ndim == 2:
            if (
                delta_dev.shape[0] == base_angles.shape[0]
                and delta_dev.shape[1] == base_angles.shape[2]
            ):
                delta_dev = delta_dev.unsqueeze(1).expand(base_angles.shape)
            else:
                delta_dev = delta_dev.reshape(base_angles.shape)
        elif tuple(delta_dev.shape) != tuple(base_angles.shape):
            delta_dev = delta_dev.reshape(base_angles.shape)
        updated = base_angles.float() + delta_dev
        if wrap_mod_2pi:
            updated = torch.remainder(updated, 2.0 * math.pi)
        merged[target_key] = updated.to(dtype=base_angles.dtype)

    return merged


def load_grt_lora_adapter(
    model_or_state_dict,
    lora_adapter_path_or_dict,
    r: int = 8,
    lora_alpha: float = 16.0,
    wrap_mod_2pi: bool = True,
):
    """Load and pre-merge a toroidal GRTRotationalLoRA adapter into an SGLang/HF GRT model or state_dict."""
    if isinstance(lora_adapter_path_or_dict, str):
        path = lora_adapter_path_or_dict
        if os.path.isdir(path):
            for cfg_name in ("grt_lora_config.json", "adapter_config.json"):
                cfg_path = os.path.join(path, cfg_name)
                if os.path.exists(cfg_path):
                    with open(cfg_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    r = int(meta.get("r", r))
                    lora_alpha = float(meta.get("lora_alpha", lora_alpha))
                    break
            st_file = os.path.join(path, "grt_lora_adapter.safetensors")
            if not os.path.exists(st_file):
                st_file = os.path.join(path, "adapter_model.safetensors")
            from safetensors.torch import load_file

            lora_sd = load_file(st_file)
        else:
            from safetensors.torch import load_file

            lora_sd = load_file(path)
    else:
        lora_sd = dict(lora_adapter_path_or_dict)

    if isinstance(model_or_state_dict, dict):
        return merge_grt_lora_weights(
            model_or_state_dict,
            lora_sd,
            r=r,
            lora_alpha=lora_alpha,
            wrap_mod_2pi=wrap_mod_2pi,
        )

    state = model_or_state_dict.state_dict()
    merged = merge_grt_lora_weights(
        state,
        lora_sd,
        r=r,
        lora_alpha=lora_alpha,
        wrap_mod_2pi=wrap_mod_2pi,
    )
    model_or_state_dict.load_state_dict(merged, strict=False)
    if hasattr(model_or_state_dict, "process_weights_after_loading"):
        model_or_state_dict.process_weights_after_loading()
    return model_or_state_dict


__all__ = [
    "GRTASRConfig",
    "GRTAudioConfig",
    "GRTConfig",
    "GRTOmniConfig",
    "GRTTTSConfig",
    "GRTVisionConfig",
    "GRTVLConfig",
    "load_grt_lora_adapter",
    "merge_grt_lora_weights",
]
