# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

import torch

from torchtune.models.convert_weights import get_mapped_key

# NOTE: This file is the same as the Qwen2 _convert_weights.py file with one key difference.
# For tied-embedding Qwen2 models, only the embedding weight is stored on the HF Hub.
# However, for Qwen3, both the embedding and output weights are stored on the Hub.
# While we handle the tying ourselves on load, we do need to duplicate the weight to save in HF's format.
# The exception is for Qwen3 4B, which matches the behavior of Qwen2.

# state dict key mappings from HF's format to torchtune's format
_FROM_HF = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attn.q_proj.bias",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.k_proj.weight",
    "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attn.k_proj.bias",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.v_proj.weight",
    "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attn.v_proj.bias",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.output_proj.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attn.q_norm.scale",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attn.k_norm.scale",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.norm.weight": "norm.scale",
    "lm_head.weight": "output.weight",
}

_FROM_HF_MOE = {
    **_FROM_HF,
    "model.layers.{}.mlp.gate.weight": "layers.{}.mlp.router.gate.weight",
}

# Matches HF's packed expert weights, for example:
# model.layers.0.mlp.experts.gate_up_proj
_HF_PACKED_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
# Matches HF's per-expert weights, for example:
# model.layers.0.mlp.experts.12.down_proj.weight
_HF_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
# Matches torchtune's grouped expert weights, for example:
# layers.0.mlp.experts.gate_proj
_TUNE_EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)$"
)


QWEN3_TIED_KEY = "lm_head.weight"
QWEN3_TUNE_EMBEDDING_KEY = "tok_embeddings.weight"


def qwen3_hf_to_tune(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to TorchTune's format, which contains the weights
    of a Qwen3 model.
    State dicts from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.
    The logic is identical to :func:`~torchtune.models.convert_weights.hf_to_tune`, but may not load
    output projection weights.

    Args:
        state_dict (dict[str, torch.Tensor]): State dict in HF's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.
        tie_word_embeddings (bool): Whether the model's input and output word embeddings should be tied.

    Returns:
        dict[str, torch.Tensor]: State dict in torchtune's format.
    """
    converted_state_dict = {}
    if head_dim is None:
        head_dim = dim // num_heads

    for key, value in state_dict.items():
        if (
            tie_word_embeddings and QWEN3_TIED_KEY in key
        ):  # Skip loading the output projection weights
            continue
        if "rotary_emb.inv_freq" in key:  # Skip loading the position embeddings
            continue

        new_key = get_mapped_key(key, _FROM_HF)
        converted_state_dict[new_key] = value
    return converted_state_dict


def qwen3_tune_to_hf(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
):
    """
    Convert a state dict from torchtune's format to HF's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (dict[str, torch.Tensor]): State dict in torchtune's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.
        tie_word_embeddings (bool): Whether the model's input and output word embeddings should be tied.

    Returns:
        dict[str, torch.Tensor]: State dict in HF's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}
    if head_dim is None:
        head_dim = dim // num_heads

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value
        if QWEN3_TUNE_EMBEDDING_KEY in key and tie_word_embeddings:
            # If the model's input and output word embeddings are tied, we need to
            # copy the input word embeddings to the output word embeddings
            converted_state_dict["lm_head.weight"] = value.detach().clone()

    return converted_state_dict


def qwen3_moe_hf_to_tune(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    num_experts: int = 128,
    dim: int = 4096,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to TorchTune's format for Qwen3 MoE models.
    State dicts from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.

    Expert weights may be stored as packed 3D tensors or per-expert 2D tensors in
    HF's format, and are split by projection in torchtune's format. For example,
    ``model.layers.0.mlp.experts.gate_up_proj`` is split into
    ``layers.0.mlp.experts.gate_proj`` and ``layers.0.mlp.experts.up_proj``.

    Args:
        state_dict (dict[str, torch.Tensor]): State dict in HF's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        num_experts (int): Number of experts in each MoE layer.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.
        tie_word_embeddings (bool): Whether the model's input and output word embeddings should be tied.

    Returns:
        dict[str, torch.Tensor]: State dict in torchtune's format.

    Raises:
        ValueError: If ``num_experts`` is not positive, or if an expert tensor
            has an unexpected shape or missing expert weights.
    """
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}.")

    converted_state_dict = {}
    if head_dim is None:
        head_dim = dim // num_heads

    for key, value in state_dict.items():
        if (
            tie_word_embeddings and QWEN3_TIED_KEY in key
        ):  # Skip loading the output projection weights
            continue
        if "rotary_emb.inv_freq" in key:  # Skip loading the position embeddings
            continue

        packed_expert_match = _HF_PACKED_EXPERT_RE.match(key)
        if packed_expert_match is not None:
            layer_num, projection = packed_expert_match.groups()
            if value.dim() != 3:
                raise ValueError(
                    f"Expected a 3D packed expert weight for {key}, got shape {value.shape}."
                )
            if value.shape[0] != num_experts:
                raise ValueError(
                    f"Expected {num_experts} experts for {key}, got {value.shape[0]}."
                )
            if projection == "gate_up_proj":
                if value.shape[1] % 2 != 0:
                    raise ValueError(
                        f"Expected an even gate/up dimension for {key}, got shape {value.shape}."
                    )
                gate_proj, up_proj = torch.chunk(value, 2, dim=1)
                converted_state_dict[
                    f"layers.{layer_num}.mlp.experts.gate_proj"
                ] = gate_proj.transpose(1, 2)
                converted_state_dict[
                    f"layers.{layer_num}.mlp.experts.up_proj"
                ] = up_proj.transpose(1, 2)
            else:
                converted_state_dict[
                    f"layers.{layer_num}.mlp.experts.down_proj"
                ] = value.transpose(1, 2)
            continue

        expert_match = _HF_EXPERT_RE.match(key)
        if expert_match is not None:
            layer_num, expert_idx, projection = expert_match.groups()
            if expert_idx != "0":
                continue

            expert_tensors = []
            for idx in range(num_experts):
                expert_key = (
                    f"model.layers.{layer_num}.mlp.experts.{idx}.{projection}.weight"
                )
                if expert_key not in state_dict:
                    raise ValueError(
                        f"Missing expert weight {expert_key} while converting layer {layer_num}."
                    )
                expert_value = state_dict[expert_key]
                if expert_value.dim() != 2:
                    raise ValueError(
                        f"Expected a 2D expert weight for {expert_key}, got shape "
                        f"{expert_value.shape}."
                    )
                expert_tensors.append(expert_value.T)
            converted_state_dict[
                f"layers.{layer_num}.mlp.experts.{projection}"
            ] = torch.stack(expert_tensors)
            continue

        new_key = get_mapped_key(key, _FROM_HF_MOE)
        converted_state_dict[new_key] = value

    return converted_state_dict


def qwen3_moe_tune_to_hf(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    num_experts: int = 128,
    dim: int = 4096,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Convert a Qwen3 MoE state dict from torchtune's format to HF's per-expert
    format. This function doesn't handle any sharding or splitting of state dicts.
    It follows the state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (dict[str, torch.Tensor]): State dict in torchtune's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        num_experts (int): Number of experts in each MoE layer.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.
        tie_word_embeddings (bool): Whether the model's input and output word embeddings should be tied.

    Returns:
        dict[str, torch.Tensor]: State dict in HF's format.

    Raises:
        ValueError: If ``num_experts`` is not positive, if a grouped expert tensor
            has an unexpected shape, or if grouped expert projections are missing
            or incompatible.
    """
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}.")

    converted_state_dict = {}
    expert_weights: dict[int, dict[str, torch.Tensor]] = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_HF_MOE.items()}
    if head_dim is None:
        head_dim = dim // num_heads

    for key, value in state_dict.items():
        expert_match = _TUNE_EXPERT_RE.match(key)
        if expert_match is not None:
            layer_num, projection = expert_match.groups()
            if value.dim() != 3:
                raise ValueError(
                    f"Expected a 3D grouped expert weight for {key}, got shape {value.shape}."
                )
            if value.shape[0] != num_experts:
                raise ValueError(
                    f"Expected {num_experts} experts for {key}, got {value.shape[0]}."
                )
            expert_weights.setdefault(int(layer_num), {})[projection] = value
            continue

        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value
        if QWEN3_TUNE_EMBEDDING_KEY in key and tie_word_embeddings:
            # If the model's input and output word embeddings are tied, we need to
            # copy the input word embeddings to the output word embeddings
            converted_state_dict["lm_head.weight"] = value.detach().clone()

    for layer_num, weights in sorted(expert_weights.items()):
        missing_projections = sorted(
            {"gate_proj", "up_proj", "down_proj"} - weights.keys()
        )
        if missing_projections:
            raise ValueError(
                f"Missing expert projections for layer {layer_num}: {missing_projections}."
            )
        gate_proj = weights["gate_proj"]
        up_proj = weights["up_proj"]
        down_proj = weights["down_proj"]
        if gate_proj.shape != up_proj.shape:
            raise ValueError(
                f"Expected gate_proj and up_proj shapes to match for layer {layer_num}, "
                f"got {gate_proj.shape} and {up_proj.shape}."
            )
        for expert_idx, tensor in enumerate(torch.unbind(gate_proj)):
            converted_state_dict[
                f"model.layers.{layer_num}.mlp.experts.{expert_idx}.gate_proj.weight"
            ] = tensor.T.contiguous()
        for expert_idx, tensor in enumerate(torch.unbind(up_proj)):
            converted_state_dict[
                f"model.layers.{layer_num}.mlp.experts.{expert_idx}.up_proj.weight"
            ] = tensor.T.contiguous()
        for expert_idx, tensor in enumerate(torch.unbind(down_proj)):
            converted_state_dict[
                f"model.layers.{layer_num}.mlp.experts.{expert_idx}.down_proj.weight"
            ] = tensor.T.contiguous()

    return converted_state_dict
