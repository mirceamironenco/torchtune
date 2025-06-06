# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional
from functools import partial

from torchtune.models.llama2._component_builders import llama2, lora_llama2, llama2_classifier, lora_llama2_classifier

from torchtune.modules import TransformerDecoder
from torchtune.models.llama2._tokenizer import Llama2Tokenizer
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.data._prompt_templates import _TemplateType
from torchtune.data._prompt_templates import _get_prompt_template


"""
Model builders build specific instantiations using component builders. For example
the llama2_7b model builder uses the llama2 component builder to create the
llama2 7B model.
"""


def llama2_7b() -> TransformerDecoder:
    """
    Builder for creating a Llama2 model initialized w/ the default 7B parameter values
    from https://arxiv.org/abs/2307.09288

    Returns:
        TransformerDecoder: Instantiation of Llama2 7B model
    """
    return llama2(
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def llama2_tokenizer(path: str, max_seq_len: Optional[int] = None, prompt_template: Optional[_TemplateType] = "torchtune.models.llama2.Llama2ChatTemplate", truncation_type: str = "right") -> Llama2Tokenizer:
    """
    Tokenizer for Llama2.

    Args:
        path (str): path to the tokenizer
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags. Default is :class:`~torchtune.models.llama2.Llama2ChatTemplate`.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".
    Returns:
        Llama2Tokenizer: Instantiation of the Llama2 tokenizer
    """
    return Llama2Tokenizer(path=path, max_seq_len=max_seq_len, prompt_template=_get_prompt_template(prompt_template) if prompt_template is not None else None, truncation_type=truncation_type)


def lora_llama2_7b(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Llama2 7B model with LoRA enabled.

    The Llama2 defaults are the same as in :func:`~torchtune.models.llama2.llama2_7b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (list[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights
        
    Returns:
        TransformerDecoder: Instantiation of Llama2 7B model with LoRA applied
    """
    return lora_llama2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_llama2_7b = partial(lora_llama2_7b, quantize_base=True)

qlora_llama2_7b.__doc__ = """
Builder for creating a Llama2 7B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama2_7b` for full API arguments.
"""


def llama2_13b() -> TransformerDecoder:
    """
    Builder for creating a Llama2 model initialized w/ the default 13B parameter values
    from https://arxiv.org/abs/2307.09288

    Returns:
        TransformerDecoder: Instantiation of Llama2 13B model
    """
    return llama2(
        vocab_size=32_000,
        num_layers=40,
        num_heads=40,
        num_kv_heads=40,
        embed_dim=5120,
        intermediate_dim=13824,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def lora_llama2_13b(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Llama2 13B model with LoRA enabled.

    The Llama2 defaults are the same as in :func:`~torchtune.models.llama2.llama2_13b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (list[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Llama2 13B model with LoRA applied
    """

    return lora_llama2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=32_000,
        num_layers=40,
        num_heads=40,
        num_kv_heads=40,
        embed_dim=5120,
        intermediate_dim=13824,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_llama2_13b = partial(lora_llama2_13b, quantize_base=True)
qlora_llama2_13b.__doc__ = """
Builder for creating a Llama2 13B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama2_13b` for full API arguments.
"""


def llama2_70b() -> TransformerDecoder:
    """
    Builder for creating a Llama2 model initialized w/ the default 70B parameter values
    from https://arxiv.org/abs/2307.09288

    Returns:
        TransformerDecoder: Instantiation of Llama2 70B model
    """
    return llama2(
        vocab_size=32_000,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        intermediate_dim=28672,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def lora_llama2_70b(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Llama2 70B model with LoRA enabled.

    The Llama2 defaults are the same as in :func:`~torchtune.models.llama2.llama2_70b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (list[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Llama2 70B model with LoRA applied
    """
    return lora_llama2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=32_000,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        max_seq_len=4096,
        intermediate_dim=28672,
        attn_dropout=0.0,
        norm_eps=1e-5,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_llama2_70b = partial(lora_llama2_70b, quantize_base=True)
qlora_llama2_70b.__doc__ = """
Builder for creating a Llama2 70B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama2_70b` for full API arguments.
"""


def llama2_reward_7b() -> TransformerDecoder:
    """
    Builder for creating a Llama2 model initialized w/ the default 7B parameter values
    from https://arxiv.org/abs/2307.09288, where the output layer is a classification layer
    projecting to a single class for reward modelling.

    Returns:
        TransformerDecoder: Instantiation of Llama2 7B model
    """
    return llama2_classifier(
        num_classes=1,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def lora_llama2_reward_7b(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Llama2 7B reward model with LoRA enabled.

    The Llama2 classifier defaults are the same as in :func:`~torchtune.models.llama2.llama2_reward_7b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (list[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Llama2 7B model with LoRA applied
    """
    return lora_llama2_classifier(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        num_classes=1,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_llama2_reward_7b = partial(lora_llama2_7b, quantize_base=True)
qlora_llama2_reward_7b.__doc__ = """
Builder for creating a Llama2 reward 7b model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama2_reward_7b` for full API arguments.
"""
