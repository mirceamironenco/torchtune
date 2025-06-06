# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from torch import nn

from torchtune import config, training, utils

logger = utils.get_logger("DEBUG")


class QuantizationRecipe:
    """
    Recipe for quantizing a Transformer-based LLM.
    Uses quantizer classes from torchao to quantize a model.

    Supported quantization modes are:
    8da4w (PyTorch 2.3+):
        torchtune.training.quantization.Int8DynActInt4WeightQuantizer
        int8 per token dynamic activation with int4 weight only per axis group quantization
        Args:
            `groupsize` (int): a parameter of int4 weight only quantization,
            it refers to the size of quantization groups which get independent quantization parameters
            e.g. 32, 64, 128, 256, smaller numbers means more fine grained and higher accuracy,
            but also higher memory overhead

    8da4w-qat (PyTorch 2.4+):
        torchtune.training.quantization.Int8DynActInt4WeightQATQuantizer
        int8 per token dynamic activation with int4 weight only per axis group quantization
        Same as "8da4w", but for quantizing QAT checkpoints
        Args:
            `groupsize` (int): a parameter of int4 weight only quantization,
            it refers to the size of quantization groups which get independent quantization parameters
            e.g. 32, 64, 128, 256, smaller numbers means more fine grained and higher accuracy,
            but also higher memory overhead
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = training.get_quantizer_mode(self._quantizer)
        training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )

    def load_checkpoint(self, checkpointer_cfg: DictConfig) -> dict[str, Any]:
        self._checkpointer = config.instantiate(checkpointer_cfg)
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    def setup(self, cfg: DictConfig) -> None:
        ckpt_dict = self.load_checkpoint(cfg.checkpointer)
        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
        )

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: dict[str, Any],
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if "qat" in self._quantization_mode:
            model = self._quantizer.prepare(model)
        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")
        return model

    @torch.no_grad()
    def quantize(self, cfg: DictConfig):
        t0 = time.perf_counter()
        if "qat" in self._quantization_mode:
            self._model = self._quantizer.convert(self._model)
        else:
            self._model = self._quantizer.quantize(self._model)
        t = time.perf_counter() - t0
        logger.info(f"Time for quantization: {t:.02f} sec")
        if self._device.type != "cpu":
            torch_device = utils.get_torch_device_namespace()
            logger.info(
                f"Memory used: {torch_device.max_memory_allocated() / 1e9:.02f} GB"
            )

    def save_checkpoint(self, cfg: DictConfig):
        ckpt_dict = self._model.state_dict()
        file_name = cfg.checkpointer.checkpoint_files[0].split(".")[0]

        output_dir = Path(cfg.checkpointer.output_dir)
        output_dir.mkdir(exist_ok=True)
        checkpoint_file = Path.joinpath(
            output_dir, f"{file_name}-{self._quantization_mode}".rstrip("-qat")
        ).with_suffix(".ckpt")

        torch.save(ckpt_dict, checkpoint_file)
        logger.info(
            "Model checkpoint of size "
            f"{os.path.getsize(checkpoint_file) / 1024**3:.2f} GiB "
            f"saved to {checkpoint_file}"
        )


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="QuantizationRecipe", cfg=cfg)
    recipe = QuantizationRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.quantize(cfg=cfg)
    recipe.save_checkpoint(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
