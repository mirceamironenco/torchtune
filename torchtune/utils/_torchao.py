# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any


try:
    from torchao.dtypes import NF4Tensor, to_nf4
except ImportError:
    try:
        from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4
    except ImportError:
        from torchao.quantization import NF4Tensor, to_nf4


def linear_nf4(*args: Any, **kwargs: Any) -> Any:
    try:
        from torchao.dtypes.nf4tensor import linear_nf4 as _linear_nf4
    except ImportError:
        from torchao.quantization.quantize_.workflows.nf4.nf4_tensor import (
            linear_nf4 as _linear_nf4,
        )

    return _linear_nf4(*args, **kwargs)


def nf4_tensor_impl(*args: Any, **kwargs: Any) -> Any:
    try:
        from torchao.dtypes.nf4tensor import implements
    except ImportError:
        from torchao.quantization.quantize_.workflows.nf4.nf4_tensor import implements

    return implements(*args, **kwargs)


__all__ = ["NF4Tensor", "linear_nf4", "nf4_tensor_impl", "to_nf4"]
