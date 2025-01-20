# coding=utf-8
# Copyright 2024 Jingze Shi. All rights reserved.
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
from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["ssd"] = [
        "SSD",
    ]
    _import_structure["dmattn"] = [
        "DMA",
    ]
    _import_structure["cdmoe"] = [
        "CDMoE",
    ]
    _import_structure["peer"] = [
        "PEER",
    ]
    _import_structure["seimoe"] = [
        "SEIMoE",
    ]
    _import_structure["mlp"] = [
        "MLP",
        "GatedMLP",
    ]


if TYPE_CHECKING:

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .ssd import SSD
        from .dma import DMA
        from .cdmoe import CDMoE
        from .peer import PEER
        from .seimoe import SEIMoE
        from .mlp import MLP, GatedMLP


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
