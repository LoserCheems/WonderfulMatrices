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
    _import_structure["configuration_cheems"] = [
        "CheemsConfig"
    ]
    _import_structure["configuration_doge"] = [
        "DogeConfig"
    ]
    _import_structure["modeling_cheems"] = [
        "CheemsForCausalLM",
        "CheemsForSequenceClassification",
        "CheemsModel",
        "CheemsPreTrainedModel",
    ]
    _import_structure["modeling_doge"] = [
        "DogeForCausalLM",
        "DogeForSequenceClassification",
        "DogeModel",
        "DogePreTrainedModel",
    ]
    _import_structure["ssd"] = [
        "SSD",
    ]
    _import_structure["dmattn"] = [
        "DMAttn",
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
        from .models.configuration_cheems import CheemsConfig
        from .models.configuration_doge import DogeConfig
        from .models.modeling_cheems import (
            CheemsForCausalLM,
            CheemsForSequenceClassification,
            CheemsModel,
            CheemsPreTrainedModel,
        )
        from .models.modeling_doge import (
            DogeForCausalLM,
            DogeForSequenceClassification,
            DogeModel,
            DogePreTrainedModel,
        )
        from .modules.ssd import SSD
        from .modules.dmattn import DMAttn
        from .modules.cdmoe import CDMoE
        from .modules.peer import PEER
        from .modules.seimoe import SEIMoE
        from .modules.mlp import MLP, GatedMLP


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
