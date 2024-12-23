# coding=utf-8
# Copyright 2024 Jingze Shi, Bingheng Wu and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on the Wonderful Matrices paper implementation.
#
#     https://arxiv.org/abs/2412.11834
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
"""PyTorch Doge Vision model."""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    # is_einx_available,
    logging,
    replace_return_docstrings,
)
from .configuration_doge_vision import DogeVisionConfig
from .modeling_doge import DogePreTrainedModel, DogeModel, DogeForCausalLM

try:
    from einx import add as einx_add
except ImportError:
    einx_add = None


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DogeConfig"


class DogePatchEmbedding(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` of shape `(batch_size, seq_len, hidden_size)` to be consumed by a Transformer.
    """

    def __init__(self, config: DogeVisionConfig):
        super().__init__()

        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.hidden_dim = config.hidden_size

        self.sequence_proj = nn.Conv2d(self.num_channels, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.state_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=config.hidden_bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        
        image_embedding = self.sequence_proj(pixel_values).flatten(2).transpose(1, 2)
        image_embedding = self.state_proj(image_embedding)
        return image_embedding


class DogeForCausalVLM(DogeForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DogeVisionConfig):
        super().__init__(config)
        self.config = config
        self.pixel_embed = DogePatchEmbedding(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            pixel_embeds = self.pixel_embed(pixel_values)
        if inputs_embeds is not None and pixel_values is not None:
            inputs_embeds = torch.cat([inputs_embeds, pixel_embeds], dim=1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds if inputs_embeds is not None else pixel_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        return outputs


@dataclass
class DogeObjectDetectionOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DogeForObjectDetection(DogePreTrainedModel):
    def __init__(self, config: DogeVisionConfig):
        super().__init__(config)
        self.config = config

        # pixel embedding
        self.pixel_embed = DogePatchEmbedding(config)

        # model backbone
        self.model = DogeModel(config)
        self.num_labels = config.num_labels

        # classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels + 1)

        # detection head
        self.detector = nn.Linear(config.hidden_size, 4)

        # Initialize weights and apply final processing
        self.post_init()
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, DogeObjectDetectionOutputWithPast]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: `'class_labels'` and `'boxes'` (the class labels and bounding boxes of an image in the
            batch respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding
            boxes in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image,
            4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
        >>> model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        ...     0
        ... ]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected remote with confidence 0.991 at location [46.48, 72.78, 178.98, 119.3]
        Detected remote with confidence 0.908 at location [336.48, 79.27, 368.23, 192.36]
        Detected cat with confidence 0.934 at location [337.18, 18.06, 638.14, 373.09]
        Detected cat with confidence 0.979 at location [10.93, 53.74, 313.41, 470.67]
        Detected remote with confidence 0.974 at location [41.63, 72.23, 178.09, 119.99]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is not None:
            inputs_embeds = self.pixel_embed(pixel_values)
        
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Take the final hidden states of the detection tokens
        hidden_states = hidden_states[:, -self.config.num_detection_tokens:, :]

        logits = self.classifier(hidden_states)
        pred_boxes = self.detector(hidden_states).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, self.device, pred_boxes, self.config, outputs_class, outputs_coord
            )
        
        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output
    
        return DogeObjectDetectionOutputWithPast(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        




