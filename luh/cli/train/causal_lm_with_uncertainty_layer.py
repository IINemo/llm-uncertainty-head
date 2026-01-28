from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

import torch
from torch.nn import BCEWithLogitsLoss

from typing import Optional
from dataclasses import dataclass


@dataclass
class CausalLMWithUncertaintyOutput(CausalLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    uncertainty: torch.FloatTensor = None


class CausalLMWithUncertaintyLayer(PreTrainedModel):
    def __init__(
        self,
        base_model,
        ue_head,
        ue_pos_weight: float,
        output_attention: bool = False,
        is_vlm: bool = False
    ):
        super().__init__(PretrainedConfig())

        self.orig_base_model = base_model
        self.ue_head = ue_head
        self._ue_pos_weight = ue_pos_weight
        self._output_attention = output_attention
        self._is_vlm = is_vlm

    def generate(self, *args, **kwargs):
        kwargs.update(
            {
                "return_dict_in_generate": True,
                "output_scores": True,
                "output_hidden_states": True,
                "output_attentions": self._output_attention
            }
        )

        out = self.orig_base_model.generate(*args, **kwargs)
        uncertainty = self.ue_head(out)
        return out, uncertainty

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        uncertainty_labels=None,
        context_lengths=None,
        return_dict=None,
        reply=None,
        **kwargs
    ):
        return_dict = (
            return_dict
            if return_dict is not None
            else self.orig_base_model.config.use_return_dict
        )
        output_hidden_states = True
        output_attentions = self._output_attention

        # Filter out arguments not supported by VLMs
        # Some VLMs (like Qwen2.5-VL) don't accept num_items_in_batch
        if self._is_vlm:
            kwargs = {k: v for k, v in kwargs.items() if k != 'num_items_in_batch'}

        outputs = self.orig_base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        logits = outputs.logits
        outputs.context_lengths = context_lengths

        uncertainty = self.ue_head({"input_ids": input_ids, "attention_mask": attention_mask}, outputs)

        if uncertainty_labels is not None:
            uncertainty_labels = uncertainty_labels[:, 1:] # Shifting labels
            uncertainty_labels = uncertainty_labels.reshape(-1)
            mask = uncertainty_labels != -100
            uncertainty_raveled = uncertainty.reshape(-1)[mask]
            uncertainty_labels = uncertainty_labels[mask]
            loss_fct = BCEWithLogitsLoss(
                pos_weight=torch.Tensor([self._ue_pos_weight]).to(
                    uncertainty_raveled.device
                )
            )
            loss = loss_fct(
                uncertainty_raveled.to(torch.float32),
                uncertainty_labels.to(torch.float32),
            )
        else:
            loss = None

        if not return_dict:
            output = (logits, uncertainty) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMWithUncertaintyOutput(
            loss=loss,
            logits=logits,
            uncertainty=uncertainty,
        )

def gradient_checkpointing_enable(self, *args, **kwargs):
    return self.base_model.gradient_checkpointing_enable(*args, **kwargs)