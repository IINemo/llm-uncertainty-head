from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

import torch
from torch.nn import BCEWithLogitsLoss

from itertools import chain

from .causal_lm_with_uncertainty_layer import CausalLMWithUncertaintyOutput


class CausalLMWithUncertaintyLayerClaim(PreTrainedModel):
    def __init__(
        self,
        base_model,
        ue_head,
        ue_pos_weight: float,
        output_attention: bool = False,
        output_router_logits: bool = False,
    ):
        super().__init__(PretrainedConfig())

        self.orig_base_model = base_model
        self.ue_head = ue_head
        self._output_attention = output_attention
        self._output_router_logits = output_router_logits
        self._ue_pos_weight = ue_pos_weight

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        context_lengths=None,
        claims=None,
        verified=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = (
            return_dict
            if return_dict is not None
            else self.orig_base_model.config.use_return_dict
        )
        output_hidden_states = True
        output_attentions = self._output_attention
        output_router_logits = self._output_router_logits

        outputs = self.orig_base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            **kwargs,
        )
        logits = outputs.logits
        outputs.context_lengths = context_lengths

        ue_head_input = {"input_ids": input_ids,
                         "attention_mask": attention_mask,
                         "claims": claims}

        uncertainty = self.ue_head(ue_head_input, outputs)

        if verified is not None:
            verified = torch.as_tensor(list(chain(*verified)), device=self.device)#verified.reshape(-1)
            mask = verified != -100
            #uncertainty_raveled = uncertainty[mask]
            uncertainty_raveled = uncertainty.reshape(-1)
            uncertainty_raveled = uncertainty_raveled[uncertainty_raveled != -100]
            uncertainty_raveled = uncertainty_raveled[mask]
            #uncertainty_raveled = torch.cat(uncertainty).reshape(-1)[mask]#uncertainty.reshape(-1)[mask]
            uncertainty_labels = verified[mask]
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

        return CausalLMWithUncertaintyOutput(
            loss=loss,
            logits=logits,
            uncertainty=uncertainty,
        )


def gradient_checkpointing_enable(self, *args, **kwargs):
    return self.base_model.gradient_checkpointing_enable(*args, **kwargs)
