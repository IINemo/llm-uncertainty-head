import torch

from .feature_extractor_base import FeatureExtractorBase


class FeatureExtractorMixtralRouter(FeatureExtractorBase):
    def __init__(self, orig_base_model, layer_nums="all", normalize=True, **kwargs):
        self._requested_layer_nums = layer_nums
        self._resolved_layer_nums = None
        self._normalize = normalize
        self._num_experts = getattr(
            orig_base_model.config, "num_local_experts", getattr(orig_base_model.config, "num_experts", None)
        )
        self._total_layers = getattr(orig_base_model.config, "num_hidden_layers", None)
        self._feature_dim_value = None

        if self._total_layers is not None:
            self._resolved_layer_nums = self._normalize_layer_ids(self._total_layers)
            if self._num_experts is not None:
                self._feature_dim_value = self._num_experts * len(self._resolved_layer_nums)

    def _normalize_layer_ids(self, total_layers):
        if self._requested_layer_nums == "all":
            return list(range(total_layers))

        if isinstance(self._requested_layer_nums, int):
            requested = [self._requested_layer_nums]
        else:
            requested = list(self._requested_layer_nums)

        normalized = []
        for l in requested:
            normalized.append(l if l >= 0 else total_layers + l)
        return normalized

    def _prepare_generation_logits(self, router_logits):
        if isinstance(router_logits[0], torch.Tensor):
            return router_logits

        per_layer_logits = []
        num_layers = len(router_logits[0])
        for layer_idx in range(num_layers):
            per_layer_logits.append(
                torch.cat([step[layer_idx] for step in router_logits], dim=1)
            )
        return per_layer_logits

    def _resolve_metadata(self, router_logits):
        if self._resolved_layer_nums is None:
            self._resolved_layer_nums = self._normalize_layer_ids(len(router_logits))
        if self._num_experts is None:
            self._num_experts = router_logits[self._resolved_layer_nums[0]].shape[-1]
        if self._feature_dim_value is None:
            self._feature_dim_value = self._num_experts * len(self._resolved_layer_nums)

    def __call__(self, llm_inputs, llm_outputs):
        if "router_logits" not in llm_outputs:
            raise ValueError("Model outputs must include router_logits. Ensure output_router_logits=True during the forward call.")

        router_logits = llm_outputs["router_logits"]
        is_training = not hasattr(llm_outputs, "sequences")
        router_logits = (
            router_logits if is_training else self._prepare_generation_logits(router_logits)
        )

        self._resolve_metadata(router_logits)

        layer_features = []
        for layer_idx in self._resolved_layer_nums:
            logits = router_logits[layer_idx]
            if is_training:
                logits = logits[:, :-1, :]
            if self._normalize:
                logits = torch.softmax(logits, dim=-1)
            layer_features.append(logits)

        return torch.cat(layer_features, dim=-1)

    def feature_dim(self):
        if self._feature_dim_value is None:
            if self._num_experts is not None and self._resolved_layer_nums is not None:
                self._feature_dim_value = self._num_experts * len(self._resolved_layer_nums)
            else:
                raise ValueError("Router feature dimension is undefined. Ensure the model config exposes num_hidden_layers and num_experts or run a forward pass first.")

        return self._feature_dim_value

    def output_router_logits(self):
        return True


def load_extractor(config, base_model):
    return FeatureExtractorMixtralRouter(base_model, **config)
