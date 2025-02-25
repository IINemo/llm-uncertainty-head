from omegaconf import OmegaConf
from pathlib import Path
from huggingface_hub import HfApi, Repository, hf_hub_download
import tempfile
from abc import abstractmethod

import torch
import torch.nn as nn

from ..utils import load_feature_extractor

import logging

log = logging.getLogger()


class UncertaintyHeadBase(nn.Module):
    def __init__(
        self,
        feature_extractor,
        cfg=None,  # Temporary we save initializing cfg in the head itself
        model_type="token"
    ):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.cfg = cfg
        self.model_type = model_type

    @abstractmethod
    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        pass

    def _get_attn_mask(self, llm_inputs, llm_outputs):
        is_training = not hasattr(llm_outputs, "sequences")
        if is_training:
            return llm_inputs["attention_mask"][:, :-1]  # no new tokens introduced
        else:
            return llm_outputs["full_attention_mask"][:, 1:]  # all feature_calculators ignore first token

    def forward(self, llm_inputs, llm_outputs):
        features = self.feature_extractor(llm_inputs, llm_outputs)
        features_attn_mask = self._get_attn_mask(llm_inputs, llm_outputs)
        return self._compute_tensors(llm_inputs, features, features_attn_mask)

    @property
    def output_attentions(self):
        return self.feature_extractor.output_attention()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        base_model,
        revision: str = "main",
        use_auth_token: str = None,
    ):
        if Path(pretrained_path).exists():  # Local path
            config_path = Path(pretrained_path) / "config.yaml"
            weights_path = Path(pretrained_path) / "weights.pth"
        else:  # Remote Hugging Face repository
            config_path = hf_hub_download(
                repo_id=pretrained_path,
                filename="config.yaml",
                revision=revision,
                use_auth_token=use_auth_token,
            )
            weights_path = hf_hub_download(
                repo_id=pretrained_path,
                filename="weights.pth",
                revision=revision,
                use_auth_token=use_auth_token,
            )

        cfg = OmegaConf.load(config_path)
        weights = torch.load(weights_path, weights_only=True)
        feature_extractor = load_feature_extractor(cfg.feature_extractor, base_model)
        ue_head_cfg = cfg.uncertainty_head if cfg.uncertainty_head is not None else dict()
        uq_head = cls(feature_extractor, cfg=cfg, **ue_head_cfg)
        incompatible_keys = uq_head.load_state_dict(weights)
        assert (
            incompatible_keys.missing_keys == []
            and incompatible_keys.unexpected_keys == []
        ), f"LuqSequenceEstimator cannot be loaded. Missing keys: {incompatible_keys.missing_keys}; Unexpected keys: {incompatible_keys}."

        return uq_head

    @property
    def output_attentions(self):
        return self.feature_extractor.output_attention()

    def save(self, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.state_dict(),
            Path(output_dir) / "weights.pth",
        )
        OmegaConf.save(self.cfg, Path(output_dir) / "config.yaml")

    def push_to_hub(self, repo_id: str, token: str = None):
        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True, token=token)

        # Use a temporary directory for local repository operations
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Clone the Hugging Face repository into the temporary directory
            repo = Repository(
                local_dir=str(tmp_path), clone_from=repo_id, use_auth_token=token
            )

            # Save model and config to the repository directory
            # output_dir = tmp_path / "model"
            # output_dir.mkdir(parents=True, exist_ok=True)
            # self.save(output_dir)
            output_dir = tmp_path
            self.save(output_dir)

            # Add and commit files to the repository
            repo.git_add(auto_lfs_track=True)
            repo.git_commit("Upload UncertaintyHead model and config")
            repo.git_push()

        log.info(f"Model pushed to Hugging Face Hub: https://huggingface.co/{repo_id}")
