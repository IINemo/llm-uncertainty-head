from .heads.saplma_head import SaplmaHead
from .heads.uncertainty_head import UncertaintyHead
from .heads.uncertainty_head_claim import UncertaintyHeadClaim
from .heads.linear_head import LinearHead
from .heads.linear_head_claim import LinearHeadClaim
from .heads.mlp_head_claim import MLPClaimHead
from .heads.uncertainty_head_claim_light import UncertaintyHeadClaimLight
from .utils import load_feature_extractor

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
import os
import inspect


class AutoUncertaintyHead:
    DEFAULT_MODEL_TYPE = "luh"

    MODEL_MAPPING = {
        "saplma": SaplmaHead,
        DEFAULT_MODEL_TYPE: UncertaintyHead,
        "claim": UncertaintyHeadClaim,
        "linear": LinearHead,
        "linear_claim": LinearHeadClaim,
        "mlp_claim": MLPClaimHead,
        "claim_light": UncertaintyHeadClaimLight,
    }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        base_model,
        revision: str = "main",
        use_auth_token: str = None,
    ):
        if os.path.isdir(pretrained_path):
            cfg = os.path.join(pretrained_path, "config.yaml")
        else:
            cfg = hf_hub_download(  # TODO: implement via hf models
                repo_id=pretrained_path,
                filename="config.yaml",
                revision=revision,
                use_auth_token=use_auth_token,
            )
            
        cfg = OmegaConf.load(cfg)
        model_class = (
            cls.MODEL_MAPPING[cfg.head_type.lower()]
            if hasattr(cfg, "head_type")
            else cls.MODEL_MAPPING[cls.DEFAULT_MODEL_TYPE]
        )
        return model_class.from_pretrained(
            pretrained_path, base_model, revision, use_auth_token
        )
    
    @classmethod
    def from_config(cls, config, base_model):
        uq_head_type = cls.MODEL_MAPPING[config.head_type]

        # Get sanitize parameter from config (default to True for backwards compatibility)
        sanitize = getattr(config, 'sanitize', True)

        feature_extractor = load_feature_extractor(
            config.feature_extractor, base_model, sanitize=sanitize
        )
        ue_head_cfg = dict() if config.uncertainty_head is None else config.uncertainty_head

        # Check if the head accepts 'sanitize' parameter
        init_signature = inspect.signature(uq_head_type.__init__)
        kwargs = {'cfg': config}
        if 'sanitize' in init_signature.parameters:
            kwargs['sanitize'] = sanitize

        uq_head = uq_head_type(
            feature_extractor,
            **ue_head_cfg,
            **kwargs,
        )

        return uq_head
