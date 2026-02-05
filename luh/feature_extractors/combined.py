import torch
import time
import warnings

from .feature_extractor_base import FeatureExtractorBase

import importlib
import logging

log = logging.getLogger(__name__)


class FeatureExtractorCombined(FeatureExtractorBase):
    def __init__(self, *feature_extractors):
        self._feature_extractors = feature_extractors

    def __call__(self, llm_inputs, llm_outputs):
        features = []
        for feature_extractor in self._feature_extractors:
            start_time = time.time()

            new_features = feature_extractor(llm_inputs, llm_outputs)
            nan_count = torch.isnan(new_features).sum()
            if nan_count != 0:
                nan_percentage = 100 * nan_count / new_features.numel()
                warnings.warn(f'Found nans in features from {feature_extractor}: {nan_count} ({nan_percentage:.2f}%), filling with zeros')
                new_features = torch.nan_to_num(new_features, nan=0.0)
            features.append(new_features)
            log.debug(f"Done extracting {feature_extractor} in {round(time.time() - start_time, 2)} seconds")

        return torch.cat(features, dim=-1)

    def feature_dim(self):
        return sum(fe.feature_dim() for fe in self._feature_extractors)
    
    def output_attention(self):
        return any(fe.output_attention() for fe in self._feature_extractors)


def load_extractor(config, base_model):
    feature_extractors = []
    for fe_cfg in config:
        fe_name = fe_cfg.name
        log.info(f"Loading feature extractor: {fe_name}")
        module = importlib.import_module(fe_name)
        feature_extractors.append(module.load_extractor(fe_cfg, base_model))

    return FeatureExtractorCombined(*feature_extractors)
