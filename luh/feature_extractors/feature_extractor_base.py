from abc import ABC, abstractmethod


class FeatureExtractorBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, llm_inputs, llm_outputs):
        pass

    @abstractmethod
    def feature_dim(self):
        pass

    def output_attention(self):
        return False
