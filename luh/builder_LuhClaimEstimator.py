from .luh_claim_estimator import LuhClaimEstimator


def load_estimator(config):
    return LuhClaimEstimator(**config)
    