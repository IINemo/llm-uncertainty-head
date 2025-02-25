from .luh_claim_estimator_dummy import LuhClaimEstimatorDummy


def load_estimator(config):
    return LuhClaimEstimatorDummy(**config)
    