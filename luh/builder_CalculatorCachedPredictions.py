from .calculator_cached_predictions import CalculatorCachedPredictions


def load_stat_calculator(config, builder):
    calc = CalculatorCachedPredictions(config.cached_predictions_path, config.dataset_path)
    return calc

