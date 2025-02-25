from .calculator_apply_uq_head import CalculatorApplyUQHead


def load_stat_calculator(config, builder):
    calc = CalculatorApplyUQHead(builder.uncertainty_head)
    return calc

