from .auto_uncertainty_head import AutoUncertaintyHead
from .calculator_infer_luh import CalculatorInferLuh


def load_stat_calculator(config, builder):
    uncertainty_head = AutoUncertaintyHead.from_pretrained(
        config.uq_head_path, 
        builder.model.model)
    builder.uncertainty_head = uncertainty_head
    return CalculatorInferLuh(
        uncertainty_head=uncertainty_head,
        tokenize=True,
        generations_cache_dir=config.get('generations_cache_dir', None),
        args_generate=config.args_generate,
        predict_token_uncertainties=config.predict_token_uncertainties
    )
