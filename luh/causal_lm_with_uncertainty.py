from .calculator_infer_luh import CalculatorInferLuh

from lm_polygraph.model_adapters import WhiteboxModelBasic


class CausalLMWithUncertainty:
    def __init__(self, llm, uhead, tokenizer, args_generate=None):
        self.llm = llm
        self.uhead = uhead

        self.calc_infer_llm = CalculatorInferLuh(self.uhead, 
                                    tokenize=False, 
                                    args_generate=args_generate,
                                    device="cuda",
                                    generations_cache_dir="")
        
        self.model_adapter = WhiteboxModelBasic(model=self.llm, 
                                   tokenizer=tokenizer, 
                                   tokenizer_args={"add_special_tokens": False, 
                                                   "return_tensors": "pt", 
                                                   "padding": True, "truncation": True},
                                    model_type="CausalLM")
        self.model_adapter.model_path = "debug"
    
    def generate(self, inputs, *args, **kwargs):
        deps = dict()
        deps.update(self.calc_infer_llm(deps, texts=inputs, model=self.model_adapter))
        return deps
