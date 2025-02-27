[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/IINemo/llm-uncertainty-head/blob/master/LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Pretrained-yellow)](https://huggingface.co/llm-uncertainty-head)


# LLM Uncertainty Head

[Installation](#installation) | [Basic usage](#basic_usage) 

Pre-trained UQ heads -- supervised auxiliary modules for LLMs that substantially enhance their ability to capture uncertainty. A powerful Transformer architecture in their design and informative features derived from LLM attention maps enable strong performance, as well as cross-lingual and cross-domain generalization.


## Installation

```
pip install git+https://github.com/IINemo/lm-polygraph.git@dev
pip install git+https://github.com/IINemo/llm-uncertainty-head.git
```

## Basic usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from luh import AutoUncertaintyHead, CausalLMWithUncertainty


model_name = "mistralai/Mistral-7B-Instruct-v0.2"
uhead_name = "llm-uncertainty-head/uhead_Mistral-7B-Instruct-v0.2"

llm = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(
    model_name)
tokenizer.pad_token = tokenizer.eos_token
uhead = AutoUncertaintyHead.from_pretrained(
    uhead_name, base_model=llm)
llm_adapter = CausalLMWithUncertainty(llm, uhead, tokenizer=tokenizer)

# prepare text ...
messages = [
    [
        {
            "role": "user", 
            "content": "How many fingers are on a coala's foot?"
        }
    ]
]

chat_messages = [tokenizer.apply_chat_template(m, tokenize=False, add_bos_token=False) for m in messages]
inputs = tokenizer(chat_messages, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to("cuda")

output = llm_adapter.generate(inputs)
output["uncertainty_logits"]
```