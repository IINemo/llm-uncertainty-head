# LLM Uncertainty Head

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/IINemo/llm-uncertainty-head/blob/master/LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Pretrained-yellow)](https://huggingface.co/llm-uncertainty-head)


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
from luh import AutoUncertaintyHead

from lm_polygraph import CausalLMWithUncertainty


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

## Training
Training UHead from data from top package directory:
```
CUDA_VISIBLE_DEVICES=0 python -m luh.cli.train.run_train_uhead \
    --config-dir=./configs \
    --config-name=run_train_uhead.yaml \
    dataset.path="<path to your dataset, e.g. hf:llm-uncertainty-head/train_akimbio_mistral>" \
    model.pretrained_model_name_or_path="<your model name, e.g.  mistralai/Mistral-7B-Instruct-v0.2>"
```

## Cite
```
@inproceedings{shelmanov2025head,
  title        = {A Head to Predict and a Head to Question: Pre-trained Uncertainty Quantification Heads for Hallucination Detection in LLM Outputs},
  author       = {Shelmanov, Artem and Fadeeva, Ekaterina and Tsvigun, Akim and Tsvigun, Ivan and Xie, Zhuohan and Kiselev, Igor and Daheim, Nico and Zhang, Caiqi and Vazhentsev, Artem and Sachan, Mrinmaya and others},
  booktitle    = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year         = {2025},
  address      = {Abu Dhabi, United Arab Emirates},
  publisher    = {Association for Computational Linguistics},
  pages        = {to appear},
  url          = {https://arxiv.org/abs/2505.08200}
}
```
