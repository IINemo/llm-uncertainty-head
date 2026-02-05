# LLM Uncertainty Head

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/IINemo/llm-uncertainty-head/blob/master/LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Pretrained-yellow)](https://huggingface.co/llm-uncertainty-head)
<a href="https://aclanthology.org/2025.emnlp-main.1809/"><img src="https://img.shields.io/badge/EMNLP-2025-red?logo=bookstack&logoColor=white" alt="EMNLP 2025"/></a>


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
from luh.calculator_infer_luh import CalculatorInferLuh
from luh.luh_estimator_dummy import LuhEstimatorDummy


model_name = "mistralai/Mistral-7B-Instruct-v0.2"
uhead_name = "llm-uncertainty-head/uhead_Mistral-7B-Instruct-v0.2"

llm = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(
    model_name)
tokenizer.pad_token = tokenizer.eos_token
uhead = AutoUncertaintyHead.from_pretrained(
    uhead_name, base_model=llm)

generation_config = GenerationConfig.from_pretrained(model_name)
args_generate = {"generation_config": generation_config,
                 "max_new_tokens": 50}
calc_infer_llm = CalculatorInferLuh(uhead, 
                                    tokenize=True, 
                                    args_generate=args_generate,
                                    device="cuda",
                                    generations_cache_dir="",
                                    predict_token_uncertainties=True)

estimator = LuhEstimatorDummy()
llm_adapter = CausalLMWithUncertainty(llm, tokenizer=tokenizer, stat_calculators=[calc_infer_llm], estimator=estimator)

# prepare text ...
messages = [
    [
        {
            "role": "user", 
            "content": "In which year did the programming language Mercury first appear? Answer with a year only."
        }
    ]
]
# The correct answer is 1995
chat_messages = [tokenizer.apply_chat_template(m, tokenize=False, add_bos_token=False) for m in messages]
inputs = tokenizer(chat_messages, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to("cuda")

output = llm_adapter.generate(inputs["input_ids"])
output["uncertainty_score"]
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

### Vision-Language Model (VLM) Training

The library also supports training uncertainty heads for Vision-Language Models (VLMs) such as Qwen2.5-VL. When training with VLMs:

1. Set `model.is_vlm: true` in your config
2. Specify the `dataset.image_column` containing images (default: "images")
3. Use a dataset with pre-tokenized inputs and image data

**Example: Training with google/gemma-3-12b-it on nhatkhangdtp/uncertainty-vlm-gemma**

```bash
CUDA_VISIBLE_DEVICES=0,1 HYDRA_CONFIG=./configs/run_train_vl_uhead.yaml python -m luh.cli.train.run_train_uhead 
```

The VLM training automatically:
- Loads the appropriate processor for handling images
- Uses VLM-specific data collators that pass `pixel_values` to the model
- Preserves image data through the training pipeline
- Maintains compatibility with text-only uncertainty head types

**Key differences from text-only training:**
- The data collator handles multimodal inputs (text + images)
- The processor (not just tokenizer) is used for image preprocessing
- Batch sizes may need to be reduced due to image memory requirements

**VLM Configuration Options:**
- `model.is_vlm`: Set to `true` to enable VLM mode
- `dataset.image_column`: Column name containing images in the dataset (default: "image")
- `training_arguments.per_device_train_batch_size`: Typically smaller for VLMs (e.g., 4 instead of 32)

## Cite
```
@inproceedings{shelmanov2025head,
  title        = {A Head to Predict and a Head to Question: Pre-trained Uncertainty Quantification Heads for Hallucination Detection in LLM Outputs},
  author       = {Shelmanov, Artem and Fadeeva, Ekaterina and Tsvigun, Akim and Tsvigun, Ivan and Xie, Zhuohan and Kiselev, Igor and Daheim, Nico and Zhang, Caiqi and Vazhentsev, Artem and Sachan, Mrinmaya and others},
  booktitle    = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year         = {2025},
  address      = {Abu Dhabi, United Arab Emirates},
  publisher    = {Association for Computational Linguistics},
  pages        = {35700--35719},
  url          = {https://aclanthology.org/2025.emnlp-main.1809/}
}
```
