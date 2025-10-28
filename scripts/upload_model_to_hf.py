from transformers import AutoModelForCausalLM
from luh import AutoUncertaintyHead
from pathlib import Path
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--head-path", type=str)
    parser.add_argument("--model-type", type=str)
    parser.add_argument("--token", type=str)
    parser.add_argument("--private", action="store_true", help="Make the model private on Hugging Face Hub")

    config = parser.parse_args()

    with open(Path(os.path.split(config.head_path)[0]) / ".hydra" / "config.yaml") as f:
        print(f.read())

    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    head_tag = config.model_name.split("/")[-1]
    head = AutoUncertaintyHead.from_pretrained(config.head_path, base_model)
    hf_name = f"llm-uncertainty-head/{config.model_type}_{head_tag}"
    head.push_to_hub(hf_name, token=config.token, private=config.private)


if __name__ == "__main__":
    main()
