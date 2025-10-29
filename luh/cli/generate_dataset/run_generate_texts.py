import hydra
from pathlib import Path
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from datasets import DatasetDict, Dataset
from luh.utils import load_any_dataset
import logging
import torch.multiprocessing as mp



log = logging.getLogger()

# Use only with instruct tuned models


@hydra.main(
    version_base=None,
    config_path=str(Path(os.environ.get("HYDRA_CONFIG", "")).parent),
    config_name=str(Path(os.environ.get("HYDRA_CONFIG", "")).name),
)
def main(config):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Check if vllm is enabled
    if config.enable_vllm:
        log.info("Loading model with vllm...")
        llm = LLM(
            model=config.model.pretrained_model_name_or_path,
            tensor_parallel_size=1,
            max_model_len=2056,
            trust_remote_code=True,
            enforce_eager=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.pretrained_model_name_or_path,
            trust_remote_code=True
        )
        log.info("Done.")
    else:
        log.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(**config.model)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.pretrained_model_name_or_path
        )
        generation_config = GenerationConfig.from_pretrained(
            config.model.pretrained_model_name_or_path
        )
        log.info("Done.")

    log.info("Loading dataset...")
    dataset_dict = load_any_dataset(config.dataset.path, config)

    result_dicts = dict()
    for split in dataset_dict.keys():
        log.info(f"Processing split {split}...")

        dataset = dataset_dict[split]
        if config.dataset.num_instances:
            if len(dataset) < config.dataset.num_instances:
                log.warning(
                    f"Dataset has fewer instances than requested: {len(dataset)} < {config.dataset.num_instances}"
                )

            dataset = dataset.select(range(min(config.dataset.num_instances, len(dataset))))

        log.info(f"Dataset size: {len(dataset)}")
        log.info("Done.")

        if config.enable_vllm:
            # vllm logic
            prompts = []
            datas = []
            for inst in dataset:
                datas.append(inst)  # Store the original instance
                prompt = inst["question"]
                message = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                prompts.append(message)

            sampling_params = SamplingParams(
                temperature=config.generation.temperature if config.generation.do_sample else 0.,
                max_tokens=config.generation.max_new_tokens,
                repetition_penalty=config.generation.repetition_penalty,
                #frequency_penalty=config.generation.diversity_penalty,
                #length_penalty=config.generation.length_penalty,
            )
            
            log.info("Generating texts in batches...")
            inputs = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
            outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)

            new_data_list = []
            for inpt, output, data in zip(inputs, outputs, datas):
                new_data = data.copy()  # Copy original instance
                
                # Access the generated text from the output
                generated_text = output.outputs[0].text  # Check if output exists

                # Use input_ids from the output directly
                input_ids = output.outputs[0].token_ids
                #print(input_ids, type(input_ids))
                new_data["input_ids"] = tokenizer(inpt, add_special_tokens=False)['input_ids'] + list(input_ids)  # Store the generated input IDs
                
                new_data["reply"] = generated_text  # Add the generated text as reply
                new_data_list.append(new_data)  # Keep the same format as the original

            # Create a Dataset from the list of dictionaries
            dataset = Dataset.from_list(new_data_list)

        else:
            # Original logic
            def generate_replies(inst):
                prompt = inst["question"]
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

                inputs = inputs.to(model.device)
                outputs = model.generate(
                    inputs,
                    num_return_sequences=1,
                    generation_config=generation_config,
                    pad_token_id=tokenizer.eos_token_id,
                    **config.generation,
                )
                inst["input_ids"] = outputs[0]
                reply = tokenizer.decode(
                    inst["input_ids"][inputs.shape[1]:], skip_special_tokens=True
                )
                inst["reply"] = reply
                return inst

            log.info("Generating texts...")
            dataset = dataset.map(generate_replies)
            log.info("Done.")

        result_dicts[split] = dataset
        log.info("Done with split.")

    output_path = Path(output_dir) / "result"
    log.info(f"Saving results to {output_path} ...")
    dd = DatasetDict(result_dicts)
    dd.save_to_disk(output_path)
    log.info("Done.")


if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)  # Set multiprocessing start method
    main()