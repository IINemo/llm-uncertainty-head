import argparse
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from luh.feature_extractors import FeatureExtractorBasicAttention


def main(args):
    ds = load_dataset(args.dataset)['train']

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=2400,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    fe = FeatureExtractorBasicAttention(
        model,
        layer_nums='all',
        head_nums='all',
        attn_history_sz=1,
        pool=False,
    )

    corrs = defaultdict(lambda: defaultdict(list))
    unc_labs = []

    for d in tqdm(ds.select(range(args.n_samples)), total=args.n_samples):
        llm_input = {
            'input_ids': torch.LongTensor([d['input_ids']]).to(args.device),
            'attention_mask': torch.ones(1, len(d['input_ids'])).to(args.device).int(),
        }
        with torch.no_grad():
            llm_output = model(**llm_input, output_attentions=True)
        attns = fe(llm_input, llm_output)[0]
        for lab, attn in zip(d['uncertainty_labels'], attns):
            if lab == -100:
                continue
            attn = attn.reshape(fe._attn_history_sz, -1, len(fe._layer_nums))[0]  # heads, layers
            for h in range(attn.shape[0]):
                for l in range(attn.shape[1]):
                    corrs[l][h].append(attn[h, l].item())
            unc_labs.append(lab)

    corrs = {(l, h): np.corrcoef(corrs[l][h], unc_labs)[0, 1]
             for l in sorted(corrs.keys())
             for h in sorted(corrs[l].keys())}

    for (l, h), corr in sorted(corrs.items()):
        print(f'Head {h} Layer {l}: correlation {corr}')

    torch.save(corrs, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze attention-head correlation with uncertainty labels.")
    parser.add_argument('--dataset', type=str, default='llm-uncertainty-head/train_akimbio_mistral',
                        help='Dataset path to load from HuggingFace')
    parser.add_argument('--model-path', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='Path to the model to load')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of samples to process')
    parser.add_argument('--output-path', type=str, default='head_correlations.torch',
                        help='Path to save correlation results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on (e.g., cuda:0 or cpu)')

    args = parser.parse_args()
    main(args)
