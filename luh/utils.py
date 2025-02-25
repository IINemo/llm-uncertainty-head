import luh.feature_extractors.combined

import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets, Sequence, Value
from collections.abc import Mapping


def load_feature_extractor(config, base_model):
    return luh.feature_extractors.combined.load_extractor(config, base_model)


def load_any_dataset(dataset_path, args):
    # Cases supported:
    #  dataset_path = '/path/to/local/dataset'
    #  dataset_path = '/path/to/local/dataset:split'
    #  dataset_path = 'hf:path/to/hf/dataset'
    #  dataset_path = 'hf:path/to/hf/dataset:split'
    #  dataset_path = [dataset_path1, dataset_path2, ...]

    if dataset_path.startswith('[') and dataset_path.endswith(']'):
        dataset_path = dataset_path[1:-1].split(',')
    if not isinstance(dataset_path, str):
        dataset_parts = [load_any_dataset(p, args) for p in dataset_path]
        dataset = concatenate_datasets(dataset_parts)
        return dataset.shuffle(seed=228)

    from_hf = False
    if dataset_path.startswith('hf:'):
        from_hf = True
        dataset_path = dataset_path[len('hf:'):]
    split_parts = dataset_path.split(':')
    split = None
    if len(split_parts) > 1:
        dataset_path, split = split_parts
    if from_hf:
        dataset = load_dataset(dataset_path, cache_dir=getattr(args, 'hf_cache', None))
    else:
        dataset = load_from_disk(dataset_path)
    if split is not None:
        dataset = dataset[split]
    # case uncertainty_labels to int64
    try:
        if 'uncertainty_labels' in dataset.features:
            dataset = dataset.cast_column(
                'uncertainty_labels',
                Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None))
    except Exception as e:
        pass
    return dataset
    

def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, Mapping):
        # Recreate the mapping preserving its type
        return obj.__class__({k: recursive_to(v, device) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [recursive_to(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(item, device) for item in obj)
    return obj
    return obj
