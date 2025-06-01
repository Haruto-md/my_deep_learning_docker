import torch.nn as nn

def get_model(cfg):
    raise ValueError(f"Unknown model name: {cfg['model_name']}")