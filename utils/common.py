import torch


def softmax_with_temperature(logits, T=3.0):
    return torch.softmax(logits / T, dim=1)
