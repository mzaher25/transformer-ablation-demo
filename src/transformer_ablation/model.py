import torch
from transformer_lens import HookedTransformer


def choose_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str, device: str):
    resolved_device = choose_device(device)
    model = HookedTransformer.from_pretrained(model_name, device=resolved_device)
    model.eval()
    return model
