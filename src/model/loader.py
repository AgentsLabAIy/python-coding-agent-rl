import torch
from transformers import AutoModelForCausalLM


def get_torch_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16

    if dtype_name == "float16":
        return torch.float16

    if dtype_name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model(model_name: str, dtype_name: str = "bfloat16"):
    dtype = get_torch_dtype(dtype_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )

    return model