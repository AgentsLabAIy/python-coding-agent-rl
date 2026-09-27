import argparse

import torch

from src.model.loader import load_model
from src.model.tokenizer import load_tokenizer
from src.observation.generation_viewer import show_generation


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )

    parser.add_argument(
        "--prompt",
        required=True,
    )

    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model)

    messages = [
        {
            "role": "user",
            "content": args.prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )

    show_generation(
        tokenizer,
        inputs["input_ids"][0],
        outputs[0],
    )


if __name__ == "__main__":
    main()