import argparse

from src.model.tokenizer import load_tokenizer
from src.observation.token_viewer import show_tokens


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )

    parser.add_argument(
        "--text",
        required=True,
    )

    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)

    show_tokens(
        tokenizer,
        args.text,
    )


if __name__ == "__main__":
    main()



#Sample invocation
#python scripts/inspect_tokenizer.py --text "def add(a, b): return a + b"