import argparse
import yaml

from src.model.loader import load_model
from src.model.tokenizer import load_tokenizer
from src.data.dataset import (
    load_training_dataset,
    validate_dataset,
)
from src.training.trainer import create_trainer


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    model_name = config["model"]["name"]

    print("=" * 70)
    print("PYTHON CODING ASSISTANT - GRPO TRAINING")
    print("=" * 70)

    print(f"\nModel: {model_name}")

    # ---------------------------------------------------------
    # TOKENIZER
    # ---------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = load_tokenizer(
        model_name
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    print("Loading model...")

    model = load_model(
        model_name,
        config["model"]["torch_dtype"],
    )

    # ---------------------------------------------------------
    # DATASET
    # ---------------------------------------------------------

    dataset_path = config["data"]["train_file"]

    print(f"\nLoading dataset: {dataset_path}")

    dataset = load_training_dataset(
        dataset_path
    )

    dataset = validate_dataset(
        dataset
    )

    print(f"Training examples: {len(dataset)}")

    # ---------------------------------------------------------
    # TRAINER
    # ---------------------------------------------------------

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
    )

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    print("\nStarting GRPO training...\n")

    trainer.train()

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    print("\nSaving model...")

    trainer.save_model(
        config["training"]["output_dir"]
    )

    tokenizer.save_pretrained(
        config["training"]["output_dir"]
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()