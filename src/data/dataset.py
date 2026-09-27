from datasets import load_dataset


def load_training_dataset(path: str):
    dataset = load_dataset(
        "json",
        data_files=path,
        split="train",
    )

    return dataset


def validate_dataset(dataset):
    required_columns = {"prompt", "solution"}

    missing = required_columns - set(dataset.column_names)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    return dataset