from trl import GRPOConfig, GRPOTrainer

from src.reward.reward_function import coding_reward


def create_trainer(
    model,
    tokenizer,
    dataset,
    config,
):
    training_config = GRPOConfig(
        output_dir=config["training"]["output_dir"],

        learning_rate=config["training"]["learning_rate"],

        num_train_epochs=config["training"]["num_train_epochs"],

        per_device_train_batch_size=config["training"][
            "per_device_train_batch_size"
        ],

        gradient_accumulation_steps=config["training"][
            "gradient_accumulation_steps"
        ],

        max_prompt_length=config["training"]["max_prompt_length"],

        max_completion_length=config["training"][
            "max_completion_length"
        ],

        num_generations=config["training"]["num_generations"],

        logging_steps=config["training"]["logging_steps"],

        save_steps=config["training"]["save_steps"],

        gradient_checkpointing=config["training"][
            "gradient_checkpointing"
        ],

        bf16=config["training"]["bf16"],
        fp16=config["training"]["fp16"],

        seed=config["training"]["seed"],

        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=coding_reward,
        train_dataset=dataset,
        args=training_config,
    )

    return trainer