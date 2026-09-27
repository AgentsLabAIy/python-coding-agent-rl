from transformers import TrainerCallback


class TrainingLoggerCallback(TrainerCallback):

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        print("\n" + "-" * 60)
        print(f"TRAINING STEP: {state.global_step}")
        print("-" * 60)

        if state.log_history:
            latest = state.log_history[-1]

            for key, value in latest.items():
                print(f"{key}: {value}")

        return control