import ast


def syntax_is_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def coding_reward(
    completions,
    solution,
    **kwargs,
):
    rewards = []

    for completion, expected in zip(completions, solution):

        if isinstance(completion, list):
            generated_text = completion[0]["content"]
        else:
            generated_text = completion

        reward = 0.0

        # Exact solution match
        if generated_text.strip() == expected.strip():
            reward += 1.0

        # Valid Python syntax
        if syntax_is_valid(generated_text):
            reward += 1.0

        # Basic code signal
        if (
            "def " in generated_text
            or "print(" in generated_text
            or "=" in generated_text
            or "for " in generated_text
            or "if " in generated_text
        ):
            reward += 0.25

        rewards.append(reward)

    return rewards