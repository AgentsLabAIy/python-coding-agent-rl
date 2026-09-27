def show_generation(
    tokenizer,
    input_ids,
    generated_ids,
):
    input_length = input_ids.shape[-1]

    completion_ids = generated_ids[input_length:]

    print("\n" + "=" * 60)
    print("GENERATION")
    print("=" * 60)

    for index, token_id in enumerate(completion_ids):
        token_id = int(token_id)

        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        print(
            f"Token #{index + 1:<3} "
            f"ID={token_id:<8} "
            f"TEXT={repr(token_text)}"
        )

    generated_text = tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
    )

    print("\nGENERATED TEXT:")
    print(generated_text)

    print("=" * 60)