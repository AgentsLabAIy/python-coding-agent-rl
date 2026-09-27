def show_tokens(tokenizer, text: str):
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    print("\n" + "=" * 60)
    print("TOKENIZATION")
    print("=" * 60)

    print("\nTEXT:")
    print(text)

    print("\nTOKENS:")

    for index, token_id in enumerate(token_ids):
        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        print(
            f"Token #{index + 1:<3} "
            f"ID={token_id:<8} "
            f"TEXT={repr(token_text)}"
        )

    print("\nTOKEN IDS:")
    print(token_ids)

    print("=" * 60)