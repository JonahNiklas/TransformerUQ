from __future__ import annotations
from typing import List


def print_generated_text_with_colors(
    generated_text: str | List[str], prompt: str | None
) -> None:
    if isinstance(generated_text, str):
        generated_text = [generated_text]

    print(prompt)
    colors = [
        "\033[91m",  # red
        "\033[92m",  # green
        "\033[93m",  # yellow
        "\033[94m",  # blue
        "\033[95m",  # magenta
        "\033[96m",  # cyan
    ]
    for i, text in enumerate(generated_text):
        print(colors[i % len(colors)] + text + "\033[0m")
