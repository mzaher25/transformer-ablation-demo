import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptExample:
    id: str
    prompt: str
    correct: str
    incorrect: str
    correct_id: int
    incorrect_id: int


def load_raw_prompts(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def token_id_or_none(model, text: str) -> int | None:
    toks = model.to_tokens(text, prepend_bos=False).squeeze(0)
    if toks.numel() != 1:
        return None
    return int(toks.item())


def build_examples(model, prompt_path: str | Path, max_examples: int | None = None) -> list[PromptExample]:
    raw_examples = load_raw_prompts(prompt_path)
    examples = []

    for raw in raw_examples:
        correct_id = token_id_or_none(model, raw["correct"])
        incorrect_id = token_id_or_none(model, raw["incorrect"])

        if correct_id is None or incorrect_id is None:
            print(f"Skipping {raw['id']}: correct/incorrect answer is not one token")
            continue

        examples.append(
            PromptExample(
                id=raw["id"],
                prompt=raw["prompt"],
                correct=raw["correct"],
                incorrect=raw["incorrect"],
                correct_id=correct_id,
                incorrect_id=incorrect_id,
            )
        )

        if max_examples is not None and len(examples) >= max_examples:
            break

    if not examples:
        raise ValueError("No valid prompts found. Use single-token correct and incorrect answers.")

    return examples
