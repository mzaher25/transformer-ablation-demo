import torch
import random
import json

from pathlib import Path
from transformer_ablation.config import InductionExample
from transformer_ablation.prompts import token_id_or_none

# Generate a set of prompts for testing induction behavior in the model
def generate_induction_prompts(model, num_examples=100, seq_len=5):

    # Need at least A B
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")

    examples = []

    vocab_size = model.cfg.d_vocab

    # Random token IDs can decode to text that doesn't re-tokenize back to a
    # single token; retry with a generous budget rather than crashing later
    # when induction_score() tries to read a scalar answer id.
    max_attempts = num_examples * 20
    attempts = 0

    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1

        # Create the initial sequence
        tokens = torch.randint(
            0,
            vocab_size,
            (seq_len,)
        )

        # Pick a token to repeat.
        # Avoid the last position because it needs a next token.
        repeat_position = random.randint(
            0,
            seq_len - 2
        )

        repeated_token = tokens[repeat_position]

        answer_token = tokens[repeat_position + 1]

        answer_text = model.tokenizer.decode(answer_token.unsqueeze(0))

        if token_id_or_none(model, answer_text) is None:
            continue

        prompt_tokens = torch.cat(
            [
                tokens,
                repeated_token.unsqueeze(0)
            ]
        )

        examples.append(
            InductionExample(
                prompt=model.tokenizer.decode(
                    prompt_tokens
                ),
                answer=answer_text,
                repeat_position=repeat_position
            )
        )

    if len(examples) < num_examples:
        print(f"Warning: only generated {len(examples)}/{num_examples} valid induction examples")

    return examples

def generate_natural_prompts():
    examples = [
        InductionExample(
            prompt="The cat sat on the mat. The cat",
            answer=" sat",
            repeat_position=1
        ),
        InductionExample(
            prompt="The dog chased the ball. The dog",
            answer=" chased",
            repeat_position=1
        ),
        InductionExample(
            prompt="Alice went to school. Alice",
            answer=" went",
            repeat_position=0
        ),
        InductionExample(
            prompt="Paris is beautiful in spring. Paris",
            answer=" is",
            repeat_position=0
        ),
    ]
    return examples

def create_custom_induction_prompt(prompt, answer, repeat_position):

    return [
        InductionExample(
            prompt=prompt,
            answer=answer,
            repeat_position=repeat_position
        )
    ]

def load_induction_prompts(path):
    with open(path, "r") as f:
        data = json.load(f)

    return [
        InductionExample(
            prompt=item["prompt"],
            answer=item["answer"],
            repeat_position=item["repeat_position"]
        )
        for item in data
    ]


def filter_valid_examples(model, examples):
    """Drop examples whose answer isn't a single token under the model's
    tokenizer, mirroring prompts.py's build_examples guard. induction_score()
    needs exactly one token to read a scalar answer id."""
    valid, skipped = [], []

    for ex in examples:
        if token_id_or_none(model, ex.answer) is None:
            skipped.append(ex)
        else:
            valid.append(ex)

    return valid, skipped