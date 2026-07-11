import torch
import random

from dataclasses import dataclass

@dataclass
class InductionExample:
    prompt: str
    answer: str
    repeat_position: int

# Generate a set of prompts for testing induction behavior in the model
def generate_induction_prompts(model, num_examples=100, seq_len=5):

    examples = []

    vocab_size = model.cfg.d_vocab

    for _ in range(num_examples):

        tokens = torch.randint(
            0,
            vocab_size,
            (seq_len,)
        )

        repeat_position = random.randint(
            0,
            seq_len-2
        )

        repeated_token = tokens[repeat_position]

        next_token = tokens[repeat_position+1]

        prompt_tokens = torch.cat(
            [
                tokens,
                repeated_token.unsqueeze(0)
            ]
        )

        examples.append(
            InductionExample(
                prompt=model.tokenizer.decode(prompt_tokens),
                answer=model.tokenizer.decode(
                    next_token.unsqueeze(0)
                )
            )
        )

    return examples