# Transformer Ablation Study

A small mechanistic interpretability project for testing how GPT-2 Small changes when you ablate transformer layers, MLP blocks, and residual stream activations.

The metric is the mean logit difference:

```text
logit(correct next token) - logit(incorrect next token)
```

A larger `drop_in_logit_diff` means the ablated component mattered more for the prompts.

## Structure

```text
transformer_ablation_module/
├── configs/
│   └── default.yaml
├── data/
│   ├── prompts.json
│   └── induction.json
├── scripts/
│   ├── run_ablation.py
│   └── app.py
├── src/
│   └── transformer_ablation/
│       ├── cli.py
│       ├── config.py
│       ├── diagram.py
│       ├── experiment.py
│       ├── hooks.py
│       ├── induction.py
│       ├── metrics.py
│       ├── model.py
│       ├── plotting.py
│       └── prompts.py
├── results/
├── figures/
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install as an editable package:

```bash
pip install -e .
```

## Run

From the project root:

```bash
PYTHONPATH=src python scripts/run_ablation.py --config configs/default.yaml
```

If installed with `pip install -e .`, you can also run:

```bash
transformer-ablation --config configs/default.yaml
```

## Interactive demo

From the project root:

```bash
PYTHONPATH=src streamlit run scripts/app.py
```

The sidebar's **Demo** selector switches between two pages:

### Layer Ablation

Pick a prompt, an ablation type, and a layer (plus a head, for `single_head`) to see the
model's top next-token predictions, generated continuation, and an architecture diagram
highlighting exactly what got zeroed, side-by-side with the unablated baseline. For the
built-in prompts you also get the correct-vs-incorrect logit difference shift. A "Run full
sweep" button reproduces the same layer-by-layer sweep as `scripts/run_ablation.py`, with an
interactive chart.

### Induction Head Ablation

Searches for GPT-2 Small's induction heads: pick a prompt source (randomly generated
repeated-token sequences or natural-language examples from `data/induction.json`, with an
option to add your own), then run a sweep across every layer/head combination. For each head
this computes:

- **Drop** — how much ablating that head hurts induction performance
- **Attention score** — how strongly the head attends from a repeated token back to its
  earlier occurrence
- **Induction score** — `drop * attention_score`, combining both signals to surface heads
  that are both causally important and attention-pattern-consistent with induction

Results are ranked and charted per layer/head.

## Outputs

```text
results/ablation_results.csv
figures/ablation_plot.png
```

## Ablations

### `whole_layer`

Zeros both attention output and MLP output for a transformer block:

```text
attn_out(layer) = 0
mlp_out(layer) = 0
```

### `mlp_only`

Zeros only the MLP output at a layer:

```text
mlp_out(layer) = 0
```

### `residual_stream`

Zeros the residual stream at the final token position before a layer:

```text
resid_pre(layer, final_position) = 0
```

### `single_head`

Zeros one attention head's output (before it's combined into `attn_out`) at a layer —
available in the interactive demo only:

```text
z(layer, head) = 0
```

## Edit prompts

Add or change layer/MLP/residual-stream ablation examples in:

```text
data/prompts.json
```

Each correct and incorrect answer should be one token under the model tokenizer. The script skips examples where this is not true.

Add or change natural-language induction examples (used by the Induction Head Ablation page) in:

```text
data/induction.json
```

Each entry needs a `prompt`, the expected `answer` continuation, and `repeat_position` (the
index of the token being repeated).

