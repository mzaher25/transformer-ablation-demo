import argparse

from .config import load_config
from .experiment import run_ablation_sweep
from .model import load_model
from .plotting import plot_layer_sweep
from .prompts import build_examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    print(f"Loading model: {cfg.model_name}")
    model = load_model(cfg.model_name, cfg.device)

    examples = build_examples(model, cfg.prompt_path, cfg.max_examples)
    print(f"Using {len(examples)} prompts")

    df = run_ablation_sweep(model, examples, cfg.ablation_types)

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_csv, index=False)
    plot_layer_sweep(df, cfg.output_plot)

    print(f"Saved CSV: {cfg.output_csv}")
    print(f"Saved plot: {cfg.output_plot}")


if __name__ == "__main__":
    main()
