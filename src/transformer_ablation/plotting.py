from pathlib import Path

import matplotlib.pyplot as plt

PALETTE = ["#c9992e", "#d97ba6", "#c65a4a"]


def plot_layer_sweep(df, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6), facecolor="#fffaf6")
    ax = plt.gca()
    ax.set_facecolor("#fffaf6")

    for color, ablation_type in zip(PALETTE, df["ablation_type"].unique()):
        sub = df[df["ablation_type"] == ablation_type]
        plt.plot(sub["layer"], sub["drop_in_logit_diff"], marker="o", label=ablation_type, color=color)

    plt.axhline(0, linestyle="--", linewidth=1, color="#9c7f8a")
    plt.xlabel("Layer")
    plt.ylabel("Drop in logit difference")
    plt.title("Transformer Component Ablation Effects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
