import pandas as pd

from .hooks import make_hooks
from .metrics import mean_logit_diff


def run_ablation_sweep(model, examples, ablation_types: list[str]) -> pd.DataFrame:
    baseline = mean_logit_diff(model, examples)
    rows = []

    for layer in range(model.cfg.n_layers):
        for ablation_type in ablation_types:
            hooks = make_hooks(ablation_type, layer)
            ablated = mean_logit_diff(model, examples, hooks=hooks)
            drop = baseline - ablated

            rows.append({
                "layer": layer,
                "ablation_type": ablation_type,
                "baseline_logit_diff": baseline,
                "ablated_logit_diff": ablated,
                "drop_in_logit_diff": drop,
            })

            print(
                f"layer={layer:02d} type={ablation_type:16s} "
                f"ablated={ablated: .4f} drop={drop: .4f}"
            )

    return pd.DataFrame(rows)
