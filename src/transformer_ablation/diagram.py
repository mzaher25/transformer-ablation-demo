STREAM_COLOR = "#d8b98e"
ATTN_FILL, ATTN_STROKE = "#fbe1ea", "#d97ba6"
MLP_FILL, MLP_STROKE = "#faf0d7", "#c9992e"
ABLATED_FILL, ABLATED_STROKE = "#fbe0da", "#c65a4a"
JOIN_FILL = "#8a6a52"
TEXT_COLOR = "#553a4a"
MUTED_TEXT = "#9c7f8a"
DIM_OPACITY = 0.4

WIDTH = 560
LAYER_HEIGHT = 52
BOX_WIDTH = 130
BOX_HEIGHT = 20
STREAM_X = 90
BOX_X = 150
TOP_MARGIN = 60
BOTTOM_MARGIN = 60


def _rect(x, y, w, h, fill, stroke, opacity=1.0, rx=6, stroke_width=1.5):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity:.2f}"/>'
    )


def _text(x, y, content, size=12, fill=TEXT_COLOR, anchor="start", weight="normal", opacity=1.0):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-family="\'Quicksand\', sans-serif" font-weight="{weight}" opacity="{opacity:.2f}">{content}</text>'
    )


def _line(x1, y1, x2, y2, stroke, width=2, dash=None, opacity=1.0):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity:.2f}"{dash_attr}/>'
    )


def _circle(cx, cy, r, fill, stroke=None, opacity=1.0, stroke_width=1.5):
    stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke else ""
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" opacity="{opacity:.2f}"{stroke_attr}/>'


def architecture_diagram_svg(n_layers: int, selected_layer: int, ablation_type: str) -> str:
    """Render an SVG of the transformer's residual stream + per-layer Attn/MLP blocks,
    highlighting whichever component the current sidebar selection zeroes out."""
    height = TOP_MARGIN + n_layers * LAYER_HEIGHT + BOTTOM_MARGIN
    parts = []

    stack_top = TOP_MARGIN
    stack_bottom = stack_top + n_layers * LAYER_HEIGHT

    # Unembed cap (output, top) and embed cap (input, bottom)
    parts.append(_text(STREAM_X, stack_top - 34, "Unembed &#8594; logits", size=12, fill=MUTED_TEXT, anchor="middle"))
    parts.append(_line(STREAM_X, stack_top - 22, STREAM_X, stack_top, STREAM_COLOR, width=2))
    parts.append(
        f'<polygon points="{STREAM_X - 5:.1f},{stack_top - 20:.1f} {STREAM_X + 5:.1f},{stack_top - 20:.1f} '
        f'{STREAM_X:.1f},{stack_top - 28:.1f}" fill="{STREAM_COLOR}"/>'
    )
    parts.append(_line(STREAM_X, stack_bottom, STREAM_X, stack_bottom + 22, STREAM_COLOR, width=2))
    parts.append(_text(STREAM_X, stack_bottom + 40, "Token + position embed", size=12, fill=MUTED_TEXT, anchor="middle"))

    for i in range(n_layers):
        # layer i drawn bottom-to-top so layer 0 sits nearest the input embedding
        row_top = stack_top + (n_layers - 1 - i) * LAYER_HEIGHT
        # resid_pre (entry, bottom of row) hits Attn first, then MLP, producing resid_post (exit, top of row)
        y_attn = row_top + 38
        y_mlp = row_top + 14
        is_selected = ablation_type != "none" and i == selected_layer
        opacity = 1.0 if (is_selected or ablation_type == "none") else DIM_OPACITY

        attn_ablated = is_selected and ablation_type == "whole_layer"
        mlp_ablated = is_selected and ablation_type in ("whole_layer", "mlp_only")
        stream_ablated = is_selected and ablation_type == "residual_stream"

        parts.append(_line(STREAM_X, row_top, STREAM_X, row_top + LAYER_HEIGHT, STREAM_COLOR, width=2, opacity=opacity))
        parts.append(_text(20, (y_attn + y_mlp) / 2 + 4, f"L{i}", size=13, fill=TEXT_COLOR, weight="bold", opacity=opacity))

        # resid_pre marker: boundary entering this layer from below
        if stream_ablated:
            by = row_top + LAYER_HEIGHT
            parts.append(_line(STREAM_X - 10, by, STREAM_X + 10, by, ABLATED_STROKE, width=4))
            parts.append(_circle(STREAM_X, by, 6, ABLATED_FILL, ABLATED_STROKE))
            parts.append(
                _text(BOX_X + BOX_WIDTH + 15, by + 4, "resid_pre = 0 (final token)",
                      size=11, fill=ABLATED_STROKE, weight="bold")
            )

        # attention join + box
        attn_fill, attn_stroke = (ABLATED_FILL, ABLATED_STROKE) if attn_ablated else (ATTN_FILL, ATTN_STROKE)
        parts.append(_circle(STREAM_X, y_attn, 4, JOIN_FILL if not attn_ablated else ABLATED_STROKE, opacity=opacity))
        parts.append(_line(STREAM_X, y_attn, BOX_X, y_attn, STREAM_COLOR, width=1.5, opacity=opacity))
        parts.append(_rect(BOX_X, y_attn - BOX_HEIGHT / 2, BOX_WIDTH, BOX_HEIGHT, attn_fill, attn_stroke, opacity=opacity))
        label = "attn_out = 0" if attn_ablated else "Attn"
        parts.append(
            _text(BOX_X + BOX_WIDTH / 2, y_attn + 4, label, size=11, anchor="middle",
                  fill=ABLATED_STROKE if attn_ablated else TEXT_COLOR, weight="bold" if attn_ablated else "normal",
                  opacity=opacity)
        )

        # mlp join + box
        mlp_fill, mlp_stroke = (ABLATED_FILL, ABLATED_STROKE) if mlp_ablated else (MLP_FILL, MLP_STROKE)
        parts.append(_circle(STREAM_X, y_mlp, 4, JOIN_FILL if not mlp_ablated else ABLATED_STROKE, opacity=opacity))
        parts.append(_line(STREAM_X, y_mlp, BOX_X, y_mlp, STREAM_COLOR, width=1.5, opacity=opacity))
        parts.append(_rect(BOX_X, y_mlp - BOX_HEIGHT / 2, BOX_WIDTH, BOX_HEIGHT, mlp_fill, mlp_stroke, opacity=opacity))
        label = "mlp_out = 0" if mlp_ablated else "MLP"
        parts.append(
            _text(BOX_X + BOX_WIDTH / 2, y_mlp + 4, label, size=11, anchor="middle",
                  fill=ABLATED_STROKE if mlp_ablated else TEXT_COLOR, weight="bold" if mlp_ablated else "normal",
                  opacity=opacity)
        )

    body = "".join(parts)
    return (
        f'<svg viewBox="0 0 {WIDTH} {height}" width="100%" height="{height}" '
        f'style="max-width: 100%;" xmlns="http://www.w3.org/2000/svg">{body}</svg>'
    )
