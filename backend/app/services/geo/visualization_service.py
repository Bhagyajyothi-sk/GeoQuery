
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_index_map(
    data,
    output_path,
    title,
    colorbar_label,
    colormap="RdYlGn"
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        data,
        cmap=colormap,
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    fig.colorbar(
        image,
        ax=ax,
        label=colorbar_label,
        fraction=0.046,
        pad=0.04
    )

    ax.set_title(title)
    ax.axis("off")

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.1
    )

    plt.close(fig)

    return str(output_path)