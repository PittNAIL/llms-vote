#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np


plt.style.use("tableau-colorblind10")


IDENTIFICATION_DATA: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "Accuracy": {
        "LLaMA 2": (0.60, 0.61, 0.60, 0.62),
        "MedAlpaca": (0.66, 0.73, 0.63, 0.68),
        "Stable Platypus 2": (0.60, 0.61, 0.60, 0.65),
        "Vicuna": (0.63, 0.67, 0.62, 0.62),
        "Models-Vote Prompting": (0.66, 0.70, 0.62, 0.68),
    },
    "Precision": {
        "LLaMA 2": (0.64, 0.65, 0.64, 0.68),
        "MedAlpaca": (0.66, 0.74, 0.63, 0.67),
        "Stable Platypus 2": (0.62, 0.63, 0.63, 0.69),
        "Vicuna": (0.63, 0.67, 0.62, 0.63),
        "Models-Vote Prompting": (0.66, 0.72, 0.62, 0.67),
    },
    "Recall": {
        "LLaMA 2": (0.61, 0.63, 0.61, 0.64),
        "MedAlpaca": (0.65, 0.72, 0.62, 0.67),
        "Stable Platypus 2": (0.61, 0.62, 0.61, 0.66),
        "Vicuna": (0.63, 0.66, 0.62, 0.63),
        "Models-Vote Prompting": (0.65, 0.69, 0.61, 0.67),
    },
    "F Score": {
        "LLaMA 2": (0.58, 0.60, 0.58, 0.60),
        "MedAlpaca": (0.65, 0.72, 0.62, 0.67),
        "Stable Platypus 2": (0.59, 0.61, 0.59, 0.64),
        "Vicuna": (0.63, 0.66, 0.62, 0.62),
        "Models-Vote Prompting": (0.65, 0.69, 0.60, 0.67),
    },
}

CLASSIFICATION_DATA: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "Accuracy": {
        "LLaMA 2": (0.71, 0.72, 0.65, 0.51),
        "MedAlpaca": (0.81, 0.81, 0.74, 0.63),
        "Stable Platypus 2": (0.70, 0.73, 0.65, 0.50),
        "Vicuna": (0.71, 0.70, 0.66, 0.56),
        "Models-Vote Prompting": (0.80, 0.81, 0.75, 0.61),
    },
    "Precision": {
        "LLaMA 2": (0.71, 0.72, 0.71, 0.67),
        "MedAlpaca": (0.77, 0.77, 0.78, 0.75),
        "Stable Platypus 2": (0.71, 0.72, 0.71, 0.68),
        "Vicuna": (0.72, 0.72, 0.72, 0.70),
        "Models-Vote Prompting": (0.76, 0.75, 0.77, 0.73),
    },
    "Recall": {
        "LLaMA 2": (0.57, 0.58, 0.52, 0.41),
        "MedAlpaca": (0.65, 0.65, 0.59, 0.51),
        "Stable Platypus 2": (0.56, 0.58, 0.52, 0.40),
        "Vicuna": (0.57, 0.56, 0.53, 0.45),
        "Models-Vote Prompting": (0.64, 0.65, 0.60, 0.49),
    },
    "F Score": {
        "LLaMA 2": (0.61, 0.62, 0.57, 0.46),
        "MedAlpaca": (0.70, 0.70, 0.67, 0.59),
        "Stable Platypus 2": (0.61, 0.63, 0.57, 0.46),
        "Vicuna": (0.63, 0.61, 0.59, 0.51),
        "Models-Vote Prompting": (0.69, 0.69, 0.67, 0.56),
    },
}


CONTEXT_SIZES: tuple[int, int, int, int] = (32, 64, 128, 256)


def main() -> None:
    """Generates a visualization for the ablation study."""

    label_loc = np.arange(len(CONTEXT_SIZES))

    data_dicts = [IDENTIFICATION_DATA, CLASSIFICATION_DATA]
    row_titles = [
        "Ablation Study: Rare Disease Identification",
        "Ablation Study: Rare Disease Classification",
    ]

    rows, cols = 2, len(IDENTIFICATION_DATA)
    fig, axs = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(16, 8),
        gridspec_kw={"hspace": 0.39, "wspace": 0.25},
        dpi=128,
    )
    width = 0.12

    for row in range(rows):
        for col, (metric, metric_vals) in enumerate(data_dicts[row].items()):
            ax = axs[row, col]
            multiplier = 0
            for attribute, measurement in metric_vals.items():
                offset = width * multiplier
                ax.bar(label_loc + offset, measurement, width, label=attribute)
                multiplier += 1

            ax.set_xlabel("Context Size")
            ax.set_ylabel(metric)
            ax.set_xticks(label_loc + (width * round(len(metric_vals) / 2)))
            ax.set_xticklabels(CONTEXT_SIZES)
            ax.set_ylim(0, 1)

            if row == 1:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="center",
                    ncol=5,
                    title="Excluded Model",
                    bbox_to_anchor=(0.5, 1),
                    fontsize=14,
                    title_fontsize=14,
                )

    for row, row_title in enumerate(row_titles):
        vertical_position = 0.91 - row * 0.448
        fig.text(0.5, vertical_position, row_title, ha="center", fontsize=12)

    plt.savefig("ablation_study.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
