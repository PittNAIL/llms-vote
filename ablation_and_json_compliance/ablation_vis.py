#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

identification_data = {
    "Accuracy": {
        "LLaMA2": (0.60, 0.61, 0.60, 0.62),
        "MedAlpaca": (0.66, 0.73, 0.63, 0.68),
        "Stable Platypus 2": (0.60, 0.61, 0.60, 0.65),
        "Vicuna": (0.63, 0.67, 0.62, 0.62),
        "Original MVP": (0.66, 0.70, 0.62, 0.68),
    },
    "Precision": {
        "LLaMA2": (0.64, 0.65, 0.64, 0.68),
        "MedAlpaca": (0.66, 0.74, 0.63, 0.67),
        "Stable Platypus 2": (0.62, 0.63, 0.63, 0.69),
        "Vicuna": (0.63, 0.67, 0.62, 0.63),
        "Original MVP": (0.66, 0.72, 0.62, 0.67),
    },
    "Recall": {
        "LLaMA2": (0.61, 0.63, 0.61, 0.64),
        "MedAlpaca": (0.65, 0.72, 0.62, 0.67),
        "Stable Platypus 2": (0.61, 0.62, 0.61, 0.66),
        "Vicuna": (0.63, 0.66, 0.62, 0.63),
        "Original MVP": (0.65, 0.69, 0.61, 0.67),
    },
    "F Score": {
        "LLaMA2": (0.58, 0.60, 0.58, 0.60),
        "MedAlpaca": (0.65, 0.72, 0.62, 0.67),
        "Stable Platypus 2": (0.59, 0.61, 0.59, 0.64),
        "Vicuna": (0.63, 0.66, 0.62, 0.62),
        "Original MVP": (0.65, 0.69, 0.60, 0.67),
    },
}

classification_data = {
    "Accuracy": {
        "LLaMA2": (0.71, 0.72, 0.65, 0.51),
        "MedAlpaca": (0.81, 0.81, 0.74, 0.63),
        "Stable Platypus 2": (0.70, 0.73, 0.65, 0.50),
        "Vicuna": (0.71, 0.70, 0.66, 0.56),
        "Original MVP": (0.80, 0.81, 0.75, 0.61),
    },
    "Precision": {
        "LLaMA2": (0.71, 0.72, 0.71, 0.67),
        "MedAlpaca": (0.77, 0.77, 0.78, 0.75),
        "Stable Platypus 2": (0.71, 0.72, 0.71, 0.68),
        "Vicuna": (0.72, 0.72, 0.72, 0.70),
        "Original MVP": (0.76, 0.75, 0.77, 0.73),
    },
    "Recall": {
        "LLaMA2": (0.57, 0.58, 0.52, 0.41),
        "MedAlpaca": (0.65, 0.65, 0.59, 0.51),
        "Stable Platypus 2": (0.56, 0.58, 0.52, 0.40),
        "Vicuna": (0.57, 0.56, 0.53, 0.45),
        "Original MVP": (0.64, 0.65, 0.60, 0.49),
    },
    "F Score": {
        "LLaMA2": (0.61, 0.62, 0.57, 0.46),
        "MedAlpaca": (0.70, 0.70, 0.67, 0.59),
        "Stable Platypus 2": (0.61, 0.63, 0.57, 0.46),
        "Vicuna": (0.63, 0.61, 0.59, 0.51),
        "Original MVP": (0.69, 0.69, 0.67, 0.56),
    },
}

width = 0.15
# the width of the bars
rows = 2
cols = len(identification_data)
context_sizes = ("32", "64", "128", "256")
x = np.arange(len(context_sizes))  # the label locations


data_dicts = [identification_data, classification_data]
row_titles = [
    "Ablation Study: Rare Disease Identification Metrics",
    "Ablation Study: Rare Disease Classification Metrics",
]

fig, axs = plt.subplots(
    nrows=rows, ncols=cols, figsize=(14, 2), gridspec_kw={"hspace": 0.4, "wspace": 0.25}
)

for row in range(rows):
    for col, (metric, metric_values) in enumerate(data_dicts[row].items()):
        ax = axs[row, col]
        multiplier = 0
        for attribute, measurement in metric_values.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            multiplier += 1

        ax.set_xlabel("Context Size")
        ax.set_ylabel(metric)
        ax.set_xticks(x + (width * round(len(metric_values) / 2)))
        ax.set_xticklabels(context_sizes)
        ax.set_ylim(0, 1)

        if row == 1:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=5,
                title="Excluded Model",
                bbox_to_anchor=(0.5, 1),
                fontsize=16,
                title_fontsize=16,
            )

for row, row_title in enumerate(row_titles):
    vertical_position = 0.9 - row * 0.44
    fig.text(0.5, vertical_position, row_title, ha="center", fontsize=14)


plt.show()
# plt.savefig("fig.png")
