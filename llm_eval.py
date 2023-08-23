#!/usr/bin/env python
import pathlib

import pandas as pd
import numpy as np

from scipy.stats import ttest_rel
from sklearn.metrics import precision_recall_fscore_support as score


CONTEXT_SIZES: tuple[int, int, int, int] = (32, 64, 128, 256)

DATASET_SIZE: int = 256

ENCODE: dict[str, int] = {
    "babesiosis": 0,
    "giant cell arteritis": 1,
    "graft versus host disease": 2,
    "cryptogenic organizing pneumonia": 3,
    "other": 4,
}


def stats(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, context_size: int) -> None:
    """Computes accuracy, precision, recall, and f-score."""

    accuracy = (y_true == y_pred).mean()
    precision, recall, fscore, _ = score(y_true, y_pred, average="macro", warn_for=())

    print(f"{model_name} ({context_size=})")
    print("================================")
    print(f"{accuracy=:.2f}")
    print(f"{precision=:.2f}")
    print(f"{recall=:.2f}")
    print(f"{fscore=:.2f}")


def rare_disease_identification_stats(dfs: dict[str, pd.DataFrame], context_size: int) -> None:
    """Generates `has_disease` statistics."""

    y_preds, y_true = {}, None
    for model_name, df in dfs.items():
        df = df[df["context_size"] == context_size]
        y_true, y_pred = df["label"], df["has_disease"]
        y_preds[model_name] = y_pred
        stats(y_true, y_pred, model_name, context_size)
        print()

    mvp = np.sum(list(y_preds.values()), axis=0)
    mvp = np.where(mvp < 2, 0, 1)
    stats(y_true, mvp, "LLMs Vote", context_size)

    against = None
    if context_size == 32:
        against = "LLaMA 2"
    elif context_size == 64:
        against = "Stable Platypus 2"
    elif context_size == 128:
        against = "LLaMA 2"
    elif context_size == 256:
        against = "Vicuna"
    else:
        raise ValueError(f"Unknown context size: {context_size}")

    pvalue = ttest_rel(y_preds[against], mvp).pvalue
    print()
    print(f"Paired t-test p-value {against} vs Models-Vote Prompting: {pvalue}")


def rare_disease_classification_stats(dfs: dict[str, pd.DataFrame], context_size: int) -> None:
    """Generates disease classification statistics."""

    y_preds = {}
    for model_name, df in dfs.items():
        df = df[df["context_size"] == context_size]
        y_true, y_pred_ = df["disease"], df["found_diseases"]

        y_true = [ENCODE[y] for y in y_true]
        y_pred = np.zeros(DATASET_SIZE)
        for idx, y in enumerate(y_pred_):
            # NOTE: Had to use `eval` hack as annotation results were stored in '[x, y, z]' format
            y = eval(y)

            val = []
            if y:
                val.extend(ENCODE.get(x.lower(), 4) for x in y)
            else:
                val.append(4)

            counts = np.bincount(val)
            top_k = np.argwhere(counts == counts.max()).flatten()
            if y_true[idx] in top_k:
                y_pred[idx] = y_true[idx]
            else:
                y_pred[idx] = top_k[0]

        y_preds[model_name] = y_pred
        stats(y_true, y_pred, model_name, context_size)
        print()

    mvp = np.stack(list(y_preds.values()), axis=1).astype(int)
    mvp = np.array([np.bincount(x).argmax() for x in mvp])
    stats(y_true, mvp, "LLMs Vote", context_size)

    against = None
    if context_size == 32:
        against = "LLaMA 2"
    elif context_size == 64:
        against = "Stable Platypus 2"
    elif context_size == 128:
        against = "Stable Platypus 2"
    elif context_size == 256:
        against = "Stable Platypus 2"
    else:
        raise ValueError(f"Unknown context size: {context_size}")

    pvalue = ttest_rel(y_preds[against], mvp).pvalue
    print()
    print(f"Paired t-test p-value {against} vs Models-Vote Prompting: {pvalue}")


def main() -> None:
    """Text generation."""

    dir = pathlib.Path("annotation")
    dfs = {
        "LLaMA 2": pd.read_csv(dir / "final_llama2_results.csv"),
        "MedAlpaca": pd.read_csv(dir / "final_medalpaca_results.csv"),
        "Stable Platypus 2": pd.read_csv(dir / "final_stable-platypus2_results.csv"),
        "Vicuna": pd.read_csv(dir / "final_vicuna_results.csv"),
    }

    for df in dfs.values():
        df["has_disease"] = df["has_disease"].map(lambda x: 0 if x == "no" else 1)

    for context_size in CONTEXT_SIZES:
        rare_disease_identification_stats(dfs, context_size)
        rare_disease_classification_stats(dfs, context_size)


if __name__ == "__main__":
    main()
