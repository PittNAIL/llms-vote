import argparse
import re

import pandas as pd

from pathlib import Path

import tqdm

from util import find_sub_list


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""

    parser = argparse.ArgumentParser("Get Context Windows of 16, 32, and 64 words")

    parser.add_argument("--data", type=str, help="path to input data", required=True)
    parser.add_argument(
        "--sizes", type=int, help="Window sizes to obtain", nargs="*", required=True
    )

    return parser.parse_args()


def context_window(text, disease, window_size):
    """This function will return the context window for a disease mention in free text,
    determined by variable "window_size"."""

    disease_tokens = disease.split()
    entry_tokens = text.split()
    entry_tokens_alphanum = [re.sub("[^A-Za-z0-9]+", "", token.lower()) for token in entry_tokens]

    disease_tokens = [re.sub("[^A-Za-z0-9]+", "", token.lower()) for token in disease_tokens]
    list_positions = [find_sub_list(disease_tokens, entry_tokens_alphanum)[0]]

    # Set bounds for the window.
    context_size = window_size // 2
    list_idx = list_positions[0]
    if window_size == len(disease_tokens):
        context = disease
    elif window_size == len(disease_tokens) + 1:
        context = entry_tokens[list_idx - 1] + " " + disease
    else:
        left_bound = list_idx - context_size
        right_bound = list_idx + context_size

        if left_bound < 0:
            left_bound = 0
            right_bound = window_size

        if right_bound > len(entry_tokens):
            right_bound = len(entry_tokens)
            left_bound = len(entry_tokens) - window_size

        context = entry_tokens[left_bound:right_bound]
        context = " ".join(context)

    return context


def main() -> None:
    """Obtain context windows of rare disease mentions in clinical notes."""
    args = parse_args()

    df = pd.read_csv(args.data)

    window_sizes = args.sizes

    for window_size in window_sizes:
        window_list = []
        for _, row in tqdm.tqdm(df.iterrows(), desc=f"Getting Window of Size {window_size}"):
            window = context_window(row["text"], row["disease"], window_size)
            window_list.append(window)

        output_dir = Path(args.data).parent
        output_file = output_dir / f"window_size_{window_size}.csv"

        filtered_df = df[["note_id", "disease", "label"]].copy()
        filtered_df["window"] = window_list

        filtered_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
