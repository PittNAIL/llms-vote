import argparse
import json
import os
import re
import tqdm

import pandas as pd

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
    context_size = int(int(window_size) / 2)
    list_idx = list_positions[0]

    left_bound = 0
    right_bound = 0

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
    args = parse_args()

    df = pd.read_csv(args.data)

    window_sizes = args.sizes

    for window_size in window_sizes:
        df["window"] = ""
        for index, row in tqdm.tqdm(df.iterrows(), desc=f"Getting Window of Size {window_size}"):
            window = context_window(row["text"], row["disease"], window_size)
            df.at[index, "window"] = window

        output_directory = os.path.dirname(args.data)

        filtered_df = df[["note_id", "window", "disease", "label"]].copy()

        filtered_df.to_csv(f"window_size_{window_size}.csv")


if __name__ == "__main__":
    main()
