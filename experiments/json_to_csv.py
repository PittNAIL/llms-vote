import argparse
import json
import os

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""

    parser = argparse.ArgumentParser("json inverted index to dataframes")

    parser.add_argument("--json-dir", type=str, help="path to json inverted indices", required=True)
    parser.add_argument("--out", type=str, help="path to write csv file", required=True)

    return parser.parse_args()


def load_json_files(json_dir: str) -> dict:
    """Loads and merges JSON files from a directory."""
    data_pos = {}
    data_neg = {}
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            with open(os.path.join(json_dir, filename), encoding="utf-8") as f:
                json_data = json.load(f)
                if "positive" in filename:
                    data_pos.update(json_data)
                elif "negative" in filename:
                    data_neg.update(json_data)
    return data_pos, data_neg


def main() -> None:
    args = parse_args()

    data_pos, data_neg = load_json_files(args.json_dir)

    df = pd.read_csv(
        r"/home/jordan/Downloads/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv",
        usecols=["note_id", "text"],
    )

    json_df_pos = pd.DataFrame(
        [(k, v) for k, values in data_pos.items() for v in values], columns=["disease", "note_id"]
    )
    json_df_pos["note_id"] = json_df_pos["note_id"].str.upper()

    new_df_pos = df.merge(json_df_pos, on="note_id")
    new_df_pos["label"] = 1

    json_df_neg = pd.DataFrame(
        [(k, v) for k, values in data_neg.items() for v in values], columns=["disease", "note_id"]
    )
    json_df_neg["note_id"] = json_df_neg["note_id"].str.upper()

    new_df_neg = df.merge(json_df_neg, on="note_id")
    new_df_neg["label"] = 0

    frames = [new_df_pos, new_df_neg]
    df_combined = pd.concat(frames)

    df_combined.to_csv(args.out, index=False)
    print(f"CSV file saved at: {args.out}")


if __name__ == "__main__":
    main()