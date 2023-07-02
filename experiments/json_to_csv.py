import argparse
import json
import logging
import pathlib


import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""

    parser = argparse.ArgumentParser("json inverted index to dataframes")

    parser.add_argument(
        "--mimic-dir", type=str, help="path to mimic-iv clinical notes", required=True
    )
    parser.add_argument("--json-dir", type=str, help="path to json inverted indices", required=True)
    parser.add_argument("--out", type=str, help="path to write csv file", required=True)

    return parser.parse_args()


def load_json_files(dir: str) -> tuple[dict, dict]:
    """Loads and merges JSON files from a directory."""

    data_pos = {}
    data_neg = {}
    for path in pathlib.Path(dir).glob("*.json"):
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
            if "positive" in str(path):
                data_pos.update(data)
            else:
                data_neg.update(data)

    return data_pos, data_neg


def main() -> None:
    args = parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"{pathlib.Path(__file__).stem}.log", level=logging.INFO)

    data_pos, data_neg = load_json_files(args.json_dir)

    logger.info("Reading MIMIC file..")
    df = pd.read_csv(args.mimic_dir, usecols=["note_id", "text"])

    data_dicts = [data_pos, data_neg]
    labels = [1, 0]
    frames = []

    for data_dict, label in zip(data_dicts, labels):
        json_df = pd.DataFrame(
            [(k, v) for k, values in data_dict.items() for v in values],
            columns=["disease", "note_id"],
        )
        json_df["note_id"] = json_df["note_id"].str.upper()
        merged = df.merge(json_df, on="note_id")
        merged["label"] = label

        frames.append(merged)

    df_combined = pd.concat(frames)

    logger.info("Writing CSV file..")
    df_combined.to_csv(args.out, index=False)
    logger.info("Complete!")


if __name__ == "__main__":
    main()
