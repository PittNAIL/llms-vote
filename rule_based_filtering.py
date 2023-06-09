#!/usr/bin/env python
# Author: David Oniani
# License: MIT
import json


def main() -> None:
    """Performs rule-based filtering."""

    with open("inverted_index.json", encoding="utf-8") as file:
        data = {dct["term"]: dct["note_ids"] for dct in json.load(file) if dct["term"] != "disease"}

    total = sum(map(len, data.values()))
    filtered_data = {}
    for term, note_ids in data.items():
        if len(term) < 4 or 0.005 <= len(note_ids) / total:
            continue
        filtered_data[term] = note_ids
    assert sum(map(len, filtered_data.values())) == 6_061

    with open("inverted_index_filtered.json", "w", encoding="utf-8") as file:
        json.dump(filtered_data, file, indent=4)


if __name__ == "__main__":
    main()
