import json

import pandas as pd

pattern_with_backslash = r'"disease\_identified": {'
pattern_without_backslash = '"disease_identified": {'

pattern_w_backslash = r'"has\_disease":'
pattern_wo_backslash = '"has_disease":'


def clean_segment(text):
    replacements = {"*": ",", "?": "null", ",\n}": "\n}", " to ": ":", "[": ""}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def remove_prompt(row):
    prompt = row["prompt"]
    response = row["response"]

    response = response.replace(pattern_with_backslash, pattern_without_backslash)
    response = response.replace(pattern_w_backslash, pattern_wo_backslash)
    response = response.replace(prompt, "")
    return response
