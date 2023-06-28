import json
import pandas as pd
from tqdm import tqdm
import re

with open("inverted_index_filtered.json") as file:
    data = json.load(file)
assert len(data) == 447

df = pd.read_csv(
    "clinical_notes_with_pos_labels.csv", usecols=["note_id", "text", "disease", "label"]
)

pbar = tqdm(total=len(df), desc="Getting Window")


def context_window(text, disease, window_size):
    disease_tokens = disease.split()
    entry_tokens = text.split()
    entry_tokens_alphanum = [re.sub("[^A-Za-z0-9]+", "", token.lower()) for token in entry_tokens]

    disease_tokens = [re.sub("[^A-Za-z0-9]+", "", token.lower()) for token in disease_tokens]
    list_positions = [find_sub_list(disease_tokens, entry_tokens_alphanum)[0]]

    context_size = int(window_size / 2)
    list_idx = list_positions[0]
    offset = len(disease_tokens)

    left_bound = 0
    right_bound = 0

    left_bound = list_idx - context_size
    right_bound = list_idx + context_size

    if left_bound < 0:
        left_bound = 0
        right_bound = 64

    if right_bound > len(entry_tokens):
        right_bound = len(entry_tokens)
        left_bound = len(entry_tokens) - 64

    context = entry_tokens[left_bound:right_bound]
    context = " ".join(context)

    return context


def find_sub_list(sl, l):
    #'''find the first occurrence of a sublist in list: #from https://stackoverflow.com/a/17870684'''
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            return ind, ind + sll - 1


df["window"] = ""
for index, row in df.iterrows():
    text = row["text"]
    disease = row["disease"]
    window = context_window(text, disease, 64)
    df.at[index, "window"] = window
    pbar.update(1)

pbar.close()


filtered_df = df[["note_id", "window", "disease", "label"]].copy()

filtered_df.to_csv("pos_label_with_window.csv")
