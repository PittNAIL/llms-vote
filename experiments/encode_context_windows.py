import argparse
import re
import sys

import numpy as np
import torch
import pandas as pd
import tqdm

from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

np.set_printoptions(threshold=sys.maxsize)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""

    parser = argparse.ArgumentParser("Encode Context Windows and Get Model Performance")

    parser.add_argument("--data", type=str, help="path to input data", required=True)

    return parser.parse_args()


args = parse_args()

df = pd.read_csv(args.data)


# Get 100 of each class just for fast testing of a sample 200 entries.
# df_label_1 = df[df['label'] == 1].head(100)
# df_label_0 = df[df['label'] == 0].head(100)

# df = pd.concat([df_label_1, df_label_0], ignore_index=True)


pbar = tqdm(total=len(df), desc="Encoding {dataset}")


# Load the BERT model and tokenizer
model_name = "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Rest of the code (iterate over tuples and compute contextual mention representations)


def find_sub_list(sl, l):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            return ind, ind + sll - 1


# Initialize an empty list to store the new rows for the new dataframe
new_data = []

# Iterate over the rows of the DataFrame df
for index, row in df.iterrows():
    mention_cue = row["disease"]  # Extract the mention cue from the "disease" column
    context_window = row["window"]  # Extract the context window from the "window" column

    # Tokenize the mention cue and context window
    mention_cue_tokenized = tokenizer.tokenize(mention_cue)
    tokenized_context_window = tokenizer.tokenize(context_window)

    # Get the index of the start and end of the wordpiece of the mention in the tokenized context window
    start_end_inds_tuple = find_sub_list(mention_cue_tokenized, tokenized_context_window)

    if start_end_inds_tuple is None:
        # Mention tokens are not found in the context window
        # Handle this case as per your requirements (e.g., skip or assign a default representation)
        continue

    # Convert the tokenized context window to input tensors
    inputs = tokenizer(context_window, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate the wordpiece representations using the BERT model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        vec = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

    # Get the wordpiece representations of the mention within the context window
    mention_wordpiece_rep_in_context = vec[0][start_end_inds_tuple[0] : start_end_inds_tuple[1] + 1]

    # Compute the contextual mention representation within the entire context window
    mention_rep_in_context = np.mean(mention_wordpiece_rep_in_context.numpy(), axis=0)

    # Append the contextual mention representation and label to the new_data list
    new_data.append([mention_rep_in_context, row["label"]])
    pbar.update(1)


# Convert the new_data list into a pandas DataFrame
new_df = pd.DataFrame(new_data, columns=["mention_representation", "label"])

mention_rep_array = np.vstack(new_df["mention_representation"])


# Convert the new_data list into a pandas DataFrame
new_df = pd.DataFrame(new_data, columns=["mention_representation", "label"])

mention_rep_array = np.vstack(new_df["mention_representation"].to_numpy())

# Print the new DataFrame
pbar.close()


def report_binary_class_dist(y):
    unique, counts = np.unique(y, return_counts=True)
    data_class_report_str = "whole data"
    for i, class_type in enumerate(unique):
        if class_type == 1:
            data_class_report_str = data_class_report_str + " pos: " + str(counts[i])
        else:
            data_class_report_str = data_class_report_str + " neg: " + str(counts[i])
    return data_class_report_str


X = mention_rep_array
y = new_df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

# set hyperparameter, this is the value set in Dr. Dong's codebase.
C = 1


clf = LogisticRegression(C=C, penalty="l2", max_iter=500)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
training_results = (
    "training precision: %s" % str(precision_score(y_pred, y_train))
    + " recall: %s" % str(recall_score(y_pred, y_train))
    + "F1: %s" % str(f1_score(y_pred, y_train))
)
print(training_results)


y_valid_pred = clf.predict(X_test)
validation_results = (
    "testing precision: %s" % str(precision_score(y_valid_pred, y_test))
    + " recall: %s" % str(recall_score(y_valid_pred, y_test))
    + "F1: %s" % str(f1_score(y_valid_pred, y_test))
)
print(validation_results)


y_pred_whole = clf.predict(X)
whole_results = (
    "whole thing precision: %s" % str(precision_score(y_pred_whole, y))
    + " recall: %s" % str(recall_score(y_pred_whole, y))
    + "F1: %s" % str(f1_score(y_pred_whole, y))
)
print(whole_results)
