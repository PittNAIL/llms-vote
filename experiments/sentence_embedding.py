import argparse
import re
import sys

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load the BERT model and tokenizer
model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenize the sentence and mention
df = pd.read_csv("window_size_64.csv")
# or 16, 32

pbar = tqdm(total=len(df), desc="Encoding dataset")

new_data = []
for index, row in df.iterrows():
    # This will encode the entire "sentence", being the full context window of 16,32,64 words, and return its
    # vector representation.
    sentence = row["window"]

    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0)
    # sentence embedding is averaged in line with what Dr. Dong does.

    new_data.append([sentence_embedding, row["label"]])

    pbar.update(1)


pbar.close()
# Convert the new_data list into a pandas DataFrame
new_df = pd.DataFrame(new_data, columns=["mention_representation", "label"])


mention_rep_array = np.vstack(new_df["mention_representation"].to_numpy())

# print(mention_rep_array)


X = mention_rep_array
y = new_df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

# set hyperparameter, this is the value set in Dr. Dong's codebase.
C = 1

# max_iter raised b/c 300 (their value) was too low, 500 was too low after I changed it when I used this script.
clf = LogisticRegression(C=C, penalty="l2", max_iter=1000)

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
