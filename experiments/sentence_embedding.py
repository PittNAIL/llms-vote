import torch
import tqdm

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from transformers import AutoModel, AutoTokenizer


MODEL_NAME: str = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"


def main() -> None:
    """Generates sentence embbeddings."""

    # Load the BERT model and tokenizer
    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize the sentence and mention
    df = pd.read_csv("window_size_64.csv")
    # or 16, 32

    new_data = []
    with torch.no_grad():
        for index, row in tqdm.tqdm(df.iterrows(), desc="Encoding dataset"):
            # This will encode the entire "sentence", being the full context window of 16,32,64
            # words, and return its vector representation.
            inputs = tokenizer(row["window"], return_tensors="pt")

            outputs = model(**inputs)
            # NOTE: sentence embedding is averaged in line with what Dr. Dong does.
            sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0)
            new_data.append([sentence_embedding, row["label"]])

    # Convert the new_data list into a pandas DataFrame
    new_df = pd.DataFrame(new_data, columns=["mention_representation", "label"])

    mention_rep_array = np.vstack(new_df["mention_representation"].to_numpy())

    # print(mention_rep_array)

    X = mention_rep_array
    y = new_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

    # set hyperparameter, this is the value set in Dr. Dong's codebase.
    C = 1

    # max_iter raised b/c 300 (their value) was too low, 500 was too low after I changed it when I
    # used this script.
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


if __name__ == "__main__":
    main()
