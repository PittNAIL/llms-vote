# Get model and train model / evaluate it (on 32 and 64 word context windows).
import pandas as pd
import numpy as np

df_32 = pd.read_csv("cw_ment_rep_32.csv")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import re

for index, row in df_32.iterrows():
    # Iterate over the columns
    for column in df_32.columns:
        # Convert the value to float
        value = row[column]
        try:
            float_value = float(value)
            df_32.at[index, column] = float_value
        except ValueError:
            # Handle any conversion errors here
            # For example, you can choose to assign NaN or a default value
            df_32.at[index, column] = float("nan")

# Apply the function to the column

X = df_32["mention_representation"]


y = df_32["label"]


def report_binary_class_dist(y):
    unique, counts = np.unique(y, return_counts=True)
    data_class_report_str = "whole data"
    for i, class_type in enumerate(unique):
        if class_type == 1:
            data_class_report_str = data_class_report_str + " pos: " + str(counts[i])
        else:
            data_class_report_str = data_class_report_str + " neg: " + str(counts[i])
    return data_class_report_str


# 17684 neg, 6057 pos

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

# print(X_train.shape[0])
# 21366 in training
# print(report_binary_class_dist(y_train))
# 15876 neg, 5490 pos in training
C = 1


clf = LogisticRegression(C=C, penalty="l2")

clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
training_results = (
    "training precision: %s" % str(precision_score(y_pred, y_train))
    + " recall: %s" % str(recall_score(y_pred, y_train))
    + "F1: %s" % str(f1_score(y_pred, y_train))
)
print(training_results)
