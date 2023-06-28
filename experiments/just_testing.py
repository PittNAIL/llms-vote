import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import tqdm
from tqdm import tqdm
import sys

np.set_printoptions(threshold=sys.maxsize)


df = pd.read_csv("context_windows_with_labels.csv")

# pbar = tqdm(total=len(df), desc="Getting Window")


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

mention_cue = df["disease"][0]
context_window = df["window"][0]
label = df["label"][0]
print(mention_cue, context_window)


mention_cue_tokenized = tokenizer.tokenize(mention_cue)
tokenized_context_window = tokenizer.tokenize(context_window)

# Get the index of the start and end of the wordpiece of the mention in the tokenized context window
start_end_inds_tuple = find_sub_list(mention_cue_tokenized, tokenized_context_window)


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
mention_rep_in_context = torch.mean(mention_wordpiece_rep_in_context, dim=0)

# Append the contextual mention representation and label to the new_data list
new_data.append([mention_rep_in_context.numpy(), label])


# Convert the new_data list into a pandas DataFrame
new_df = pd.DataFrame(new_data, columns=["mention_representation", "label"])

# Print the new DataFrame
print(new_df)
print(type(new_df["mention_representation"]))
print(new_df["mention_representation"][0].mean())

new_df.to_csv("bigtester.csv")


# Iterate over the rows of the DataFrame df
