import json
import pandas as pd


f = open("inverted_index_filtered.json", encoding="utf-8")
g = open("inverted_index_positive.json", encoding="utf-8")
h = open("inverted_index_negative.json", encoding="utf-8")

df = pd.read_csv(
    r"/home/jordan/Downloads/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv"
)


data = json.load(f)
data_pos = json.load(g)
data_neg = json.load(h)


note_dataframe = pd.DataFrame()


json_df = pd.DataFrame(
    [(k, v) for k, values in data.items() for v in values], columns=["disease", "note_id"]
)
json_df["note_id"] = json_df["note_id"].str.upper()


new_df = df.merge(json_df, on="note_id")

# new_df.to_csv('clinical_notes_with_mention_labels.csv')

json_df_pos = pd.DataFrame(
    [(k, v) for k, values in data_pos.items() for v in values], columns=["disease", "note_id"]
)
json_df_pos["note_id"] = json_df_pos["note_id"].str.upper()


new_df_pos = df.merge(json_df_pos, on="note_id")
new_df_pos["label"] = 1
new_df_pos.to_csv("clinical_notes_with_pos_labels.csv")

json_df_neg = pd.DataFrame(
    [(k, v) for k, values in data_neg.items() for v in values], columns=["disease", "note_id"]
)
json_df_neg["note_id"] = json_df_neg["note_id"].str.upper()


new_df_neg = df.merge(json_df_neg, on="note_id")
new_df_neg["label"] = 0
new_df_neg.to_csv("clinical_notes_with_neg_labels.csv")


# json_df.to_csv('clinical_notes_with_mention_labels.csv')


# filtered_df = json_df.groupby("note_id").agg({"disease": list}).reset_index()
# filtered_df["note_id"] = filtered_df["note_id"].str.upper()
# print(df.columns)
