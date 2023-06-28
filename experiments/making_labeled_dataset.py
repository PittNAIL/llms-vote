import pandas as pd
import tqdm
from tqdm import tqdm

# df = pd.read_csv('/home/jordan/Downloads/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv')

# pbar = tqdm(total=len(df), desc="Processing")

with open("/home/jordan/fsl-for-rare-disease/data/rare_disease.txt") as file:
    diseases = file.read().splitlines()

df1 = pd.read_csv("pos_label_with_window.csv")
df2 = pd.read_csv("neg_label_with_window.csv")

frames = [df1, df2]

df3 = pd.concat(frames)
df3.to_csv("context_windows_with_labels_64.csv")
