import pathlib
import json

import pandas as pd

from results_util import clean_segment, remove_prompt


def extract_json(response_text):
    dis_idx = response_text.rfind('"disease_identified":')
    has_idx = response_text.rfind('"has_disease"', dis_idx)
    dis_curl = response_text.find("}", dis_idx)
    has_curl = response_text.find("}", has_idx)
    segment = "{\n" + response_text[dis_idx : dis_curl + 1] + ","
    if '"has_disease": "no"' in response_text[dis_curl : dis_curl + 30]:
        segment += "\n" + '"has_disease": "no"'
    elif ('"has_disease": "yes"' in response_text[dis_curl : dis_curl + 30]) | (
        '"has_disease" :"Yes"' in response_text[dis_curl : dis_curl + 30]
    ):
        segment += "\n" + '"has_disease": "yes"'
    segment += "\n}"
    if '"disease_identified":' in segment and '"has_disease"' in segment:
        pass
    else:
        dis_idx = response_text.find('"disease_identified":')
        has_idx = response_text.find('"has_disease"', dis_idx)
        dis_curl = response_text.find("}", dis_idx)
        has_curl = response_text.find("}", has_idx)
        segment = (
            "{\n"
            + response_text[dis_idx : dis_curl + 1]
            + ",\n"
            + response_text[has_idx:has_curl]
            + "\n}"
        )  # W: Line too long (105/100) # E: line too long (105 > 79 characters)
        hd_idx = segment.find('"has_disease"') + len('"has_disease"')
        partial = segment[:hd_idx]
        if "yes" in segment[hd_idx:]:
            partial += ': "yes" \n}'
        else:
            partial += ': "no" \n}'
        segment = partial
        segment = clean_segment(segment)

    return segment


def get_info(json_str):
    try:
        json_obj = json.loads(json_str)
        found_diseases = [
            disease for disease, value in json_obj["disease_identified"].items() if value == 1
        ]
        present = json_obj["has_disease"]
        return pd.Series({"found_diseases": found_diseases, "present": present})
    except Exception as e:
        return pd.Series({"found_diseases": [], "present": None})


dir = pathlib.Path.cwd()

model_prefixes = ["llama2", "vicuna", "medalpaca", "stable-platypus2"]


def main() -> None:
    for model in model_prefixes:
        matching_files = [file.name for file in dir.glob(f"{model}*") if file.is_file()]
        results = pd.DataFrame()
        for file in matching_files:
            file_name = pathlib.PurePosixPath(file).stem
            df = pd.read_csv(file)
            df["response"] = df.apply(remove_prompt, axis=1)
            df["json"] = df["response"].apply(extract_json)
            df["model"], df["context_size"] = file_name.split("_")
            filt_df = df[df["json"] != "{\n\n}"]
            new_cols = filt_df["json"].apply(get_info)
            df[["found_diseases", "has_disease"]] = new_cols
            df["relevant_information_identification"] = df["has_disease"].apply(
                lambda x: 1 if pd.notna(x) else None
            )
            df["relevant_information_classification"] = df["has_disease"].apply(
                lambda x: 1 if pd.notna(x) else None
            )
            dropped_cols = ["prompt", "window"]
            copied_df = df.drop(columns=dropped_cols)
            copied_df["json_incomplete"] = 0
            results = pd.concat([results, copied_df])
        no_info = results.groupby("context_size")["has_disease"].apply(lambda x: x.isnull().sum())
        print(f"{model} Count of No Output:")
        print(no_info)
        results.to_csv(f"results_{model}.csv")


if __name__ == "__main__":
    main()
