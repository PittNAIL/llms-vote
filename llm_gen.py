#!/usr/bin/env python
import pathlib

import torch

from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import pipeline, set_seed


set_seed(1337)


MODELS: dict[str, str] = {
    "Stable-Platypus2": "garage-bAInd/Stable-Platypus2-13B",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "medalpaca": "medalpaca/medalpaca-7b",
    "vicuna": "lmsys/vicuna-7b-v1.3",
}


DATASETS: dict[str, str] = {
    "32": "data/annotated_32.csv",
    "64": "data/annotated_64.csv",
    "128": "data/annotated_128.csv",
    "256": "data/annotated_256.csv",
}

MAX_NEW_TOKENS: int = 1_024
NUM_RETURN_SEQUENCES: int = 1
TOP_K: int = 10


def main() -> None:
    """Text generation."""

    prompt_dir = pathlib.Path("prompt")

    for model_id, model_path in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        generator = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )

        with open(prompt_dir / f"{model_id}.txt") as file:
            PROMPT = file.read()

        for dataset_id, dataset_path in DATASETS.items():
            dataset = load_dataset(
                "csv",
                data_files={dataset_id: dataset_path},
                split=f"{dataset_id}",
            )
            dataset = dataset.map(lambda s: {"prompt": PROMPT.replace("$INPUT$", s["window"])})
            dataset = dataset.map(
                lambda sample: {
                    "response": generator(
                        sample["prompt"],
                        max_new_tokens=MAX_NEW_TOKENS,
                        num_return_sequences=NUM_RETURN_SEQUENCES,
                        top_k=TOP_K,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                    )[0]["generated_text"]
                }
            )

            dataset.to_csv(f"{model_id}_{dataset_id}.csv")


if __name__ == "__main__":
    main()
