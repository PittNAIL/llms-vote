# llms-vote

Code for the Paper "Large Language Models Vote: Prompting for Rare Disease Identification"

The preprint is available on [arXiv](https://arxiv.org/abs/2308.12890)

## Environment

```console
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing the Dataset

### Generating a List of Rare Diseases

Run the following to generate a list of rare diseases using [ORDO ontology][ordo] version 4.2:

```console
mkdir data
./gen_rare_disease.py --ordo data/ORDO_en_4.2.owl --out data/rare_disease.txt
```

### Generating Inverted Index

Get [MIMIC-IV][mimic-iv] first. Then [install Rust][rust] and run the following:

```console
cd inverted_index
cargo run --release -- ../data/rare_disease.txt ../data/discharge.csv ../data/inverted_index.json
```

## Implementation

[llms-vote][llms-vote] has been implemented by [David Oniani][david] and Jordan Hilsman.

[david]: https://oniani.ai
[llms-vote]: https://github.com/oniani/llms-vote
[mimic-iv]: https://physionet.org/content/mimiciv/2.2/
[ordo]: https://www.orphadata.com/ordo/
[rust]: https://www.rust-lang.org/tools/install
