## Environment

```console
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Generating a List of Rare Diseases

Run the following to generate a list of rare diseases using [ORDO ontology][ordo]:

```console
$ mkdir data
$ ./gen_rare_disease.py --ordo data/ORDO_en_4.2.owl --out data/rare_disease.txt
```

[ordo]: https://www.orphadata.com/ordo/
