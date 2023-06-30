#!/usr/bin/env python
# Author: David Oniani
# License: MIT
#
# ORDO has the following codes:
#   Clinical entity: Orphanet_C001
#   Group of disorders: Orphanet_557492
#   Disorder: Orphanet_557493
#   Subtype of a disorder: Orphanet_557494
#
import argparse
import logging
import pathlib

import rdflib
import tqdm


QUERY: str = """
PREFIX ordo: <http://www.orpha.net/ORDO/Orphanet_>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?clsLabel WHERE {
    ?cls rdfs:subClassOf ordo:557493 ; rdfs:label ?clsLabel
}
"""


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""

    parser = argparse.ArgumentParser("Pretrained SNN Benchmarking")

    parser.add_argument("--ordo", type=str, help="path to the ORDO ontology file", required=True)
    parser.add_argument("--out", type=str, help="path to write the file", required=True)

    return parser.parse_args()


def main() -> None:
    """Extracts rare disease terms and writes to a JSON file."""

    args = parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"{pathlib.Path(__file__).stem}.log", level=logging.INFO)

    logger.info("Loading and parsing ORDO ontology...")
    g = rdflib.Graph()
    g.parse(args.ordo)
    logger.info("Loading and parsing finished!")

    logger.info("Writing to a file...")
    with open(args.out, "w", encoding="utf-8") as file:
        for (elem,) in tqdm.tqdm(g.query(QUERY)):
            file.write(f"{elem.value}\n")
    logger.info("Writing finished!")


if __name__ == "__main__":
    main()
