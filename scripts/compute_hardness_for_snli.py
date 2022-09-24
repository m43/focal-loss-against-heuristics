import json
import os

from datasets import load_dataset

from src.constants import LABEL_TO_INTEGER

SNLI_TEST_HARD_JSONL = f"data/raw/SNLI Test Hard/snli_1.0_test_hard.jsonl"
SNLI_TEST_HARD_OUTPUT_ANNOTATIONS = f"./data/snli_test_hardness.csv"


def main():
    """
    Script that computes the hardness annotations for the SNLI test subset
    based on the annotations from `https://nlp.stanford.edu/projects/snli/`.
    A datapoint is considered "hard" if a simple fastText classifier (Joulin et al., 2017),
    misclassified it, as done in (Gururangan et al., 2018). This is because the classifier
    is rather simple and must rely on syntactic heuristics/biases in the dataset.

    Run script as `python -m scripts.compute_hardness_for_snli`.
    """

    assure_paths = [
        SNLI_TEST_HARD_JSONL,
    ]
    for path in assure_paths:
        if not os.path.exists(path):
            print(f"Could not find: {os.path.abspath(path)}")
            print()

            print(f"Download and extract the mismatched and matched datasetd from: "
                  f"`https://nlp.stanford.edu/projects/snli/`")
            print()

            print("Or use the direct download link (September 21, 2022):")
            print("https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl")
            exit()

    snli_test = load_dataset("snli", split="test")

    lookup = {}
    for idx, datapoint in enumerate(snli_test):
        key = datapoint["premise"] + " -- " + datapoint["hypothesis"] + " -- " + str(datapoint["label"])
        lookup[key] = {"hardness": 0, "idx": idx, "pair_id": ""}

    with open(SNLI_TEST_HARD_JSONL, 'r') as jsonl_file:
        snli_test_hardness_raw = list(map(json.loads, jsonl_file))
    for raw_datapoint in snli_test_hardness_raw:
        query = raw_datapoint["sentence1"] + " -- " + raw_datapoint["sentence2"] + " -- " + str(LABEL_TO_INTEGER[
                                                                                                    raw_datapoint[
                                                                                                        "gold_label"]])
        assert query in lookup, f"Query: {query}"
        lookup[query]["hardness"] = 1
        lookup[query]["pair_id"] = raw_datapoint["pairID"]
    assert len(snli_test_hardness_raw) == sum([v["hardness"] for k, v in lookup.items()])

    output_tuples = sorted([(v["idx"], v["pair_id"], v["hardness"]) for k, v in lookup.items()])
    with open(SNLI_TEST_HARD_OUTPUT_ANNOTATIONS, "w") as f:
        f.write(f"idx,pairID,hardness\n")
        for idx, pair_id, hard in output_tuples:
            assert "," not in pair_id
            f.write(f"{idx},{pair_id},{hard}\n")

    print(f"Done. Find the outputted hardness annotations in: {os.path.abspath(SNLI_TEST_HARD_OUTPUT_ANNOTATIONS)}")


if __name__ == '__main__':
    main()
