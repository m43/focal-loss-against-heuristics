import os

import nltk
from tqdm import tqdm

from src.constants import LABEL_TO_INTEGER
from src.dataset.datamodule import ExperimentDataModule

MNLI_MATCHED_HARD_WITH_HARD_TEST = f"./data/MNLIMatchedHardWithHardTest/"
MNLI_MISMATCHED_HARD_WITH_HARD_TEST = f"./data/MNLIMismatchedHardWithHardTest/"
TOKENIZER_SED = "./data/tokenizer.sed"

MNLI_MATCHED_OUTPUT_ANNOTATIONS = f"./data/mnli_validation_matched_hardness.csv"
MNLI_MISMATCHED_OUTPUT_ANNOTATIONS = f"./data/mnli_validation_mismatched_hardness.csv"


def load_hard_mnli(dataset_path):
    label_list = []
    premise_list = []
    hypothesis_list = []

    with open(os.path.join(dataset_path, "labels.test"), "r") as f:
        for line in f.readlines():
            line = line.lower().strip()
            if len(line) == 0:
                continue
            label = LABEL_TO_INTEGER[line]
            label_list.append(label)

    with open(os.path.join(dataset_path, "s1.test"), "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            premise_list.append(line)

    with open(os.path.join(dataset_path, "s2.test"), "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            hypothesis_list.append(line)

    assert len(label_list) == len(premise_list) == len(hypothesis_list)

    datapoints = []
    for l, p, h in zip(label_list, premise_list, hypothesis_list):
        datapoints.append({"label": l, "premise": p, "hypothesis": h})

    return datapoints


def main():
    """
    Script that computes the harness annotations for the MNLI validation dataset
    based on the annotations from `https://github.com/rabeehk/robust-nli`.
    A datapoint is considered "hard" if a simple fastText classifier (Joulin et al., 2017),
    misclassified it, as done in (Gururangan et al., 2018). This is because the classifier
    is rather simple and must rely on syntactic heuristics/biases in the dataset.

    Run script as `python -m scripts.compute_hardness_from_robustnli`.
    """

    assure_paths = [
        MNLI_MATCHED_HARD_WITH_HARD_TEST,
        MNLI_MISMATCHED_HARD_WITH_HARD_TEST,
    ]
    for path in assure_paths:
        if not os.path.exists(path):
            print(f"Could not find: {os.path.abspath(path)}")
            print()

            print(f"Download and extract the mismatched and matched datasetd from: "
                  f"`https://github.com/rabeehk/robust-nli`")
            print()

            print("Or use these direct download links (August 7, 2022):")
            print("https://www.dropbox.com/s/bidxvrd8s2msyan/MNLIMismatchedHardWithHardTest.zip?dl=1")
            print("https://www.dropbox.com/s/3aktzl4bhmqti9n/MNLIMatchedHardWithHardTest.zip?dl=1")
            exit()

    datamodule = ExperimentDataModule(batch_size=32, tokenizer_model_max_length=128)
    datamodule.setup()

    mnli_m_hard = load_hard_mnli(MNLI_MATCHED_HARD_WITH_HARD_TEST)
    mnli_mm_hard = load_hard_mnli(MNLI_MISMATCHED_HARD_WITH_HARD_TEST)

    datapoints_dataset_pairs = [
        (mnli_m_hard, datamodule.mnli_dataset['validation_matched'], MNLI_MATCHED_OUTPUT_ANNOTATIONS),
        (mnli_mm_hard, datamodule.mnli_dataset['validation_mismatched'], MNLI_MISMATCHED_OUTPUT_ANNOTATIONS),
    ]
    for datapoints, dataset, output_path in datapoints_dataset_pairs:
        hardness = [0] * len(dataset)
        penn_tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
        premises = [" ".join(penn_tokenizer.tokenize(p)).replace(" ", "") for p in dataset["premise"]]
        hypotheses = [" ".join(penn_tokenizer.tokenize(p)).replace(" ", "") for p in dataset["hypothesis"]]
        keys = [p + "--" + h + "--" + str(l.item()) for p, h, l in zip(premises, hypotheses, dataset["label"])]
        for dp in tqdm(datapoints):
            query = dp["premise"].replace(" ", "") + "--" + dp["hypothesis"].replace(" ", "") + "--" + str(dp["label"])
            query = query.replace(u'\xa0', u'')
            cnt = keys.count(query)
            idx = keys.index(query)

            if idx == 3393:
                assert cnt == 2
                assert keys[3393] == keys[6860]
                hardness[3393] = 1
                hardness[6860] = 1
                continue

            assert cnt == 1
            hardness[idx] = 1

        assert sum(hardness) == len(datapoints)

        with open(output_path, "w") as f:
            f.write(f"idx,pairID,hardness\n")
            for idx, pair_id, hard in zip(dataset["idx"], dataset["pairID"], hardness):
                f.write(f"{idx},{pair_id},{hard}\n")


if __name__ == '__main__':
    main()
