from enum import IntEnum

HEURISTIC_TO_INTEGER = {
    'lexical_overlap': 0,
    'subsequence': 1,
    'constituent': 2
}

DATASET_TO_INTEGER = {
    "mnli_train": 0,
    "mnli_validation_matched": 1,
    "mnli_validation_mismatched": 2,
    "hans_train": 3,
    "hans_validation": 4,
}


class SampleType(IntEnum):
    STANDARD = 0
    TRIVIAL = 1
    NOISE = 2
    HEURISTIC_NE = 3
    HEURISTIC_E = 4
