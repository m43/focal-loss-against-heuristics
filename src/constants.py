from enum import IntEnum

HEURISTIC_TO_INTEGER = {
    'none': -1,
    'lexical_overlap': 0,
    'subsequence': 1,
    'constituent': 2
}
INTEGER_TO_HEURISTIC = {v: k for k, v in HEURISTIC_TO_INTEGER.items()}

DATASET_TO_INTEGER = {
    "mnli_train": 0,
    "mnli_validation_matched": 1,
    "mnli_validation_mismatched": 2,
    "hans_train": 3,
    "hans_validation": 4,
}
INTEGER_TO_DATASET = {v: k for k, v in DATASET_TO_INTEGER.items()}
MNLI_DATASET_INTEGER_IDENTIFIERS = [
    DATASET_TO_INTEGER["mnli_train"],
    DATASET_TO_INTEGER["mnli_validation_matched"],
    DATASET_TO_INTEGER["mnli_validation_mismatched"],
]
HANS_DATASET_INTEGER_IDENTIFIERS = [
    DATASET_TO_INTEGER["hans_train"],
    DATASET_TO_INTEGER["hans_validation"],
]


class HandcraftedType(IntEnum):
    STANDARD = 0
    TRIVIAL = 1
    NOISE = 2
    HEURISTIC_NE = 3
    HEURISTIC_E = 4
