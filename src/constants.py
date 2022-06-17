from enum import IntEnum

HEURISTIC_TO_INTEGER = {
    'lexical_overlap': 0,
    'subsequence': 1,
    'constituent': 2
}


class SampleType(IntEnum):
    STANDARD = 0
    TRIVIAL = 1
    NOISE = 2
    HEURISTIC_NE = 3
    HEURISTIC_E = 4
