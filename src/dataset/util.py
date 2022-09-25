import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from src.constants import HandcraftedType, HEURISTIC_TO_INTEGER, DATASET_TO_INTEGER
from src.utils.util import get_logger

log = get_logger(__name__)


class HandcraftedTypeSingleton(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(HandcraftedTypeSingleton, cls).__new__(cls)
            cls.instance.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("bert-base-uncased")
        return cls.instance

    def compute_handcrafted_type(self, sample):
        handcrafted_type = HandcraftedType.STANDARD
        if sample['premise'] == sample['hypothesis']:
            handcrafted_type = HandcraftedType.TRIVIAL if sample['label'] == 0 else HandcraftedType.NOISE
        else:
            tokenized_premise = self.tokenizer(sample['premise'], add_special_tokens=False)['input_ids']
            tokenized_hypothesis = self.tokenizer(sample['hypothesis'], add_special_tokens=False)['input_ids']
            if all(token in tokenized_premise for token in tokenized_hypothesis):
                handcrafted_type = HandcraftedType.HEURISTIC_E if sample['label'] == 0 else HandcraftedType.HEURISTIC_NE

        return handcrafted_type.value


class HANSUtils:
    @staticmethod
    def process_hans(batch, tokenizer):
        res = tokenizer(batch['premise'], batch['hypothesis'])
        res['heuristic'] = [HEURISTIC_TO_INTEGER[sample] for sample in batch['heuristic']]
        res['handcrafted_type'] = [
            HandcraftedType.HEURISTIC_E.value if (sample == 0) else HandcraftedType.HEURISTIC_NE.value
            for sample in batch['label']
        ]
        return res

    @staticmethod
    def setup_hans(batch_size, tokenizer):
        hans_dataset_validation = load_dataset("hans", split='validation')
        hans_dataset_validation = hans_dataset_validation.map(
            HANSUtils.process_hans,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={'tokenizer': tokenizer},
        )
        n = len(hans_dataset_validation)
        idx_col = np.arange(0, n)
        dataset_col = np.repeat(DATASET_TO_INTEGER[f"hans_validation"], n)
        hans_dataset_validation = hans_dataset_validation.add_column("idx", idx_col)
        hans_dataset_validation = hans_dataset_validation.add_column("dataset", dataset_col)
        hans_dataset_validation.set_format(
            type='torch',
            columns=['idx', 'dataset', 'input_ids', 'token_type_ids', 'attention_mask', 'label', 'handcrafted_type',
                     'heuristic']
        )
        log.info(f"Hans validation dataset loaded, datapoints: {len(hans_dataset_validation)}")
        return hans_dataset_validation
