from typing import Union

import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from src.constants import HandcraftedType, HEURISTIC_TO_INTEGER, DATASET_TO_INTEGER, BERT_IDENTIFIER, T5_IDENTIFIER, \
    INTEGER_TO_LABEL, T5_LABEL_PAD_LENGTH
from src.utils.util import get_logger

log = get_logger(__name__)


def datasetdict_map_with_fingerprint(datasetdict: DatasetDict, fingerprint: str, **kwargs) -> DatasetDict:
    return DatasetDict({
        k: dataset.map(new_fingerprint=f"_{k}_{fingerprint}", **kwargs)
        for k, dataset in datasetdict.items()
    })


def dataset_map_with_fingerprint(dataset: Dataset, split: str, fingerprint: str, **kwargs) -> Dataset:
    return dataset.map(new_fingerprint=f"_{split}_{fingerprint}", **kwargs)


def set_dataset_format(model_name: str, dataset: Union[Dataset, DatasetDict], additional_columns=[]):
    if model_name == BERT_IDENTIFIER:
        dataset.set_format(
            type='torch',
            columns=['idx', 'dataset', 'input_ids', 'token_type_ids', 'attention_mask', 'label',
                     'handcrafted_type'] + additional_columns
        )
    elif model_name == T5_IDENTIFIER:
        dataset.set_format(
            type='torch',
            columns=['idx', 'dataset', 'input_ids', 'attention_mask', 'target_input_ids', 'target_attention_mask',
                     'label', 'handcrafted_type'] + additional_columns
        )
    else:
        raise ValueError(
            f"Model name value must be {BERT_IDENTIFIER} or {T5_IDENTIFIER}, '{model_name}' not recognized!")


def tokenize_sample_for_model_name(sample, tokenizer, model_name):
    res = {}
    if model_name == BERT_IDENTIFIER:
        res.update(tokenizer(sample['premise'], sample['hypothesis']))
    elif model_name == T5_IDENTIFIER:
        tokenized_label = tokenizer(INTEGER_TO_LABEL[sample['label']],
                                    max_length=T5_LABEL_PAD_LENGTH, pad_to_max_length=True)
        res["target_input_ids"] = tokenized_label.input_ids
        res["target_attention_mask"] = tokenized_label.attention_mask
        res.update(tokenizer('mnli premise: ' + sample['premise'], 'hypothesis: ' + sample['hypothesis']))
    else:
        raise ValueError(
            f"Model name value must be {BERT_IDENTIFIER} or {T5_IDENTIFIER}, '{model_name}' not recognized!")
    return res


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
    def process_hans(batch, model_name, tokenizer):
        if model_name == BERT_IDENTIFIER:
            res = tokenizer(batch['premise'], batch['hypothesis'])
        elif model_name == T5_IDENTIFIER:
            tokenized_label = tokenizer([INTEGER_TO_LABEL[label] for label in batch['label']],
                                        max_length=T5_LABEL_PAD_LENGTH, pad_to_max_length=True)
            res = {'target_input_ids': tokenized_label.input_ids,
                   'target_attention_mask': tokenized_label.attention_mask}
            res.update(tokenizer(['mnli premise: ' + input for input in batch['premise']],
                                 ['hypothesis: ' + input for input in batch['hypothesis']]))
        else:
            raise ValueError(
                f"Model name value must be {BERT_IDENTIFIER} or {T5_IDENTIFIER}, '{model_name}' not recognized!")

        res['heuristic'] = [HEURISTIC_TO_INTEGER[sample] for sample in batch['heuristic']]
        res['handcrafted_type'] = [
            HandcraftedType.HEURISTIC_E.value if (sample == 0) else HandcraftedType.HEURISTIC_NE.value
            for sample in batch['label']
        ]
        return res

    @staticmethod
    def setup_hans(batch_size, model_name, tokenizer):
        hans_dataset_validation = dataset_map_with_fingerprint(
            dataset=load_dataset("hans", split='validation'),
            fingerprint=model_name,
            split='validation',
            batched=True,
            batch_size=batch_size,
            function=HANSUtils.process_hans,
            fn_kwargs={'tokenizer': tokenizer, 'model_name': model_name},
            # ~~~ Uncomment for map debugging or when changing map code ~~~ #
            # load_from_cache_file=False
        )
        n = len(hans_dataset_validation)
        idx_col = np.arange(0, n)
        dataset_col = np.repeat(DATASET_TO_INTEGER[f"hans_validation"], n)
        hans_dataset_validation = hans_dataset_validation.add_column("idx", idx_col)
        hans_dataset_validation = hans_dataset_validation.add_column("dataset", dataset_col)
        set_dataset_format(model_name, hans_dataset_validation, additional_columns=["heuristic"])
        log.info(f"Hans validation dataset loaded, datapoints: {len(hans_dataset_validation)}")
        return hans_dataset_validation
