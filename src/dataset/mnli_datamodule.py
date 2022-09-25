from typing import Optional

import numpy as np
import pytorch_lightning as pl
from datasets import load_dataset, concatenate_datasets, ClassLabel
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorWithPadding

from src.constants import *
from src.dataset.util import HandcraftedTypeSingleton, HANSUtils
from src.model.nlitransformer import PRETRAINED_MODEL_ID
from src.utils.util import get_logger

log = get_logger(__name__)


@DATAMODULE_REGISTRY
class MNLIWithHANSDatamodule(pl.LightningDataModule):
    """
    PyTorch Lightning datamodule to load and preprocess the MultiNLI and HANS datasets.
    """

    def __init__(
            self,
            batch_size: int,
            num_hans_train_examples: int = 0,
            num_workers: int = 4,
            tokenizer_model_max_length: int = 512
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_hans_train_examples = num_hans_train_examples
        self.num_workers = num_workers
        self.tokenizer_str = PRETRAINED_MODEL_ID
        self.tokenizer_model_max_length = tokenizer_model_max_length

        self.tokenizer = None
        self.collator_fn = None
        self.hans_dataset = None
        self.mnli_dataset = None
        self.collator = None

    def prepare_data(self):
        load_dataset("hans", split='train')
        load_dataset("hans", split='validation')
        load_dataset("multi_nli")

    @staticmethod
    def load_tokenizer(tokenizer_str, tokenizer_model_max_length):
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_str)
        tokenizer.model_max_length = tokenizer_model_max_length
        return tokenizer

    @staticmethod
    def _process_mnli(sample, tokenizer):
        res = tokenizer(sample['premise'], sample['hypothesis'])
        res['handcrafted_type'] = HandcraftedTypeSingleton().compute_handcrafted_type(sample)
        return res

    def _setup_mnli(self):
        self.mnli_dataset = load_dataset("multi_nli").map(
            self._process_mnli,
            fn_kwargs={'tokenizer': self.tokenizer},
        )
        for subset in self.mnli_dataset.keys():
            n = len(self.mnli_dataset[subset])
            idx_col = np.arange(0, n)
            dataset_col = np.repeat(DATASET_TO_INTEGER[f"mnli_{subset}"], n)
            self.mnli_dataset[subset] = self.mnli_dataset[subset].add_column("idx", idx_col)
            self.mnli_dataset[subset] = self.mnli_dataset[subset].add_column("dataset", dataset_col)
        self.mnli_dataset.set_format(
            type='torch',
            columns=['idx', 'dataset', 'input_ids', 'token_type_ids', 'attention_mask', 'label', 'handcrafted_type']
        )
        log.info(f"MNLI dataset splits loaded:")
        log.info(f"   len(self.mnli_dataset['train'])={len(self.mnli_dataset['train'])}")
        log.info(f"   len(self.mnli_dataset['validation_matched'])={len(self.mnli_dataset['validation_matched'])}")
        log.info(
            f"   len(self.mnli_dataset['validation_mismatched'])={len(self.mnli_dataset['validation_mismatched'])}")

    def _sprinkle_train_with_hans(self):
        hans_dataset_train = load_dataset("hans", split='train').map(
            HANSUtils.process_hans,
            batched=True,
            batch_size=self.batch_size,
        )
        n = len(hans_dataset_train)
        idx_col = np.arange(0, n)
        dataset_col = np.repeat(DATASET_TO_INTEGER[f"hans_train"], n)
        hans_dataset_train = hans_dataset_train.add_column("idx", idx_col)
        hans_dataset_train = hans_dataset_train.add_column("dataset", dataset_col)

        # rename features to match MNLI
        features = hans_dataset_train.features.copy()
        features['label'] = ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'])
        hans_dataset_train = hans_dataset_train.map(
            lambda batch: batch,
            batched=True,
            batch_size=self.batch_size,
            features=features
        )
        log.info(f"Hans train dataset loaded, datapoints: {len(hans_dataset_train)}")

        hans_dataset_train = hans_dataset_train.shuffle()
        hans_dataset_train = hans_dataset_train.select(range(self.num_hans_train_examples))
        self.mnli_dataset['train'] = concatenate_datasets([self.mnli_dataset['train'], hans_dataset_train])

        self.mnli_dataset.set_format(
            type='torch',
            columns=['idx', 'dataset', 'input_ids', 'token_type_ids', 'attention_mask', 'label', 'handcrafted_type']
        )

        log.info(f"HANS training examples added to the MNLI training dataset splits loaded:")
        log.info(f"   len(self.mnli_dataset['train'])={len(self.mnli_dataset['train'])}")

    def _setup_hans(self):
        self.hans_dataset_validation = HANSUtils.setup_hans(self.batch_size, self.tokenizer)

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = self.load_tokenizer(self.tokenizer_str, self.tokenizer_model_max_length)

        self._setup_mnli()
        self._setup_hans()
        if self.num_hans_train_examples > 0:
            self._sprinkle_train_with_hans()

        self.collator = DataCollatorWithPadding(self.tokenizer, padding='longest', return_tensors="pt")
        self.collator_fn = lambda x: self.collator(x).data

    def train_dataloader(self):
        return DataLoader(self.mnli_dataset['train'],
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collator_fn)

    def val_dataloader(self):
        mnli_val1_dataloader = DataLoader(self.mnli_dataset['validation_matched'],
                                          batch_size=self.batch_size,
                                          collate_fn=self.collator_fn)
        mnli_val2_dataloader = DataLoader(self.mnli_dataset['validation_mismatched'],
                                          batch_size=self.batch_size,
                                          collate_fn=self.collator_fn)
        hans_dataloader = DataLoader(self.hans_dataset_validation,
                                     batch_size=self.batch_size,
                                     collate_fn=self.collator_fn)
        return [mnli_val1_dataloader, mnli_val2_dataloader, hans_dataloader]

    def teardown(self, stage: Optional[str] = None):
        pass
