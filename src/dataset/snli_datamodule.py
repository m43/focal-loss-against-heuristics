from typing import Optional

import numpy as np
import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorWithPadding

from src.constants import *
from src.dataset.mnli_datamodule import HandcraftedTypeSingleton
from src.model.nlitransformer import PRETRAINED_MODEL_ID
from src.utils.util import get_logger

log = get_logger(__name__)


@DATAMODULE_REGISTRY
class SNLIDatamodule(pl.LightningDataModule):
    """
    PyTorch Lightning datamodule to load and preprocess the SNLI dataset.
    """

    def __init__(
            self,
            batch_size: int,
            num_workers: int = 4,
            tokenizer_model_max_length: int = 512
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_str = PRETRAINED_MODEL_ID
        self.tokenizer_model_max_length = tokenizer_model_max_length

        self.tokenizer = None
        self.snli_dataset = None
        self.collator_fn = None
        self.collator = None

    def prepare_data(self):
        load_dataset("snli")

    @staticmethod
    def load_tokenizer(tokenizer_str, tokenizer_model_max_length):
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_str)
        tokenizer.model_max_length = tokenizer_model_max_length
        return tokenizer

    @staticmethod
    def _process_snli(sample, tokenizer):
        res = tokenizer(sample['premise'], sample['hypothesis'])
        res['handcrafted_type'] = HandcraftedTypeSingleton().compute_handcrafted_type(sample)
        return res

    def _setup_snli(self):
        self.snli_dataset = load_dataset("snli").map(self._process_snli, fn_kwargs={'tokenizer': self.tokenizer})

        # Add idx and dataset columns
        for subset in self.snli_dataset.keys():
            n = len(self.snli_dataset[subset])
            idx_col = np.arange(0, n)
            dataset_col = np.repeat(DATASET_TO_INTEGER[f"snli_{subset}"], n)
            self.snli_dataset[subset] = self.snli_dataset[subset].add_column("idx", idx_col)
            self.snli_dataset[subset] = self.snli_dataset[subset].add_column("dataset", dataset_col)

        # Filter out labels of -1 (but after the idx column has been assigned)
        # Dataset instances which don't have any gold label are marked with -1 label
        for subset in self.snli_dataset.keys():
            self.snli_dataset[subset] = self.snli_dataset[subset].filter(lambda x: x["label"] != -1)

        self.snli_dataset.set_format(
            type='torch',
            columns=['idx', 'dataset', 'input_ids', 'token_type_ids', 'attention_mask', 'label', 'handcrafted_type']
        )
        log.info(f"SNLI dataset splits loaded:")
        log.info(f"   len(self.snli_dataset['train'])={len(self.snli_dataset['train'])}")
        log.info(f"   len(self.snli_dataset['validation'])={len(self.snli_dataset['validation'])}")
        log.info(f"   len(self.snli_dataset['test'])={len(self.snli_dataset['test'])}")

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = self.load_tokenizer(self.tokenizer_str, self.tokenizer_model_max_length)
        self._setup_snli()
        self.collator = DataCollatorWithPadding(self.tokenizer, padding='longest', return_tensors="pt")
        self.collator_fn = lambda x: self.collator(x).data

    def train_dataloader(self):
        return DataLoader(self.snli_dataset['train'],
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collator_fn)

    def val_dataloader(self):
        snli_validation = DataLoader(self.snli_dataset['validation'],
                                     batch_size=self.batch_size,
                                     collate_fn=self.collator_fn)
        snli_test = DataLoader(self.snli_dataset['test'],
                               batch_size=self.batch_size,
                               collate_fn=self.collator_fn)
        return [snli_validation, snli_test]

    def teardown(self, stage: Optional[str] = None):
        pass
