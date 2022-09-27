from typing import Optional

import numpy as np
import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorWithPadding

from src.constants import *
from src.dataset.util import HandcraftedTypeSingleton, HANSUtils, datasetdict_map_with_fingerprint, set_dataset_format, \
    tokenize_sample_for_model_name
from src.utils.util import get_logger

log = get_logger(__name__)


@DATAMODULE_REGISTRY
class SNLIDatamodule(pl.LightningDataModule):
    """
    PyTorch Lightning datamodule to load and preprocess the SNLI dataset.
    """

    def __init__(
            self,
            model_name: str,
            batch_size: int,
            num_workers: int = 4,
            tokenizer_model_max_length: int = 512
    ):
        super().__init__()

        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_str = PRETRAINED_IDS[model_name]
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
    def _process_snli(sample, model_name, tokenizer):
        res = tokenize_sample_for_model_name(sample, tokenizer, model_name)
        res['handcrafted_type'] = HandcraftedTypeSingleton().compute_handcrafted_type(sample)
        return res

    def _setup_snli(self):
        self.snli_dataset = load_dataset("snli")

        # Filter out labels of -1 (but after the idx column has been assigned)
        # Dataset instances which don't have any gold label are marked with -1 label
        for subset in self.snli_dataset.keys():
            self.snli_dataset[subset] = self.snli_dataset[subset].filter(lambda x: x["label"] != -1)

        self.snli_dataset = datasetdict_map_with_fingerprint(
            datasetdict=self.snli_dataset,
            fingerprint=self.model_name,
            function=self._process_snli,
            fn_kwargs={'tokenizer': self.tokenizer, 'model_name': self.model_name},
            # ~~~ Uncomment for map debugging or when changing map code ~~~ #
            # load_from_cache_file=False
        )

        # Add idx and dataset columns
        for subset in self.snli_dataset.keys():
            n = len(self.snli_dataset[subset])
            idx_col = np.arange(0, n)
            dataset_col = np.repeat(DATASET_TO_INTEGER[f"snli_{subset}"], n)
            self.snli_dataset[subset] = self.snli_dataset[subset].add_column("idx", idx_col)
            self.snli_dataset[subset] = self.snli_dataset[subset].add_column("dataset", dataset_col)

        set_dataset_format(self.model_name, self.snli_dataset)

        log.info(f"SNLI dataset splits loaded:")
        log.info(f"   len(self.snli_dataset['train'])={len(self.snli_dataset['train'])}")
        log.info(f"   len(self.snli_dataset['validation'])={len(self.snli_dataset['validation'])}")
        log.info(f"   len(self.snli_dataset['test'])={len(self.snli_dataset['test'])}")

    def _setup_hans(self):
        self.hans_dataset_validation = HANSUtils.setup_hans(self.batch_size, self.model_name, self.tokenizer)

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = self.load_tokenizer(self.tokenizer_str, self.tokenizer_model_max_length)
        self._setup_snli()
        self._setup_hans()
        self.collator = DataCollatorWithPadding(self.tokenizer, padding='longest', return_tensors="pt")
        self.collator_fn = lambda x: self.collator(x).data

    def train_dataloader(self):
        snli_train_dataloader = DataLoader(self.snli_dataset['train'],
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           collate_fn=self.collator_fn)
        return snli_train_dataloader

    def val_dataloader(self):
        snli_validation = DataLoader(self.snli_dataset['validation'],
                                     batch_size=self.batch_size,
                                     collate_fn=self.collator_fn)
        snli_test = DataLoader(self.snli_dataset['test'],
                               batch_size=self.batch_size,
                               collate_fn=self.collator_fn)
        hans_validation = DataLoader(self.hans_dataset_validation,
                                     batch_size=self.batch_size,
                                     collate_fn=self.collator_fn)
        return [snli_validation, snli_test, hans_validation]

    def teardown(self, stage: Optional[str] = None):
        pass
