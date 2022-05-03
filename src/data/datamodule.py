import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorWithPadding

from math import ceil

@DATAMODULE_REGISTRY
class ExperimentDataModule(pl.LightningDataModule):

    def __init__(self, batch_size:int, tokenizer:str, num_workers:int=4):
        super().__init__()

        assert (tokenizer in ['bert-base-cased'])

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_str = tokenizer

    def prepare_data(self):

        self.tokenizer:PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.tokenizer)

        # note that this batch size is the processing batch size for tokenization,
        # not the training batch size, I used the same because I'm lazy

        self.hans_dataset= load_dataset("hans", split='validation').map(
            lambda batch : self.tokenizer(
                batch['premise'],
                batch['hypothesis'],
            ),
            batched=True,
            batch_size=self.batch_size,
        ).set_format(
            type='torch',
            columns = ['input_ids', 'token_type_ids', 'attention_mask', 'label']
        )

        self.mnli_dataset = load_dataset("multi_nli").map(
            lambda batch : self.tokenizer(
                batch['premise'],
                batch['hypothesis'],
            ),
            batched=True,
            batch_size=self.batch_size,
        ).set_format(
            type='torch',
            columns = ['input_ids', 'token_type_ids', 'attention_mask', 'label']
        )

        self.collator = DataCollatorWithPadding(self.tokenizer, padding='longest')


    def setup(self, stage:str):
        pass

    def train_dataloader(self):
        return DataLoader(self.mnli_dataset['train'],
                          batch_size=self.batch_size,
                          collate_fn=self.collator) #type:ignore

    def val_dataloader(self):
        mnli_val_dataloader = DataLoader(self.mnli_dataset['validation_matched'],
                                         batch_size=self.batch_size,
                                         collate_fn=self.collator) #type:ignore

        hans_dataloader = DataLoader(self.hans_dataset,
                                     batch_size=self.batch_size,
                                     collate_fn=self.collator) #type:ignore

        return [mnli_val_dataloader, hans_dataloader]

    def teardown(self, stage: Optional[str] = None):
        pass
