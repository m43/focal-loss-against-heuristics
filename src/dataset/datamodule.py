from typing import Optional

import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorWithPadding

from src.constants import HEURISTIC_TO_INTEGER, SampleType
from src.model.nlitransformer import PRETRAINED_MODEL_ID
from src.utils.util import get_logger

log = get_logger(__name__)


@DATAMODULE_REGISTRY
class ExperimentDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, num_workers: int = 4):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_str = PRETRAINED_MODEL_ID

        # attributes that may be downloaded and are initialized
        # in prepare data
        self.tokenizer = None
        self.hans_dataset = None
        self.mnli_dataset = None
        self.collator = None

    def prepare_data(self):
        load_dataset("hans", split='validation')
        load_dataset("multi_nli")

    def setup(self, stage: str):
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.tokenizer_str)

        # note that this batch size is the processing batch size for tokenization,
        # not the training batch size, I used the same because I'm lazy

        def tokenize_hans(batch):
            res = self.tokenizer(
                batch['premise'],
                batch['hypothesis']
            )
            res['heuristic'] = [HEURISTIC_TO_INTEGER[sample] for sample in batch['heuristic']]
            return res

        def process_mnli(sample):
            sample_type = SampleType.STANDARD
            if sample['premise'] == sample['hypothesis']:
                sample_type = SampleType.TRIVIAL if sample['label'] == 0 else SampleType.NOISE

            tokenized_premise = self.tokenizer(sample['premise'], add_special_tokens=False)['input_ids']
            tokenized_hypothesis = self.tokenizer(sample['hypothesis'], add_special_tokens=False)['input_ids']
            if all(token in tokenized_premise for token in tokenized_hypothesis):
                sample_type = SampleType.HEURISTIC_E if sample['label'] == 0 else SampleType.HEURISTIC_NE

            new_attributes = {'type': sample_type.value}
            new_attributes.update(self.tokenizer(sample['premise'], sample['hypothesis']))

            return new_attributes

        self.hans_dataset = load_dataset("hans", split='validation').map(
            tokenize_hans,
            batched=True,
            batch_size=self.batch_size,
        )
        self.hans_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'heuristic']
        )
        log.info(f"Hans validation dataset loaded, datapoints: {len(self.hans_dataset)}")

        self.mnli_dataset = load_dataset("multi_nli").map(process_mnli)
        self.mnli_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'type']
        )
        log.info(f"MNLI dataset splits loaded:")
        log.info(f"   len(self.mnli_dataset['train'])={len(self.mnli_dataset['train'])}")
        log.info(f"   len(self.mnli_dataset['validation_matched'])={len(self.mnli_dataset['validation_matched'])}")

        self.collator = DataCollatorWithPadding(self.tokenizer, padding='longest', return_tensors="pt")
        self.collator_fn = lambda x: self.collator(x).data

    def train_dataloader(self):
        return DataLoader(self.mnli_dataset['train'],
                          batch_size=self.batch_size,
                          collate_fn=self.collator_fn)  # type:ignore

    def val_dataloader(self):
        mnli_val_dataloader = DataLoader(self.mnli_dataset['validation_matched'],
                                         batch_size=self.batch_size,
                                         collate_fn=self.collator_fn)  # type:ignore

        hans_dataloader = DataLoader(self.hans_dataset,
                                     batch_size=self.batch_size,
                                     collate_fn=self.collator_fn)  # type:ignore
        return [mnli_val_dataloader, hans_dataloader]

    def teardown(self, stage: Optional[str] = None):
        pass
