import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW, Adam
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, \
    PreTrainedTokenizerBase, AutoTokenizer, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput

from src.constants import *
from src.model.focalloss import FocalLoss
from src.utils.util import get_logger

PRETRAINED_MODEL_ID = "bert-base-uncased"

log = get_logger(__name__)


class BertForNLI(LightningModule):
    """
    A PyTorch Lightning module that is a wrapper around
    a HuggingFace BERT for sequence classification model.
    The BERT model has a classification head on top and
    will be used to perform Natural Language Inference (NLI).

    Besides wrapping BERT, this class provides the functionality
    of training on MultiNLI dataset and evaluating on MultiNLI
    and HANS dataset. It also adds verbose logging.

    The module uses a linear warmup of configurable length
    and can be configured to either use a polynomial
    or a linear learning rate decay schedule.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        print("-" * 72)
        print(f"self.hparams={self.hparams}")
        print("-" * 72)

        self.bert: BertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_ID,
            hidden_dropout_prob=self.hparams["hidden_dropout_prob"],
            attention_probs_dropout_prob=self.hparams["attention_probs_dropout_prob"],
            classifier_dropout=self.hparams["classifier_dropout"],
            num_labels=3,
        )
        print(self.bert.config)

        assert isinstance(self.bert, BertForSequenceClassification)

        # initialized in self.setup()
        self.loss_criterion = FocalLoss(self.hparams.focal_loss_gamma)

    def forward(self, input_ids, attention_mask, token_type_ids, label=None, **kwargs) -> SequenceClassifierOutput:
        output: SequenceClassifierOutput = self.bert.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return output

    def training_step(self, batch, batch_idx):
        results = self._step(batch)
        self._step_log(batch, batch_idx, "Train", results)
        return results

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        results = self._step(batch)
        self._step_log(batch, batch_idx, "Valid", results, dataloader_idx)
        return results

    def _step(self, batch):
        output = self.forward(**batch)

        onehot_labels = F.one_hot(batch["labels"], num_classes=3).float()
        loss = self.loss_criterion(output.logits, onehot_labels)
        pred = output.logits.argmax(dim=-1)
        true_pred = (pred == batch["labels"]).float()
        prob = output.logits.softmax(-1)
        true_prob = prob.gather(-1, batch["labels"].unsqueeze(-1)).squeeze(-1)
        if "heuristic" in batch:  # HANS has the heuristic type, MNLI does not
            heuristic = batch["heuristic"]
        else:
            heuristic = true_pred.new_ones(true_pred.shape) * -1.0

        results = {
            "loss": loss.mean(),
            "acc": true_pred.mean(),
            "count": float(len(pred)),
            "datapoint_idx": batch["idx"],
            "datapoint_dataset": batch["dataset"],
            "datapoint_label": batch["labels"],
            "datapoint_handcrafted_type": batch["handcrafted_type"],
            "datapoint_heuristic": heuristic,
            "datapoint_loss": loss,
            "datapoint_pred": pred,
            "datapoint_true_pred": true_pred,
            "datapoint_prob": prob,
            "datapoint_true_prob": true_prob,
        }
        return results

    def _step_log(self, batch, batch_idx, prefix, results, dataloader_idx=0):
        dataset = results["datapoint_dataset"][0].item()
        dataset_str = INTEGER_TO_DATASET[dataset]

        if batch_idx == 0 or batch_idx == -1 and self.global_rank == 0 and self.current_epoch in [0, 1]:
            self._log_batch_for_debugging(f"{prefix}/Batch/batch-{batch_idx}_dataloader-{dataloader_idx}", batch)

        if dataset in MNLI_DATASET_INTEGER_IDENTIFIERS:
            log_kwargs = {
                'on_step': True,
                'on_epoch': True,
                'prog_bar': True,
                'logger': True,
                'add_dataloader_idx': False,
            }
            self.log(f"{prefix}/{dataset_str}_loss", results["loss"], **log_kwargs)
            self.log(f"{prefix}/{dataset_str}_acc", results["acc"], **log_kwargs)
            self.log(f"{prefix}/{dataset_str}_datapoint_count", results["count"], reduce_fx="sum", **log_kwargs)

    def _log_batch_for_debugging(self, log_key, batch):
        def jsonify(value):
            if isinstance(value, torch.Tensor):
                return value.tolist()
            return value

        debug_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_ID)
        batch = dict(batch)  # do not modify the original batch dict
        batch["txt"] = debug_tokenizer.batch_decode(batch["input_ids"])

        batch_json = json.dumps({k: jsonify(v) for k, v in batch.items()})
        log.info(f"{log_key}:\n{batch_json}")

        batch_df = pd.DataFrame({k: [str(jsonify(e)) for e in v] for k, v in batch.items()})
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger: WandbLogger = logger
                logger.log_text(f"{log_key}", dataframe=batch_df)

    def training_epoch_end(self, outputs):
        results_list = outputs
        self._epoch_end("Train", results_list)

    def validation_epoch_end(self, outputs):
        for results_list in outputs:
            self._epoch_end("Valid", results_list)

    def _epoch_end(self, split: str, results_list):
        dataset = results_list[0]["datapoint_dataset"][0].item()
        dataset_str = INTEGER_TO_DATASET[dataset]

        # Parse results from the results list
        results = {}
        for k in results_list[0].keys():
            if not k.startswith("datapoint"):
                # We are not interested in accumulated batch metrics
                # (avg loss per batch, avg acc per batch, etc.)
                # since we will recompute them with all datapoints
                continue
            values = [x[k] for x in results_list]
            results[k] = torch.cat(values).detach().cpu().numpy()

        # Add additional information to the results
        n = len(results["datapoint_idx"])
        results["epoch"] = np.repeat(self.current_epoch, n)
        results["step"] = np.repeat(self.global_step, n)
        results["datapoint_heuristics_str"] = np.array([
            INTEGER_TO_HEURISTIC[h]
            for h in results["datapoint_heuristic"]
        ])
        results["datapoint_handcrafted_type_str"] = np.array([
            HandcraftedType(t).name.title()
            for t in results["datapoint_handcrafted_type"]
        ])

        # Additional logs per dataset
        if dataset in MNLI_DATASET_INTEGER_IDENTIFIERS:
            self._log_mnli_epoch_end(split, dataset_str, results)
        if dataset in HANS_DATASET_INTEGER_IDENTIFIERS:
            self._log_hans_epoch_end(split, dataset_str, results)

        # Create a DataFrame to be used in post-run logs processing to create visuals for the paper report
        for k in results.keys():
            if results[k].ndim > 1:  # Data for a pd.DataFrame must be 1-dimensional
                results[k] = [str(x) for x in results[k]]
        df = pd.DataFrame(results)

        # Log the dataframe to wandb
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger: WandbLogger = logger
                csv_path = os.path.join(
                    logger.experiment.dir,
                    f"{split}_{dataset_str}_epoch_end_df_epoch-{self.current_epoch}_step-{self.global_step}.csv"
                )
                df.to_csv(csv_path)
                artifact = wandb.Artifact(
                    name=f"{logger.experiment.name}-{split}-{dataset_str}_epoch_end_df",
                    type="df",
                    metadata={"epoch": self.current_epoch, "step": self.global_step},
                )
                artifact.add_file(csv_path, "df.csv")
                logger.experiment.log_artifact(artifact)

    def _log_mnli_epoch_end(self, prefix, dataset_str, results):
        handcrafted_types = results["datapoint_handcrafted_type"],
        losses = results["datapoint_loss"],
        true_preds = results["datapoint_true_pred"],

        for handcrafted_type in HandcraftedType:
            mask = handcrafted_types == handcrafted_type.value
            loss_per_type = losses[mask].mean()
            acc_per_type = true_preds[mask].mean()
            log_kwargs = {
                'on_step': False,
                'on_epoch': True,
                'prog_bar': True,
                'logger': True,
                'add_dataloader_idx': False,
            }
            self.log(f"{prefix}/HandcraftedType/{dataset_str}_{handcrafted_type.name.lower()}_loss", loss_per_type,
                     **log_kwargs)
            self.log(f"{prefix}/HandcraftedType/{dataset_str}_{handcrafted_type.name.lower()}_accuracy", acc_per_type,
                     **log_kwargs)

    def _log_hans_epoch_end(self, split, dataset_str, results):
        n = len(results["datapoint_idx"])
        loss = results["datapoint_loss"].mean()
        acc = results["datapoint_true_pred"].sum() / len(results["datapoint_pred"])

        # Sanity check
        acc2 = (results["datapoint_pred"] == results["datapoint_label"]).sum() / len(results["datapoint_pred"])
        assert acc == acc2

        log_kwargs = {
            'on_step': False,
            'on_epoch': True,
            'prog_bar': True,
            'logger': True,
            'add_dataloader_idx': False,
        }
        self.log(f"{split}/{dataset_str}_loss", loss, **log_kwargs)
        self.log(f"{split}/{dataset_str}_acc", acc, **log_kwargs)
        self.log(f"{split}/{dataset_str}_count", float(n), **log_kwargs)

        heuristics = results["datapoint_heuristic"]
        labels = results["datapoint_label"]
        losses = results["datapoint_loss"]
        preds = results["datapoint_pred"]

        for target_label, label_description in enumerate(["entailment", "non_entailment"]):
            for heuristic_name, heuristic_idx in HEURISTIC_TO_INTEGER.items():
                mask = (heuristics == heuristic_idx) & (labels == target_label)
                if mask.sum() == 0:
                    # This might be true during the sanity checks of PyTorch Lightning
                    # where not the whole dataset would be used, but a small subset.
                    # This way we avoid NaN and polluting our metrics.
                    continue

                loss = losses[mask].mean()
                acc = (preds[mask] == labels[mask]).mean()
                self.log(f"Valid/Hans_loss/{label_description}_{heuristic_name}", loss, **log_kwargs)
                self.log(f"Valid/Hans_acc/{label_description}_{heuristic_name}", acc, **log_kwargs)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        model = self.bert
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "name": "1_w-decay",
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "name": "2_no-decay",
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.optimizer_name == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                eps=self.hparams.adam_epsilon,
            )
        elif self.hparams.optimizer_name == "adam":
            optimizer = Adam(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                eps=self.hparams.adam_epsilon,
            )
        else:
            raise ValueError(f"Invalid optimizer_name given: {self.hparams.optimizer_name}")

        train_steps = self.trainer.estimated_stepping_batches

        if self.hparams.warmup_ratio is not None and self.hparams.warmup_steps is not None:
            raise ValueError("Either warmup_steps or warmup_ratio should be given, but not both.")

        if self.hparams.warmup_steps:
            warmup_steps = self.hparams.warmup_steps
        elif self.hparams.warmup_ratio:
            warmup_steps = train_steps * self.hparams.warmup_ratio
        else:
            raise ValueError("Either warmup_steps or warmup_ratio should be given, but none were given.")

        if self.hparams.scheduler_name == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=train_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.hparams.scheduler_name == "polynomial":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=train_steps,
                lr_end=0.0,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        else:
            raise ValueError(f"Invalid scheduler_name given: {self.hparams.optimizer_name}")

        return [optimizer], [scheduler]
