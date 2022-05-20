import pdb

import torch
from datasets import load_metric
from pytorch_lightning import LightningModule
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModelForSequenceClassification, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from src.constants import HEURISTIC_TO_INTEGER
from src.model.focalloss import FocalLoss

PRETRAINED_MODEL_ID = "bert-base-uncased"


class BertForNLI(LightningModule):
    def __init__(
            self,
            focal_loss_gamma: float,
            learning_rate: float,
            batch_size: int,
            weight_decay: float,
            adam_epsilon: float,
            warmup_steps: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.bert_config = AutoConfig.from_pretrained(PRETRAINED_MODEL_ID)
        self.bert: BertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_ID, num_labels=3)
        assert isinstance(self.bert, BertForSequenceClassification)

        self.classifier = nn.Linear(self.bert_config.hidden_size, 3)

        # initialized in self.setup()
        self.loss_criterion = FocalLoss(self.hparams.focal_loss_gamma)
        self.metric = load_metric("accuracy")

    def forward(self, input_ids, attention_mask, token_type_ids, label=None, **kwargs) -> SequenceClassifierOutput:
        output: SequenceClassifierOutput = self.bert.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return output

    def mnli_step(self, batch):
        output = self.forward(**batch)

        onehot_labels = torch.nn.functional.one_hot(batch["labels"], num_classes=3).float()
        loss = self.loss_criterion(output.logits, onehot_labels).mean()
        preds = output.logits.argmax(dim=-1)
        metrics_dict = self.metric.compute(predictions=preds, references=batch["labels"])

        results = {
            "loss": loss,
            "mnli_loss": loss,
            **{f"mnli_{k}": v for k, v in metrics_dict.items()},
        }
        return results

    def hans_step(self, batch):
        output = self.forward(**batch)

        onehot_labels = torch.nn.functional.one_hot(batch["labels"], num_classes=3).float()
        loss = self.loss_criterion(output.logits, onehot_labels)
        preds = output.logits.argmax(dim=-1)
        labels = batch["labels"]

        return {"loss": loss, "preds": preds, "labels": labels}

    def training_step(self, batch, batch_idx):
        results = self.mnli_step(batch)
        results_to_log = {f"train_{k}": v for k, v in results.items()}
        self.log_dict(results_to_log, on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)
        return results

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            results = self.mnli_step(batch)
            results_to_log = {f"val_{k}": v for k, v in results.items()}
            self.log_dict(results_to_log, on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)
            return results
        else:
            return self.hans_step(batch)

    def validation_epoch_end(self, outputs):
        hans_results = outputs[1]

        preds = torch.cat([x["preds"] for x in hans_results]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in hans_results]).detach().cpu().numpy()
        loss = torch.cat([x["loss"] for x in hans_results]).detach().cpu().numpy()
        loss_mean = loss.mean()

        metrics = {f"val_{k}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()}
        self.log("val_loss", loss_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)

        for heuristic_name, heuristic_idx in HEURISTIC_TO_INTEGER.items():
            mask = labels == heuristic_idx
            loss_heuristic_mean = loss[mask].mean()
            metrics = {f"val_{k}": v
                       for k, v in self.metric.compute(predictions=preds[mask], references=labels[mask]).items()}
            self.log(f"val_loss_{heuristic_name}", loss_heuristic_mean,
                     on_step=False, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        model = self.bert
        no_decay = ["bias", "LayerNorm.weight"]
        normal_lr = ["classifier"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and any(nlr in n for nlr in normal_lr)
                ],
                "learning_rate": self.hparams.learning_rate,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and any(nlr in n for nlr in normal_lr)
                ],
                "learning_rate": self.hparams.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and not any(nlr in n for nlr in normal_lr)
                ],
                "learning_rate": self.hparams.learning_rate / 100,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and not any(nlr in n for nlr in normal_lr)
                ],
                "learning_rate": self.hparams.learning_rate / 100,
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
