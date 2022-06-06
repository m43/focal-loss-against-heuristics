import io
import json

import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, BertForSequenceClassification, \
    PreTrainedTokenizerBase, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput

from src.constants import HEURISTIC_TO_INTEGER, SampleType
from src.model.focalloss import FocalLoss
from src.utils.util import get_logger

PRETRAINED_MODEL_ID = "bert-base-uncased"

log = get_logger(__name__)


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

        # initialized in self.setup()
        self.loss_criterion = FocalLoss(self.hparams.focal_loss_gamma)

    def forward(self, input_ids, attention_mask, token_type_ids, label=None, **kwargs) -> SequenceClassifierOutput:
        output: SequenceClassifierOutput = self.bert.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return output

    def mnli_step(self, batch):
        output = self.forward(**batch)

        onehot_labels = F.one_hot(batch["labels"], num_classes=3).float()
        loss = self.loss_criterion(output.logits, onehot_labels)
        preds = output.logits.argmax(dim=-1)
        true_preds = (preds == batch["labels"])

        results = {
            "mnli_loss": loss.mean(),
            "mnli_datapoint_loss": loss,
            "mnli_datapoint_type": batch["type"],
            "mnli_acc": true_preds.mean(),
            "mnli_true_preds": preds,
            "mnli_datapoint_count": len(preds),
        }
        return results

    def hans_step(self, batch):
        output = self.forward(**batch)

        onehot_labels = F.one_hot(batch["labels"], num_classes=3).float()
        loss = self.loss_criterion(output.logits, onehot_labels)
        preds = output.logits.argmax(dim=-1)
        labels = batch["labels"]
        heuristic = batch["heuristic"]

        return {"hans_loss": loss, "preds": preds, "labels": labels, "heuristic": heuristic}

    def training_step(self, batch, batch_idx):
        results = self.mnli_step(batch)

        self.log(f"Train/mnli_loss", results["mnli_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"Train/mnli_acc", results["mnli_acc"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"Train/mnli_datapoint_count", results["mnli_datapoint_count"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, add_dataloader_idx=False, reduce_fx="sum")

        if batch_idx == 0 or batch_idx == -1 and self.global_rank == 0 and self.current_epoch in [0, 1]:
            self._log_batch_for_debugging(f"Train/Batch/batch_{batch_idx}", batch)

        # Loss to be used by the optimizer
        results["loss"] = results["mnli_loss"]
        return results

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            results = self.mnli_step(batch)
            self.log(f"Valid/mnli_loss", results["mnli_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     add_dataloader_idx=False)
            self.log(f"Valid/mnli_acc", results["mnli_acc"], on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     add_dataloader_idx=False)
            self.log(f"Valid/mnli_datapoint_count", results["mnli_datapoint_count"], on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, add_dataloader_idx=False, reduce_fx="sum")
        else:
            results = self.hans_step(batch)

        if batch_idx == 0 or batch_idx == -1 and self.global_rank == 0 and self.current_epoch in [0, 1]:
            self._log_batch_for_debugging(f"Valid/Batch/batch-{batch_idx}_dataloader-{dataloader_idx}", batch)

        return results

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

    @staticmethod
    def log_plot_to_wandb(wandb_logger, log_key, backend=plt, dpi=200):
        with io.BytesIO() as f:
            backend.savefig(f, dpi=dpi, format='png')
            im = Image.open(f)
            wandb_logger.log({log_key: wandb.Image(im)})

    def _log_loss_histogram(self, losses, split, metric_name, log_df=False, bins=72, height=4.5, dpi=200):
        df = pd.DataFrame({metric_name: losses})
        grid = sns.displot(data=df, x=metric_name, height=height, bins=bins)
        grid.fig.suptitle(f"[Epoch {self.current_epoch}] {split} {metric_name} histogram", fontsize=16)
        grid.set_titles(col_template="{col_name}", row_template="{row_name}")
        grid.tight_layout()

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log({f'{split}/Verbose/{metric_name}_histogram': wandb.Histogram(losses)})
                self.log_plot_to_wandb(logger.experiment, f'{split}/Verbose/{metric_name}_histogram_seaborn',
                                       backend=grid, dpi=dpi)
                if log_df:
                    logger.experiment.log({f"{split}/Verbose/{metric_name}_df": df})

        plt.close()

    def _log_mnli_metrics_per_sample_type(self, prefix:str, types, losses, true_preds):
        for sample_type in SampleType:
            mask = types == sample_type.value
            loss_per_type = losses[mask].mean()
            acc_per_type = true_preds[mask].mean()
            self.log(f"{prefix}/mnli_{sample_type.name.lower()}_loss", loss_per_type, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)
            self.log(f"{prefix}/mnli_{sample_type.name.lower()}_accuracy", acc_per_type, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True)

    def training_epoch_end(self, outputs):
        # MNLI
        mnli_results = outputs

        types = torch.cat([x["mnli_datapoint_type"] for x in mnli_results]).detach().cpu().numpy()
        losses = torch.cat([x["mnli_datapoint_loss"] for x in mnli_results]).detach().cpu().numpy()
        true_preds = torch.cat([x["mnli_true_preds"] for x in mnli_results]).detach().cpu().numpy()

        self._log_mnli_metrics_per_sample_type("Train", types, losses, true_preds)
        self._log_loss_histogram(losses, "Train", "mnli_loss", log_df=True)

    def validation_epoch_end(self, outputs):
        # MNLI
        mnli_results = outputs[0]

        types = torch.cat([x["mnli_datapoint_type"] for x in mnli_results]).detach().cpu().numpy()
        losses = torch.cat([x["mnli_datapoint_loss"] for x in mnli_results]).detach().cpu().numpy()
        true_preds = torch.cat([x["mnli_true_preds"] for x in mnli_results]).detach().cpu().numpy()

        self._log_mnli_metrics_per_sample_type("Train", types, losses, true_preds)
        self._log_loss_histogram(losses, "Valid", "mnli_loss", log_df=False)

        # HANS
        hans_results = outputs[1]

        preds = torch.cat([x["preds"] for x in hans_results]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in hans_results]).detach().cpu().numpy()
        heuristics = torch.cat([x["heuristic"] for x in hans_results]).detach().cpu().numpy()
        losses = torch.cat([x["hans_loss"] for x in hans_results]).detach().cpu().numpy()
        loss = losses.mean()

        acc = (preds == labels).sum() / len(preds)
        self.log("Valid/hans_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Valid/hans_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Valid/hans_count", float(len(preds)), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._log_loss_histogram(losses, "Valid", "hans_loss", log_df=False)

        for target_label, label_description in enumerate(["entailment", "non_entailment"]):
            for heuristic_name, heuristic_idx in HEURISTIC_TO_INTEGER.items():
                mask = (heuristics == heuristic_idx) & (labels == target_label)
                if mask.sum() == 0:
                    # that way we avoid NaN and polluting our metrics
                    continue

                loss = losses[mask].mean()
                acc = (preds[mask] == labels[mask]).mean()
                self.log(f"Valid/Hans_loss/{label_description}_{heuristic_name}", loss, on_step=False, on_epoch=True,
                         prog_bar=True,
                         logger=True)
                self.log(f"Valid/Hans_acc/{label_description}_{heuristic_name}", acc, on_step=False, on_epoch=True,
                         prog_bar=True, logger=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        model = self.bert
        no_decay = ["bias", "LayerNorm.weight"]
        normal_lr = ["classifier"]
        optimizer_grouped_parameters = [
            {
                "name": "1_w-decay_normal-lr",
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and any(nlr in n for nlr in normal_lr)
                ],
                "lr": self.hparams.learning_rate,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "name": "2_no-decay_normal-lr",
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and any(nlr in n for nlr in normal_lr)
                ],
                "lr": self.hparams.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "name": "3_w-decay_lower-lr",
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and not any(nlr in n for nlr in normal_lr)
                ],
                "lr": self.hparams.learning_rate / 100,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "name": "4_no-decay_lower-lr",
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and not any(nlr in n for nlr in normal_lr)
                ],
                "lr": self.hparams.learning_rate / 100,
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
