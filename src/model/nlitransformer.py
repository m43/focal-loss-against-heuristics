import torch
from pytorch_lightning import LightningModule

from torch import nn
from transformers import DistilBertPreTrainedModel, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from src.constants import HEURISTIC_TO_INTEGER
from src.model.focalloss import FocalLoss

PRETRAINED_MODEL_ID = "bert-base-uncased"


class BertForNLI(LightningModule):

    def __init__(self, gamma:float):
        super().__init__()

        self.bert_config = AutoModel.from_pretrained(PRETRAINED_MODEL_ID)
        self.bert: BertModel = AutoModel.from_pretrained(PRETRAINED_MODEL_ID)
        assert isinstance(self.bert, BertModel)

        self.classifier = nn.Linear(self.bert_config.dim, 3)

        # initialized in self.setup()
        self.total_steps = None
        self.loss_criterion = FocalLoss(gamma, reduction='mean')

    def forward(self, input_ids, attention_mask, token_type_ids, label=None, **kwargs):

        bert_output:BaseModelOutputWithPoolingAndCrossAttentions = self.bert.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls_repr = bert_output[1]

        logits = self.classifier(cls_repr)
        return logits

    def training_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        def hans_step(hans_batch):

            heuristics_masks = {
                'lexical_overlap': hans_batch['heuristic'] == HEURISTIC_TO_INTEGER['lexical_overlap'],
                'subsequence': hans_batch['heuristic'] == HEURISTIC_TO_INTEGER['subsequence'],
                'constituent': hans_batch['heuristic'] == HEURISTIC_TO_INTEGER['constituent']
            }

            logits = self.forward(**hans_batch)

            logits_per_heuristic = {k+"_logits" : logits[mask] for k, mask in heuristics_masks}
            labels_per_heuristic = {k+"_labels" : hans_batch['labels'][mask] for k, mask in heuristics_masks}

            return {**logits_per_heuristic, labels_per_heuristic}

        def mnli_step(mnli_batch):
            logits = self.forward(**mnli_batch)
            return {"mnli_logits" : logits, "mnli_label"}

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):

        # NOT DONE YET

        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        return loss

    def setup(self, stage=None) -> None:

        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps ???
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        # might want to give smaller learning rate to all parameters that were already trained

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

