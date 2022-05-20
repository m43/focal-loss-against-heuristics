import torch
from pytorch_lightning import LightningModule

from torch import nn
from transformers import DistilBertPreTrainedModel, DistilBertModel
from transformers import AutoModel

from src.model.nliembedding import NliEmbeddings

PRETRAINED_MODEL_ID = "distilbert-base-uncased"


class DistilBertForNLI(LightningModule):

    def __init__(self):
        super().__init__()

        self.distilbert_config = AutoModel.from_pretrained(PRETRAINED_MODEL_ID)
        self.distilbert: DistilBertModel = AutoModel.from_pretrained(PRETRAINED_MODEL_ID)
        assert isinstance(self.distilbert, DistilBertModel)

        self.embeddings = NliEmbeddings(src=self.distilbert.embeddings)

        self.classifier = nn.Linear(self.distilbert_config.dim, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, label=None):
        embeds = self.embeddings.forward(input_ids=input_ids, token_type_ids=token_type_ids)

        distilbert_output = self.distilbert.forward(
            inputs_embeds=embeds,
            attention_mask=attention_mask
        )

        cls_repr = distilbert_output['last_hidden_state'][:0]  # (bs, emb)
        logits = self.classifier(cls_repr)

        return logits

    def training_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        def hans_step(hans_batch):


        def mnli_step():
            ...


        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

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

        # NOT DONE YET

        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        # NOT DONE YET

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

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

