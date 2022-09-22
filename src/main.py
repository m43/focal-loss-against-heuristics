import argparse
import logging
from datetime import datetime

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from src.dataset.datamodule import ExperimentDataModule
from src.model.nlitransformer import BertForNLI, T5ForNLI
from src.utils.util import nice_print, HORSE, get_logger

log = get_logger(__name__)

logging.basicConfig(level=logging.INFO)


def get_parser_main_model():
    """
    Get the argument parser used to configure the run.

    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()

    # logging
    parser.add_argument('--experiment_name', type=str, default='bertfornli-exp1')
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Checkpoint used to restore training state")
    parser.add_argument('--experiment_version', type=str, default=None)

    # experiment configuration
    parser.add_argument('--model_name', type=str, default='BERT', help='name of the model (BERT or T5)')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=20, help='number of dataloader workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--early_stopping_patience', type=int, default=50)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    parser.add_argument('--num_hans_train_examples', type=int, default=0, help='number of HANS train examples')

    # model hparams
    parser.add_argument('--bert_hidden_dropout_prob', type=float, default=0.1,
                        help='The dropout probability for all fully connected layers in the embeddings, encoder, '
                             'and pooler.')
    parser.add_argument('--bert_attention_probs_dropout_prob', type=float, default=0.1,
                        help='The dropout ratio for the attention probabilities.')
    parser.add_argument('--bert_classifier_dropout', type=float, default=None,
                        help='The dropout ratio for the classification head.')
    parser.add_argument('--focal_loss_gamma', type=float, default=0.0, help='gamma used in focal loss')
    parser.add_argument('--optimizer_name', type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument('--scheduler_name', type=str, default="linear", choices=["linear", "polynomial"])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--warmup_steps', type=int, default=None, help='number of warmup steps')
    parser.add_argument('--warmup_ratio', type=float, default=None, help='ratio of warmup over all epochs')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--gradient_clip_val', type=float, default=None, help='gradient clipping value')
    parser.add_argument('--tokenizer_model_max_length', type=int, default=512, help='number of warmup steps')

    return parser


def main(config):
    """
    Configure and run the training and evaluation of the `model.nlitrasformer.BertForNLI` model.

    :param config: The run configuration.
    """
    # 0. Ensure reproducibility of results
    pl.seed_everything(72, workers=True)

    # 1. Prepare datamodule
    dm = ExperimentDataModule(
        model_name=config.model_name,
        batch_size=config.batch_size,
        num_hans_train_examples=config.num_hans_train_examples,
        num_workers=config.num_workers,
        tokenizer_model_max_length=config.tokenizer_model_max_length,
    )

    # 2. Prepare loggers
    if config.experiment_version is None:
        config.experiment_version = f"{config.experiment_name}" \
                                    f"__gamma={config.focal_loss_gamma}" \
                                    f"_e={config.n_epochs}" \
                                    f"_b={config.batch_size}" \
                                    f"_lr={config.lr}"
    config.experiment_version += f"_{datetime.now().strftime('%m.%d_%H.%M.%S')}"

    wandb_logger = WandbLogger(
        project=config.experiment_name,
        version=config.experiment_version.replace("=", "-"),
        settings=wandb.Settings(start_method='fork'),
    )
    tb_logger = TensorBoardLogger("logs", name=config.experiment_name, version=config.experiment_version, )
    csv_logger = CSVLogger("logs", name=config.experiment_name, version=config.experiment_version, )
    loggers = [wandb_logger, tb_logger, csv_logger]
    log.info(config)

    # 3. Prepare model
    if config.model_name == 'BERT':
        nlitransformer = BertForNLI(
            hidden_dropout_prob=config.bert_hidden_dropout_prob,
            attention_probs_dropout_prob=config.bert_attention_probs_dropout_prob,
            classifier_dropout=config.bert_classifier_dropout,
            focal_loss_gamma=config.focal_loss_gamma,
            learning_rate=config.lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            warmup_ratio=config.warmup_ratio,
            scheduler_name=config.scheduler_name,
            optimizer_name=config.optimizer_name,
        )
    elif config.model_name == 'T5':
        nlitransformer = T5ForNLI(
            focal_loss_gamma=config.focal_loss_gamma,
            learning_rate=config.lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            warmup_ratio=config.warmup_ratio,
            scheduler_name=config.scheduler_name,
            optimizer_name=config.optimizer_name,
        )
    else:
        raise ValueError(f"Model value must be 'BERT' or 'T5', '{config.model_name}' not recognized!")

    wandb_logger.watch(nlitransformer, log="all")

    # 4. Prepare callbacks
    early_stopping_metric = "Valid/mnli_validation_matched_loss_epoch"
    early_stopping_callback = EarlyStopping(
        monitor=early_stopping_metric, mode="min",
        patience=config.early_stopping_patience,
        check_on_train_epoch_end=False
    )
    model_checkpoint_callback = ModelCheckpoint(monitor=early_stopping_metric, save_last=True, verbose=True, )
    learning_rate_monitor_callback = LearningRateMonitor(logging_interval='step')
    callbacks = [
        early_stopping_callback,
        model_checkpoint_callback,
        learning_rate_monitor_callback,
    ]

    # 5. Run
    if torch.cuda.is_available() and config.gpus != 0:
        trainer = Trainer(
            max_epochs=config.n_epochs,
            default_root_dir="logs",
            logger=loggers,
            callbacks=callbacks,
            gradient_clip_val=config.gradient_clip_val,
            gpus=config.gpus,
            precision=config.precision,
            accelerator="gpu",
            # strategy="dp",
            strategy=DDPStrategy(process_group_backend="gloo"),
            accumulate_grad_batches=config.accumulate_grad_batches,
            val_check_interval=1 / 3,
            num_sanity_val_steps=0,

            # ~~~ Uncomment for fast debugging ~~~ #
            # limit_train_batches=50,
            # limit_val_batches=50,
        )
    else:
        print("\n\n*** Using CPU ***\n\n")
        trainer = Trainer(
            max_epochs=config.n_epochs,
            default_root_dir="logs",
            logger=loggers,
            callbacks=callbacks,
            # limit_train_batches=5,
            # limit_val_batches=5,
        )
    trainer.fit(nlitransformer, dm, ckpt_path=config.checkpoint_path)

    wandb.save(model_checkpoint_callback.best_model_path)
    wandb.save(model_checkpoint_callback.last_model_path)

    log.info(f"best_model_path={model_checkpoint_callback.best_model_path}")
    log.info(f"best_model_score={model_checkpoint_callback.best_model_score}")
    log.info(f"last_model_path={model_checkpoint_callback.last_model_path}")


if __name__ == "__main__":
    nice_print(HORSE)
    args = get_parser_main_model().parse_args()
    main(args)
