"""
Script that evaluates takes a BERT checkpoint from WANDB and evaluates it on HANS.
The HANS dataset is loaded by hand (i.e., without using 🤗HuggingFace datasets).

Run for example like: `python -m scripts.evaluate_bert_wandb_checkpoint_on_hans`
"""
import os.path
import os.path

import seaborn as sns
import wandb
from tqdm import tqdm

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

import torch
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from collections import OrderedDict


def load_bert_from_checkpoint(ckpt_path: str, device):
    # Load BERT's state dict from the PyTorch Lightning module
    pl_checkpoint = torch.load(ckpt_path, map_location=device)
    bert_state_dict = OrderedDict()
    for k, v in pl_checkpoint["state_dict"].items():
        if not k.startswith("bert."):
            print(f"Skipping key `{k}` in the PyTorch Lightning state dict")
            continue
        bert_state_dict[k[len("bert."):]] = v

    # Load BERT from the state dict
    print("*** Creating a clean BERT (Ignore warnings) ***")
    bert: BertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                                             num_labels=3)
    print("*** Loading BERT weights from checkpoint (There should be no warnings) ***")
    bert.load_state_dict(bert_state_dict)
    print("*** Loaded ***")
    return bert


import requests
from os import makedirs
from os.path import join, exists
import logging
from os.path import dirname
from collections import namedtuple
from typing import List


def ensure_dir_exists(filename):
    """Make sure the parent directory of `filename` exists"""
    makedirs(dirname(filename), exist_ok=True)


def download_to_file(url, output_file):
    """Download `url` to `output_file`, intended for small files."""
    ensure_dir_exists(output_file)
    with requests.get(url) as r:
        r.raise_for_status()
        with open(output_file, 'wb') as f:
            f.write(r.content)


SOURCE_DIR = "./data"

# Directory to store HANS data
HANS_SOURCE = join(SOURCE_DIR, "hans")

HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"

TextPairExample = namedtuple("TextPairExample", ["id", "premise", "hypothesis", "label"])


def load_hans() -> List[TextPairExample]:
    out = []

    logging.info("Loading hans...")
    src = join(HANS_SOURCE, "heuristics_evaluation_set.txt")
    if not exists(src):
        logging.info("Downloading source to %s..." % HANS_SOURCE)
        download_to_file(HANS_URL, src)

    with open(src, "r") as f:
        f.readline()
        lines = f.readlines()

    for line in lines:
        parts = line.split("\t")
        label = parts[0]
        if label == "entailment":
            label = 0
        elif label == "non-entailment":
            label = 1
        else:
            raise RuntimeError()
        s1, s2, pair_id = parts[5:8]
        out.append(TextPairExample(pair_id, s1, s2, label))
    return out


hans = load_hans()
print(hans[0])

from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorWithPadding

tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("bert-base-uncased")
collator = DataCollatorWithPadding(tokenizer, padding='longest', return_tensors="pt")
collator_fn = lambda x: collator(x).data

from math import ceil


def evaluate_bert_on_hans(bert, batch_size=256):
    csv_str = ""
    correct = 0
    total = 0
    pbar = tqdm(range(ceil(len(hans) / batch_size)))
    for batch_idx in pbar:
        batch = hans[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        hypotheses = [hans_datapoint.hypothesis for hans_datapoint in batch]
        premises = [hans_datapoint.premise for hans_datapoint in batch]
        labels = [hans_datapoint.label for hans_datapoint in batch]
        labels = torch.tensor(labels).to(bert.device)

        model_inputs = collator_fn(tokenizer(premises, hypotheses))
        model_inputs = {k: v.to(bert.device) for k, v in model_inputs.items()}
        output: SequenceClassifierOutput = bert.forward(**model_inputs)

        prob = output.logits.softmax(-1).detach().clone()

        # **************  This resolves an important bug  **************
        # pred = output.logits.argmax(dim=-1)
        prob[:, 1] += prob[:, 2]
        prob = prob[:, :2]
        pred = prob.argmax(dim=-1)
        # **************************************************************

        true_pred = (pred == labels).float()
        true_prob = prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        for j in range(len(batch)):
            csv_str += f"{batch[j].id}\t{pred[j].item()}\t{prob[j].tolist()}\n"

        correct += true_pred.sum()
        total += len(true_pred)
        pbar.set_description(f"{correct}/{total} :: {correct / total * 100:.2f}%")

    from datetime import datetime
    out_file = f"hans_result__{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}.csv"
    with open(out_file, "w") as f:
        f.write(csv_str)
    # from google.colab import files
    # files.download(out_file)

    return (correct / total).item()


def evaluate_runpath_on_hans(run_path, device="cuda"):
    wapi = wandb.Api()
    wrun = wapi.run(run_path)
    ckpt_files = [f for f in wrun.files() if "ckpt" in f.name]
    for ckpt_file in ckpt_files:
        if not os.path.exists(ckpt_file.name):
            ckpt_file.download("./")

    best, last = (f.name for f in ckpt_files)
    assert "last" in last
    print(f"best --> {best}")
    print(f"last --> {last}")
    bert_best = load_bert_from_checkpoint(best, device).to(device)
    bert_last = load_bert_from_checkpoint(last, device).to(device)

    best_acc = evaluate_bert_on_hans(bert_best)
    last_acc = evaluate_bert_on_hans(bert_last)

    print()
    print(f"Best: {best_acc * 100:.4f}")
    print(f"Last: {last_acc * 100:.4f}")

    return {"best_acc": best_acc, "last_acc": last_acc}


run_path = "user72/bertfornli-test/S7.03_gamma-0.0_adamw-1e-06_lr-2e-05_e-10_precision-16_08.28_17.55.48"

# evaluate_runpath_on_hans(run_path)
#
# ### S1 and S2 TURNED AROUND
# ## With bug (+ HANS labels flipped)
# # Best: 52.1000
# # Last: 46.9933
#
# ## With bug:
# # Best: 47.4867
# # Last: 44.9233
#
# ## Without bug:
# # Best: 47.7567
# # Last: 52.3100
#
# ### S1 and S2 in correct order
# ## With bug:
# # Best: 49.8800
# # Last: 52.7867
#
# ## Without bug:
# # Best: 52.5100
# # Last: 65.4967
#
# run_path = "user72/bertfornli-exp1/S10.02_gamma-10.0_adamw-1e-06_bs-1x32_lr-2e-05_wd-0.01_e-10_prec-16_08.29_00.07.17"
#
# evaluate_runpath_on_hans(run_path)
#
# ### S1 and S2 TURNED AROUND
# ## With bug
# # Best: 47.8200
# # Last: 47.0633
#
# ## Without bug
# # Best: 49.8800
# # Last: 48.1533
#
# ### S1 and S2 in correct order
# ## With bug:
# # Best: 49.9833
# # Last: 49.3567
#
# ## Without bug:
# # Best: 54.5367
# # Last: 57.3133
#
# run_path = "epfl-optml/bertfornli-exp1/e-03__bs-32_09.24_02.01.13"
# evaluate_runpath_on_hans(run_path, device="cuda")
# # Best: 56.3900
# # Last: 60.1733
#
# run_path = "epfl-optml/bertfornli-exp1/e-03__bs-08_09.24_03.22.44"
# evaluate_runpath_on_hans(run_path, device="cuda")
# # Best: 54.6933
# # Last: 60.1733


run_paths = [
    "epfl-optml/nli/S1.01.A_e-03_model-bert_dataset-mnli_gamma-0.0_seed-72_09.25_01.53.02",
    "epfl-optml/nli/S1.01.B_e-04__model-bert_dataset-mnli_gamma-0.0_seed-24_09.25_01.53.02",
    "epfl-optml/nli/S1.01.C_e-10__model-bert_dataset-mnli_gamma-0.0_seed-24_09.25_01.53.02",
    "epfl-optml/nli/S1.01.C_e-10__model-bert_dataset-mnli_gamma-0.0_seed-72_09.25_01.53.02",
    "epfl-optml/nli/S1.01.D_e-3_p-32__model-bert_dataset-mnli_gamma-0.0_seed-24_09.25_01.53.02",
    "epfl-optml/nli/S1.01.E_eps-8_model-bert_dataset-mnli_gamma-0.0_seed-24_09.25_01.53.02",
    "epfl-optml/nli/S1.01.F_mahabadi_eps8-bs8-wmp0-wd0-p32__model-bert_dataset-mnli_gamma-0.0_seed-24_09.25_01.53.02",
    "epfl-optml/nli/S1.01.G_clark_lr-5e-5__model-bert_dataset-mnli_gamma-0.0_seed-24_09.25_01.53.02",
    "epfl-optml/nli/S1.02_model-bert_dataset-mnli_gamma-0.5_seed-72_09.25_01.53.02",
    "epfl-optml/nli/S1.03_model-bert_dataset-mnli_gamma-1.0_seed-72_09.25_01.53.02",
    "epfl-optml/nli/S1.04_model-bert_dataset-mnli_gamma-2.0_seed-72_09.25_01.53.04",
    "epfl-optml/nli/S1.05_model-bert_dataset-mnli_gamma-5.0_seed-72_09.25_01.53.12",
    "epfl-optml/nli/S1.06_model-bert_dataset-mnli_gamma-10.0_seed-72_09.25_01.53.12",
    "epfl-optml/nli/S1.07_model-bert_dataset-snli_gamma-0.0_seed-72_09.25_01.53.12",
    "epfl-optml/nli/S1.08_model-bert_dataset-snli_gamma-0.5_seed-72_09.25_01.53.32",
    "epfl-optml/nli/S1.09_model-bert_dataset-snli_gamma-1.0_seed-72_09.25_02.57.08",
    "epfl-optml/nli/S1.10_model-bert_dataset-snli_gamma-2.0_seed-72_09.25_02.57.08",
    "epfl-optml/nli/S1.11_model-bert_dataset-snli_gamma-5.0_seed-72_09.25_03.02.37",
    "epfl-optml/nli/S1.12_model-bert_dataset-snli_gamma-10.0_seed-72_09.25_03.16.14",
]
results = {}
for run_path in run_paths:
    print(f"*" * 72)
    print(f"run_path={run_path}")
    print(f"*" * 72)
    results[run_path] = evaluate_runpath_on_hans(run_path, device="cuda")

print(results)
print()

print("run_path,best_acc,last_acc")
for run_path, x in results.items():
    print(f"{run_path},{x['best_acc']},{x['last_acc']}")
