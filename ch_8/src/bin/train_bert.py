import argparse
import pathlib
import sys

import pandas as pd
import torch
from transformers import AdamW
from transformers import get_constant_schedule_with_warmup, get_constant_schedule
import transformers

from sklearn import metrics
from sklearn import model_selection

sys.path.append("./../../")
from src.utils import io
from src.datasets import bert_dataset
from src.models import bert
from src.bin import engine_bert


device = "cuda" if torch.cuda.is_available() else "cpu"

def main(config):
    """
        Entry point for training
        :param config: A SimpleNamespace config object
    """
    df = pd.read_csv(config.train_filepath, usecols=config.data["columns"])
    feats, labels = df.loc[:,"feature"].values, df.loc[:,"labels"].values
    feats_train, feats_val, labels_train, labels_val = model_selection.train_test_split(feats, labels, train_size=0.02, random_state=42, stratify=labels)

    feats_label_pair_train = zip(feats_train, labels_train)
    feats_label_pair_val = zip(feats_val, labels_val)

    tokenizer = transformers.BertTokenizer.from_pretrained(config.bert_model, do_lower=True)

    train_dataset = bert_dataset.BertClassificationDataset(feats_label_pair_train, tokenizer)
    val_dataset = bert_dataset.BertClassificationDataset(feats_label_pair_val, tokenizer)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=True, num_workers=config.num_workers)
    
    model = bert.BertClassificationModel(config)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    ##Optimizer
    params_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params":[p for n,p in params_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay":0.001
        },
        {
            "params":[p for n,p in params_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay":0.0
        }
    ]

    #num_train_steps = int(len(train_data_loader)*config.epochs/config.train_batch_size)

    optimizer = AdamW(optimizer_parameters, lr=config.lr)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0, last_epoch=-1)

    for epoch in range(config.epochs):
        engine_bert.train_step(train_data_loader, model, optimizer, scheduler, criterion, device, epoch)
        true_labels, pred_labels = engine_bert.eval_step(val_data_loader, model, device, epoch)

        roc_auc = metrics.roc_auc_score(true_labels, pred_labels)
        print (f"epoch: {epoch}, val auc_roc is : {roc_auc}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Use this script to train bert on Indiamart dataset")
    parser.add_argument("config_filepath", metavar="fc", type=pathlib.Path, help="Configuration filepath")
    args = parser.parse_args()

    config = io.load_config(args.config_filepath)
    main(config)
