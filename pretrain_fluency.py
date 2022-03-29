from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler
import torch
import pandas as pd, numpy as np
import datetime
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
import argparse
import os
from model_generator import Fluency
from parameters import Parameters


## WITH LR=1e-4, VAL LOSS BOTTOMS OUT AT EPOCH #8

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--train_test_split', type=float, default=0.8)
parser.add_argument('--n_epochs', type=int, default=15)
parser.add_argument('--learning_rate', type=float, default=1e-4)

args = parser.parse_args()
train_test_split = args.train_test_split
batch_size = args.batch_size
n_epochs = args.n_epochs
learning_rate = args.learning_rate
optim_every = int(32 / batch_size)

milestones = list(range(5, n_epochs))
gamma = 0.8

params = Parameters()

df = pd.read_csv(params.dataset_filename, low_memory=False)
msgs = [msg for msg in df.msg_clean.tolist() if isinstance(msg, str)]

N = len(msgs)
N_train = int(N * train_test_split)
N_val = N - N_train
d_train, d_val = torch.utils.data.dataset.random_split(msgs, [N_train, N_val])

dl_train = DataLoader(dataset=d_train, batch_size=batch_size, sampler=RandomSampler(d_train))
dl_val = DataLoader(dataset=d_val, batch_size=batch_size, sampler=RandomSampler(d_val))

fluency = Fluency()
crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.AdamW(fluency.model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

if not os.path.isdir(params.model_today_dir):
    os.mkdir(params.model_today_dir)
if not os.path.isdir(params.base_fluency_dir):
    os.mkdir(params.base_fluency_dir)

params.write_params(params.base_fluency_dir)
fluency.tokenizer.save_pretrained(params.fluency_tokenizer_dir)
lines = ['\nepoch,train_loss,val_loss']
params.write_params(params.base_fluency_dir, lines)
for i_epoch in range(n_epochs):
    total_train_loss = 0
    total_val_loss = 0
    
    fluency.model.train()
    for ib, batch in enumerate(tqdm(dl_train, desc='Train, epoch #{} - LR={}'.format(i_epoch, optimizer.param_groups[0]["lr"]))):
        inputs, outputs = fluency.preprocess_input(batch)
        res = fluency.model(**inputs)
        logits = res.logits
        loss = crit(logits.view(-1,len(fluency.tokenizer)), outputs.input_ids.view(-1))
        loss.backward()
        total_train_loss += loss.item()
        if ib % optim_every == 0:
            optimizer.step()
            optimizer.zero_grad()
    scheduler.step()
    avg_train_loss = total_train_loss / (batch_size * len(dl_train))
    
    fluency.model.eval()
    with torch.no_grad():
        for ib, batch in enumerate(tqdm(dl_val, desc='Val, epoch #{}'.format(i_epoch))):
            inputs, outputs = fluency.preprocess_input(batch)
            res = fluency.model(**inputs)
            logits = res.logits
            loss = crit(logits.view(-1, len(fluency.tokenizer)), outputs.input_ids.view(-1))
            total_val_loss += loss.item()
        avg_val_loss = total_val_loss / (batch_size * len(dl_val))

    lines = ['\n'+','.join([str(item) for item in [i_epoch, avg_train_loss, avg_val_loss]])]
    params.write_params(params.base_fluency_dir, lines)
    print(lines)
    fluency.model.save_pretrained(params.fluency_model_dir.format(i_epoch))