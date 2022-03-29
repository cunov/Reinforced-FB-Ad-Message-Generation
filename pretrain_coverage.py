from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, BertTokenizer, BertForMaskedLM
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import DataLoader, RandomSampler
import pandas as pd, numpy as np
import torch
from ast import literal_eval
from tqdm import tqdm
import datetime
import sys
import os
# import multiprocessing as mp
# from IPython.display import clear_output
# import time
# import matplotlib.pyplot as plt
import argparse
from model_coverage import KeywordCoverage
from parameters import Parameters

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=2e-5)

args = parser.parse_args()
batch_size = args.batch_size
n_epochs = args.n_epochs
learning_rate = args.learning_rate
optim_every = int(32 / batch_size)

params = Parameters()

def collate_func(triplets):
    return [desc for desc,msg,loc in triplets], [msg for desc,msg,loc in triplets], [loc for desc,msg,loc in triplets]


kw_cov = KeywordCoverage()
                         
                         
df = pd.read_csv(params.dataset_filename, low_memory=False)
good_idxs = df.index[np.array([isinstance(x, str) for x in df.desc_clean_msg_stripped]) * np.array([isinstance(x,str) for x in df.msg_clean])]
descs = df.desc_clean_msg_stripped[good_idxs]
msgs = df.msg_clean[good_idxs]
locs = [literal_eval(x) for x in df.desc_clean_msg_stripped_locs[good_idxs]]

dataset = [(desc, msg, loc) for desc,msg,loc in zip(descs, msgs, locs)]
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=RandomSampler(dataset), drop_last=True, collate_fn=collate_func)

optimizer = torch.optim.AdamW(kw_cov.bert.parameters(), lr=learning_rate)

if not os.path.isdir(params.model_today_dir):
    os.mkdir(params.model_today_dir)
if not os.path.isdir(params.base_coverage_dir):
    os.mkdir(params.base_coverage_dir)

kw_cov.bert.train()
n_items = batch_size * len(dataloader)
params.write_params(params.base_coverage_dir)
kw_cov.tokenizer.save_pretrained(params.coverage_tokenizer_dir)
lines = ['\nepoch,avg_loss,avg_acc']
params.write_params(params.base_coverage_dir, lines)
for i_epoch in range(n_epochs):
    total_loss = 0
    total_acc = 0
    for ib, batch in enumerate(tqdm(dataloader, desc='Epoch #{}'.format(i_epoch))):
        contents, summaries, locs = batch
        loss, acc = kw_cov.get_loss_acc(contents, summaries, locs)
        loss.backward()
        total_loss += loss.item()
        total_acc += acc
        if ib % optim_every == 0:
            optimizer.step()
            optimizer.zero_grad()

    avg_loss = total_loss / n_items
    avg_acc = total_acc / n_items
    lines = ['\n'+','.join([str(item) for item in [i_epoch, avg_loss, avg_acc]])]
    params.write_params(params.base_coverage_dir, lines)
    print(lines)
    kw_cov.bert.save_pretrained(params.coverage_model_dir.format(i_epoch))