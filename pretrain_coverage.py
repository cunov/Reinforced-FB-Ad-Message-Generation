from torch.utils.data import DataLoader
import pandas as pd, numpy as np
import torch
from ast import literal_eval
from tqdm import tqdm
# import time
import os
import argparse
from model_coverage import KeywordCoverage
from parameters import Parameters

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_epochs', type=int, default=15)
parser.add_argument('--learning_rate', type=float, default=5e-5)

args = parser.parse_args()
batch_size = args.batch_size
n_epochs = args.n_epochs
learning_rate = args.learning_rate
optim_every = int(32 / batch_size)

params = Parameters()

def collate_func(triplets):
    return list(zip(*triplets))
    # return [desc for desc,msg,loc in triplets], [msg for desc,msg,loc in triplets], [loc for desc,msg,loc in triplets]

df = pd.read_csv(params.dataset_filename, low_memory=False)
good_idxs = df.index[np.array([isinstance(x, str) for x in df.desc_clean_msg_stripped]) * np.array([isinstance(x,str) for x in df.msg_clean])]
descs = df.desc_clean_msg_stripped[good_idxs]
msgs = df.msg_clean[good_idxs]
locs = [literal_eval(x) for x in df.desc_clean_msg_stripped_locs[good_idxs]]

kw_cov = KeywordCoverage(_type='default', descs=list(descs))
                         
dataset = [(desc, msg, loc) for desc,msg,loc in zip(descs, msgs, locs)]
train_test_split = 0.8
N_train = int(len(dataset) * train_test_split)
N_val = len(dataset) - N_train
d_train, d_val = torch.utils.data.dataset.random_split(dataset, [N_train, N_val])
dl_train = DataLoader(dataset=d_train, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_func)
val_batch_size = 8
dl_val = DataLoader(dataset=d_val, batch_size=val_batch_size, shuffle=True, drop_last=True, collate_fn=collate_func)

param_optimizer = list(kw_cov.bert.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
scheduler = None
milestones = list(range(5, n_epochs))
gamma = 0.8
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


if not os.path.isdir(params.model_today_dir):
    os.mkdir(params.model_today_dir)
if not os.path.isdir(params.base_coverage_dir):
    os.mkdir(params.base_coverage_dir)

params.write_params(params.base_coverage_dir)

if scheduler is None:
    lines = ['\nLearning Rate = {}, init = {}'.format(learning_rate, kw_cov.init)]
else:
    lines = ['\nLearning rate = {}\t milestones = {}\tgamma = {}, init = {}'.format(learning_rate, milestones, gamma, kw_cov.init)]
params.write_params(params.base_coverage_dir, lines)

lines = ['\nepoch,avg_train_loss,avg_train_acc_with,avg_train_acc_without,avg_val_loss,avg_val_acc_with,avg_val_acc_without']
params.write_params(params.base_coverage_dir, lines)

kw_cov.tokenizer.save_pretrained(params.coverage_tokenizer_dir)
for i_epoch in range(n_epochs):
    train_loss = 0
    train_right_with = 0
    train_masks_with = 0
    train_right_without = 0
    train_masks_without = 0
    
    train_scores_without = []
    kw_cov.bert.train()
    for ib, batch in enumerate(tqdm(dl_train, desc='Train Epoch #{} - LR={}'.format(i_epoch, optimizer.param_groups[0]["lr"]))):
        contents, summaries, locs = batch

        loss, n_right, n_masks = kw_cov.get_loss_acc(contents, summaries, locs)

        loss.backward()
        train_loss += float(loss.item())
        train_right_with += float(n_right)
        train_masks_with += float(n_masks)

        if int(ib+1) % optim_every == 0:
            _, n_right, n_masks = kw_cov.get_loss_acc(contents, [''] * len(contents), locs, grad_enabled=False)
            train_right_without += float(n_right)
            train_masks_without += float(n_masks)
            
            optimizer.step()
            optimizer.zero_grad()


    avg_train_loss = train_loss / len(dl_train)
    avg_train_acc_with = train_right_with / train_masks_with
    avg_train_acc_without = train_right_without / train_masks_without

    val_loss = 0
    val_right_with = 0
    val_masks_with = 0
    val_right_without = 0
    val_masks_without = 0
    kw_cov.bert.eval()
    with torch.no_grad():
        for ib, batch in enumerate(tqdm(dl_val, desc='Val Epoch #{} - LR={}'.format(i_epoch, optimizer.param_groups[0]["lr"]))):
            contents, summaries, locs = batch
            loss, n_right, n_masks = kw_cov.get_loss_acc(contents, summaries, locs, grad_enabled=False)
            val_loss += float(loss.item())
            val_right_with += float(n_right)
            val_masks_with += float(n_masks)

            _, n_right, n_masks = kw_cov.get_loss_acc(contents, [''] * len(contents), locs, grad_enabled=False)
            val_right_without += float(n_right)
            val_masks_without += float(n_masks)

    avg_val_loss = val_loss / len(dl_val)
    avg_val_acc_with = val_right_with / val_masks_with
    avg_val_acc_without = val_right_without / val_masks_without

    lines = ['\n'+','.join([str(item) for item in [i_epoch, avg_train_loss, avg_train_acc_with, avg_train_acc_without, avg_val_loss, avg_val_acc_with, avg_val_acc_without]])]
    params.write_params(params.base_coverage_dir, lines)
    print(lines)
    kw_cov.bert.save_pretrained(params.coverage_model_dir.format(i_epoch))
    if scheduler is not None:
        scheduler.step()