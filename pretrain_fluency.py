import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import pandas as pd, numpy as np
from tqdm import tqdm
import argparse
import os
from model_generator import Fluency
from parameters import Parameters


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--train_test_split', type=float, default=0.8)
parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=5e-5)

args = parser.parse_args()
train_test_split = args.train_test_split
batch_size = args.batch_size
n_epochs = args.n_epochs
learning_rate = args.learning_rate
optim_every = int(32 / batch_size)

milestones = list(range(7, n_epochs))
gamma = 0.95

params = Parameters()

df = pd.read_csv(params.dataset_filename, low_memory=False)
msgs = [msg for msg in df.msg_clean.tolist() if isinstance(msg, str) and len(msg.split(' ')) > 10]

N = len(msgs)
N_train = int(N * train_test_split)
N_val = N - N_train
d_train, d_val = torch.utils.data.dataset.random_split(msgs, [N_train, N_val])

fluency = Fluency(load_pretrained='fluency')

class Dataset:
    def __init__(self, msgs):
        inputs, outputs = fluency.preprocess_input(msgs)
        self.input_ids = inputs.input_ids
        self.attn_mask = inputs.attention_mask
        self.outputs = outputs.input_ids
        
    def __getitem__(self, i):
        return self.input_ids[i].to(params.device), self.attn_mask[i].to(params.device), self.outputs[i].to(params.device)
    
    def __len__(self):
        return len(self.outputs)

print('Creating train dataset...',end='')
dataset_train = Dataset(list(d_train))
print('done\nCreating val dataset...',end='')
dataset_val = Dataset(list(d_val))
print('done\nCreating dataloaders...',end='')
dl_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
dl_val = DataLoader(dataset=dataset_val, batch_size=32, shuffle=True, drop_last=True)

crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.AdamW(fluency.model.parameters(), lr=learning_rate)
scheduler = None
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

if not os.path.isdir(params.model_today_dir):
    os.mkdir(params.model_today_dir)
if not os.path.isdir(params.base_fluency_dir):
    os.mkdir(params.base_fluency_dir)

params.write_params(params.base_fluency_dir)
if scheduler is not None:
    lines = ['\nlearning rate = {}\t milestones = {}\tgamma = {}'.format(learning_rate, milestones, gamma)]
else:
    lines = ['\nlearning rate = {}'.format(learning_rate)]
params.write_params(params.base_fluency_dir, lines)
fluency.tokenizer.save_pretrained(params.fluency_tokenizer_dir)
lines = ['\nepoch,train_loss,val_loss']
params.write_params(params.base_fluency_dir, lines)
for i_epoch in range(n_epochs):
    total_train_loss = 0
    total_val_loss = 0
    fluency.model.train()
    for ib, batch in enumerate(tqdm(dl_train, desc='Train, epoch #{} - LR={}'.format(i_epoch, optimizer.param_groups[0]["lr"]))):
        input_ids, attn_mask, outputs = batch
        res = fluency.model(input_ids=input_ids, attention_mask=attn_mask)
        logits = res.logits
        loss = crit(logits.view(-1,len(fluency.tokenizer)), outputs.view(-1))
        loss.backward()
        total_train_loss += loss.item()

        if int(ib + 1) % optim_every == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    if scheduler is not None:
        scheduler.step()
    avg_train_loss = total_train_loss / len(dl_train)
    
    fluency.model.eval()
    with torch.no_grad():
        for ib, batch in enumerate(tqdm(dl_val, desc='Val, epoch #{}'.format(i_epoch))):
            input_ids, attn_mask, outputs = batch
            res = fluency.model(input_ids=input_ids, attention_mask=attn_mask)
            logits = res.logits
            loss = crit(logits.view(-1, len(fluency.tokenizer)), outputs.view(-1))
            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(dl_val)

    lines = ['\n'+','.join([str(item) for item in [i_epoch, avg_train_loss, avg_val_loss]])]
    params.write_params(params.base_fluency_dir, lines)
    print(lines)
    fluency.model.save_pretrained(params.fluency_model_dir.format(i_epoch))