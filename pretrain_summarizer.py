from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler
import torch
import pandas as pd, numpy as np
import datetime
from tqdm import tqdm
import argparse
from parameters import Parameters
import os

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--train_batch_size', type=int, default=2)
args = parser.parse_args()
n_epochs = args.n_epochs
train_batch_size = args.train_batch_size

params = Parameters()
max_input_length = params.max_input_length
max_output_length = params.max_output_length

bos_token, eos_token, pad_token ='<|startoftext|>', '<|endoftext|>', '<|pad|>'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos_token, eos_token=eos_token, pad_token=pad_token) #gpt2-medium
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
model.resize_token_embeddings(len(tokenizer))

device = 'cuda'
model.to(device)

df = pd.read_csv('C:/Users/Colton/OneDrive/School/Thesis/Adfenix/play/df_masks.csv', low_memory=False)
descs = [x for x in df.desc_clean if isinstance(x, str)] # truncate like cut300?

def preprocess_batch(descs, device=device, max_input_len=max_input_length, max_output_len=max_output_length):
    bodies = descs
    summaries = descs
    inputs = tokenizer(bodies, truncation=True, max_length=max_input_len)
    summ_inp = tokenizer([tokenizer.bos_token + summ for summ in summaries], truncation=True, max_length=max_output_len)
    inputs['input_ids'] = [torch.LongTensor(input_id + summ_in) for input_id,summ_in in zip(inputs.input_ids, summ_inp.input_ids)]
    inputs['attention_mask'] = [torch.LongTensor(attn_input + attn_summ) for attn_input,attn_summ in zip(inputs.attention_mask, summ_inp.attention_mask)]
    inputs['input_ids'] = torch.nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    inputs['attention_mask'] = torch.nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    targets = [tokenizer.encode(summ, truncation=True, max_length=max_output_len-1) for summ in summaries]
    targets = [torch.LongTensor(input_id + [tokenizer.eos_token_id]) for input_id in targets]
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)

    return inputs.to(device), targets.to(device)

train_test_split = 0.8
val_batch_size = 4
learning_rate = 2e-5
optim_every = int(32 / train_batch_size)

N = len(descs)
N_train = int(N * train_test_split)
N_val = N - N_train
d_train, d_val = torch.utils.data.dataset.random_split(descs, [N_train, N_val])

dl_train = DataLoader(dataset=d_train, batch_size=train_batch_size, sampler=RandomSampler(d_train), collate_fn=preprocess_batch)
dl_val = DataLoader(dataset=d_val, batch_size=val_batch_size, sampler=RandomSampler(d_val), collate_fn=preprocess_batch)

crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

if not os.path.isdir(params.model_today_dir):
    os.mkdir(params.model_today_dir)
if not os.path.isdir(params.base_summarizer_dir):
    os.mkdir(params.base_summarizer_dir)


params.write_params(params.base_summarizer_dir)
tokenizer.save_pretrained(params.summarizer_tokenizer_dir)
lines = ['\nepoch,train_loss,val_loss']
params.write_params(params.base_summarizer_dir, lines)

for i_epoch in range(n_epochs):
    total_train_loss = 0
    total_val_loss = 0
    
    model.train()
    for ib, batch in enumerate(tqdm(dl_train, desc='Train, epoch #{} - LR={}'.format(i_epoch, optimizer.param_groups[0]["lr"]))):
        inputs, targets = batch
        
        outputs = model(**inputs)
        logits = outputs.logits
        logits_shifted = logits[:,-targets.shape[1]:]
        loss = crit(logits_shifted.contiguous().view(-1, len(tokenizer)), targets.contiguous().view(-1))
        loss.backward()
        total_train_loss += loss.item()
        if ib % optim_every == 0:
            optimizer.step()
            optimizer.zero_grad()
    avg_train_loss = total_train_loss / (train_batch_size * len(dl_train))
    
    model.eval()
    with torch.no_grad():
        for ib, batch in enumerate(tqdm(dl_val, desc='Val, epoch #{}'.format(i_epoch))):
            inputs, targets = batch

            outputs = model(**inputs)
            logits = outputs.logits
            logits_shifted = logits[:,-targets.shape[1]:]
            loss = crit(logits_shifted.contiguous().view(-1, len(tokenizer)), targets.contiguous().view(-1))

            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / (val_batch_size * len(dl_val))
        
    lines = ['\n' + ','.join([str(x) for x in [i_epoch, avg_train_loss, avg_val_loss]])]
    print(lines)
    lines.append(lines)
    model.save_pretrained(params.summarizer_model_dir.format(i_epoch))
    params.write_params(params.base_summarizer_dir, lines)