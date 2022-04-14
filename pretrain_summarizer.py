from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm
import argparse
from parameters import Parameters
import os
# from rouge import Rouge

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--train_batch_size', type=int, default=4)
parser.add_argument('--model_init', type=bool, default=False)

args = parser.parse_args()
n_epochs = args.n_epochs
train_batch_size = args.train_batch_size
model_init = args.model_init

params = Parameters()
max_input_length = params.max_input_length
max_output_length = params.max_output_length - 15

if not model_init:
    bos_token, eos_token, pad_token ='<|startoftext|>', '<|endoftext|>', '<|pad|>'
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', bos_token=bos_token, eos_token=eos_token, pad_token=pad_token)
    print('Loading default model...',end='')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.resize_token_embeddings(len(tokenizer))
    print('done')
else:
    print('Loading previously trained model, tokenizer...',end='')
    model = GPT2LMHeadModel.from_pretrained(params.trained_summarizer_model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(params.trained_summarizer_tokenizer_dir)
    print('done')

device = 'cuda'
model.to(device)

df = pd.read_csv(params.dataset_filename, low_memory=False)
descs = [' '.join(x.split(' ')[:500]) for x in df.desc_clean if isinstance(x, str)]

def preprocess_batch(descs, device=device, max_input_len=max_input_length, max_output_len=max_output_length):
    inputs = tokenizer(descs, truncation=True, max_length=max_input_len)
    summ_inp = tokenizer([tokenizer.bos_token + desc for desc in descs], truncation=True, max_length=max_output_len)
    inputs['input_ids'] = [torch.LongTensor(input_id + summ_in) for input_id,summ_in in zip(inputs.input_ids, summ_inp.input_ids)]
    inputs['attention_mask'] = [torch.LongTensor(attn_input + attn_summ) for attn_input,attn_summ in zip(inputs.attention_mask, summ_inp.attention_mask)]
    inputs['input_ids'] = torch.nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    inputs['attention_mask'] = torch.nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True, padding_value=0)

    outputs = inputs['input_ids']
    bos_token_idx = torch.where(outputs.eq(tokenizer.bos_token_id))[1].tolist()
    first_pad_idx = [torch.where(inpid.eq(tokenizer.pad_token_id))[0][0].item() if (tokenizer.pad_token_id in inpid) else len(inpid) for inpid in outputs]
    targets = []
    for i,output in enumerate(outputs):
        targets.append(torch.cat((output[bos_token_idx[i]+1:first_pad_idx[i]], torch.IntTensor([tokenizer.eos_token_id]))))
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)

    return inputs.to(device), targets.to(device)

train_test_split = 0.9
val_batch_size = 6
learning_rate = 2e-5
optim_every = int(32 / train_batch_size)
# LR = 1e-4: val loss epoch #0 = 2.06, epoch #1 = 0.73

N = len(descs)
N_train = int(N * train_test_split)
N_val = N - N_train
d_train, d_val = torch.utils.data.dataset.random_split(descs, [N_train, N_val])

class Dataset:
    def __init__(self, descs):
        inputs, targets = preprocess_batch(descs, device='cpu')
        self.input_ids = [inpid for inpid in inputs.input_ids]
        self.attn_masks = [attn_mask for attn_mask in inputs.attention_mask]
        self.targets = targets
        
    def __getitem__(self, i):
        return self.input_ids[i].to(params.device), self.attn_masks[i].to(params.device), self.targets[i].to(params.device)
    
    def __len__(self):
        return len(self.targets)

print('Creating train dataset...',end='')
dataset_train = Dataset(list(d_train))
print('done\nCreating val dataset...',end='')
dataset_val = Dataset(list(d_val))
print('done\nCreating dataloaders...',end='')
dl_train = DataLoader(dataset=dataset_train, batch_size=train_batch_size, drop_last=True, shuffle=True)
dl_val = DataLoader(dataset=dataset_val, batch_size=val_batch_size, drop_last=True, shuffle=True)
print('done')


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
scheduler = None
milestones = list(range(5,n_epochs))
gamma = 0.9
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

if not os.path.isdir(params.model_today_dir):
    os.mkdir(params.model_today_dir)
if not os.path.isdir(params.base_summarizer_dir):
    os.mkdir(params.base_summarizer_dir)


params.write_params(params.base_summarizer_dir)
if scheduler is None:
    lines = ['\nLearning Rate = {}'.format(learning_rate)]
else:
    lines = ['\nLearning Rate = {}\nMilestones = {}\nGamma = {}'.format(learning_rate, milestones, gamma)]
params.write_params(params.base_summarizer_dir, lines)
lines = ['\nepoch,train_loss,val_loss']
params.write_params(params.base_summarizer_dir, lines)

tokenizer.save_pretrained(params.summarizer_tokenizer_dir)
# rouge = Rouge()

for i_epoch in range(n_epochs):
    total_train_loss = 0
    total_val_loss = 0
    
    model.train()
    for ib, batch in enumerate(tqdm(dl_train, desc='Train, epoch #{} - LR={}'.format(i_epoch, optimizer.param_groups[0]["lr"]))):
        input_ids, attn_mask, targets = batch # 5%
        outputs = model(input_ids=input_ids, attention_mask=attn_mask) # 25%
        
        logits = outputs.logits 
        logits_shifted = logits[:,-targets.shape[1]:]
        loss = crit(logits_shifted.contiguous().view(-1, len(tokenizer)), targets.contiguous().view(-1))
        
        loss.backward() # 70%

        total_train_loss += loss.item()
        if ib % optim_every == 0:
            optimizer.step()
            optimizer.zero_grad()

    avg_train_loss = total_train_loss / len(dl_train)

    model.eval()
    with torch.no_grad():
        for ib, batch in enumerate(tqdm(dl_val, desc='Val, epoch #{}'.format(i_epoch))):
            input_ids, attn_mask, targets = batch

            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits
            logits_shifted = logits[:,-targets.shape[1]:]
            loss = crit(logits_shifted.contiguous().view(-1, len(tokenizer)), targets.contiguous().view(-1))

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(dl_val)
        
    lines = ['\n' + ','.join([str(x) for x in [i_epoch, avg_train_loss, avg_val_loss]])]
    print(lines)
    model.save_pretrained(params.summarizer_model_dir.format(i_epoch))
    params.write_params(params.base_summarizer_dir, lines)

    if not scheduler is None:
        scheduler.step()