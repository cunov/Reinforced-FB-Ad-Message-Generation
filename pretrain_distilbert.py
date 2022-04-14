from transformers import AutoModelForMaskedLM, DistilBertTokenizer
import torch
from torch.utils.data import DataLoader, RandomSampler
import pandas as pd
import random
from tqdm import tqdm
import argparse
from parameters import Parameters
import os

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--train_test_split', type=float, default=0.9)

args = parser.parse_args()
learning_rate = args.learning_rate
n_epochs = args.n_epochs
batch_size = args.batch_size
train_test_split = args.train_test_split

optim_every = int(32 / batch_size)

params = Parameters()
device = params.device
max_input_length = params.max_input_length

model_name = "distilbert-base-uncased"

tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

model.to(device)

df = pd.read_csv(params.dataset_filename, low_memory=False)
print('Model, tokenizer, dataframe loaded')

def random_word(tokens, tokenizer):
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = tokenizer.mask_token

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]
            # -> rest 10% randomly keep current token
            output_label.append(tokenizer.vocab[token])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label

def _truncate_seq_pair(tokens_a, tokens_b, max_length=max_input_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_example_to_features(tokens_a, tokens_b, max_seq_length, tokenizer):
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens_a, t1_label = random_word(tokens_a, tokenizer)
    tokens_b, t2_label = random_word(tokens_b, tokenizer)
    # concatenate lm labels and account for CLS, SEP, SEP

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #

    tokens =      [tokenizer.cls_token] + tokens_a +             [tokenizer.sep_token] + tokens_b +              [tokenizer.sep_token]
    segment_ids = [0] +      (len(tokens_a) * [0]) + [0] +       (len(tokens_b) * [1]) + [1] 
    lm_label_ids = [-1] + t1_label + [-1] + t2_label + [-1]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    pad_amount = max_seq_length - len(input_ids)
    input_mask = [1] * len(input_ids) + [0] * pad_amount
    input_ids += [0] * pad_amount
    segment_ids += [0] * pad_amount
    lm_label_ids += [-1] * pad_amount

    return input_ids, input_mask, segment_ids, lm_label_ids

def collate_func(bodies):
    bodies_tokenized = [tokenizer.tokenize(body) for body in bodies]

    max_length = max_input_length
    half_length = int(max_length/2)

    mid_point = int(len(bodies)/2)
    batch_ids, batch_mask, batch_segments, batch_lm_label_ids, batch_is_next = [], [], [], [], []
    for i in range(mid_point):
        is_next = 1 if random.random() < 0.5 else 0

        tokens_a = bodies_tokenized[i]
        if is_next == 0:
            tokens_b = bodies_tokenized[i]
        else:
            tokens_b = bodies_tokenized[i+mid_point]
        half_length_a = min(half_length, int(len(tokens_a) / 2))
        half_length_b = min(half_length, int(len(tokens_b) / 2))
        max_length_b = min(max_length, int(len(tokens_b)))
        tokens_a = tokens_a[:half_length_a]
        tokens_b = tokens_b[half_length_b:max_length_b]
        input_ids, input_mask, segment_ids, lm_label_ids = convert_example_to_features(tokens_a, tokens_b, max_length, tokenizer)

        batch_ids.append(input_ids)
        batch_mask.append(input_mask)
        batch_segments.append(segment_ids)
        batch_lm_label_ids.append(lm_label_ids)
        batch_is_next.append(is_next)

    batch_ids = torch.LongTensor(batch_ids)
    batch_mask = torch.LongTensor(batch_mask)
    batch_segments = torch.LongTensor(batch_segments)
    batch_lm_label_ids = torch.LongTensor(batch_lm_label_ids)
    batch_is_next = torch.LongTensor(batch_is_next)

    return batch_ids, batch_mask, batch_segments, batch_lm_label_ids, batch_is_next

descs = [x for x in df.desc_clean.tolist() if isinstance(x,str)]

N = len(descs)
N_train = int(N * train_test_split)
N_val = N - N_train
d_train, d_val = torch.utils.data.dataset.random_split(descs, [N_train, N_val])
dl_train = DataLoader(dataset=d_train, batch_size=batch_size, sampler=RandomSampler(d_train), drop_last=True, collate_fn=collate_func)
dl_val = DataLoader(dataset=d_val, batch_size=batch_size, sampler=RandomSampler(d_val), drop_last=True, collate_fn=collate_func)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
scheduler = None
milestones = list(range(5,n_epochs))
gamma = 0.9
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

# def f1_score(mat):
#     precision = (mat[0,0] / mat[0,:].sum()).item()
#     recall = (mat[0,0] / mat[:,0].sum()).item()
#     return 2*precision*recall / (precision + recall)

if not os.path.isdir(params.model_today_dir):
    os.mkdir(params.model_today_dir)
if not os.path.isdir(params.base_bert_dir):
    os.mkdir(params.base_bert_dir)

params.write_params(params.base_bert_dir)
if scheduler is not None:
    lines = ['\nlearning rate = {}\t milestones = {}\tgamma = {}'.format(learning_rate, milestones, gamma)]
else:
    lines = ['\nlearning rate = {}'.format(learning_rate)]
params.write_params(params.base_bert_dir, lines)
lines = ['\nepoch,avg_train_loss,avg_train_mlm_acc,avg_train_nsp_f1,avg_val_loss,avg_val_mlm_acc,avg_val_nsp_f1']
params.write_params(params.base_bert_dir, lines)
tokenizer.save_pretrained(params.bert_tokenizer_dir)
for i_epoch in range(n_epochs):
    loss_sum = 0
    mlm_right_sum = 0
    n_preds = 0
    nsp_confusion = torch.zeros((2,2)).to('cuda')
    for ib, batch in enumerate(tqdm(dl_train, desc='Train, Epoch #{} - LR={}'.format(i_epoch, optimizer.param_groups[0]["lr"]))):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
        # out = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        out = model(input_ids=input_ids, attention_mask=input_mask)
        mlm_logits = out.logits # mlm_logits = out.prediction_logits
        # is_next_logits = out.seq_relationship_logits
        
        loss = crit(mlm_logits.view(-1, tokenizer.vocab_size), lm_label_ids.view(-1))
        # loss += crit(is_next_logits.view(-1, 2), is_next.view(-1))
        loss.backward()

        # nsp_guesses = torch.argmax(is_next_logits, dim=1)
        # for i in range(2):
        #     for j in range(2):
        #         nsp_confusion[i,j] += (nsp_guesses.eq(i) * is_next.eq(j)).float().sum().item()

        num_predicts_mlm = (~lm_label_ids.eq(-1)).sum().item()
        mlm_right = (lm_label_ids.view(-1).eq(torch.argmax(mlm_logits,dim=2).view(-1)).float().sum()).item()
        
        loss_sum += loss.item()
        mlm_right_sum += mlm_right
        n_preds += num_predicts_mlm
        if ib % optim_every == 0:
            optimizer.step()
            optimizer.zero_grad()
            
    n_items = batch_size * len(dl_train)
    avg_train_loss = loss_sum / n_items
    avg_train_mlm_acc = mlm_right_sum / n_preds
    # avg_train_nsp_f1 = f1_score(nsp_confusion)
    avg_train_loss = loss_sum / n_items

    loss_sum = 0
    mlm_right_sum = 0
    n_preds = 0
    nsp_confusion = torch.zeros((2,2)).to('cuda')
    with torch.no_grad():
        for ib, batch in enumerate(tqdm(dl_val, desc='Val, Epoch #{} - LR={}'.format(i_epoch, optimizer.param_groups[0]["lr"]))):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
            out = model(input_ids=input_ids, attention_mask=input_mask)#, token_type_ids=segment_ids)
            mlm_logits = out.logits #mlm_logits = out.prediction_logits
            # is_next_logits = out.seq_relationship_logits
            
            loss = crit(mlm_logits.view(-1, tokenizer.vocab_size), lm_label_ids.view(-1))
            # loss += crit(is_next_logits.view(-1, 2), is_next.view(-1))

            # nsp_guesses = torch.argmax(is_next_logits, dim=1)
            # for i in range(2):
            #     for j in range(2):
            #         nsp_confusion[i,j] += (nsp_guesses.eq(i) * is_next.eq(j)).float().sum().item()

            num_predicts_mlm = (~lm_label_ids.eq(-1)).sum().item()
            mlm_right = (lm_label_ids.view(-1).eq(torch.argmax(mlm_logits,dim=2).view(-1)).float().sum()).item()
           
            loss_sum += loss.item()
            mlm_right_sum += mlm_right
            n_preds += num_predicts_mlm
        
    n_items = batch_size * len(dl_val)
    avg_val_mlm_acc = mlm_right_sum / n_preds
    # avg_val_nsp_f1 = f1_score(nsp_confusion)
    avg_val_loss = loss_sum / n_items
    
    tmp = [i_epoch, avg_train_loss, avg_train_mlm_acc, avg_val_loss, avg_val_mlm_acc] #avg_train_nsp_f1, avg_val_loss, avg_val_mlm_acc, avg_val_nsp_f1]
    lines = ['\n' + ','.join([str(x) for x in tmp])]
    print(lines)
    model.save_pretrained(params.bert_model_dir.format(i_epoch))
    params.write_params(params.base_bert_dir, lines)

    if scheduler is not None:
        scheduler.step()