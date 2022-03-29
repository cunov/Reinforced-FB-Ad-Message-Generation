import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler
import torch
import pandas as pd
from tqdm import tqdm
from model_generator import Fluency
from model_coverage import KeywordCoverage
import argparse
from parameters import Parameters
import torch.optim as optim
from model_guardrails import PatternPenalty, LengthPenalty, RepeatPenalty
import sys
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=2e-5)

args = parser.parse_args()
batch_size = args.batch_size
n_epochs = args.n_epochs
learning_rate = args.learning_rate

params = Parameters()

def score(sampled_summs, argmax_summs, sampled_lens, argmax_lens, contents, locs, scorers):
    sampled_scores, argmax_scores = 0, 0
    importance, model = scorers['coverage']['importance'], scorers['coverage']['model']
    sampled_acc, acc_wo = model.score(sampled_summs, contents, locs)
    argmax_acc, _ = model.score(argmax_summs, contents, locs, acc_wo)
    sampled_scores += importance * sampled_acc
    argmax_scores += importance * argmax_acc

    importance, model = scorers['fluency']['importance'], scorers['fluency']['model']
    sampled_scores += importance * model.score(sampled_summs)
    argmax_scores += importance * model.score(argmax_summs)

    importance, model = scorers['patpen']['importance'], scorers['patpen']['model']
    sampled_scores -= importance * model.score(sampled_summs)
    argmax_scores -= importance * model.score(argmax_summs)

    importance, model = scorers['lengthpen']['importance'], scorers['lengthpen']['model']
    sampled_scores -= importance * model.score(sampled_lens, sampled_lens)
    argmax_scores -= importance * model.score(argmax_lens, argmax_lens)

    importance, model = scorers['reppen']['importance'], scorers['reppen']['model']
    sampled_scores -= importance * model.score(sampled_summs)
    argmax_scores -= importance * model.score(argmax_summs)

    sampled_scores = sampled_scores.to('cuda')
    argmax_scores = argmax_scores.to('cuda')
    return sampled_scores, argmax_scores

summarizer = Fluency(load_pretrained='summarizer')
summarizer.model.to(summarizer.device)

optimizer = optim.AdamW(summarizer.model.parameters(), lr=learning_rate)

scorers = {}
scorers['coverage'] = {'importance':10.0, 'model':KeywordCoverage(load_pretrained=True)}
scorers['fluency'] = {'importance':10.0, 'model':Fluency(load_pretrained='fluency')}
scorers['patpen'] = {'importance':5.0, 'model':PatternPenalty()}
scorers['lengthpen'] = {'importance':2.0, 'model':LengthPenalty(params.max_output_length)}
scorers['reppen'] = {'importance':2.0, 'model':RepeatPenalty()}

# scorers = [{'name':'coverage', 'importance':10.0, 'sign':1.0, 'model':KeywordCoverage(load_pretrained=True)},
#            {'name':'fluency', 'importance':10.0, 'sign':1.0, 'model':Fluency(load_pretrained='fluency')},
#            {'name':'patpen', 'importance':5.0, 'sign':-1.0, 'model':PatternPenalty()},
#            {'name':'lengthpen', 'importance':2.0, 'sign':-1.0, 'model':LengthPenalty(params.max_output_length)},
#            {'name':'reppen', 'importance':2.0, 'sign':-1.0, 'model':RepeatPenalty()}]

df = pd.read_csv(params.dataset_filename, low_memory=False)
descs = []
locs = []
for i,(desc,loc) in enumerate(zip(df.desc_clean, df.desc_clean_msg_stripped_locs)):
    if isinstance(desc,str) and isinstance(loc,str) and isinstance(ast.literal_eval(loc), list):
        descs.append(desc)
        locs.append(ast.literal_eval(loc))

def collate_func(batch):
    descs, locs = [item[0] for item in batch], [item[1] for item in batch]
    return zip(descs, locs)

dataset = list(zip(descs, locs))
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_func)

summarizer.model.train()
for i_epoch in range(n_epochs):
    for ib, batch in enumerate(tqdm(dataloader, desc='Epoch #{}'.format(i_epoch))):
        descs, locs = batch
        sampled_summaries, sampled_logprobs, sampled_tokens, input_past, sampled_end_idxs = summarizer.decode_batch(descs, return_logprobs=True, sample=True)
        with torch.no_grad():
            argmax_summaries, argmax_end_idxs = summarizer.decode_batch(descs, input_past=input_past)

        selected_logprobs = torch.sum(sampled_logprobs, dim=1)
        batch_size, seq_len = sampled_logprobs.shape

        # thread coverage.tokenized descs here?

        total_sampled_scores, total_argmax_scores = score(sampled_summaries, argmax_summaries, sampled_end_idxs, argmax_end_idxs, descs, locs, scorers)
        # total_sampled_scores = torch.FloatTensor(total_sampled_scores).to()
        # total_argmax_scores = torch.FloatTensor(total_argmax_scores, device='cuda')

        loss = torch.mean((total_argmax_scores - total_sampled_scores) * selected_logprobs)
        loss.backward()
        print(str(round(loss.item(),4)))
