import torch.optim as optim
from torch import save as save_checkpoint, load as load_checkpoint
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm
from model_generator import Fluency
from model_coverage import KeywordCoverage
import argparse
from parameters import Parameters
import torch.optim as optim
from model_guardrails import PatternPenalty, LengthPenalty, RepeatPenalty, InvalidCharacter, Hotness
import ast
import numpy as np
import time
import threading, queue
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=2e-5)

args = parser.parse_args()
batch_size = args.batch_size
n_epochs = args.n_epochs
learning_rate = args.learning_rate

params = Parameters()

def score(sampled_summs, argmax_summs, tokenized_sampled, tokenized_argmax, tokenized_descs, sampled_lens, argmax_lens, contents, locs, scorers, prebuild_withouts, prebuild_summaries, is_train=True, i_epoch=-1, product=False):
    sampled = {}
    argmax = {}
    N = len(sampled_summs)
    
    sign, importance, model = scorers['coverage']['sign'], scorers['coverage']['importance'], scorers['coverage']['model']
    sampled_acc, acc_wo = model.score(sampled_summs, contents, locs, prebuild_withouts=prebuild_withouts, prebuild_summaries=prebuild_summaries)
    argmax_acc, _ = model.score(argmax_summs, contents, locs, accs_without=acc_wo)
    sampled['coverage'] = sign * importance * sampled_acc
    argmax['coverage'] = sign * importance * argmax_acc

    sign, importance, model = scorers['fluency']['sign'], scorers['fluency']['importance'], scorers['fluency']['model']
    sampled['fluency'] = sign * importance * model.score(sampled_summs)
    argmax['fluency'] = sign * importance * model.score(argmax_summs)

    sampled_words_list = [word_tokenize(summ) for summ in sampled_summs]
    argmax_words_list = [word_tokenize(summ) for summ in argmax_summs]

    sign, importance, model = scorers['patpen']['sign'], scorers['patpen']['importance'], scorers['patpen']['model']
    sampled['patpen'] = sign * importance * model.score(sampled_words_list)
    argmax['patpen'] = sign * importance * model.score(argmax_words_list)

    sign, importance, model = scorers['lengthpen']['sign'], scorers['lengthpen']['importance'], scorers['lengthpen']['model']
    sampled['lengthpen'] = sign * importance * model.score(sampled_words_list)
    argmax['lengthpen'] = sign * importance * model.score(argmax_words_list)

    sign, importance, model = scorers['reppen']['sign'], scorers['reppen']['importance'], scorers['reppen']['model']
    sampled['reppen'] = sign * importance * model.score(sampled_words_list)
    argmax['reppen'] = sign * importance * model.score(argmax_words_list)
    
    # sign, importance, model = scorers['badchar']['sign'], scorers['badchar']['importance'], scorers['badchar']['model']
    # sampled['badchar'] = sign * importance * model.score(sampled_summs)
    # argmax['badchar'] = sign * importance * model.score(argmax_summs)

    sign, importance, model = scorers['hotness']['sign'], scorers['hotness']['importance'], scorers['hotness']['model']
    sampled['hotness'] = sign * importance * model.score(tokenized_sampled, tokenized_descs)
    argmax['hotness'] = sign * importance * model.score(tokenized_argmax, tokenized_descs)

    sampled_scores, argmax_scores = 0, 0
    if not product: # alpha*coverage + beta*fluency
        sampled_scores += sampled['coverage'] + sampled['fluency']
        argmax_scores += argmax['coverage'] + argmax['fluency']
    else: # (alpha + beta)*coverage*fluency
        alpha = scorers['coverage']['importance']
        beta = scorers['fluency']['importance']

        sampled_scores += (alpha + beta) * (sampled['coverage'] / alpha) * (sampled['fluency'] / beta)
        argmax_scores += (alpha + beta) * (argmax['coverage'] / alpha) * (argmax['fluency'] / beta)

    for _model in ['patpen','lengthpen','reppen','hotness']:#,'badchar']:
        sampled_scores += sampled[_model]
        argmax_scores += argmax[_model]
    sampled_scores = sampled_scores.to('cuda')
    argmax_scores = argmax_scores.to('cuda')
    
    sampled = {key:(val / scorers[key]['importance']).sum().item() for key,val in sampled.items()}
    argmax = {key:(val / scorers[key]['importance']).sum().item() for key,val in argmax.items()}
    sampled['sampled'] = True
    argmax['sampled'] = False
    sampled['is_train'] = is_train
    argmax['is_train'] = is_train
    sampled['epoch'] = i_epoch
    argmax['epoch'] = i_epoch
    sampled['batch_size'] = N
    argmax['batch_size'] =  N
    sampled['n_words'] = sum(len(summ.split(' ')) for summ in sampled_summs)
    argmax['n_words'] = sum(len(summ.split(' ')) for summ in argmax_summs)
    

    return sampled_scores, argmax_scores, [sampled, argmax]

summarizer = Fluency(load_pretrained='summarizer')
summarizer.model.to(summarizer.device)

param_optimizer = list(summarizer.model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
optim_every = int(8)
scheduler = None

df = pd.read_csv(params.dataset_filename, low_memory=False)
descs = []
locs = []
for i,(desc,loc) in enumerate(zip(df.desc_clean, df.desc_clean_msg_stripped_locs)):
    if isinstance(desc,str) and isinstance(loc,str) and isinstance(ast.literal_eval(loc), list):
        descs.append(' '.join(desc.split(' ')[:500]))
        locs.append(ast.literal_eval(loc))

scorers = {}
scorers['coverage'] = {'sign':1.0, 'importance':10.0, 'model':KeywordCoverage(_type='', descs=descs)}
scorers['fluency'] = {'sign':1.0, 'importance':2.0, 'model':Fluency(load_pretrained='fluency')}
scorers['patpen'] = {'sign':-1.0, 'importance':5.0, 'model':PatternPenalty()}
scorers['lengthpen'] = {'sign':-1.0, 'importance':5.0, 'model':LengthPenalty(params.max_output_length)}
scorers['reppen'] = {'sign':-1.0, 'importance':2.0, 'model':RepeatPenalty()}
# scorers['badchar'] = {'sign':-1.0, 'importance':2.0, 'model':InvalidCharacter()}
scorers['hotness'] = {'sign':1.0, 'importance':5.0, 'model':Hotness()}
scorers['coverage']['model'].bert.eval()
scorers['fluency']['model'].model.eval()

use_product = False



def collate_func(batch):
    return list(zip(*batch))

dataset = list(zip(descs, locs))
N = len(dataset)
N_train = N - 500
N_val = N - N_train
d_train, d_val = torch.utils.data.dataset.random_split(dataset, [N_train, N_val])
dl_train = DataLoader(dataset=d_train, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_func)
dl_val = DataLoader(dataset=d_val, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_func)

prebuild_withouts_queue = queue.Queue()
prebuild_summaries_queue = queue.Queue()
def background_worker(descs, msgs, locs, q, return_unmaskeds_summ_toks=False):
    q.put(scorers['coverage']['model'].build_inputs(descs, msgs, locs, return_unmaskeds_summ_toks))

# def background_tokenizer(descs, msgs, q):
#     q.put(scorers['coverage']['model'].tokenizer(texts, truncation=True, max_length=params.max_input_length))

def get_time(last_time):
    return time.time() - last_time, time.time()

scaler = torch.cuda.amp.GradScaler()
summarizer.tokenizer.save_pretrained(params.summary_loop_tokenizer_dir)
checkpoint_every = 600 # seconds
time_checkpoint = time.time()
total_score_history = []
checkpoint_lookback = 400
best_checkpoint_score = None
checkpoint_model_file = params.summary_loop_model_dir.format('checkpoint_model.pt')
checkpoint_optimizer_file = params.summary_loop_model_dir.format('checkpoint_optimizer.pt')

texts = ["""A prewar Rosario Candela architectural masterpiece, this sun-flooded corner 14 into 11 room grand duplex was expertly renovated by renowned architect Mary Burnham of MBB Architects in partnership with designer Rachel Laxer and home builder Josh Weiner of Silver Lining. The largest home in the building, Apartment 10/11A has five bedrooms, five full bathrooms and two powder rooms. The residence has undergone a complete state-of-the-art restoration and modernization with every room being completely gutted and put back together with no expense being spared and utilizing the best of materials and systems while restoring the elegance of historic details. Perfect for both entertaining and everyday living, there are ceilings up to 10' 8", two wood burning fireplaces, a balcony, raised doorways, perfectly restored original oak wood flooring as well as new rift quarter sawn white oak flooring, new windows, custom millwork, plaster moldings, bespoke E.R. Butler & Co. hardware, specialty paint finishes, and radiant heat in the kitchen, mud room and all bathrooms. All mechanical elements of this sophisticated home have been fully modernized including new electrical, new plumbing and a state-of-the-art Savant Pro Home Technology system which integrates the extensive audio/video components, Lutron lighting system, motorized shades and HVAC. The Savant system allows you to control the entire home through your cell phone or the multiple iPads in the residence. There is a five-zone HVAC system with temperature controlled zones for AC and each room having a control for heat. The home has Cat5e cabling, a Panasonic phone and intercom system, wired WAPs throughout for excellent WiFi connectivity, in-ceiling speakers throughout and it is wired for televisions in seven rooms. A semi-private elevator landing opens onto a regal 31-foot long entrance gallery that spans the beautiful enfilade of formal spaces which face south, east and west. Off the gallery is an exquisite powder room, an oversized coat closet, and a discreet china closet hidden under the staircase. The sun-drenched living room boasts three oversized windows which face south. This magnificent room is 28' 7" by 20' 2" and has the first wood burning fireplace. Adjacent is a palatial 26-foot long corner library with four windows facing south and east which has been meticulously restored and has the original distinguished wood paneling, original Tudor style plaster molded ceiling and frieze, and the second wood burning fireplace. Next is a corner formal dining room which has been transformed into an incredible entertainment room which has four windows facing south and west and has a built-in banquette for dining. The corner eat-in kitchen is open to this room and can be closed off by concealed pocket doors if you want to formally entertain. The kitchen has custom Bulthaup cabinetry, quartzite countertops by Walker Zanger, porcelain tile flooring by Fibra Collection, and a center island clad in a Luce Di Luna slab with seating. It is equipped with top-of-the-line appliances including a Subzero refrigerator and two freezer drawers, a Subzero wine refrigerator, two Subzero refrigerator drawers, two Gaggenau ovens, a Gaggenau five burner range with a vented hood, a Gaggenau warming drawer, two Miele dishwashers, a Franke sink and an endless amount of storage. Off of the kitchen is an incredible large windowed mudroom with built-in wood storage lockers and shelving as well as a second windowed powder room, a balcony and a back staircase leading to the 11th floor.""",
            """Spacious 3 bedroom, 2 bathroom home with beautiful East River views and oversized balcony with direct access from both living room and primary bedroom. Large kitchen with pass-through to living room. Living room with plenty of room for dining, piano and more!
Primary bedroom is almost 400 sq ft and could accommodate a king sized bed, sitting room and home office. Closets are abundant and one walk-in could easily convert to 3rd bathroom or laundry room.
Two additional bedrooms could also accommodate king sized beds, desks etc. 
W/D allowed. Cats & Dog friendly. 45 East End Ave permits up to 65% Financing."""]

with open(params.base_summary_loop_dir + 'scorer_importance.csv','w+') as f:
    lines = []
    for key in scorers.keys():
        lines.append(key + ',' + str(scorers[key]['importance']) + '\n')
    f.writelines(lines)

rows = []
for i_epoch in range(n_epochs):
    summarizer.model.train()
    for ib, batch in enumerate(tqdm(dl_train, desc='Train, epoch #{} - LR={}'.format(i_epoch, optimizer.param_groups[0]["lr"]))):
        with torch.cuda.amp.autocast():
            descs, locs = batch
            prebuild_withouts_thread = threading.Thread(target=background_worker, args=(descs, [""] * len(descs), locs, prebuild_withouts_queue, False))
            prebuild_withouts_thread.start()

            sampled_summaries, sampled_logprobs, input_past, sampled_end_idxs = summarizer.decode_batch(descs, return_logprobs=True, sample=True) # 32%

            prebuild_summaries_thread = threading.Thread(target=background_worker, args=(descs, sampled_summaries, locs, prebuild_summaries_queue, True))
            prebuild_summaries_thread.start()

            selected_logprobs = torch.sum(sampled_logprobs, dim=1)
            with torch.no_grad():
                argmax_summaries, argmax_end_idxs = summarizer.decode_batch(descs, input_past=input_past) # 28%
                tokenized_argmax = scorers['coverage']['model'].tokenizer(argmax_summaries, truncation=True, max_length=params.max_input_length).input_ids

            prebuild_withouts_thread.join()
            prebuild_summaries_thread.join()
            prebuild_withouts = prebuild_withouts_queue.get()
            prebuild_summaries_tmp = prebuild_summaries_queue.get()
            prebuild_summaries = prebuild_summaries_tmp[:-2]
            tokenized_descs, tokenized_sampled = prebuild_summaries_tmp[-2:]



            total_sampled_scores, total_argmax_scores, new_rows = score(sampled_summaries, argmax_summaries, tokenized_sampled, tokenized_argmax, tokenized_descs, sampled_end_idxs, argmax_end_idxs, descs, locs, scorers, 
                                                                prebuild_withouts, prebuild_summaries, True, i_epoch, product=use_product) # 10%
            rows += new_rows
            total_score_history.append(total_sampled_scores.mean().item())
            loss = torch.mean((total_argmax_scores - total_sampled_scores) * selected_logprobs) / optim_every

        scaler.scale(loss).backward() # 27%
        if int(ib+1) % optim_every == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if time.time() - time_checkpoint > checkpoint_every and len(total_score_history) > checkpoint_lookback:
            time_checkpoint = time.time()
            current_score = np.mean(total_score_history[-checkpoint_lookback:])
            print('\n===== CHECKPOINT =====')
            print('Best Score: ', best_checkpoint_score)
            print('Current Score: ', current_score)
            if best_checkpoint_score is not None:
                if current_score < min(1.2*best_checkpoint_score, 0.8*best_checkpoint_score):
                    print('*@@@@@ REVERTING @@@@@')
                    summarizer.model.load_state_dict(load_checkpoint(checkpoint_model_file))
                    optimizer.load_state_dict(load_checkpoint(checkpoint_optimizer_file))
            if best_checkpoint_score is None or current_score > best_checkpoint_score:
                print('!!!!! Saved New HIGH Score !!!!!')
                save_checkpoint(summarizer.model.state_dict(), checkpoint_model_file)
                save_checkpoint(optimizer.state_dict(), checkpoint_optimizer_file)
                best_checkpoint_score = current_score
            with torch.no_grad():
                sampled_summaries, _ = summarizer.decode_batch(texts, sample=True)
                argmax_summaries, _ = summarizer.decode_batch(texts)

                input_ids = [summarizer.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=params.max_input_length).to(params.device) for text in texts]
                input_lengths = [len(input_id[0]) for input_id in input_ids]
                outputs = [summarizer.model.generate(input_id, do_sample=True, top_k=50, max_new_tokens=params.max_output_length)[0] for input_id in input_ids]
                top_k_summaries = [summarizer.tokenizer.decode(output[input_len:]) for output,input_len in zip(outputs,input_lengths)]
                
                print('------ SAMPLED ------\n** '+'\n** '.join(sampled_summaries))
                print('------ ARGMAX ------\n** '+'\n** '.join(argmax_summaries))
                print('------ TOP 50 ------\n** '+'\n** '.join(top_k_summaries))
                del _

            df_tmp = pd.DataFrame(data=rows) # data frame of raw, unweighted sum of batch scores
            if i_epoch == 0:
                df_tmp.to_csv(params.base_summary_loop_dir+'dat.csv', index=False)
            else:
                df_tmp.to_csv(params.base_summary_loop_dir+'dat.csv', mode='a', index=False, header=False)
            del df_tmp
            rows = []


    summarizer.model.eval()
    with torch.no_grad():
        for ib, batch in enumerate(tqdm(dl_val, desc='Val, epoch #{} - LR={}'.format(i_epoch, optimizer.param_groups[0]["lr"]))):
            descs, locs = batch
            prebuild_withouts_thread = threading.Thread(target=background_worker, args=(descs, [""] * len(descs), locs, prebuild_withouts_queue, False))
            prebuild_withouts_thread.start()

            sampled_summaries, sampled_logprobs, input_past, sampled_end_idxs = summarizer.decode_batch(descs, return_logprobs=True, sample=True)
            prebuild_summaries_thread = threading.Thread(target=background_worker, args=(descs, sampled_summaries, locs, prebuild_summaries_queue, True))
            prebuild_summaries_thread.start()

            selected_logprobs = torch.sum(sampled_logprobs, dim=1)
            argmax_summaries, argmax_end_idxs = summarizer.decode_batch(descs, input_past=input_past)
            tokenized_argmax = scorers['coverage']['model'].tokenizer(argmax_summaries, truncation=True, max_length=params.max_input_length).input_ids
            prebuild_withouts_thread.join()
            prebuild_summaries_thread.join()
            prebuild_withouts = prebuild_withouts_queue.get()
            prebuild_summaries_tmp = prebuild_summaries_queue.get()
            prebuild_summaries = prebuild_summaries_tmp[:-2]
            tokenized_descs, tokenized_sampled = prebuild_summaries_tmp[-2:]

            total_sampled_scores, total_argmax_scores, new_rows = score(sampled_summaries, argmax_summaries, tokenized_sampled, tokenized_argmax, tokenized_descs, sampled_end_idxs, argmax_end_idxs, descs, locs, scorers, 
                                                                prebuild_withouts, prebuild_summaries, False, i_epoch, product=use_product)
            rows += new_rows

    if scheduler is not None:
        scheduler.step()
    
    # summarizer.model.save_pretrained(params.summary_loop_model_dir.format(i_epoch))