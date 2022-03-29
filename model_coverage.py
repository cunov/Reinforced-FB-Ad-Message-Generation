from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np
import pandas as pd
import sys
from nltk import word_tokenize, sent_tokenize
from parameters import Parameters

params = Parameters()

class KeywordExtractor():
    def __init__(self, hotwords_filename=params.hotwords_filename, bert_tokenizer_dir=params.trained_bert_tokenizer_dir):
        with open(hotwords_filename, 'r') as f:
            self.hotwords = {}
            for line in f.readlines():
                word = line.split(',')[0]
                val = line.split(',')[1][:-1]
                self.hotwords[word] = float(val)
                
        # self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_dir)
        self.hotword_tokens = {key : self.tokenizer.encode(key)[1:-1] for key,_ in self.hotwords.items()}
        # model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        # self.ner = pipeline('ner', model=model, tokenizer=self.tokenizer, device=0, aggregation_strategy='simple')
        
    def find_potential_masks(self, text, locs, loc_score_thresh=params.loc_score_thresh):
        hot_masks = []
        loc_masks = []
        for sentence in sent_tokenize(text):
            for word in word_tokenize(sentence):
                if word.lower() in self.hotwords.keys():
                    hot_masks.append({'word':word, 
                                      'prob':self.hotwords[word.lower()], 
                                      'tokens':self.hotword_tokens[word.lower()]})
        # entities = self.ner(text)
        for i,entity in enumerate(locs):
            if entity['entity_group'] == 'LOC' and entity['score'] > loc_score_thresh and not '#' in entity['word']:
                    loc_masks.append({'word':entity['word'], 
                                      'prob':entity['score'], 
                                      'tokens':self.tokenizer.encode(entity['word'])[1:-1]})

        return hot_masks, loc_masks
    
    def mask_doc(self, text, tokenized_text, locs, masking_scheme=params.masking_scheme, n_hot_masks=params.n_hot_masks, n_loc_masks=params.n_loc_masks):
        hot_masks, loc_masks = self.find_potential_masks(text, locs)
        to_mask = [item['tokens'] for item in loc_masks[:n_loc_masks]]
        
        if masking_scheme == 'random':
            hot_mask_idx = np.arange(len(hot_masks))
            np.random.shuffle(hot_mask_idx)
            for item in [hot_masks[idx] for idx in hot_mask_idx[:n_hot_masks]]:
                to_mask.append(item['tokens'])
        elif masking_scheme == 'probabalistic':
            n_masked_hotwords = 0
            while n_masked_hotwords < n_hot_masks and n_masked_hotwords < len(hot_masks):
                for item in hot_masks:
                    if np.random.random() < item['prob']:
                        n_masked_hotwords += 1
                        to_mask.append(item['tokens'])
                        if n_masked_hotwords >= n_hot_masks or n_masked_hotwords >= len(hot_masks):
                            break
        elif masking_scheme == 'prioritize_clicks':
            probs = [item['prob'] for item in hot_masks]
            sorted_idxs = np.flip(np.argsort(probs))
            n_masked_hotwords = 0
            for item in [hot_masks[idx] for idx in sorted_idxs]:
                to_mask.append(item['tokens'])
                if n_masked_hotwords >= n_hot_masks:
                    break
                    
        masked_text = [tok for tok in tokenized_text]
        for seq in to_mask:
            seq = np.asarray(seq)
            for i in range(len(masked_text) - len(seq)):
                if (masked_text[i:i+len(seq)] == seq).all():
                    masked_text[i:i+len(seq)] = [self.tokenizer.mask_token_id] * len(seq)
        
        return masked_text
    
class KeywordCoverage():
    def __init__(self, max_output_length=params.max_output_length, max_input_length=params.max_input_length,
                 hotwords_filename=params.hotwords_filename,
                 load_pretrained = False,
                 bert_tokenizer_dir=params.trained_bert_tokenizer_dir, bert_model_dir=params.trained_bert_model_dir,
                 masking_scheme=params.masking_scheme, n_hot_masks=params.n_hot_masks, n_loc_masks=params.n_loc_masks, 
                 device='cuda'):
        self.device = device
        self.kw_ex = KeywordExtractor(hotwords_filename, bert_tokenizer_dir)
        if load_pretrained:
            self.tokenizer = BertTokenizer.from_pretrained(params.trained_coverage_tokenizer_dir)
            self.bert = BertForMaskedLM.from_pretrained(params.trained_coverage_model_dir)
            print('Loaded pretrained coverage model, tokenizer')
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_dir)
            self.bert = BertForMaskedLM.from_pretrained(bert_model_dir).to(self.device)
            print('Loaded default coverage model, tokenizer')
        self.crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.masking_scheme, self.n_hot_masks, self.n_loc_masks = masking_scheme, n_hot_masks, n_loc_masks
        self.max_output_length = max_output_length
        self.max_input_length = max_input_length
        self.bert.to(self.device)
        
    def build_inputs(self, contents, summaries, locations):
        N = len(contents)
        input_ids, labels = [], []
        for content, summary, locs in zip(contents, summaries, locations):
            unmasked = torch.LongTensor(
                self.tokenizer.encode(content, truncation=True, max_length=self.max_input_length)
            )
            masked = torch.LongTensor(
                self.kw_ex.mask_doc(content, unmasked, locs,
                    self.masking_scheme, self.n_hot_masks, self.n_loc_masks)
            )
            summ_tok = torch.LongTensor(
                self.tokenizer.encode(summary, truncation=True, max_length=self.max_output_length)
            )
            # while len(summ_tok) + len(masked) >= self.tokenizer.model_max_length:
            #     if len(summ_tok) > self.max_output_length:
            #         summ_tok = summ_tok[:self.max_output_length-1]
            #     masked = masked[:self.tokenizer.model_max_length-len(summ_tok)-2]
            #     unmasked = unmasked[:self.tokenizer.model_max_length-len(summ_tok)-2]
            
            input_ids.append(torch.cat((summ_tok, torch.LongTensor([self.tokenizer.sep_token_id]), masked)))
            labels.append(torch.cat((summ_tok, torch.LongTensor([self.tokenizer.sep_token_id]), unmasked)))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        is_masked = input_ids.eq(torch.LongTensor([self.tokenizer.mask_token_id]))
        labels = (labels*is_masked) + (~is_masked * torch.LongTensor([-1]))
        attn_mask = 1 * torch.eq(input_ids, self.tokenizer.pad_token_id)

        input_ids = input_ids.to(self.device)
        is_masked = is_masked.to(self.device)
        labels = labels.to(self.device)
        attn_mask = attn_mask.to(self.device)        
        return input_ids, is_masked, labels, attn_mask
    
    def get_loss_acc(self, contents, summaries, locations):
        input_ids, is_masked, labels, attn_mask = self.build_inputs(contents, summaries, locations)

        outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits
        loss = self.crit(logits.view(-1, self.tokenizer.vocab_size), labels.view(-1))
            
        n_masks = is_masked.sum(dim=1)
        with torch.no_grad(): # move this up above model call?
            preds = torch.argmax(logits, dim=2)
            accs = (preds.eq(labels) * is_masked).sum(dim=1).float() / n_masks
            accs = torch.nan_to_num(accs, nan=0.0)
        return loss, accs.mean().item()

    def score(self, summaries, contents, locs, accs_without=None):
        with torch.no_grad():
            input_ids, is_masked, labels, attn_mask = self.build_inputs(contents, summaries, locs)
            
            outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
            preds = torch.argmax(outputs.logits, dim=2)
            n_masks = torch.sum(is_masked, dim=1).float()
            problem = False
            for idx in (n_masks==0).nonzero():
                print(contents[idx])
                problem = True
            if problem:
                sys.exit('================\nFOUND CONTENTS WITH NO MASKS\n============')
            accs_with = torch.sum(preds.eq(labels).long() * is_masked, dim=1).float() / n_masks
            accs_with = torch.nan_to_num(accs_with, nan=0)
            if accs_without is None:
                input_ids, is_masked, labels, attn_mask = self.build_inputs(contents, [""] * len(input_ids), locs)
                outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
                preds = torch.argmax(outputs.logits, dim=2)
                n_masks = torch.sum(is_masked, dim=1).float()
                accs_without = torch.sum(preds.eq(labels).long() * is_masked, dim=1).float() / n_masks
                accs_without = torch.nan_to_num(accs_without, nan=0)
            return accs_with - accs_without, accs_without