from transformers import BertTokenizer, BertForMaskedLM, BertTokenizerFast
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, DistilBertTokenizerFast
import torch
import numpy as np
import sys
from parameters import Parameters
from collections import Counter

params = Parameters()
STOP_WORDS = set(["'", '#', '%', '+', ".", "!", "?", ",", '"', '-', ')', '(', 'we', 'our', 'you', 'he', 'him', 'she', 'her', 'it', "it's", 'its', 'they', 'their', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'the', 'and', 'or', 'as', 'of', 'at', 'by', 'to', 'not', 'so', "'s", "in", "for", "with", "on", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

class KeywordExtractor():
    def __init__(self, hotwords_filename=params.hotwords_filename, bert_tokenizer_dir=params.trained_bert_tokenizer_dir, descs=None):
        with open(hotwords_filename, 'r') as f:
            self.hotwords = {}
            s = 0.0
            for line in f.readlines():
                token = int(line.split(',')[0])
                word = line.split(',')[1]
                val = line.split(',')[2][:-1]
                self.hotwords[token] = {'word':word, 'val':float(val)}
                s += float(val)
            for key in self.hotwords.keys():
                self.hotwords[key]['val'] /= s

        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_dir)
        tokenizer_tmp = BertTokenizerFast.from_pretrained(bert_tokenizer_dir)
        
        if params.masking_scheme == 'tfidf':
            print('Building IDF...',end='')
            tokenized_descs = tokenizer_tmp(descs, truncation=True, max_length=params.max_input_length).input_ids
            N = len(descs)
            document_freq = Counter()
            for tokenized_desc in tokenized_descs:
                document_freq.update([token for token in set(tokenized_desc)])
            tokens, doc_freqs = zip(*document_freq.most_common())
            self.idf = {token:np.log( N / (1+doc_freq)) for token,doc_freq in zip(tokens,doc_freqs)}
            print('done.')
        # self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        # model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        # self.ner = pipeline('ner', model=model, tokenizer=self.tokenizer, device=0, aggregation_strategy='simple')
        
    def find_potential_masks(self, tokens, locs, loc_score_thresh=params.loc_score_thresh):
        hot_masks = []
        loc_masks = []
        for token in tokens:
            token = token.item()
            if token in self.hotwords.keys():
                hot_masks.append({'word':self.hotwords[token]['word'], 
                                  'prob':self.hotwords[token]['val'], 
                                  'tokens':token})
        for entity in locs:
            if entity['entity_group'] == 'LOC' and entity['score'] > loc_score_thresh and not '#' in entity['word']:
                    loc_masks.append({'word':entity['word'], 
                                      'prob':entity['score'], 
                                      'tokens':self.tokenizer.encode(entity['word'])[1:-1]})

        return hot_masks, loc_masks
    
    def mask_doc(self, tokenized_text, locs, masking_scheme=params.masking_scheme, n_hot_masks=params.n_hot_masks, n_loc_masks=params.n_loc_masks, n_tfidf_masks=params.n_tfidf_masks):
        
        hot_masks, loc_masks = self.find_potential_masks(tokenized_text, locs)
        to_mask = [item['tokens'] for item in loc_masks[:n_loc_masks]]
        if masking_scheme != 'tfidf':
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
                            to_mask.append(item['tokens'])
                            n_masked_hotwords += 1
                            if n_masked_hotwords >= n_hot_masks or n_masked_hotwords >= len(hot_masks):
                                break
            elif masking_scheme == 'prioritize_clicks':
                probs = [item['prob'] for item in hot_masks]
                sorted_idxs = np.flip(np.argsort(probs))
                n_masked_hotwords = 0
                for item in [hot_masks[idx] for idx in sorted_idxs]:
                    to_mask.append(item['tokens'])
                    n_masked_hotwords += 1
                    if n_masked_hotwords >= n_hot_masks:
                        break
        else:
            n_locs = len(to_mask)
            term_freq = Counter(tokenized_text.tolist())
            tfidf = {}
            N = len(tokenized_text)
            for token,tf in term_freq.items():
                if len(self.tokenizer.decode([token])) > 3:
                    tfidf[token] = (tf / N) * self.idf[token]
            tokens, tfidf = zip(*tfidf.items())
            idxs = np.flip(np.argsort(tfidf))
            for idx in idxs:
                if not tokens[idx] in to_mask:
                    to_mask.append(tokens[idx])
                if len(to_mask) >= n_locs + n_tfidf_masks:
                    break

        masked_text = [tok if not tok in to_mask else self.tokenizer.mask_token_id for tok in tokenized_text]
        
        return masked_text
    
class KeywordCoverage():
    def __init__(self, max_output_length=params.max_output_length, max_input_length=params.max_input_length,
                 hotwords_filename=params.hotwords_filename,
                 _type='default',
                 bert_tokenizer_dir=params.trained_bert_tokenizer_dir, bert_model_dir=params.trained_bert_model_dir,
                 masking_scheme=params.masking_scheme, n_hot_masks=params.n_hot_masks, n_loc_masks=params.n_loc_masks, 
                 device='cuda', descs=None):
        self.device = device
        if _type == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained(bert_tokenizer_dir)
            self.bert = DistilBertForMaskedLM.from_pretrained(bert_model_dir).to(self.device)
            self.kw_ex = KeywordExtractor(hotwords_filename, bert_tokenizer_dir=bert_tokenizer_dir)
            self.init = bert_model_dir
            print('Loaded', bert_model_dir)
        elif _type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_dir)
            self.bert = BertForMaskedLM.from_pretrained(bert_model_dir).to(self.device)
            self.kw_ex = KeywordExtractor(hotwords_filename, bert_tokenizer_dir=bert_tokenizer_dir)
            self.init = bert_model_dir
            print('Loaded', bert_model_dir)
        elif _type == 'default':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
            self.kw_ex = KeywordExtractor(hotwords_filename, bert_tokenizer_dir='distilbert-base-uncased', descs=descs)
            self.init = 'distilbert-base-uncased'
            print('Loaded distilbert-base-uncased')
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained(params.trained_coverage_tokenizer_dir)
            self.bert = DistilBertForMaskedLM.from_pretrained(params.trained_coverage_model_dir)
            self.init = params.trained_coverage_model_dir
            # self.tokenizer = DistilBertTokenizer.from_pretrained(params.trained_coverage_tokenizer_dir)
            # self.bert = DistilBertForMaskedLM.from_pretrained(params.trained_coverage_model_dir) 
            self.kw_ex = KeywordExtractor(hotwords_filename, bert_tokenizer_dir=params.trained_coverage_tokenizer_dir, descs=descs)
            print('Loaded fine-tuned coverage model, tokenizer')
        self.crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.masking_scheme, self.n_hot_masks, self.n_loc_masks = masking_scheme, n_hot_masks, n_loc_masks
        self.max_output_length = max_output_length
        self.max_input_length = max_input_length
        self.bert.to(self.device)
        self.scale_factor = params.coverage_scaler
        
    def build_inputs(self, contents, summaries, locations, return_unmaskeds_summ_toks=False):
        # this is very expensive function
        input_ids, labels = [], []
        unmaskeds = [torch.LongTensor(x) for x in self.tokenizer(contents, truncation=True, max_length=self.max_input_length).input_ids]
        summ_toks = [torch.LongTensor(x) for x in self.tokenizer(summaries, truncation=True, max_length=self.max_output_length).input_ids]
        for locs, unmasked, summ_tok in zip(locations, unmaskeds, summ_toks):
            # unmasked = torch.LongTensor(
            #     self.tokenizer.encode(content, truncation=True, max_length=self.max_input_length)
            # )
            masked = torch.LongTensor(
                self.kw_ex.mask_doc(unmasked, locs,
                    self.masking_scheme, self.n_hot_masks, self.n_loc_masks)
            )
            # summ_tok = torch.LongTensor(
            #     self.tokenizer.encode(summary, truncation=True, max_length=self.max_output_length)
            # )
            
            input_ids.append(torch.cat((summ_tok, masked[1:])))
            labels.append(torch.cat((summ_tok, unmasked[1:])))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        is_masked = input_ids.eq(torch.LongTensor([self.tokenizer.mask_token_id]))
        labels = (labels*is_masked) + (~is_masked * torch.LongTensor([-1]))
        attn_mask = 1 * ~torch.eq(input_ids, self.tokenizer.pad_token_id)

        input_ids = input_ids.to(self.device)
        is_masked = is_masked.to(self.device)
        labels = labels.to(self.device)
        attn_mask = attn_mask.to(self.device)
        if return_unmaskeds_summ_toks:
            return input_ids, is_masked, labels, attn_mask, unmaskeds, summ_toks
        else:
            return input_ids, is_masked, labels, attn_mask
        
    
    def get_loss_acc(self, contents, summaries, locations, grad_enabled=True):
        if grad_enabled:    
            input_ids, is_masked, labels, attn_mask = self.build_inputs(contents, summaries, locations)

            outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits
            loss = self.crit(logits.view(-1, self.tokenizer.vocab_size), labels.view(-1))
        else:
            with torch.no_grad():
                input_ids, is_masked, labels, attn_mask = self.build_inputs(contents, summaries, locations)

                outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
                logits = outputs.logits
                loss = self.crit(logits.view(-1, self.tokenizer.vocab_size), labels.view(-1))

        with torch.no_grad():        
            preds = torch.argmax(logits, dim=2)
            n_right = (preds.eq(labels) * is_masked).sum().float()
            n_masks = is_masked.sum()
        return loss, n_right, n_masks

    def score_non_threaded(self, summaries, contents, locs, accs_without=None):
        
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

            score = torch.nn.functional.relu(accs_with - accs_without) / self.scale_factor

            return score, accs_without

    def score_with_only(self, summaries, contents, locs, accs_without=None):
        
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
            return accs_with

    def score(self, summaries, contents, locs, prebuild_withouts=None, prebuild_summaries=None, accs_without=None):
        
        with torch.no_grad():
            if prebuild_summaries is not None:
                input_ids, is_masked, labels, attn_mask = prebuild_summaries
            else:
                input_ids, is_masked, labels, attn_mask = self.build_inputs(contents, summaries, locs)
            outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
            preds = torch.argmax(outputs.logits, dim=2)
            n_masks = torch.sum(is_masked, dim=1).float()
            accs_with = torch.sum(preds.eq(labels).long() * is_masked, dim=1).float() / n_masks
            accs_with = torch.nan_to_num(accs_with, nan=0)
            if accs_without is None:
                input_ids, is_masked, labels, attn_mask = prebuild_withouts
                outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
                preds = torch.argmax(outputs.logits, dim=2)
                n_masks = torch.sum(is_masked, dim=1).float()
                accs_without = torch.sum(preds.eq(labels).long() * is_masked, dim=1).float() / n_masks
                accs_without = torch.nan_to_num(accs_without, nan=0)
            
            # score = torch.nn.functional.relu(accs_with - accs_without) / self.scale_factor
            score = accs_with - accs_without
            return score, accs_without