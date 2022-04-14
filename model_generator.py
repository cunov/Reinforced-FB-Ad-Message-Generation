from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from parameters import Parameters

params = Parameters()

class Fluency():
    def __init__(self, load_pretrained='',
                fluency_scaler=params.fluency_scaler, device=params.device,
                max_output_length=params.max_output_length, max_input_length=params.max_input_length):
        if load_pretrained=='fluency':
            self.model = GPT2LMHeadModel.from_pretrained(params.trained_fluency_model_dir)
            self.tokenizer = GPT2Tokenizer.from_pretrained(params.trained_fluency_tokenizer_dir)
            print('Loaded fine-tuned fluency model, tokenizer')
        elif load_pretrained=='summarizer':
            self.model = GPT2LMHeadModel.from_pretrained(params.trained_summarizer_model_dir)
            self.tokenizer = GPT2Tokenizer.from_pretrained(params.trained_summarizer_tokenizer_dir)
            print('Loaded pretrained summarizer model, tokenizer')
        else:
            bos_token, eos_token, pad_token ='<|startoftext|>', '<|endoftext|>', '<|pad|>'
            tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', bos_token=bos_token, eos_token=eos_token, pad_token=pad_token)
            model = GPT2LMHeadModel.from_pretrained('distilgpt2')
            model.resize_token_embeddings(len(tokenizer))
            self.model = model
            self.tokenizer = tokenizer
            print('Loaded default fluency model, tokenizer')
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = device
        self.fluency_scaler = fluency_scaler
        self.model.to(device)
        self.special_token_ids = [self.tokenizer.encode(x) for x in list(self.tokenizer.special_tokens_map.values())]

    def preprocess_input(self, batch_msgs):
        inputs = self.tokenizer([self.tokenizer.bos_token + msg for msg in batch_msgs], return_tensors='pt', padding=True, truncation=True, max_length=self.max_output_length).to(self.device)
        outputs = self.tokenizer([msg + self.tokenizer.eos_token for msg in batch_msgs], return_tensors='pt', padding=True, truncation=True, max_length=self.max_output_length).to(self.device)
        for i,inpid in enumerate(outputs.input_ids):
            for j,item in enumerate(inpid):
                if item == self.tokenizer.pad_token_id:
                    outputs['input_ids'][i][j] = -1 
            if not self.tokenizer.eos_token_id in inpid:
                outputs['input_ids'][i][-1] = self.tokenizer.eos_token_id
        return inputs, outputs

    def score(self, batch_msgs):
        with torch.no_grad():
            summ_inp, summ_out = self.preprocess_input(batch_msgs)
            outputs = self.model(**summ_inp)
            logits = outputs.logits
            crit = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = crit(logits.view(-1, len(self.tokenizer)), summ_out.input_ids.view(-1)).view(summ_out.input_ids.shape)
            non_pads = ~torch.eq(summ_inp.input_ids, self.tokenizer.pad_token_id)
            non_pad_cnts = torch.sum(non_pads, dim=1).to(self.device)
            loss_per = torch.sum(loss, dim=1) / non_pad_cnts
            
            score = (self.fluency_scaler - loss_per) / self.fluency_scaler
            return score

    def decode_batch(self, bodies, input_past=None, sample=False, return_logprobs=False):
        N = len(bodies)
        current = self.tokenizer([self.tokenizer.bos_token] * N, return_tensors='pt').input_ids.to(self.device)
        build_up = None
        total_logprobs = []
        with torch.no_grad():
            if input_past is None:
                inputs = self.tokenizer(bodies, return_tensors='pt', padding=True, truncation=True, max_length=self.max_input_length).to(self.device)
                input_past = self.model(**inputs)
            past = input_past.past_key_values

        while build_up is None or (build_up.shape[1] < self.max_output_length and not all([self.tokenizer.eos_token_id in build for build in build_up])):
            outputs = self.model(current, past_key_values=past)
            logits = outputs.logits
            past = outputs.past_key_values
            probs = torch.nn.functional.softmax(logits, dim=2).squeeze(1)
            logprobs = torch.nn.functional.log_softmax(logits, dim=2)
            if sample:
                current = torch.multinomial(probs, 1)
            else:
                current = torch.argmax(logprobs, dim=2)
            
            if build_up is None:
                build_up = current
            else:
                build_up = torch.cat((build_up, current), dim=1)
            
            if return_logprobs:
                selected_logprobs = logprobs[:, 0, current.squeeze()].unsqueeze(1)
                total_logprobs.append(selected_logprobs)

        build_up = [[token for token in build if not token in self.special_token_ids] for build in build_up]
        end_idxs = [self.max_output_length+1 if self.tokenizer.eos_token_id not in build else build.index(self.tokenizer.eos_token_id) for build in build_up]
        outputs = [self.tokenizer.decode(build[:idx]) for build,idx in zip(build_up, end_idxs)]
        if return_logprobs:
            return outputs, torch.cat(total_logprobs, dim=1), input_past, end_idxs
        else:
            return outputs, end_idxs