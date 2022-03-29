from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
from parameters import Parameters

params = Parameters()

class Fluency():
    def __init__(self, load_pretrained='',
                fluency_scaler=params.fluency_scaler, device='cuda',
                max_output_length=params.max_output_length, max_input_length=params.max_input_length):
        if load_pretrained=='fluency':
            self.model = GPT2LMHeadModel.from_pretrained(params.trained_fluency_model_dir)
            self.tokenizer = GPT2Tokenizer.from_pretrained(params.trained_fluency_tokenizer_dir)
            print('Loaded pretrained fluency model, tokenizer')
        elif load_pretrained=='summarizer':
            self.model = GPT2LMHeadModel.from_pretrained(params.trained_summarizer_model_dir)
            self.tokenizer = GPT2Tokenizer.from_pretrained(params.trained_summarizer_tokenizer_dir)
            print('Loaded pretrained summarizer model, tokenizer')
        else:
            bos_token, eos_token, pad_token ='<|startoftext|>', '<|endoftext|>', '<|pad|>'
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos_token, eos_token=eos_token, pad_token=pad_token) #gpt2-medium
            configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
            model = GPT2LMHeadModel.from_pretrained('gpt2', config=configuration)
            model.resize_token_embeddings(len(tokenizer))
            self.model = model
            self.tokenizer = tokenizer
            print('Loaded default fluency model, tokenizer')
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = device
        self.fluency_scaler = fluency_scaler
        self.model.to(device)

    def preprocess_input(self, batch_msgs):
        inputs = self.tokenizer([self.tokenizer.bos_token + msg for msg in batch_msgs], return_tensors='pt', padding=True, truncation=True, max_length=self.max_output_length).to(self.device)
        outputs = self.tokenizer([msg + self.tokenizer.eos_token for msg in batch_msgs], return_tensors='pt', padding=True, truncation=True, max_length=self.max_output_length).to(self.device)
        for i,inpid in enumerate(outputs.input_ids):
            if self.tokenizer.eos_token_id in inpid:
                continue
            outputs['input_ids'][i][-1] = self.tokenizer.eos_token_id
        return inputs, outputs

    def score(self, batch_msgs):
        summ_inp, summ_out = self.preprocess_input(batch_msgs)
        
        with torch.no_grad():
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
        scores = torch.zeros((N)).to(self.device)
        total_logprobs = []

        if input_past is None:
            inputs, outputs = self.preprocess_input(bodies)
            input_past = self.model(**inputs)
            # _, input_past = self.model(input_ids=inputs, past_key_values=None)
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
                selected_logprobs = logprobs[torch.arange(N), 0, current.squeeze()].unsqueeze(1)
                total_logprobs.append(selected_logprobs)
                
            not_finished = (~torch.any(build_up==self.tokenizer.eos_token_id, dim=1)).float()
            scores += not_finished * logprobs[torch.arange(N), :, current.squeeze(1)].squeeze()

        build_up = [build.tolist() for build in build_up]
        end_idxs = [self.max_output_length+1 if self.tokenizer.eos_token_id not in build else build.index(self.tokenizer.eos_token_id) for build in build_up]
        outputs = [self.tokenizer.decode(build) for build in build_up]
        if return_logprobs:
            return outputs, torch.cat(total_logprobs, dim=1), build_up, input_past, end_idxs
        else:
            return outputs, end_idxs