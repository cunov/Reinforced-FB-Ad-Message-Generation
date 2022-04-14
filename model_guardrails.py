import nltk, math
from collections import Counter
import torch
from parameters import Parameters
import re

params = Parameters()
STOP_WORDS = set(["'", '#', '%', '+', ".", "!", "?", ",", '"', '-', ')', '(', 'we', 'our', 'you', 'he', 'him', 'she', 'her', 'it', "it's", 'its', 'they', 'their', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'the', 'and', 'or', 'as', 'of', 'at', 'by', 'to', 'not', 'so', "'s", "in", "for", "with", "on", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

class PatternPenalty:
    # Penalize using the same n-gram >history_length/2 times in the last history_length summaries
    def __init__(self, history_length=30):
        self.stop_words = STOP_WORDS
        self.history_words = []
        self.ngram_history = []
        self.history_length = history_length
        self.device = Parameters().device
        self.n = 2

    def score(self, words_list):
        batch_ngrams = []
        for words in words_list:
            n_grams = nltk.bigrams(words)
            self.ngram_history.append(n_grams)
            batch_ngrams.append(n_grams)

        self.ngram_history = self.ngram_history[-self.history_length:] # Trim

        ngram_counter = Counter([ng for ngrams in self.ngram_history for ng in ngrams])

        scores = []
        for ngrams in batch_ngrams:
            if any(ngram_counter[ng] > 0.5*self.history_length for ng in ngrams):
                scores.append(1.0)
            else:
                scores.append(0.0)
        return torch.LongTensor(scores).to(self.device)

class LengthPenalty:
    def __init__(self, target_length):
        self.target_length = float(target_length)
        self.device = Parameters().device

    def score(self, words_list):
        scores = []
        for words in words_list:
            L = len(words)
            if L < 20:
                scores.append(1.0)
            elif L > 50:
                scores.append(1.0)
            else:
                scores.append(0.0)

        return torch.LongTensor(scores).to(self.device)

class RepeatPenalty:
     # No word should be used more than 5 times and no non-stop word should be used more than 3 times
    def __init__(self):
        self.stop_words = STOP_WORDS
        self.device = Parameters().device

    def score(self, words_list):
        scores = []
        for words in words_list:
            L = len(words)
            # N_1 = max(2, math.ceil(L * 0.1))
            word_counts = Counter([w.lower() for w in words])
            non_stop_word_counts = Counter([w.lower() for w in words if not w.lower() in self.stop_words])
            try:
                if word_counts.most_common(1)[0][1] > 5 or non_stop_word_counts.most_common(1)[0][1] > 3:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            except:
                # Empty summary, will receive low fluency and coverage
                scores.append(1.0)
        return torch.LongTensor(scores).to(self.device)

class InvalidCharacter:
    def __init__(self):
        self.invalid_chars = ['\n','\t','\r','*','#','@','~','\\','^','%','`','_',
                                '<|pad|>','{','}','[',']','=']
        self.device = Parameters().device
        self.regex = re.compile(r"[A-Za-z]\.[A-Za-z]") # likely a link e.g. http://www.abc.com, abc.com, abc.co.uk, abc.co

    def score(self, summaries):
        scores = []
        for summary in summaries:
            if any(char in summary for char in self.invalid_chars) or self.regex.search(summary):
                scores.append(1)
            else:
                scores.append(0)
        return torch.LongTensor(scores).to(self.device)


class Hotness:
    def __init__(self):
        with open(params.hotwords_filename,'r') as f:
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
            self.hot_tokens_set = set(self.hotwords.keys())
            self.device = params.device

    def score(self, tokenized_summs, tokenized_descs):
        scores = []
        for summary,desc in zip(tokenized_summs, tokenized_descs):
            hot_tokens_in_desc = set(desc).intersection(self.hot_tokens_set)
            score = 0.0
            for token in hot_tokens_in_desc:
                if token in summary:
                    score += self.hotwords[token]['val']
            if score > 0.015:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return torch.LongTensor(scores).to(self.device)



