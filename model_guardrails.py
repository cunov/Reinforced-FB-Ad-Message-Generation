import nltk, math
from collections import Counter
import torch
from parameters import Parameters

STOP_WORDS = set(["'", ".", "!", "?", ",", '"', '-', 'we', 'our', 'you', 'he', 'him', 'she', 'her', 'it', "it's", 'its', 'they', 'their', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'the', 'and', 'or', 'as', 'of', 'at', 'by', 'to', 'not', 'so', "'s", "in", "for", "with", "on"])

class PatternPenalty:
    # Depending on how many words are used a large fraction of the last X summaries
    def __init__(self, history_length=30):
        self.stop_words = STOP_WORDS
        self.history_words = []
        self.ngram_history = []
        self.history_length = history_length
        self.device = Parameters().device

    def score(self, summaries, lengths=None):
        batch_words = []
        batch_ngrams = []
        for summary in summaries:
            words = nltk.tokenize.word_tokenize(summary.lower())
            gram = 2
            n_grams = [tuple(words[i:(i+gram)]) for i in range(len(words)-gram+1)]

            word_set = set(words)-self.stop_words
            word_set = [w for w in word_set if len(w) > 1]
            self.history_words.append(word_set)
            self.ngram_history.append(n_grams)
            batch_words.append(word_set)
            batch_ngrams.append(n_grams)

        self.history_words = self.history_words[-self.history_length:] # Trim
        self.ngram_history = self.ngram_history[-self.history_length:] # Trim

        word_counter = Counter([w for words in self.history_words for w in words])
        ngram_counter = Counter([ng for ngrams in self.ngram_history for ng in ngrams])

        scores = []
        for words, ngrams in zip(batch_words, batch_ngrams):
            score = 0.0

            if any(word_counter[w] > 0.5*self.history_length for w in words):
                score = 1.0
            if any(ngram_counter[ng] > 0.5*self.history_length for ng in ngrams):
                score = 1.0
                # print(">>>",ngram_counter.most_common(8))
            scores.append(score)
        return torch.LongTensor(scores).to(self.device)

class LengthPenalty:
    # Depending on how many words are used a large fraction of the last X summaries
    def __init__(self, target_length):
        self.target_length = float(target_length)
        self.device = Parameters().device

    def score(self, summaries, lengths=None):
        # In lengths, the number of tokens. Is -1 if the summary did not produce an END token, which will be maximum penalty, by design.
        # scores = [1.0-L/self.target_length for L in lengths]
        scores = [1.0 if L > self.target_length else 1.0-L/self.target_length for L in lengths] # This lets it go beyond for free

        return torch.LongTensor(scores).to(self.device)

class RepeatPenalty:
    # Shouldn't use non-stop words several times in a summary. Fairly constraining.
    def __init__(self):
        self.stop_words = STOP_WORDS
        self.device = Parameters().device

    def score(self, summaries, lengths=None):
        scores = []
        for summary in summaries:
            words = nltk.tokenize.word_tokenize(summary.lower())
            L = len(words)
            N_1 = max(2, math.ceil(L * 0.1)) # No word should be used more than 10% of the time
            word_counts = Counter([w for w in words if w.lower() not in self.stop_words])
            if word_counts.most_common(1)[0][1] > N_1:

            # all_word_counts = Counter([w for w in words if len(w) > 1])
            # if len(word_counts) > 0 and len(all_word_counts) > 0 and (word_counts.most_common(1)[0][1] > N_1 or all_word_counts.most_common(1)[0][1] > N_2):
                # print(L, N_1, N_2)
                # print("Repeat penalty:", word_counts.most_common(3), all_word_counts.most_common(3))
                scores.append(1.0)
            else:
                scores.append(0.0)
        return torch.LongTensor(scores).to(self.device)