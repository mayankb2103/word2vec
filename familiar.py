import nltk
import numpy as np

# nltk.download('nps_chat')
import pandas
from nltk import bigrams
from nltk.corpus import webtext
fx=webtext.raw(webtext.fileids()[0])
from nltk.corpus import nps_chat
chat=nps_chat.posts(nps_chat.fileids()[0])
print(len(chat))
fx=fx.replace("\n"," ")
fx=fx.replace("\r","")
fxl=fx.split(" ")
print(fxl[:100])

def generate_co_occurrence_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}

    # Create bigrams from all words in corpus
    bi_grams = list(bigrams(corpus))

    # Frequency distribution of bigrams ((word1, word2), num_occurrences)
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))

    # Initialise co-occurrence matrix
    # co_occurrence_matrix[current][previous]
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))

    # Loop through the bigrams taking the current and previous word,
    # and the number of occurrences of the bigram.
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_index[current]
        pos_previous = vocab_index[previous]
        co_occurrence_matrix[pos_current][pos_previous] = count
    co_occurrence_matrix = np.matrix(co_occurrence_matrix)

    # return the matrix and the index
    return co_occurrence_matrix, vocab_index

matrix, vocab_index = generate_co_occurrence_matrix(fxl[:15])
data_matrix = pandas.DataFrame(matrix, index=vocab_index,
                             columns=vocab_index)
print(data_matrix)