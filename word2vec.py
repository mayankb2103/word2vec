from nltk.corpus import webtext
import text_preprocess
from collections import defaultdict

class word2vec():
    def __init__(self,parameters):
        self.window_size=parameters["train"]["window"]
        self.epoch=parameters["train"]["epoch"]
        self.learning_rate=parameters["train"]["lr"]




    def pre_process(self, corpus,ishtml=False,ispara=False):
        print(len(corpus))
        if(ishtml==True):
            corpus=text_preprocess.strip_html(corpus)

        corpus=text_preprocess.replace_contractions(corpus)
        corpus=corpus.lower()

        if ispara:
            linewordcorpus = []
            corpline=corpus.splitlines()
            for i in corp:
                wordcorpus=text_preprocess.tokenize(corpus)
                wordcorpus = text_preprocess.remove_non_ascii(wordcorpus)
                wordcorpus = text_preprocess.remove_punctuation(wordcorpus)
                wordcorpus = text_preprocess.replace_numbers(wordcorpus)
                linewordcorpus.append(wordcorpus)
        else:
            wordcorpus = text_preprocess.tokenize(corpus)
            wordcorpus = text_preprocess.remove_non_ascii(wordcorpus)
            wordcorpus = text_preprocess.remove_punctuation(wordcorpus)
            wordcorpus = text_preprocess.replace_numbers(wordcorpus)
            linewordcorpus=[wordcorpus]
        print(len(linewordcorpus))
        return linewordcorpus
    def gen_training_data(self,linewordcorpus):
        word_cnts=defaultdict(int)
        for line in linewordcorpus:
            for word in line:
                word_cnts[word] += 1

        self.unique_word=len(word_cnts.keys())
        self.word_list=sorted(list(word_cnts.keys()),reverse=False)
        print(self.word_list)
        self.word_index=dict((word,i) for i,word in enumerate(self.word_list))
        self.index_word = dict((i,word) for i,word in enumerate(self.word_list))
        training_data=[]
        for sentence in linewordcorpus:
            sentence_lenth=len(sentence)
            for i,word in enumerate(sentence):
                target_word=self.word_onehot(word)
                word_context = []
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j != i and j <= sentence_lenth - 1 and j >= 0:
                        word_context.append(self.word_onehot(sentence[j]))
                # print(word_context)
                training_data.append([target_word, word_context])



        return np.array(training_data)


    def word_onehot(self, word):
        word_vec=[0]*self.unique_word
        word_vec[self.word_index[word]]=1
        return word_vec


fx=webtext.raw(webtext.fileids()[0])
corpus=fx
settings={"train":{"window":2,"epoch":5,"lr":0.01 }}
w2=word2vec(settings)
pre_pr=w2.pre_process(corpus,ispara=True)
train_data=w2.gen_training_data(pre_pr)
print(train_data.shape)













