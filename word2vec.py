from nltk.corpus import webtext
import text_preprocess
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import time
class word2vec():
    def __init__(self,parameters):
        self.window_size=parameters["train"]["window"]
        self.epoch=parameters["train"]["epoch"]
        self.learning_rate=parameters["train"]["lr"]


    def pre_process(self, corpus,ishtml=False,ispara=False):
        # print(len(corpus))
        if(ishtml==True):
            corpus=text_preprocess.strip_html(corpus)

        corpus=text_preprocess.replace_contractions(corpus)
        corpus=corpus.lower()

        if ispara:
            linewordcorpus = []
            corpline=corpus.splitlines()
            # print(len(corpline))
            for i in corpline:
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
        # print(len(linewordcorpus))
        return linewordcorpus

    def gen_training_data(self,linewordcorpus):

        word_cnts=defaultdict(int)
        for line in linewordcorpus:
            for word in line:
                word_cnts[word] += 1

        self.unique_word=len(word_cnts.keys())

        self.word_list=sorted(list(word_cnts.keys()),reverse=False)


        print(len(linewordcorpus[0]),self.unique_word)
        self.word_index=dict((word,i) for i,word in enumerate(self.word_list))
        self.index_word = dict((i,word) for i,word in enumerate(self.word_list))
        training_data=[]
        for sentence in linewordcorpus:
            sentence_lenth=len(sentence)
            for i,word in enumerate(tqdm(sentence, desc="gen_data")):
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

    def weight_initializers(self,  w1_shape, w2_shape):
        return np.random.uniform(-1,1,w1_shape), np.random.uniform(-1,1,w2_shape)


    def train(self, training_data):
        self.hidden_size=12
        self.w1,self.w2=self.weight_initializers((self.unique_word, self.hidden_size), (self.hidden_size, self.unique_word))
        lenth=len(training_Data)
        # print(self.w1.shape, self.w2.shape)
        ep=tqdm(range(self.epoch),desc="Epoch")
        for i in ep:

            self.loss=0
            for cnt, word in enumerate(training_Data):
                word_target, context_word=word
                # print(context_word)
                # print(cnt,end=" ")
                y_pred, oup, hid= self.forward_pass(word_target)
                # print(y_pred.shape, oup.shape, hid.shape)
                Err=np.sum([np.subtract(y_pred,word) for word in context_word],axis=0)
                self.backprop(Err, hid, word_target)

                self.loss+= -np.sum([oup[wc.index(1)] for wc in context_word]) + \
                            len(context_word)*np.log(np.sum(np.exp(oup)))

            ep.set_description("{:.04f}".format(self.loss/lenth))
        # print(Err.shape, hid.shape, len(word_target))
        # print(len(word_target))


    def backprop(self, Errr, hid, word_target):
        dW2=np.outer(hid,Errr)

        # dW1=np.outer(word_target, )
        dW1=np.outer(word_target, np.dot(self.w2,Errr.T))

        self.w2-=(self.learning_rate) * dW2
        self.w1-=(self.learning_rate)*dW1



    def softmax(self, x):
        e_x=np.exp(x-np.max(x))
        return e_x/e_x.sum(axis=0)

    def forward_pass(self, inpt):
        hidden=np.dot(inpt, self.w1)
        output=np.dot(hidden,self.w2)

        y=self.softmax(output)
        return y,output,  hidden

    def word_vec(self,word):
        word_ind=self.word_index[word]
        return self.w1[word_ind]

    def vec_sim(self,word, top_n):
        vet_w1=self.word_vec(word)

        word_sim={}
        for i in range(self.unique_word):
            vet_w2=self.w1[i]
            theta_sum=np.dot(vet_w1, vet_w2)
            theta_den=np.linalg.norm(vet_w1)+np.linalg.norm(vet_w2)
            theta=theta_sum/theta_den

            word=self.index_word[i]
            word_sim[word]=theta
        sort_word=sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)


        for word, sim in sort_word[:top_n]:
            print(word,sim)












fx=webtext.raw(webtext.fileids()[0])
corpus=fx[:1000]
print(corpus)
settings={"train":{"window":2,"epoch":3000,"lr":0.01 }}
w2=word2vec(settings)
pre_pr=w2.pre_process(corpus,ispara=False)
# print(corpus)
training_Data=w2.gen_training_data(pre_pr)
w2.train(training_Data)
t_word="phoenix"
print(w2.word_vec(t_word))
w2.vec_sim(t_word,5)
# print(training_Data.size*training_Data.itemsize)








