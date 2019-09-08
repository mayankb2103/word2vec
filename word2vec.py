from nltk.corpus import webtext
import text_preprocess
class word2vec():
    def __init__(self,parameters):
        self.window_size=parameters["train"]["window"]
        self.epoch=parameters["train"]["epoch"]
        self.learning_rate=parameters["train"]["lr"]




    def pre_process(self, corpus,ishtml=False):
        if(ishtml==True):
            corpus=text_preprocess.strip_html(corpus)

        corpus=text_preprocess.replace_contractions(corpus)
        corpus=corpus.lower()


        wordcorpus=text_preprocess.tokenize(corpus)

        wordcorpus = text_preprocess.remove_non_ascii(wordcorpus)
        wordcorpus=text_preprocess.remove_punctuation(wordcorpus)
        wordcorpus=text_preprocess.replace_numbers(wordcorpus)

        return wordcorpus
    def gen_training_data(self):
        pass


fx=webtext.raw(webtext.fileids()[0])
corpus=fx[:500]
settings={"train":{"window":2,"epoch":5,"lr":0.01 }}
w2=word2vec(settings)
a=w2.pre_process(corpus)















