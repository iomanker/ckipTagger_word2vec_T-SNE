# -*- coding: utf-8 -*-
import os
import sys
import re
import logging
from gensim.models import word2vec
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontProperties
from ckiptagger import data_utils, WS
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def visualized(x,y,labels,filename):
    myfont = FontProperties(fname=r'./NotoSansCJK-Light.ttc')
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontproperties=myfont)
    plt.savefig(filename)

def word_embedding(modelfilename,num_randomchoice,saveimgfilename):
    model = word2vec.Word2Vec.load(modelfilename)
    logging.info("Loaded Word2Vec Model")
    labels = []
    tokens = []
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    logging.info("Tokens,labels append completely")
    tsne = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)
    logging.info("Builded T-SNE")
    new_values = tsne.fit_transform(tokens)
    logging.info("Finishing to transform")
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    choices = np.random.choice(len(x), num_randomchoice)
    x = np.asarray(x)[choices]
    y = np.asarray(y)[choices]
    logging.info("X,Y append completely")
    visualized(x,y,labels,saveimgfilename)

def make_word2vec(inputfilename,savefilename,num_limit):
    sentences = word2vec.LineSentence(inputfilename, limit=num_limit)
    model = word2vec.Word2Vec(sentences, size=250)
    model.save(savefilename)
    
def word_segmentation(inputfilename,outputfilename,stopstep):
    # You need to download ckip dataset at first.
    ckip_ws = WS("./data", disable_cuda=False)
    jieba_stopword_set = set()
    pattern = re.compile("[A-Za-z0-9]+")
    with open('jieba_dict/stopwords.txt','r',encoding='utf-8') as stopwords:
        for stopword in stopwords:
            jieba_stopword_set.add(stopword.strip('\n'))
    sentence_output = open(outputfilename,'w',encoding='utf-8')
    with open(inputfilename, 'r', encoding='utf-8') as content:
        for idx, line in enumerate(content):
            line = line.strip('\n')
            word_sentence_list = ckip_ws([line])
            for word in word_sentence_list[0]:
                if word not in jieba_stopword_set and\
                 pattern.match(word) is None:
                    sentence_output.write(word+' ')
            sentence_output.write('\n')
            if (idx + 1) % 10 == 0:
                logging.info("Executed {} of lines for word segmentations.".format(idx+1))
            if (idx+1) % stopstep == 0:
                break
    sentence_output.close()

if __name__ == "__main__":
    # './utils/wiki_zh_tw.txt'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    stopstep = 50
    word_segmentation("./utils/wiki_zh_tw.txt","wiki_seg.txt",stopstep)
    make_word2vec("wiki_seg.txt","word2vec_{}.model".format(stopstep),stopstep)
    word_embedding("word2vec_{}.model".format(stopstep),300,"tsne_result.png")