# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:59:16 2019

@author: LADYMARTINHA
"""
#https://xang1234.github.io/textrank/ == corpus
import sys
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
import string
from nltk.stem import WordNetLemmatizer 
import networkx as nx
import itertools
from collections import Counter 

fIn = "corpus.txt"
stopwords_punctuation  = set(stopwords.words('english')).union(string.punctuation)
tag = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']

def main():
    file_content = open_file()
    nodes = extractKeyphrasesTextRank(str(file_content), tag) 
    graph = buildGraph(nodes)
    pagerank_scores = nx.pagerank(graph)
    print(pagerank_scores)
    return get_top_x(pagerank_scores, 5)

def open_file():
    try:
        with open(fIn, 'r') as f:
          file_content = f.read()
          print("read file " + fIn)
          return file_content
        if not file_content:
          print("no data in file " + fIn)
    except IOError as e:
       print("I/O error({0}): {1}".format(e.errno, e.strerror))
    except: #handle other exceptions such as attribute errors
       print("Unexpected error:", sys.exc_info()[0])
       
def extractKeyphrasesTextRank(file_content, tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']):
    # tokenize into sentences
    sent_tokens = sent_tokenize(file_content)
    
    # remove stopwords, punctuation also makes lowering and lemmanize
    sent_tokens_clean =[]
    lmtzr = WordNetLemmatizer()
    for sent in sent_tokens:
        lst=[]
        for word in word_tokenize(sent):
            if word.lower() not in stopwords_punctuation:
                lst.append(lmtzr.lemmatize(word.lower()))
        sent_tokens_clean.append(' '.join(lst))
    #print(sent_tokens_clean)
    
    # assign POS tags to the words in the text
    sent_tokens_filtered = []
    for sent in sent_tokens_clean:    
        tagged_sents = nltk.pos_tag(nltk.word_tokenize(sent))
        for item in tagged_sents:
            if item[1] in tags and sent not in sent_tokens_filtered:
                sent_tokens_filtered.append(sent)
    #print(sent_tokens_filtered)
        
    #List with uni- bi- and trigrams per sentence
    words_nodes=list()
    for sent in sent_tokens_filtered:
        sent_grams=list()
        unigram = list(itertools.combinations(nltk.word_tokenize(sent), 1))
        for gram in unigram:
            sent_grams.append(' '.join(gram))
        bigram = list(itertools.combinations(nltk.word_tokenize(sent), 2))
        for gram in bigram:
            sent_grams.append(' '.join(gram))
        trigram = list(itertools.combinations(nltk.word_tokenize(sent), 3))
        for gram in trigram:
            sent_grams.append(' '.join(gram))
        words_nodes +=[sent_grams]
    #print(words_nodes)
    return words_nodes
      
def buildGraph(nodes): 
    gr = nx.Graph()  # initialize an undirected graph
    #print("nodes>>>>", nodes)
    for sent in nodes:
        gr.add_nodes_from(sent)
        for gram1 in range(0, len(sent)):
            for gram2 in range(gram1, len(sent)):
                if sent[gram1] != sent[gram2] and sent[gram1] not in sent[gram2] and sent[gram2] not in sent[gram1]:
                     gr.add_edge(sent[gram1], sent[gram2], weight=1)
                     #print("add edge>>", sent[gram1], sent[gram2])
    return gr

def get_top_x(pagerank_scores, x):
    k = Counter(pagerank_scores)
    top_x = k.most_common(x)
    return top_x
