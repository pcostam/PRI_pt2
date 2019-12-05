# -*- coding: utf-8 -*-
"""
EXERCISE_1
"""
import sys
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
import string
import networkx as nx
import regex as re
import itertools

#https://xang1234.github.io/textrank/ == corpus
fIn = "corpus.txt"
stopwords_punctuation  = set(stopwords.words('english')).union(string.punctuation)

def main():
    file_content = open_file()
    nodes = extractKeyphrasesTextRank(str(file_content))
    graph = buildGraph(nodes, exercise2 = False)  
    
    nstart = dict()
    nodes_set = set(itertools.chain.from_iterable(nodes))
    for gram in nodes_set:
        nstart[gram] = 1/len(nodes_set)
    
    pagerank_scores = nx.pagerank(graph, max_iter = 50, nstart = nstart)
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
    
    # remove stopwords, punctuation and digits also makes lowering
    sent_tokens_clean =[]
    for sent in sent_tokens:
        lst=[]
        for word in word_tokenize(sent):
            if (word.lower() not in stopwords_punctuation) and not(re.match('\d+', word)):
                lst.append(word.lower())
        sent_tokens_clean.append(' '.join(lst))
        
    # assign POS tags to the words in the text
    sent_tokens_filtered = []
    for sent in sent_tokens_clean:    
        tagged_sents = nltk.pos_tag(nltk.word_tokenize(sent))
        for item in tagged_sents:
            if item[1] in tags and sent not in sent_tokens_filtered:
                sent_tokens_filtered.append(sent)
                
    #List with uni- bi- and trigrams per sentence
    words_nodes=list()
    for sent in sent_tokens_filtered:
                
        sent_grams=list()
        unigram = list(nltk.ngrams(nltk.word_tokenize(sent), 1))
        for gram in unigram:
            sent_grams.append(' '.join(gram))
        #print(uni_grams)
        bigram = list(nltk.ngrams(nltk.word_tokenize(sent), 2))
        for gram in bigram:
            sent_grams.append(' '.join(gram))
            
        trigram = list(nltk.ngrams(nltk.word_tokenize(sent), 3))
        for gram in trigram:
            sent_grams.append(' '.join(gram))
            
        words_nodes.append(sent_grams)           
    return words_nodes
      
def buildGraph(nodes, edge_weights = [], exercise2 = False): 
    gr = nx.Graph()  # initialize an undirected graph
    for sent in nodes:
        gr.add_nodes_from(sent)
        if  exercise2 == False:
            for gram1 in range(0, len(sent)):
                for gram2 in range(gram1, len(sent)):
                    if sent[gram1] != sent[gram2] and sent[gram1] not in sent[gram2] and sent[gram2] not in sent[gram1]:
                            gr.add_edge(sent[gram1], sent[gram2], weight = 1)
    if exercise2 == True:
        gr.add_weighted_edges_from(edge_weights)
    return gr

from collections import Counter 
def get_top_x(pagerank_scores, x):
    k = Counter(pagerank_scores)
    top_x = k.most_common(x)
    return top_x

#if __name__ == "__main__":
#    main()

#https://www.geeksforgeeks.org/page-rank-algorithm-implementation/