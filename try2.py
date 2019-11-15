# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 21:55:00 2019

@author: LADYMARTINHA
"""
from nltk.stem import PorterStemmer
import xml.dom.minidom
import networkx as nx
import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
import itertools
from  more_itertools import unique_everseen
import numpy as np

try1 = __import__('try1')

def main():
    
    train_set, test_set  = get_dataset("test", t="word", stem_or_not_stem = "not stem")
    true_labels = json_references(stem_or_not_stem = "not stem")
    
    
    prior_weights = get_prior_weights(train_set, test_set, variant = "tfidf")
    #print(prior_weights)
    
#    for doc in docs:
#        #nodes = try1.extractKeyphrasesTextRank(doc) 
#        #talvez se possa separar na equação
#        
#        
#        graph = try1.buildGraph(nodes)
#        pagerank_scores = nx.pagerank(graph, max_iter = 50)
    
    
#Process XML file. Preprocesses text
#Input:path to file, token (optional)"word" or "lemma", token(optional)"stem" or "not stem"
#Output: list where each element corresponds to a document (string) and list of lists where 
#intern lists has the test documents (strings) as elements
#Notes: can't be "lemma" and "stem" at the same time
def get_dataset(folder, t="word", stem_or_not_stem = "not stem"):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\Inspec\\" + folder
    ps = PorterStemmer()
    test_set = dict()
    files = list()
    docs = dict()
    file_counter = 0
    i = 0
    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))
                
    for f in files:
        i += 1
        text = str()
        base_name=os.path.basename(f)
        key = os.path.splitext(base_name)[0]
        doc = xml.dom.minidom.parse(f)

        # get a list of XML tags from the document and print each one
        sentences = doc.getElementsByTagName("sentence")
      
        for sentence in sentences:
                tokens = sentence.getElementsByTagName("token")
                sentence_string = ""
                for token in tokens:
                    word = token.getElementsByTagName(t)[0].firstChild.data
                    if (t == 'word' and stem_or_not_stem == 'stem'):
                        word = ps.stem(word)
                        
                    sentence_string = sentence_string + " " + word
          
                text += sentence_string
        
       #add dictionary. key is name of file.
        if(file_counter <= 375):
            docs[key] = text
        else:
            test_set[key] = [text]

        file_counter += 1

        
    return docs.values(), test_set

#From "\Inspec\references" extracts the real keyphrases of the documents
#Input: token (optional) "stem" or "not stem"
#Output: dictionary with n-grams where n < 4.
#Notes: Key is filename; value is a list containing lists of keyphrases. 
def json_references(stem_or_not_stem = 'not stem'):
    data = dict()    
    
    path = os.path.dirname(os.path.realpath('__file__')) + "\\Inspec\\references"
    if stem_or_not_stem == 'not stem':
        filename = os.path.join(path,"test.uncontr.json")
    elif stem_or_not_stem == 'stem':
        filename = os.path.join(path,"test.uncontr.stem.json")
        
    with open(filename) as f:    
        docs = json.load(f)
        
        for key, value in docs.items():
            aux_list = []
            for gram in value:
                size = len(gram[0].split(" "))
                if(size==1 or size == 2 or size == 3):
                    aux_list.append(gram[0])
            value = aux_list
            data[key] = value

    return data

def get_prior_weights(train_set, test_set, variant = "length_and_position"):
    words_nodes = []
    
    prior_weights=[]
    
    for doc in test_set.values():
        words_nodes += list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(doc[0])))
    words_nodes = list(unique_everseen(words_nodes))
        
    if variant == "length_and_position":
        for doc in test_set.values():
            words_nodes = try1.extractKeyphrasesTextRank(doc[0])
            count_sent = 1
            for sent in words_nodes:
                length_position_sentence = []
                for gram in sent:
                    length_position_sentence.append(len(gram.split()) + (1/count_sent)) #fator inversamente proporcional
                prior_weights.append(length_position_sentence)
                count_sent += 1
            
    if variant == "tfidf":
       normalized_prior_weights = []
       #words_nodes = list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(doc)))
       vectorizer = TfidfVectorizer(use_idf = True, 
                                    analyzer = 'word', 
                                    ngram_range = (1,3), 
                                    stop_words = 'english',
                                    token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b", 
                                    lowercase = True,
                                    vocabulary = iter(list(unique_everseen(words_nodes))))
                                    
       X = vectorizer.fit_transform(train_set)
       
       for doc in test_set.values():        
           Y = vectorizer.transform(doc)
       unnormalized_prior_weights = Y.toarray()
       
       sum_prior_weights = np.sum(unnormalized_prior_weights, axis=0)
       print(sum_prior_weights.shape)
       for sum_docs in sum_prior_weights:
           prior_weights.append(sum_docs/len(test_set.values()))
       print(len(prior_weights))
       #print(X.toarray())
       #print(vectorizer.get_feature_names())
    
#    if variant == "bm25":
#        for sent in words:
#            #TODO
#        
#    else:
#        print(">>UNKNOWN>>PRIOR WEIGHTS>>VARIANT")
        
    return prior_weights 
          
        
runtime = {}
runtime["co-occurrences"] = None       
def get_edge_weights(gr, variant = "co-occurrences"):
    if runtime["co-occurrences"] is None:
        for candidate in gr.nodes:
            runtime["co-occurrences"].append(gr.degree[candidate])
        
    
#    if variant == "co-occurrences":
#        
#        
#    if variant == "embeddings":
        
    
    
    
#if __name__ == "__main__":
#    main()
    

#ver similaridade no espaço de cada palavra
#weigth