# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 21:55:00 2019

@author: 
"""
from nltk.stem import PorterStemmer
import xml.dom.minidom
import networkx as nx
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer 
import itertools
from  more_itertools import unique_everseen
import numpy as np

try1 = __import__('try1')

def main():
    
    train_set, test_set  = get_dataset("test", t="word", stem_or_not_stem = "not stem")
    true_labels = json_references(stem_or_not_stem = "not stem")
    
    
    prior_weights = get_prior_weights(train_set, test_set, variant = "bm25")
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
                    if word == "." or word == "%" or word == "'s" or word == ",":   #MUDANÇA
                        sentence_string = sentence_string + word
                    else:
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
    
    prior_weights = []
            
    if variant == "length_and_position":      
        #MAL
        for doc in test_set.values():                     
            words_nodes = try1.extractKeyphrasesTextRank(doc[0])
            words_nodes = unique_everseen(words_nodes)
            count_sent = 1
            #print(doc)
            #print(len(words_nodes))      6
            prior_weights_sent = []
            for sent in words_nodes:
                #print(sent)             #6
                length_position_sentence = []
                for gram in sent:
                    #print("gram", gram)
                    length_position_sentence.append(len(gram.split()) + (1/count_sent)) #fator inversamente proporcional
                #print(length_position_sentence)
                prior_weights_sent.append(length_position_sentence)
                count_sent += 1
                #print("sent", sent, prior_weights_sent)

            prior_weights.append(prior_weights_sent)
        print(prior_weights)
        print(len(list(itertools.chain.from_iterable(prior_weights))))
        raise
        
    elif(variant == "tfidf" or variant == "bm25"):

        for doc in list(train_set):
            words_nodes += list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(doc)))
        words_nodes = list(unique_everseen(words_nodes))
        
        if variant == "tfidf":
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
        
        if variant == "bm25":
            bm25 = BM25(train_set)
            
            get_score = bm25._get_scores()
            
            print(get_score)
            raise
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
############################################BM25############################################
import math
from six import iteritems
from six.moves import range
from scipy.sparse import lil_matrix

PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class BM25(object):
    """Implementation of Best Matching 25 ranking function.
    Attributes
    ----------
    corpus_size : int
        Size of corpus (number of documents).
    avgdl : float
        Average length of document in `corpus`.
    doc_freqs : list of dicts of int
        Dictionary with terms frequencies for each document in `corpus`. Words used as keys and frequencies as values.
    idf : dict
        Dictionary with inversed documents frequencies for whole `corpus`. Words used as keys and frequencies as values.
    doc_len : list of int
        List of document lengths.
    """

    def __init__(self, corpus):
        """
        Parameters
        ----------
        corpus : list of list of str
            Given corpus.
        """
        self.corpus_size = 0
        self.avgdl = 0
        self.no_terms = 0
        self.terms= []
        self.doc_freqs = []
        self.idf = []
        self.doc_len = []
        self._initialize(corpus)

    def _initialize(self, corpus):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        nd = {}  # word -> number of documents with word
        num_doc = 0
        
        for document in corpus:
            self.corpus_size += 1
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in self.terms:
                        self.terms.append(word)
                        self.no_terms +=1
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = float(num_doc) / self.corpus_size
        # collect idf sum to calculate an average idf for epsilon value
        
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        
        for document in corpus:
            inv_doc_freq = {}
            negative_idfs = []
            idf_sum = 0
            for word in document:
                for word, freq in iteritems(nd):
                    inv_doc_freq[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
                    idf_sum += inv_doc_freq[word]
                    if inv_doc_freq[word] < 0:
                        negative_idfs.append(word)
                
                self.average_idf = float(idf_sum) / len(inv_doc_freq)
                for word, freq in iteritems(nd):
                    for word in negative_idfs:
                        inv_doc_freq[word] = EPSILON * self.average_idf
            self.idf.append(inv_doc_freq)
        self.tfidf_matrix = lil_matrix((self.corpus_size, self.no_terms), dtype=float)
        
    def _get_scores(bm25, document):
        """Helper function for retrieving bm25 scores of given `document`
        in relation to every item in corpus.
        
        Parameters
        ----------
        bm25 : BM25 object
            BM25 object fitted on the corpus where documents are retrieved.
        document : list of str
            Document to be scored.
            
        Returns
        -------
        list of float
            BM25 scores.
        """
        return bm25.get_scores(document)
    
    def get_scores(self, document):
        """Computes and returns BM25 scores of given `document` in relation to
        every item in corpus.
        Parameters
        ----------
        document : list of str
            Document to be scored.
        Returns
        -------
        list of float
            BM25 scores.
        """
        for index in range(self.corpus_size):
            scores = self.get_score(document, index)
        return scores
    
    def get_score(self, document, index):
        """Computes BM25 score of given `document` in relation to item of corpus selected by `index`.
        Parameters
        ----------
        document : list of str
            Document to be scored.
        index : int
            Index of document in corpus selected to score with `document`.
        Returns
        -------
        float
            BM25 score.
        """
        score = 0
        doc_freqs = self.doc_freqs[index]
        for word in document:
            if word not in doc_freqs:
                continue
            score += (self.idf[word] * doc_freqs[word] * (PARAM_K1 + 1)
                      / (doc_freqs[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.doc_len[index] / self.avgdl)))
        return score
        
    

#    def get_score(self, doc_index, index_term):
#        score = 0
# 
#        word = self.terms[index_term]
#        if word in self.idf[doc_index].keys():
#            if word in self.doc_freqs[doc_index].keys():           
#                term_freqs = self.doc_freqs[doc_index][word]
#                score = (self.idf[doc_index][word]) * (term_freqs * (PARAM_K1 + 1)
#                        / (term_freqs + PARAM_K1 * (1- PARAM_B + PARAM_B * self.doc_len[doc_index]/ self.avgdl)))
#        return score
#    
#    def get_scores(self):
#        for doc_index in range(0, self.corpus_size):
#            for term_index in range(0, self.no_terms):
#                tfidf = self.get_score(doc_index, term_index)  #* len(" ".split(self.terms[term_index]))
#                self.tfidf_matrix[doc_index, term_index] = tfidf
#        #print("tfidf_matrix", self.tfidf_matrix)
#        return self.tfidf_matrix