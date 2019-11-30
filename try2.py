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
from sklearn.feature_extraction.text import TfidfVectorizer 
import itertools
from  more_itertools import unique_everseen
from scipy.sparse import lil_matrix
import numpy as np
import gensim



try1 = __import__('try1')
try3 = __import__('try3')

def main():
    
    train_set, test_set = get_dataset("test", t="word", stem_or_not_stem = "not stem")
    true_labels = json_references(stem_or_not_stem = "not stem")
    
    # train model WHEN USE EMBEDDINGS 
    ##### put this in other place 
    #####---> PUS AQUI PARA NAO TAR SEMPRE A TREINAR 
    #####
    print("start train")
    model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
    print("end train")
     
#    prior_weights = get_prior_weights(train_set, test_set, variant = "length_and_position")
#    edge_weights = get_edge_weights(test_set, variant = "co-occurrences")
    #print(prior_weights)
    all_mAP = list()
    for key, test_doc in test_set.items():
        
        true_labels_doc = true_labels[key]
        print("key", key)
        
        if len(true_labels_doc) >= 5:
            
            test_doc = ' '.join(list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(test_doc[0]))))
  
            prior_weights = get_prior_weights(train_set, test_doc, variant = "length_and_position")
            edge_weights = get_edge_weights(train_set, test_doc, variant = "embeddings", model=model, test_doc_sent=test_set_sent[key])
            
            nodes = try1.extractKeyphrasesTextRank(test_doc) 
            
            print("start build graph")
            graph = try1.buildGraph(nodes, edge_weights, exercise2 = True)
            print("end build graph")
            
            pagerank_scores = nx.pagerank(graph, personalization = prior_weights, max_iter = 50)
            
            doc_top_5 = try1.get_top_x(pagerank_scores, 5)
            predicted_labels_doc = [x[0] for x in doc_top_5]
            
              
            avg_precision_per_doc = try3.average_precision_score(true_labels_doc, predicted_labels_doc)
            print("true>>", true_labels_doc)
            print("predi>> ", predicted_labels_doc)
            print("pred>>", avg_precision_per_doc)        
            all_mAP.append(avg_precision_per_doc)
    
    mAP = np.array(all_mAP)
    print(mAP)
    mAP = np.mean(mAP)
    print(mAP)
    
    
#Process XML file. Preprocesses text
#Input:path to file, token (optional)"word" or "lemma", token(optional)"stem" or "not stem"
#Output: list where each element corresponds to a document (string) and list of lists where 
#intern lists has the test documents (strings) as elements
#Notes: can't be "lemma" and "stem" at the same time
def get_dataset(folder, t="word", stem_or_not_stem = "not stem"):
    path = os.path.dirname(os.path.realpath('__file__')) + "\\Inspec\\" + folder
    ps = PorterStemmer()
    test_set = dict()
    test_set_sent = dict()
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
        sentence_text = list()
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
                    if word == "." or word == "%" or word == "'s" or word == "," or word == "'":   #MUDANÇA
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

def get_prior_weights(train_set, test_doc, variant = "length_and_position"):
    words_nodes = []
    #prior_weights = []
    
    score = dict()
            
    if variant == "length_and_position":      
        
        #for doc in test_set.values():                     
        words_nodes = try1.extractKeyphrasesTextRank(test_doc)
        words_nodes = unique_everseen(words_nodes)
        count_sent = 1
        #print(doc)
        #print(len(words_nodes))      6
        ##prior_weights_sent = []
        for sent in words_nodes:
            #print(sent)             #6
            ##length_position_sentence = []
            for gram in sent:
                #print("gram", gram)
                ##length_position_sentence.append(len(gram.split()) + (1/count_sent)) #fator inversamente proporcional
            #print(length_position_sentence)
                score[gram] = len(gram.split()) + (1/count_sent) #NEW
            ##prior_weights_sent.append(length_position_sentence)
            count_sent += 1
            #print("sent", sent, prior_weights_sent)

        ##prior_weights.append(prior_weights_sent)
        #print(prior_weights)
        #print(len(list(itertools.chain.from_iterable(prior_weights))))
        #print("score>>", score)
        
        
    elif(variant == "tfidf" or variant == "bm25"):
  
        for doc in list(train_set):
            words_nodes += list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(doc)))
        #words_nodes = list(unique_everseen(words_nodes))
        
        if variant == "tfidf":
           #words_nodes = list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(doc)))
           vectorizer = TfidfVectorizer(use_idf = True, 
                                        analyzer = 'word', 
                                        ngram_range = (1,3), 
                                        stop_words = 'english',
                                        token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b", 
                                        lowercase = True,
                                        vocabulary = list(unique_everseen(words_nodes)))
        
                                        
           X = vectorizer.fit_transform(train_set)
           #this is a mapping of index to
           feature_names = vectorizer.get_feature_names()                      

                
           Y = vectorizer.transform([test_doc])
           #y = Y.tocoo()
           #print(y.shape[1])
           sorted_items = sort_coo(Y.tocoo())
           
           
           score = extract_from_vector(feature_names,sorted_items)
           #print(len(score))
#           unnormalized_prior_weights = Y.toarray()                 
#           
#           sum_prior_weights = np.sum(unnormalized_prior_weights, axis=0)
#           print(sum_prior_weights.shape)
#           for sum_docs in sum_prior_weights:
#               prior_weights.append(sum_docs/len(test_set.values()))
#           print(len(prior_weights))
           #print(X.toarray())
           #print(vectorizer.get_feature_names())
        
        if variant == "bm25":
            bm25 = BM25(words_nodes)
            #words_nodes = list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(test_doc)))
            score = bm25.get_score(test_doc)
            #print(get_score)
        
    else:
        print(">>UNKNOWN>>PRIOR WEIGHTS>>VARIANT")
        
    return score 

#sort the tf-idf vectors by descending order of scores          
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

#extract keyphrases and scores in a coherent format along the rest of the variants
def extract_from_vector(feature_names, sorted_items):
    """get the feature names and tf-idf score of top n items"""
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results       

def isNgram(term):
    if(len(term.split()) >= 2):    
        return True
    return False


from sklearn.feature_extraction.text import CountVectorizer    
def get_edge_weights(train_set, test_doc, variant = "co-occurrences", model="", test_doc_sent=[]):
    final_weights = []
    words_nodes = []
    for doc in train_set:
        words_nodes += try1.extractKeyphrasesTextRank(doc)
        #words_nodes = unique_everseen(words_nodes)
    
    if variant == "co-occurrences":
        vectorizer = CountVectorizer(binary = True,
                                     analyzer = 'word', 
                                     ngram_range = (1,3), 
                                     stop_words = 'english',
                                     token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b", 
                                     lowercase = True,
                                     vocabulary = list(unique_everseen(itertools.chain.from_iterable(words_nodes)))
                                     )
        vectorizer._validate_vocabulary()                
        #https://stackoverflow.com/questions/35562789/how-do-i-calculate-a-word-word-co-occurrence-matrix-with-sklearn
        #https://github.com/scikit-learn/scikit-learn/issues/10901
        
        test_doc_candidates =  try1.extractKeyphrasesTextRank(test_doc)
        print("test_doc_candidates", test_doc_candidates)
        test_doc_normalized = [' '.join(sentence) for sentence in test_doc_candidates]
        print("test_doc_normalized", test_doc_normalized)
        X = vectorizer.fit_transform(test_doc_normalized)
        X = lil_matrix(X)
        Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
        #Xc[Xc > 0] = 1 # run this line if you don't want extra within-text cooccurence (see below) bem explicado no link above 
        Xc.setdiag(0) #  fill same word cooccurence to 0
        Xc = lil_matrix(Xc)
#        print(vectorizer.get_feature_names())
#        print("Xc", Xc)
#        raise
        feature_names = vectorizer.get_feature_names()
        final_weights = format_weights(Xc.tocoo(), feature_names)
        
    if variant == "embeddings":
        #print("train len",len(words_nodes))
        #print("test len", len(try1.extractKeyphrasesTextRank(test_doc)[0]))
        #print("test_doc_sent", test_doc_sent)
        
        # Tokenize(split) the sentences into candidates
        #test_doc_candidates = [try1.extractKeyphrasesTextRank(sentence)[0] for sentence in test_doc_sent]
        #print("test_doc_candidates", test_doc_candidates)
    
        #### THIS MODEL COMMENTED IS USING JUST OUR TRAIN
        #model = gensim.models.Word2Vec(words_nodes, min_count=1)
        wv = model.wv
        del model
        
        #vocab_dict = gensim.corpora.Dictionary(test_doc_candidates)
        #sm = wv.similarity_matrix(vocab_dict)
        #print("shape sm", sm.shape)
        #print("no terms", len(vocab_dict.token2id))
        vocabulary =  list(set(try1.extractKeyphrasesTextRank(test_doc)[0]))
        size = len(vocabulary)
        print("size vocab", size)
     
        matrix = lil_matrix((size, size), dtype=float)
        row_m = -1
        print("start csc matrix populate")
        for term_i in vocabulary:
            col_m = -1
            row_m += 1
            for term_j in vocabulary:
                col_m += 1
                i = 0
                acc = 0
                try:
                   grams_i = term_i.split()
                   grams_j = term_j.split()
                   
                   for g_i in grams_i:
                       for g_j in grams_j:       
                           i += 1
                           acc += wv.similarity(g_i, g_j)
                           sim = acc/i
                           matrix[row_m, col_m] = sim
                except KeyError as e:
                    continue
   
                    
      
        print("end") 
        #print("matrix", matrix)
        print("start format")
        #final_weights = format_weights(wv.similarity_matrix(vocab_dict).tocoo(), list(wv.vocab.keys()))
        final_weights = format_weights(matrix.tocoo(), vocabulary)
        print("end format")
        
                

            
        
    return final_weights
        
        #print(Xc.toarray()) # print out matrix in dense format

#    if variant == "embeddings":
# https://www.irit.fr/publis/SIG/2018_SAC_MRR.pdf
#        https://gluon-nlp.mxnet.io/examples/word_embedding/word_embedding.html
#https://kavita-ganesan.com/easily-access-pre-trained-word-embeddings-with-gensim/#.Xd1ikej7Q2w QUEEN
#https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/        
#    if variant == "distance_between_candidates":
        
def format_weights(Xc, feature_names):
    listTuplesWeights = []
    print("start format weight")
    for  line, column, data in zip(Xc.row, Xc.col, Xc.data):
        if(line != column):
            listTuplesWeights.append(tuple([feature_names[line], feature_names[column], data]))
    print("end format weight")
    return listTuplesWeights
    #for candidate in        

#if __name__ == "__main__":
#    main()
    

#ver similaridade no espaço de cada palavra
#weigth
############################################BM25############################################
import math
from six import iteritems

PARAM_K1 = 1.5
PARAM_B = 0.75


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
        self.idf = dict()
        self.doc_len = []
        self._initialize(corpus)

    def _initialize(self, corpus):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        nd = {}  # word -> number of documents with word
        num_doc = 0
        
        print("_initialize")
        
        for document in corpus:
            self.corpus_size += 1
            self.doc_len.append(len(document))
            num_doc += len(document)
            term_docs = list()
            
            for word in document:
                if word not in nd:
                    nd[word] = 1
                elif word in nd and word not in term_docs :
                    nd[word] += 1 
                term_docs.append(word)
                    
                if word not in self.terms:
                        self.terms.append(word)
                        self.no_terms +=1
     
        self.avgdl = float(num_doc) / self.corpus_size
        # collect idf sum to calculate an average idf for epsilon value
        
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        #for document in corpus:
            #negative_idfs = []
            #idf_sum = 0
            #for word in document:
        for word, freq in iteritems(nd):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
                    #idf_sum += idf[word]
                    #if inv_doc_freq[word] < 0:
                     #   negative_idfs.append(word)
                
                #self.average_idf = float(idf_sum) / len(inv_doc_freq)
                
                #for word, freq in iteritems(nd):
                 #   for word in negative_idfs:
                  #      inv_doc_freq[word] = EPSILON * self.average_idf
            #self.idf.append(inv_doc_freq)
        
        print("end_initialize")
    
    def get_score(self, document):
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
        frequencies = dict()
        terms = list()
        no_terms = 0
        score = dict()
        for word in document:
            no_terms +=1
            if word not in terms:
                    terms.append(word)
                   
            if word not in frequencies.keys():
                frequencies[word] = 0
            frequencies[word] += 1
                    
        for word in frequencies.keys():
            term_freq = frequencies[word]/no_terms    
            idf = 0.25
            if word in self.idf.keys():
                idf = self.idf[word]
            score[word] = (idf * term_freq * (PARAM_K1 + 1) / (term_freq + PARAM_K1 * (1 - PARAM_B + PARAM_B * no_terms / self.avgdl)))
        return score
