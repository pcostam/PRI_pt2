"""Exercise 3"""
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer 
import math
import operator
try1 = __import__('try1')
try2 = __import__('try2')
import numpy as np 
from  more_itertools import unique_everseen
import itertools

def main():
    train_set, test_set  = try2.get_dataset("test", t="word", stem_or_not_stem = "not stem")
    train_set = list(train_set)
    true_labels = try2.json_references(stem_or_not_stem = "not stem")
     
    words_nodes = list()
    for doc in train_set:
        words_nodes += list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(doc)))
    words_nodes = list(unique_everseen(words_nodes))
    
    
    #doc = try1.open_file()
    vectorizer = tf_idf_train(train_set, words_nodes, maxdf=0.5, mindf=2)
    vectorizer_tf = do_tf_train(train_set, words_nodes, maxdf=0.5, mindf=2)
    
    i = 0
    all_ap_RRF = list()
    all_ap_CombSum = list()
    all_ap_CombMNZ = list()
    for key, test_doc in test_set.items():
        print("key", key)
        y_true = true_labels[key]
        print("y_true", y_true)
        tfidf_vector = vectorizer.transform(test_doc)
     
        tf_vector    = vectorizer_tf.transform(test_doc)
        idf   = vectorizer.idf_
     
        tfidf_name = map_name_score(tfidf_vector.tocoo(), vectorizer.get_feature_names())
        tf_name    = map_name_score( tf_vector.tocoo(), vectorizer_tf.get_feature_names())
        idf_name   = dict(zip( vectorizer.get_feature_names(), idf))
        
        
        rankers = [tfidf_name, tf_name, idf_name]
        RRF     = RRFScore(rankers)
        CombSum = CombSumScore(rankers)
        CombMNZ = CombMNZScore(rankers)
        
        RRF_sorted     = sorted(RRF.items()    , key = operator.itemgetter(1), reverse = True)
        CombSum_sorted = sorted(CombSum.items(), key = operator.itemgetter(1), reverse = True)
        CombMNZ_sorted = sorted(CombMNZ.items(), key = operator.itemgetter(1), reverse = True)
        
        y_pred_RRF = [i[0] for i in RRF_sorted[:5]] 
        y_pred_CombSum = [i[0] for i in CombSum_sorted[:5]]
        y_pred_CombMNZ = [i[0] for i in CombMNZ_sorted[:5]]
        
        print(y_pred_RRF)
        print(y_pred_CombSum)
        print(y_pred_CombMNZ)
        
        RRF_avg_precision     = average_precision_score(y_true, y_pred_RRF)
        all_ap_RRF.append(RRF_avg_precision)
        
        CombSum_avg_precision = average_precision_score(y_true, y_pred_CombSum)
        all_ap_CombSum.append(CombSum_avg_precision)
        
        CombMNZ_avg_precision = average_precision_score(y_true, y_pred_CombMNZ)
        all_ap_CombMNZ.append(CombSum_avg_precision)
        
        #print(">>>RRF: "    ,RRF[:5])
        #print(">>>CombSum: ",CombSum[:5])
        #print(">>>CombMNZ: ",CombMNZ[:5])     
        print("RRF_avg_precision", RRF_avg_precision)
        print("CombSum_avg_precision", CombSum_avg_precision)
        print("CombMNZ_avg_precision", CombMNZ_avg_precision)
    
        #print("RRF:", RRF)
        #print("CombSum:", CombSum)
        #print("CombMNZ:", CombMNZ)
        
    
        if(i == 80):
          break
        else:
            i += 1
            
    mean_average_score_RRF = np.array(all_ap_RRF)/len(all_ap_RRF)
    mean_average_score_CombSum = np.array(all_ap_CombSum)/len(all_ap_CombSum)
    mean_average_score_CombMNZ = np.array(all_ap_CombSum)/len(all_ap_CombMNZ)
    
    #print("RRF_avg_precision",  mean_average_score_RRF)
    #print("CombSum_avg_precision", mean_average_score_CombSum )
    #print("CombMNZ_avg_precision",  mean_average_score_CombMNZ)
    
def vector_scores(test_vector, feature_names):
    test_vector = test_vector.toarray()
   
    for i in range(0, test_vector.shape[0]):
        for j in range(0, test_vector.shape[1]):
            if test_vector[i,j] != 0:
                    test_vector[i,j] =  test_vector[i,j] * len(feature_names[j].split())
                    
    test_vector = sparse.csr_matrix(test_vector)
    return test_vector
def CombMNZScore(rankers):
    dictCombMNZ = dict()
    #number of ranks where term occurs
    dictCountRank = dict()
    for ranker in rankers:
        for key, rank in ranker.items():
            if key not in dictCountRank.keys():
                dictCountRank[key] = 1
            else:
                dictCountRank[key] += 1
            
    for ranker in rankers:
        for key, rank in ranker.items():
              if key not in dictCombMNZ.keys():
                  dictCombMNZ[key] = dictCountRank[key]*rank
              else:
                  dictCombMNZ[key] += dictCountRank[key]*rank
    return dictCombMNZ
    
def CombSumScore(rankers):
    dictCombSum = dict()
    for ranker in rankers:
        for key, rank in ranker.items():
              if key not in dictCombSum.keys():
                  dictCombSum[key] = rank
              else:
                  dictCombSum[key] += rank
    return dictCombSum
    
def RRFScore(rankers):
    dictRRF = {}
    for ranker in rankers:
        for key, rank in ranker.items():
              if key not in dictRRF.keys():
                    dictRRF[key] = 1/(50+rank)
              else:
                    dictRRF[key] += 1/(50+rank)                
    return dictRRF

def do_tf_train(doc, vocab, maxdf=1, mindf=1):
    vectorizer_tf = TfidfVectorizer(use_idf = False, 
                                           analyzer = 'word', 
                                           ngram_range=(1,3), 
                                           stop_words = 'english',
                                           lowercase = True,
                                           norm = 'l1',
                                           max_df=maxdf,
                                           min_df=mindf,
                                           vocabulary=vocab)
    vectorizer_tf.fit_transform(doc)
        
    return vectorizer_tf


#Creates vectorizer and fits it to the docs
#@input:train set, parameter (optional) removes the n most frequent words 
#@return: vectorizer 
def tf_idf_train(docs, vocab,  maxdf = 1, mindf = 1):      
    vectorizer_tfidf = TfidfVectorizer(use_idf = True, 
                                           analyzer = 'word', 
                                           ngram_range=(1,3), 
                                           stop_words = 'english', 
                                           lowercase = True,
                                           max_df = maxdf,
                                           min_df = mindf,
                                           norm = 'l1',
                                           vocabulary=vocab)
        
    vectorizer_tfidf.fit_transform(docs)
    #print("test_vector", test_vector)
    
    return vectorizer_tfidf

def map_name_score(test_vector, feature_names):  
    dictTfidf = {}
    for col, data in zip(test_vector.col, test_vector.data):
        name = feature_names[col]
        dictTfidf[name] = data
    return dictTfidf


def average_precision_score(y_true, y_pred):
    nr_relevants = 0
    i = 0
    ap_at_sum = 0
  
    for el in y_pred:
        i += 1
        
        #is relevant
        if el in y_true:
            nr_relevants += 1
            ap_at_sum += nr_relevants/i
        
    if(nr_relevants == 0):
        return 0
    
    return ap_at_sum/nr_relevants