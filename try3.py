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
import networkx as nx

def main():
    train_set, test_set  = try2.get_dataset("test", t="word", stem_or_not_stem = "not stem")
    train_set = list(train_set)
    true_labels = try2.json_references(stem_or_not_stem = "not stem")
     
    vectorizer, vectorizer_tf, bm25 = do_train(train_set)
   
    all_ap_RRF = list()
    all_ap_CombSum = list()
    all_ap_CombMNZ = list()
    
    features = ["bm25", "idf", "tf", "tfidf", "centrality"]
    f = open("test.txt", "x")
    for L in range(0, len(features)+1):
        for combination_features in itertools.combinations(features, L):
            print(">>>>>>features used:", combination_features)
    
            if combination_features == ():
                continue
            f.write("features used:" + str(combination_features) + "\n")
            for key, test_doc in test_set.items():
                
                y_true = true_labels[key]
                
                if(len(y_true) >= 5):        
                    print("key", key)
                  
                    print("y_true", y_true)
                    RRF_sorted, CombSum_sorted, CombMNZ_sorted = do_score(train_set, test_doc, vectorizer, vectorizer_tf, bm25, combination_features=combination_features)
                 
                    
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
                    
            
                    
            mean_average_score_RRF = np.mean(np.array(all_ap_RRF))
            mean_average_score_CombSum = np.mean(np.array(all_ap_CombSum))
            mean_average_score_CombMNZ = np.mean(np.array(all_ap_CombSum))
            
            print("RRF_mean_avg_precision",  mean_average_score_RRF)
            print("CombSum_mean_avg_precision", mean_average_score_CombSum )
            print("CombMNZ_mean_avg_precision",  mean_average_score_CombMNZ)
            
            
            f.write("\n RRF MAP " + str(mean_average_score_RRF) + "\n" + "CombSum MAP " + str(mean_average_score_CombSum) + "\n CombMNZ MAP" + str(mean_average_score_CombMNZ) + "\n")
    f.close()

def do_train(train_set):
    words_nodes = generate_vocabulary(train_set)
    vectorizer = tf_idf_train(train_set, words_nodes)
    vectorizer_tf = do_tf_train(train_set, words_nodes)
     #bm25
    bm25 = try2.BM25(train_set)
    
    return vectorizer, vectorizer_tf, bm25
    
def generate_vocabulary(train_set):
    words_nodes = list()
    for doc in train_set:
        words_nodes += list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(doc)))
    words_nodes = list(unique_everseen(words_nodes))
    return words_nodes
    
def do_score(train_set, test_doc, vectorizer, vectorizer_tf, bm25, combination_features = ''):  
       test_doc_candidates = list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(test_doc[0])))
       
       tfidf_vector = vectorizer.transform(test_doc_candidates)
     
       tf_vector    = vectorizer_tf.transform(test_doc_candidates)
       idf   = vectorizer.idf_
  
       rankers = []
       if "tf" in combination_features:
           tf_name    = map_name_score( tf_vector.tocoo(), vectorizer_tf.get_feature_names())
           rankers.append(tf_name)
           
       if "tfidf" in combination_features:
           tfidf_name = map_name_score(tfidf_vector.tocoo(), vectorizer.get_feature_names())
           rankers.append(tfidf_name)
           
       if "idf" in combination_features:
           idf_name   = dict(zip( vectorizer.get_feature_names(), idf))
           rankers.append(idf_name)
           
       if "bm25" in combination_features:
           bm25_scores = bm25.get_score(test_doc_candidates)
           rankers.append(bm25_scores)
           
       if "centrality" in combination_features:
           nodes = ' '.join(list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(test_doc[0]))))
           edge_weights = try2.get_edge_weights(train_set, nodes, variant = "co-occurrences")
           nodes = try1.extractKeyphrasesTextRank(nodes) 
           graph = try1.buildGraph(nodes, edge_weights, exercise2 = True)
           centrality_scores = nx.degree_centrality(graph)
           rankers.append(centrality_scores)
                  
       RRF     = RRFScore(rankers)
       CombSum = CombSumScore(rankers)
       CombMNZ = CombMNZScore(rankers)
        
       RRF_sorted     = sorted(RRF.items()    , key = operator.itemgetter(1), reverse = True)
       CombSum_sorted = sorted(CombSum.items(), key = operator.itemgetter(1), reverse = True)
       CombMNZ_sorted = sorted(CombMNZ.items(), key = operator.itemgetter(1), reverse = True)
       
       return RRF_sorted, CombSum_sorted, CombMNZ_sorted
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
