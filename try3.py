"""Exercise 3"""
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer 
import math
import operator
try1 = __import__('try1')
try2 = __import__('try2')

def main():
    train_set, test_set  = try2.get_dataset("test", t="word", stem_or_not_stem = "not stem")
    train_set = list(train_set)
    #true_labels = try2.json_references(stem_or_not_stem = "not stem")
    
    #doc = try1.open_file()
    vectorizer = tf_idf_train(train_set)
    vectorizer_tf = do_tf_train(train_set)
        
    for test_doc in list(test_set.values())[:15]:
        tfidf = vectorizer.transform(test_doc)
        tf    = vectorizer_tf.transform(test_doc)
        idf   = vectorizer.idf_
        
        tfidf = dict(zip(vectorizer.get_feature_names(), tfidf.data.tolist()))
        tf    = dict(zip(vectorizer_tf.get_feature_names(), tf.data.tolist()))
        idf   = dict(zip(vectorizer.get_feature_names(), idf.data.tolist()))
        
        
        rankers = [tfidf, tf, idf]
        RRF     = RRFScore(rankers)
        CombSum = CombSumScore(rankers)
        CombMNZ = CombMNZScore(rankers)
        
        RRF     = sorted(RRF.items()    , key = operator.itemgetter(1), reverse = True)
        CombSum = sorted(CombSum.items(), key = operator.itemgetter(1), reverse = True)
        CombMNZ = sorted(CombMNZ.items(), key = operator.itemgetter(1), reverse = True)
        
        print(">>>RRF: "    ,RRF[:5])
        print(">>>CombSum: ",CombSum[:5])
        print(">>>CombMNZ: ",CombMNZ[:5])            
    
        #print("RRF:", RRF)
        #print("CombSum:", CombSum)
        #print("CombMNZ:", CombMNZ)
    
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

def do_tf_train(doc):
    vectorizer_tf = TfidfVectorizer(use_idf = False, 
                                           analyzer = 'word', 
                                           ngram_range=(1,3), 
                                           stop_words = 'english',
                                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b", 
                                           lowercase = True,
                                           norm = 'l1')
    vectorizer_tf.fit_transform(doc)
        
    return vectorizer_tf


#Creates vectorizer and fits it to the docs
#@input:train set, parameter (optional) removes the n most frequent words 
#@return: vectorizer 
def tf_idf_train(docs, maxdf = 1, mindf = 1):      
    vectorizer_tfidf = TfidfVectorizer(use_idf = True, 
                                           analyzer = 'word', 
                                           ngram_range=(1,3), 
                                           stop_words = 'english',
                                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b", 
                                           lowercase = True,
                                           max_df = maxdf,
                                           min_df = mindf,
                                           norm = 'l1')
        
    vectorizer_tfidf.fit_transform(docs)
    #print("test_vector", test_vector)
    
    return vectorizer_tfidf

def do_tfidf_test(test_vector, feature_names):  
    dictTfidf = {}
    for col, data in zip(test_vector.col, test_vector.data):
        name = feature_names[col]
        dictTfidf[name] = data
    return dictTfidf