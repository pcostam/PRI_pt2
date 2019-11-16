
"""
Exercise 3
"""
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer 
import math
try1 = __import__('try1')
try2 = __import__('try2')

def main():
    train_set, test_set  = try2.get_dataset("test", t="word", stem_or_not_stem = "not stem")
    #true_labels = try2.json_references(stem_or_not_stem = "not stem")
    num = 0
    for doc in train_set:
        #doc = try1.open_file()
        tfidf = tf_idf([doc])
        tf = do_tf([doc])
        idf = do_idf([doc], 4)
        
        rankers = [tfidf, tf, idf]
        RRF = RRFScore(rankers)
        CombSum = CombSumScore(rankers)
        CombMNZ = CombMNZScore(rankers)
        
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>DOC NUM:", num)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("RRF:", RRF)
        print("CombSum:", CombSum)
        print("CombMNZ:", CombMNZ)
        num +=1

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

def do_count(doc):
     vectorizer_tf = TfidfVectorizer(binary=True, 
                                            use_idf = False, 
                                           analyzer = 'word', 
                                           ngram_range=(1,3), 
                                           stop_words = 'english',
                                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b", 
                                           lowercase = True,
                                           norm = None)
     #term-document matrix
     matrix = vectorizer_tf.fit_transform(doc)
     matrix = sparse.csr_matrix(matrix)
     matrix = matrix.tocoo()
     #print("matrix", matrix)
  
     feature_names = vectorizer_tf.get_feature_names()
     countDict = dict()
     for col, data in zip(matrix.col, matrix.data):
        name = feature_names[col]
        if name not in countDict.keys():
            countDict[name] = data
        else:
            countDict[name] += data
    

     return countDict
 
def do_idf(docs, total_no_docs):
    count = do_count(docs)
    dictIdf = dict()
    for term_name, count in count.items():
         dictIdf[term_name] = math.log(total_no_docs/count)
    return dictIdf

def do_tf(doc):
    vectorizer_tf = TfidfVectorizer(use_idf = False, 
                                           analyzer = 'word', 
                                           ngram_range=(1,3), 
                                           stop_words = 'english',
                                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b", 
                                           lowercase = True,
                                           norm = 'l1')
    matrix = vectorizer_tf.fit_transform(doc)
    test_vector = sparse.csr_matrix(matrix[0])
    test_vector = test_vector.tocoo()
    
    dictTf = {}
    feature_names = vectorizer_tf.get_feature_names()
    for col, data in zip(test_vector.col, test_vector.data):
        name = feature_names[col]
        dictTf[name] = data
    return dictTf
    
def tf_idf(doc):
    #TRAIN
    #print("doc", doc)
    test_vector, vectorizer = tf_idf_train(doc)
    feature_names = vectorizer.get_feature_names()
    tfidfDict = do_tfidf(test_vector, feature_names)
    #print("tfIdfDict", tfidfDict)
    return tfidfDict
  
def do_tfidf(test_vector, feature_names):  
    dictTfidf = {}
    for col, data in zip(test_vector.col, test_vector.data):
        name = feature_names[col]
        dictTfidf[name] = data
    return dictTfidf

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
        
    matrix = vectorizer_tfidf.fit_transform(docs)
    test_vector = sparse.csr_matrix(matrix[0])
    test_vector = test_vector.tocoo()
    #print("test_vector", test_vector)

        
    return test_vector, vectorizer_tfidf
  

