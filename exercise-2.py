# -*- coding: utf-8 -*-
"""
EXERCISE_2
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
from collections import Counter

try1 = __import__('exercise-1')
try3 = __import__('exercise-3')

def main():
    
    train_set, test_set  = get_dataset("test", t="word", stem_or_not_stem = "not stem")
    true_labels = json_references(stem_or_not_stem = "not stem")
    
#    import gensim
#    print("start train")
#    model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
#    print("end train")
    
#    #### THIS MODEL COMMENTED IS USING JUST OUR TRAIN 
    words_nodes = []
    for doc in train_set:
        words_nodes += try1.extractKeyphrasesTextRank(doc)
    import gensim
    model = gensim.models.Word2Vec(words_nodes, min_count=1)
    wv = model.wv
    del model

    all_mAP = list()
    for key, test_doc in test_set.items():
        
        true_labels_doc = true_labels[key]
        
        if len(true_labels_doc) >= 5:
            
            prior_weights = get_prior_weights(train_set, test_doc[0], variant = "length_and_position")   #length_and_position/tfidf/bm25/likelihood
            edge_weights = get_edge_weights(train_set, test_doc[0], variant = "co-occurrences", model=wv)    #co-occurrences/embeddings/edit_distance/levenshtein_ratio_and_distance/co-occurrences_plus_embeddings

            nodes = try1.extractKeyphrasesTextRank(test_doc[0])
            
            graph = try1.buildGraph(nodes, edge_weights, exercise2 = True)
            
            N = 1/len(graph.nodes())
            nstart = dict.fromkeys(graph.nodes() , N)
            if "fuzzy control" not in nstart.keys():
                print("NÃO ESTOU CÁ!!")

            pagerank_scores = nx.pagerank(graph, personalization = prior_weights, max_iter = 50, nstart = nstart) #, nstart = nstart
            
            doc_top_5 = try1.get_top_x(pagerank_scores, 5)
            predicted_labels_doc = [x[0] for x in doc_top_5]
            
            avg_precision_per_doc = try3.average_precision_score(true_labels_doc, predicted_labels_doc)
            print("true_labels >>", true_labels_doc)
            print("predicted_labels >> ", predicted_labels_doc)
            print("avg_precision >>", avg_precision_per_doc)        
            all_mAP.append(avg_precision_per_doc)
    
    mAP = np.array(all_mAP)
    mAP = np.mean(mAP)
    print("Mean Average Precision >>", mAP)
    
    
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

#################################get_prior_weights#############################

def get_prior_weights(train_set, test_doc, variant = "length_and_position"):
    words_nodes = []
        
    score = dict()
            
    if variant == "length_and_position":      
                           
        words_nodes = try1.extractKeyphrasesTextRank(test_doc)
        #words_nodes = unique_everseen(words_nodes)
        
        count_sent = 1
        for sent in words_nodes:
            for gram in sent:
                score[gram] = len(gram.split()) + (1/count_sent) 

            count_sent += 1

    elif(variant == "tfidf" or variant == "bm25"):
        test_doc = ' '.join(list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(test_doc))))

        for doc in list(train_set):
            words_nodes += list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(doc)))
        
        if variant == "tfidf":
           vectorizer = TfidfVectorizer(use_idf = True, 
                                        analyzer = 'word', 
                                        ngram_range = (1,3), 
                                        stop_words = 'english',
                                        token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b", 
                                        lowercase = True,
                                        vocabulary = list(unique_everseen(words_nodes)))
        
           X = vectorizer.fit_transform(train_set)
           
           feature_names = vectorizer.get_feature_names()                      

                
           Y = vectorizer.transform([test_doc])
           
           sorted_items = sort_coo(Y.tocoo())
           
           
           score = extract_from_vector(feature_names,sorted_items)
        
        if variant == "bm25":
            bm25 = BM25(train_set)
            words_nodes = list(itertools.chain.from_iterable(try1.extractKeyphrasesTextRank(test_doc)))
            score = bm25.get_score(words_nodes)    
            
    elif variant == "likelihood":
        words_nodes = try1.extractKeyphrasesTextRank(test_doc)
        words_nodes = list(itertools.chain.from_iterable(words_nodes))
        
        word_count = Counter(words_nodes)          # count the words

        sumWords = sum(word_count.values())       # sum total words
        
        for gram in list(unique_everseen(words_nodes)):
            score[gram] = word_count[gram]/sumWords
        
    else:
        print(">>UNKNOWN>>PRIOR WEIGHTS>>", variant)
        
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
    
    # word index and corresponding scores
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

##############################get_edge_weights#################################
   
from sklearn.feature_extraction.text import CountVectorizer    
def get_edge_weights(train_set, test_doc, variant = "co-occurrences", model = ""):
    final_weights = []
    
    test_doc_candidates =  try1.extractKeyphrasesTextRank(test_doc)

    if variant == "co-occurrences":
        
        words_nodes = []
        for doc in train_set:
            words_nodes += try1.extractKeyphrasesTextRank(doc)
        
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
                
        X = vectorizer.fit_transform(test_doc_candidates[0])
        X = lil_matrix(X)
        Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
        Xc[Xc > 0] = 1 # run this line if you don't want extra within-text cooccurence (see below) bem explicado no link above 
        Xc.setdiag(0) #  fill same word cooccurence to 0
        Xc = lil_matrix(Xc)

        feature_names = vectorizer.get_feature_names()
        final_weights = format_weights(Xc.tocoo(), feature_names)
        
    elif variant == "embeddings":
        #https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/
        feature_names = list(unique_everseen(itertools.chain.from_iterable(test_doc_candidates)))
        
        similarity_matrix = lil_matrix((len(feature_names), len(feature_names)), dtype=float)
        
        test_doc = try1.extractKeyphrasesTextRank(test_doc)
        
        for sent in test_doc:
            row_m = -1
            for gram1 in feature_names:
                 
                col_m = -1
                row_m += 1
                
                for gram2 in feature_names:
                    col_m += 1
                    
                    if gram1 in sent and gram2 in sent and gram1 != gram2: 
                    
                        i = 0
                        acc = 0
                        
                        grams_i = gram1.split()
                        grams_j = gram2.split()
                        
                        for g_i in grams_i:
                            for g_j in grams_j:
                                i += 1
                                try:
                                    acc += model.similarity(g_i, g_j)
                                except KeyError as e:
                                    continue
                        similarity = acc/i
                        similarity_matrix[row_m, col_m] = similarity
                            
        final_weights = format_weights(similarity_matrix.tocoo(), feature_names)
 
    elif variant == "edit_distance":
       import nltk
       feature_names = list(unique_everseen(itertools.chain.from_iterable(test_doc_candidates)))  
            
       edit_distance_matrix = lil_matrix((len(feature_names), len(feature_names)), dtype=float)         
       
       test_doc = try1.extractKeyphrasesTextRank(test_doc)
       for sent in test_doc:
           row_m = -1
           for gram1 in feature_names:
                 
                col_m = -1
                row_m += 1
                
                for gram2 in feature_names:
                    col_m += 1
                    
                    if gram1 in sent and gram2 in sent and gram1 != gram2: 
                    #http://www.nltk.org/howto/metrics.html
                    #https://www.nltk.org/api/nltk.metrics.html
                    #https://www.datacamp.com/community/tutorials/fuzzy-string-python
                    #edit_distance - 0.0153571428571
                    #binary_distance - 0.0280357142857
                    #ratio
                        #edit_distance_matrix[row_m,col_m] = fuzz.token_set_ratio(gram1, gram2)
                        edit_distance_matrix[row_m,col_m] = 1/(nltk.edit_distance(gram1, gram2))
            
       final_weights = format_weights(edit_distance_matrix.tocoo(), feature_names)
       
    elif variant == "binary_distance":
       import nltk
       feature_names = list(unique_everseen(itertools.chain.from_iterable(test_doc_candidates)))  
            
       edit_distance_matrix = lil_matrix((len(feature_names), len(feature_names)), dtype=float)         
       
       test_doc = try1.extractKeyphrasesTextRank(test_doc)
       for sent in test_doc:
           row_m = -1
           for gram1 in feature_names:
                 
                col_m = -1
                row_m += 1
                
                for gram2 in feature_names:
                    col_m += 1
                    
                    if gram1 in sent and gram2 in sent and gram1 != gram2: 
                        edit_distance_matrix[row_m,col_m] = 1/(nltk.binary_distance(gram1, gram2))
            
       final_weights = format_weights(edit_distance_matrix.tocoo(), feature_names)
    
    elif variant == "levenshtein_ratio_and_distance":
       feature_names = list(unique_everseen(itertools.chain.from_iterable(test_doc_candidates)))  
            
       levenshtein_ratio_and_distance_matrix = lil_matrix((len(feature_names), len(feature_names)), dtype=float)         
       
       test_doc = try1.extractKeyphrasesTextRank(test_doc)
       for sent in test_doc:
           row_m = -1
           for gram1 in feature_names:
                 
                col_m = -1
                row_m += 1
                
                for gram2 in feature_names:
                    col_m += 1
                    
                    if gram1 in sent and gram2 in sent and gram1 != gram2: 
                        levenshtein_ratio_and_distance_matrix[row_m,col_m] = levenshtein_ratio_and_distance(gram1, gram2, ratio_calc = True)
        
       final_weights = format_weights(levenshtein_ratio_and_distance_matrix.tocoo(), feature_names)
       
    elif variant == "ratio":
       from fuzzywuzzy import fuzz
       feature_names = list(unique_everseen(itertools.chain.from_iterable(test_doc_candidates)))  
            
       edit_distance_matrix = lil_matrix((len(feature_names), len(feature_names)), dtype=float)         
       
       test_doc = try1.extractKeyphrasesTextRank(test_doc)
       for sent in test_doc:
           row_m = -1
           for gram1 in feature_names:
                 
                col_m = -1
                row_m += 1
                
                for gram2 in feature_names:
                    col_m += 1
                    
                    if gram1 in sent and gram2 in sent and gram1 != gram2: 
                        edit_distance_matrix[row_m,col_m] = fuzz.ratio(gram1, gram2)
            
       final_weights = format_weights(edit_distance_matrix.tocoo(), feature_names)
        
    elif variant == "co-occurrences_plus_embeddings":
        #https://www.irit.fr/publis/SIG/2018_SAC_MRR.pdf
        dict_cooccurrences = get_edge_weights(train_set, test_doc, variant="co-occurrences", model=model)
        
        dict_embeddings = get_edge_weights(train_set, test_doc, variant="embeddings", model=model)
        
        final_weights = { k: dict_cooccurrences.get(k, 0) + dict_embeddings.get(k, 0) for k in set(dict_cooccurrences) & set(dict_embeddings) }

    else:
        print(">>UNKNOWN>>EDGE WEIGHTS>>", variant)

    return final_weights
        
def format_weights(Xc, feature_names):
    listTuplesWeights = []
    for  line, column, data in zip(Xc.row, Xc.col, Xc.data):
        if(line != column):
            listTuplesWeights.append(tuple([feature_names[line], feature_names[column], data]))
    return listTuplesWeights     
        

#if __name__ == "__main__":
#    main()

#######################################BM25####################################
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

        for word, freq in iteritems(nd):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
    
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
    
###############################################################################
def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio