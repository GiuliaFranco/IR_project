from collections import defaultdict
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TreebankWordTokenizer
from rank_bm25 import BM25Okapi
import distance
from math import log, sqrt
import numpy as np
import operator
import re
import itertools

################################
#  vector space model class    #
#  3 implementations           #
################################


# VSM from scratch : Naive way to implement vector space model using list of lists. 

class VectorSpaceModel_scratch():
    
    def __init__(self,indirizzo,indirizzoq):
        self.name_file_doc=indirizzo   # document file
        self.name_file_q=indirizzoq    # query file
        self.articles = []    # store all the documents and query texts
        self.doc_mat = []     # store document term frequency matrix
        self.query_mat = []   # store query term frequency matrix
        self.bagofwords=[]    # overall bagofword
        self.documentsTFIDF=[]  # store document TFIDF matrix
        self.queryTFIDF=[]      # store query TFIDF matrix
        numdoc=0          # number of documents
        numquery=0        # number of query
        
 
    def import_dataset(self,kind):

#    This function import all the articles in the TIME corpus,
#    returning list of lists where each sub-list contains all the
#    terms present in the document as a string.

        if kind=="doc": 
            w="*TEXT"
        else: 
            w="*FIND"    
        lines = [line.rstrip('\n') for line in open(self.name_file_doc)]
        self.articles = [list(y) for x, y in itertools.groupby(lines, lambda z: z.startswith(w)) if not x]
        for i in range(0,len(self.articles)):
            self.articles[i] = " ".join(self.articles[i])
            self.articles[i] = re.sub(r'[^a-zA-Z\s]+', '', self.articles[i])
        self.numdoc=len(self.articles)
        
    def add_query(self,kind):
#    This function import all the query in the TIME.QUE,
#    returning list of lists where each sub-list contains all the
#    terms present in the query as a string.
        if kind=="doc": 
            w="*TEXT"
        else: 
            w="*FIND"    
        lines = [line.rstrip('\n') for line in open(self.name_file_q)]
        articles = [list(y) for x, y in itertools.groupby(lines, lambda z: z.startswith(w)) if not x]
        for i in range(0,len(articles)):
            articles[i] = " ".join(articles[i])
            articles[i] = re.sub(r'[^a-zA-Z\s]+', '', articles[i])            
        self.numquery=len(articles)
        self.articles=self.articles+articles

    def remove_stop_words(self):
#    Utility function to remove stop words provided in file 
#    "TIME.STP" in the orginal directory file.

        stops = [line.rstrip('\n') for line in open('TIME.STP')]
        stops=list(filter(None, stops))
        for i in range(0,len(self.articles)):
            self.articles[i] = [x for x in self.articles[i] if x not in stops]

                    

    def bag_of_words(self):
#    This function create global bag_of_words.
        for i in range(0,len(self.articles)):
            self.articles[i]=self.articles[i].split()
        self.remove_stop_words()
        self.bagofwords = list(set(sum(self.articles, [])))
            
            
    def TermDocumentMatrix(self):
        
# This function create a global term document matrix from the bag_of_word 
# and finally divides between query columns and documents columns.

        allq=[]
        for i in range(0,self.numdoc+self.numquery):
            allq.append(dict.fromkeys(self.bagofwords, 0))
            for word in self.articles[i]:
                allq[i][word]+=1
        self.doc_mat = allq[:self.numdoc]
        self.query_mat = allq[self.numdoc:]
        return allq

    def TF_document_query(self):
#   This function calculate Term frequency for both documents columns   
#   and query columns.

        allq=self.TermDocumentMatrix()
        alltf=[]
        for i in range(0,self.numdoc+self.numquery):
            tfDict = {}
            for word, count in allq[i].items():
                tfDict[word] = count/float(len(self.articles[i]))
            alltf.append(tfDict)
        documentsTF=alltf[:self.numdoc]
        queryTF=alltf[self.numdoc:]
        return documentsTF,queryTF

    def IDF_aux(self,mat,N):
#    Auxiliary function for Inverse document frequency calculation.

        dictIDF = dict.fromkeys(mat[0].keys(), 0)
        for doc in mat:
            for word, val in doc.items():
                if val > 0:
                    dictIDF[word] += 1
        for word, val in dictIDF.items():
            if val > 0:
                dictIDF[word] = math.log10(N / float(val))
        return dictIDF
    
    def IDF_document_query(self):
#   This function calculate Inverse document frequency for both documents columns   
#   and query columns.
        
        documentsIDF ={}
        queryIDF ={}
        N_d = len(self.doc_mat)
        N_q = len(self.query_mat)
        documentsIDF = self.IDF_aux(self.doc_mat,N_d)
        queryIDF = self.IDF_aux(self.query_mat,N_q)
        return documentsIDF,queryIDF
    
    def TFIDF_aux(self,N,dTF,dIDF,dTFIDF):
#    Auxiliary function for TF-IDF calculation.
        
        for i in range(0,N):
            tfidf = {}
            for word, val in dTF[i].items():
                tfidf[word] = val*dIDF[word]
            dTFIDF.append(tfidf)
            del tfidf
    
    def TFIDF_document_query(self):
#   This function calculate TFIDF for both documents columns   
#   and query columns.

        documentsTF,queryTF=self.TF_document_query()
        documentsIDF,queryIDF=self.IDF_document_query()
        self.TFIDF_aux(self.numdoc,documentsTF,documentsIDF,self.documentsTFIDF)
        self.TFIDF_aux(self.numquery,queryTF,queryIDF,self.queryTFIDF)
                              
    def get_document_vector(self,docid):
#    This function returns, for a document represented as a vector in v, all the
#    non-zero weights (both normalized and not) and the corresponding terms

        #non_zero_terms = [x for x in self.documentsTFIDF[docid].keys() if self.documentsTFIDF[docid][x] > 0]
        terms = [x for x in self.documentsTFIDF[docid].keys()]
        vect = [(x, self.documentsTFIDF[docid][x]) for x in terms]
        vect.sort(key=lambda x: x[1], reverse=True)
        length = sqrt(sum([x[1]**2 for x in vect]))
        normalized = {k: tfidf/length for k, tfidf in vect}
        return vect,normalized

    def get_query_vector(self,docid):
#    This function returns, for a query represented as a vector in v, all the
#    non-zero weights (both normalized and not) and the corresponding terms

        #non_zero_terms = [x for x in self.queryTFIDF[docid].keys() if self.queryTFIDF[docid][x] > 0]
        terms = [x for x in self.documentsTFIDF[docid].keys()]
        vect = [(x, self.queryTFIDF[docid][x]) for x in terms]
        vect.sort(key=lambda x: x[1], reverse=True)
        length = sqrt(sum([x[1]**2 for x in vect]))  
        normalized = {k: tfidf/length for k, tfidf in vect}
        return vect,normalized   
            
    def show_document_vector(self,docid):
#    This function prints, for a document represented as a vector in v, all the
#    non-zero weights (both normalized and not) and the corresponding terms

        vect,normalized=self.get_document_vector(docid)
        for (term, tfidf) in vect:
            print(f"{term}:\t{tfidf}\t(normalized: {normalized[term]})")
            
    def show_query_vector(self,docq):
#    This function prints, for a query represented as a vector in v, all the
#    non-zero weights (both normalized and not) and the corresponding terms

        vect,normalized=self.get_query_vector(docq)
        for (term, tfidf) in vect:
            print(f"{term}:\t{tfidf}\t(normalized: {normalized[term]})")
        
# Implementation of the vector space model using positional index.

class VectorSpaceModel_pos():
    
    def __init__(self,indirizzo,indirizzoq):
        self.name_file_doc=indirizzo  # document file
        self.name_file_q=indirizzoq   # query file
        self.articles = []      # store all the documents and query texts
        self.documentsTFIDF = []   # store document TFIDF vectors
        self.queryTFIDF = []  # store query TFIDF vectors
        self.bagofwords=[]     # overall bagofword
        numdoc=0               # number of documents
        numquery=0             # number of query

    def remove_stop_words(self):
#    Utility function to remove stop words provided in file 
#    "TIME.STP" in the orginal directory file.

        stops = [line.rstrip('\n') for line in open('TIME.STP')]
        stops=list(filter(None, stops))
        for i in range(0,len(self.articles)):
            self.articles[i] = [x for x in self.articles[i] if x not in stops]

            
    def import_dataset(self,kind):
#    This function import all the articles in the TIME corpus,
#    returning list of lists where each sub-list contains all the
#    terms present in the document as a string.
        if kind=="doc": 
            w="*TEXT"
        else: 
            w="*FIND"    
        lines = [line.rstrip('\n') for line in open(self.name_file_doc)]
        self.articles = [list(y) for x, y in itertools.groupby(lines, lambda z: z.startswith(w)) if not x]
        for i in range(0,len(self.articles)):
            self.articles[i] = " ".join(self.articles[i])
            self.articles[i] = re.sub(r'[^a-zA-Z\s]+', '', self.articles[i])
            self.articles[i] = self.articles[i].split()
        self.numdoc=len(self.articles)
        
    def add_query(self,kind):
#    This function import all the query in the TIME.QUE,
#    returning list of lists where each sub-list contains all the
#    terms present in the query as a string.
        if kind=="doc": 
            w="*TEXT"
        else: 
            w="*FIND"    
        lines = [line.rstrip('\n') for line in open(self.name_file_q)]
        articles = [list(y) for x, y in itertools.groupby(lines, lambda z: z.startswith(w)) if not x]
        for i in range(0,len(articles)):
            articles[i] = " ".join(articles[i])
            articles[i] = re.sub(r'[^a-zA-Z\s]+', '', articles[i])
            articles[i] = articles[i].split()
            
        self.numquery=len(articles)
        self.articles=self.articles+articles
   
    def make_positional_index(self):
# A more advanced version of make_inverted_index. Here each posting is
#    non only a document id, but a list of positions where the term is
#    contained in the article.
        index = defaultdict(dict)
        for docid, self.article in enumerate(self.articles):
            for pos, term in enumerate(self.article):
                try:
                    index[term][docid].append(pos)
                except KeyError:
                    index[term][docid] = [pos]
        return index
    
    def make_positional_index_mod(self):
# A more advanced version of make_inverted_index. Here each posting is
#    non only a document id, but a list of positions where the term is
#    contained in the article.
        p_index=self.make_positional_index()
        index_doc = defaultdict(dict)
        index_que = defaultdict(dict)
        for v in p_index.items():
            doc_d={}
            doc_q={}
            for i in v[1].items():
                if i[0]<self.numdoc:
                    doc_d[i[0]]=i[1]
                if i[0]>=self.numdoc:
                    doc_q[i[0]]=i[1]
            index_doc[v[0]]=doc_d
            index_que[v[0]]=doc_q
            del doc_d,doc_q
            
        return index_doc,index_que
    
    def documents_as_vectors(self):
#    Here we generate a list of dictionaries. Each element of the list
#    represents a document and each document has an associated dict where
#    to each term the corresponding tf-idf is associated. Since this function
#    creates a structure of size O(#documents \times #terms), it can
#    be used only for small collections.

        p_index = self.make_positional_index_mod()[0]
        n=self.numdoc
        idf = {}
        for term in p_index.keys():
            try:
                idf[term] = log(n/len(p_index[term]))
            except ZeroDivisionError:
                idf[term] = 0
        for docid in range(0,n):
        # We create a dictionary with a key for each dimension (term)
            v = {}
            for term in p_index.keys():
                try:
                    tfidf = len(p_index[term][docid]) * idf[term]
                except KeyError:
                    tfidf = 0
                v[term] = tfidf
            self.documentsTFIDF.append(v)
            
    def query_as_vectors(self):
#    Here we generate a list of dictionaries. Each element of the list
#    represents a query and each query has an associated dict where
#    to each term the corresponding tf-idf is associated. Since this function
#    creates a structure of size O(#queries \times #terms), it can
#    be used only for small collections.

        p_index = self.make_positional_index_mod()[1]
        n=self.numquery
        idf = {}
        for term in p_index.keys():
            try:
                idf[term] = log(n/len(p_index[term]))
            except ZeroDivisionError:
                idf[term] = 0
        for docid in range(self.numdoc,self.numdoc+n):
        # We create a dictionary with a key for each dimension (term)
            v = {}
            for term in p_index.keys():
                try:
                    tfidf = len(p_index[term][docid]) * idf[term]
                except KeyError:
                    tfidf = 0
                v[term] = tfidf
            self.queryTFIDF.append(v)
            
    def get_document_vector(self,docid):
#    This function returns, for a document/query represented as a vector in v, all the
#    non-zero weights (both normalized and not) and the corresponding terms

        non_zero_terms = [x for x in self.documentsTFIDF[docid].keys() if self.documentsTFIDF[docid][x] > 0]
        vect = [(x, self.documentsTFIDF[docid][x]) for x in non_zero_terms]
        vect.sort(key=lambda x: x[1], reverse=True)
        length = sqrt(sum([x[1]**2 for x in vect]))
        normalized = {k: tfidf/length for k, tfidf in vect}
        return vect,normalized
        
    def get_query_vector(self,docid):
#    This function returns, for a document/query represented as a vector in v, all the
#    non-zero weights (both normalized and not) and the corresponding terms

        non_zero_terms = [x for x in self.queryTFIDF[docid].keys() if self.queryTFIDF[docid][x] > 0]
        vect = [(x, self.queryTFIDF[docid][x]) for x in non_zero_terms]
        vect.sort(key=lambda x: x[1], reverse=True)
        length = sqrt(sum([x[1]**2 for x in vect]))
        normalized = {k: tfidf/length for k, tfidf in vect}
        return vect,normalized
            
    def show_document_vector(self,docid):
#    This function prints, for a document represented as a vector in v, all the
#    non-zero weights (both normalized and not) and the corresponding terms

        vect,normalized=self.get_doc_vector(docid)
        for (term, tfidf) in vect:
            print(f"{term}:\t{tfidf}\t(normalized: {normalized[term]})")
            
    def show_query_vector(self,docq):
#    This function prints, for a query represented as a vector in v, all the
#    non-zero weights (both normalized and not) and the corresponding terms

        vect,normalized=self.get_query_vector(docq)
        for (term, tfidf) in vect:
            print(f"{term}:\t{tfidf}\t(normalized: {normalized[term]})")
        
# Implementation of vector space model using predefined implementation (sklearn library, TfidfVectorizer() function).

class VectorSpaceModel_pre():
    
    def __init__(self,indirizzo,indirizzoq):
        self.name_file_doc=indirizzo  # document file
        self.name_file_q=indirizzoq   # query file
        self.articles = []      # store all the documents and query texts
        self.documentsTFIDF = []   # store document TFIDF vectors
        self.queryTFIDF = []  # store query TFIDF matrix
        self.bagofwords=[]     # overall bagofword
        numdoc=0               # number of documents
        numquery=0             # number of query

    def remove_stop_words(self):
#    Utility function to remove stop words provided in file 
#    "TIME.STP" in the orginal directory file.

        stops = [line.rstrip('\n') for line in open('TIME.STP')]
        stops=list(filter(None, stops))
        for i in range(0,len(self.articles)):
            self.articles[i] = [x for x in self.articles[i] if x not in stops]

            
    def import_dataset(self,kind):
#    This function import all the articles in the TIME corpus,
#    returning list of lists where each sub-list contains all the
#    terms present in the document as a string.
        if kind=="doc": 
            w="*TEXT"
        else: 
            w="*FIND"    
        lines = [line.rstrip('\n') for line in open(self.name_file_doc)]
        self.articles = [list(y) for x, y in itertools.groupby(lines, lambda z: z.startswith(w)) if not x]
        for i in range(0,len(self.articles)):
            self.articles[i] = " ".join(self.articles[i])
            self.articles[i] = re.sub(r'[!#?,.:";]', '', self.articles[i])
            self.articles[i] = re.sub(r'[^a-zA-Z\s]+', '', self.articles[i])
        self.numdoc=len(self.articles)
        
    def add_query(self,kind):
#    This function import all the query in the TIME.QUE,
#    returning list of lists where each sub-list contains all the
#    terms present in the query as a string.
        if kind=="doc": 
            w="*TEXT"
        else: 
            w="*FIND"    
        lines = [line.rstrip('\n') for line in open(self.name_file_q)]
        articles = [list(y) for x, y in itertools.groupby(lines, lambda z: z.startswith(w)) if not x]
        for i in range(0,len(articles)):
            articles[i] = " ".join(articles[i])
            articles[i] = re.sub(r'[!#?,.:";]', '', articles[i])
            articles[i] = re.sub(r'[^a-zA-Z\s]+', '', articles[i])
            
        self.numquery=len(articles)
        self.articles=self.articles+articles
        
    def bag_of_words(self):
        
#    This function create global bag_of_words.

        for i in range(0,len(self.articles)):
            self.articles[i]=self.articles[i].split()
        self.remove_stop_words()
        self.bagofwords = list(set(sum(self.articles, [])))
 
   
    def TFIDFmatrix(self):
        
# This function creates a TFIDF pandas dataframe for documents and query exploiting
# the TfidfVectorizer function provided by sklearn.

        a=self.articles[:self.numdoc].copy()
        b=self.articles[self.numdoc:].copy()
        self.bag_of_words()
        vectorizer = TfidfVectorizer(vocabulary=self.bagofwords,
                             lowercase = False,
                             ngram_range = (1,1),
                             analyzer='word',
                             )

        words = vectorizer.fit_transform(a)
        self.documentsTFIDF = pd.DataFrame(words.todense()).rename(columns=dict(zip(vectorizer.vocabulary_.values(),
            vectorizer.vocabulary_.keys())))
        words1 = vectorizer.fit_transform(b)
        self.queryTFIDF = pd.DataFrame(words1.todense()).rename(columns=dict(zip(vectorizer.vocabulary_.values(),
            vectorizer.vocabulary_.keys())))
            
    def get_document_vector(self,docid):
        
#    This function returns, for a document/query represented as a vector in v, all the
#    non-zero weights (both normalized and not) and the corresponding terms

        vec_doc=self.documentsTFIDF.iloc[docid].to_dict()
        non_zero_terms = [x for x in vec_doc.keys() if vec_doc[x] > 0]
        vect = [(x, vec_doc[x]) for x in non_zero_terms]
        vect.sort(key=lambda x: x[1], reverse=True)
        length = sqrt(sum([x[1]**2 for x in vect]))
        normalized = {k: tfidf/length for k, tfidf in vect}
        return vect,normalized
        
    def get_query_vector(self,docid):
        
#    This function returns, for a document/query represented as a vector in v, all the
#    non-zero weights (both normalized and not) and the corresponding terms

        vec_q=self.queryTFIDF.iloc[docid].to_dict()
        non_zero_terms = [x for x in vec_q.keys() if vec_q[x] > 0]
        vect = [(x, vec_q[x]) for x in non_zero_terms]
        vect.sort(key=lambda x: x[1], reverse=True)
        length = sqrt(sum([x[1]**2 for x in vect]))
        normalized = {k: tfidf/length for k, tfidf in vect}
        return vect,normalized
            
    def show_document_vector(self,docid):
        
#    This function prints, for a document represented as a vector in v, all the
#    non-zero weights (both normalized and not) and the corresponding terms

        vect,normalized=self.get_doc_vector(docid)
        for (term, tfidf) in vect:
            print(f"{term}:\t{tfidf}\t(normalized: {normalized[term]})")
            
    def show_query_vector(self,docq):
        
#    This function prints, for a query represented as a vector in v, all the
#    non-zero weights (both normalized and not) and the corresponding terms

        vect,normalized=self.get_query_vector(docq)
        for (term, tfidf) in vect:
            print(f"{term}:\t{tfidf}\t(normalized: {normalized[term]})")
        

