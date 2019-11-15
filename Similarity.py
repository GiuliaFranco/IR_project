import distance
from math import log, sqrt
import numpy as np
import operator
import itertools

##########################
#			 #
#   Similarity measures  #
#			 #
##########################

class Similarity():
    
# we want in input a  VectorSpaceModel object (with both documents and eventual query inside) 
# in order to compute similarity using different kind of measueres. The constructor can take 
# as input any of the three implementations of VSM.

    coll_1=[]   #collection 1: all documents from the VectorSpaceModel objects
    coll_2=[]   #collection 2: all query from the VectorSpaceModel objects
    doc_corpus=[]  #all tokenized documents 
    q_corpus=[]    #all tokenized queries 
    res_cos = {}  #store the results for comparison of cosine_similarity for a query 
                  #and all the documents.
    res_mat = {}   #store the results for comparison of matching for a query 
                  #and all the documents.
    res_jac= {}    #store the results for comparison of jaccard_similarity for a query 
                  #and all the documents.

    def __init__(self,doc):
            
# Take as constructor the VectorSpaceModel object and create document and query collection.
        self.doc_corpus=doc.articles[:doc.numdoc]
        self.q_corpus=doc.articles[doc.numdoc:]
        for coll_id_1 in range(0,doc.numdoc):
            self.coll_1.append(doc.get_document_vector(coll_id_1)[1])
        for coll_id_2 in range(0,doc.numquery):
            self.coll_2.append(doc.get_query_vector(coll_id_2)[1])
        
    def cosine_similarity(self,id_doc,id_q):
        
# Take in input the id of the document and query and compute the cosine similarity 
# over them.
        
        vec_1=self.coll_1[id_doc]
        vec_2=self.coll_2[id_q]
        value_vec = [vec_1[key] for key in vec_1.keys() & vec_2.keys()]
        value_q = [vec_2[key] for key in vec_1.keys() & vec_2.keys()]
        a=np.array(value_vec)
        b=np.array(value_q)
        cos = np.dot(a, b)
        return cos
    
    def matching(self,id_doc,id_q):
        
# Take in input the id of the document and query and compute the matching score (sum over
# the weights of common words) over them.        
        
        vec_1=self.coll_1[id_doc]
        vec_2=self.coll_2[id_q]
        value_vec = [vec_1[key] for key in vec_1.keys() & vec_2.keys()]
        score = sum(value_vec)
        return score

    def jaccard_similarity(self,id_doc,id_q):
        
# Take in input the id of the document and query and compute the jaccard similarity 
# over them.

        vec_1=self.coll_1[id_doc]
        vec_2=self.coll_2[id_q]
        intersection = len([key for key in vec_1.keys() & vec_2.keys()])
        union = (len(vec_1) + len(vec_2)) - intersection
        return float(intersection) / union
        
# Utility functions in order to exploit the previous methods between a selected query
# and all the documents.
    def print_utility(self,sorted_res,method,id_q):
        print("overall comparison for query " + str(id_q) +":\n")
        for j in range(0,len(sorted_res)):            
            print("doc: "+str(sorted_res[j][0])+" "+str(method)+" :"+str(sorted_res[j][1])+"\n")

    def compare_all_cosine(self,id_q):
        for i in range(0,len(self.coll_1)):
            self.res_cos[i]=self.cosine_similarity(i,id_q)
        sorted_res = sorted(self.res_cos.items(), key=operator.itemgetter(1),reverse=True)
        fin=[]
        for j in range(0,len(sorted_res)):            
            fin.append(sorted_res[j][0])
        return fin,sorted_res
        
    def compare_all_matching(self,id_q):
        for i in range(0,len(self.coll_1)):
            self.res_mat[i]=self.matching(i,id_q)
        sorted_res = sorted(self.res_mat.items(), key=operator.itemgetter(1),reverse=True)
        fin=[]
        for j in range(0,len(sorted_res)):            
            fin.append(sorted_res[j][0])
        return fin,sorted_res 
    
    def compare_all_jaccard(self,id_q):
        for i in range(0,len(self.coll_1)):
            self.res_jac[i]=self.jaccard_similarity(i,id_q)
        sorted_res = sorted(self.res_jac.items(), key=operator.itemgetter(1),reverse=True)
        fin=[]
        for j in range(0,len(sorted_res)):            
            fin.append(sorted_res[j][0])
        return fin,sorted_res
    
    def calling(self,id_q):
        m=input("choose the method between\n 1:Cosine Similarity\n 2:matching\n 3:Jaccard\n ")
        if m=="1": 
            self.print_utility(self.compare_all_cosine(id_q)[1],"cosine",id_q)            
        if m=="2": 
            self.print_utility(self.compare_all_matching(id_q)[1],"matching",id_q)
        if m=="3":
            self.print_utility(self.compare_all_jaccard(id_q)[1],"jaccard",id_q)
            
    def comparison_VSM_BM25(self,id_q,N):
        
# This function compares the results obtained with the VSM (using cosine similarity measure) 
# with the one obtained using BM25 (probabilistic model). 

        a=bm25.get_top_n(self.q_corpus[id_q], self.doc_corpus, n=N)
        c_1=self.compare_all_cosine(id_q)[0][:N]
        print("here the results for BM25:\n ")
        for i in range(0,len(a)):
            print(self.doc_corpus.index(a[i]))
        print("\nhere the results for VSM:\n ")
        for i in range(0,len(c_1)):
            print(c_1[i])

