from collections import defaultdict
import numpy as np
import VSM 
import Similarity as SI
from math import log, sqrt

#############################
#                           #
#  Relevance feedback class #
#		 	    #
#############################


# Class implemented for Relevance feedback, both Rocchio algorithm and pseudorelevance feeedback.
# The constructor implementation is meant to be able to take as input 2 kind of vector space model (scratch and positional index VSM).
 
class Relevance_feedback():
    
    relevance={}   # relevance feedback (for each query) taken from file"TIME.REL"
    
    def __init__(self,indirizzo,doc):
    # Take as input the address of the feedback ("TIME.REL") and a VSM.
    # The function fills up a dictionary: the key is the queryID and the values will be a list of relevant documentsID for each.

        self.VSM=doc
        self.SIM=SI.Similarity(doc)
        lines = [line.strip() for line in open(indirizzo)]
        lines = list(filter(None, lines))
        for i in range(0,doc.numquery):
            l=[int(s) for s in " ".join(lines[i].split()).split(' ')]
            self.relevance[l[0]] = l[1:]
            
            
            
    def get_new_query_vector(self,new_query_dict):

	# Auxiliary function used in order to convert the modified query (from Rocchio o pseudoRelevance) into a readable form for 
        # Similarity class (normalization and non zero terms).

        non_zero_terms = [x for x in new_query_dict.keys() if new_query_dict[x] > 0]
        vect = [(x, new_query_dict[x]) for x in non_zero_terms]
        vect.sort(key=lambda x: x[1], reverse=True)
        length = sqrt(sum([x[1]**2 for x in vect]))
        normalized = {k: tfidf/length for k, tfidf in vect}
        return vect,normalized
  
    def Rocchio_alg(self,idq,N,alpha=1,beta=0.75,gamma=0.15):

	# Rocchio algorithm takes as input the queryID, the number of iterations and the hyperparameters (default values specified). 
	# From the relevance dict it retreives the relevant docIDs (so not relevant ones as well) and use them in order to extract the vectors
	# defined in VSM class form queryTFIDF and documentsTFIDF. Finally it performs N times the update of the original query and returns the modified one.

        list_docids=self.relevance[idq+1]
        relevant_doc=[]
        original_query=np.array(list(self.VSM.queryTFIDF[idq].values()))
        for i in list_docids:
            relevant_doc.append(self.VSM.documentsTFIDF[i-1])
        not_relenvant_doc=[x for x in self.VSM.documentsTFIDF if x not in relevant_doc]
        relevant_doc=[np.array(list(xi.values())) for xi in relevant_doc]
        not_relenvant_doc=[np.array(list(xi.values())) for xi in not_relenvant_doc]
        for i in range(0,N):
            new_query=alpha*original_query+beta*sum(relevant_doc)/len(relevant_doc)-gamma*sum(not_relenvant_doc)/len(not_relenvant_doc)
            original_query=new_query
        new_query[new_query < 0] = 0
        new_query_dict=dict(zip(list(self.VSM.queryTFIDF[idq].keys()),list(new_query)))
        return new_query_dict
    
    def Pseudo_relevance(self,idq,k,N,alpha=1,beta=0.75,gamma=0.15):

        # Takes as input the queryID, the k number of top relevant documents to consider. the number of iterations and the hyperparameters (default values specified).
	# The function extracts the the vector of the query using the queryID and applies the cosine similarity measure in order to obtain the first k relevant 
	# documents (the others will be not relevant). Finally it updates N times the original query and returns the modified one.

        relevant_doc=[]
        original_query=np.array(list(self.VSM.queryTFIDF[idq].values()))
        list_docids=self.SIM.compare_all_cosine(idq)[0][:k]
        for i in list_docids:
            relevant_doc.append(self.VSM.documentsTFIDF[i])
        not_relenvant_doc=[x for x in self.VSM.documentsTFIDF if x not in relevant_doc]
        relevant_doc=[np.array(list(xi.values())) for xi in relevant_doc]
        not_relenvant_doc=[np.array(list(xi.values())) for xi in not_relenvant_doc]
        for i in range(0,N):
            new_query=alpha*original_query+beta*sum(relevant_doc)/len(relevant_doc)-gamma*sum(not_relenvant_doc)/len(not_relenvant_doc)
            original_query=new_query
        new_query[new_query < 0] = 0
        new_query_dict=dict(zip(list(self.VSM.queryTFIDF[idq].keys()),list(new_query)))
        return new_query_dict
    
    def get_ROCCHIO_res(self,idq,N):
	# This function returns the results for Rocchio algorithms by making a final similarity measure (the user can choose between cosine similarity,matching and 
	# jaccard similarity) between the updated query and all the documents.

        self.SIM.coll_2[idq]=self.get_new_query_vector(self.Rocchio_alg(idq,N))[1]
        self.SIM.calling(idq)
    
    def get_Pseudo_res(self,idq,k,N):

	# This function returns the results for Pseudo_relevance algorithms by making a final similarity measure (the user can choose between cosine similarity,
 	# matching and  jaccard similarity) between the updated query and all the documents.

        self.SIM.coll_2[idq]=self.get_new_query_vector(self.Pseudo_relevance(idq,k,N))[1]
        self.SIM.calling(idq)
