import VSM
import Relevance_feedback as RF
import Similarity as SI

## I implemented 3 vector space model in order to have a time-guide comparison criteria. 
## The last implementation,the one that exploits the sklearn library, is not meant for
## usage with other classes.


###################
#    First VSM    #
###################

doc=VSM.VectorSpaceModel_scratch("TIME.ALL","TIME.QUE")
doc.import_dataset("doc")
doc.add_query("query")
doc.bag_of_words()
doc.TFIDF_document_query()


rel=RF.Relevance_feedback("TIME.REL",doc)
rel.get_ROCCHIO_res(2,10)


###################
#  Second  VSM    #
###################



doc1=VSM.VectorSpaceModel_pos("TIME.ALL","TIME.QUE")
doc1.import_dataset("doc")
doc1.add_query("query")
doc1.remove_stop_words()
doc1.documents_as_vectors()
doc1.query_as_vectors()


rel1=RF.Relevance_feedback("TIME.REL",doc1)
rel1.get_ROCCHIO_res(2,10)
