Usage guideline:

---------------------------------
1- VSM: scratch implementation  -
---------------------------------

Define the VSM, fill it up with documents and queries, calculate TFIDF vectors.

doc=VectorSpaceModel_scratch("TIME.ALL","TIME.QUE")
doc.import_dataset("doc")
doc.add_query("query")
doc.bag_of_words()
doc.TFIDF_document_query()

Define the Similarity class and call one of the available similarity measures.

sim=Similarity(doc)
qID=2
sim.calling(qID)

Define the Relevance feedback class and get the results for Rocchio and pseudo-relevance feedback.

rel=Relevance_feedback("TIME.REL",doc)
rel.get_Pseudo_res(2,10,100)
rel.get_ROCCHIO_res(2,10)



----------------------------------------
2- VSM: positional idx implementation  -
----------------------------------------

Define the VSM, fill it up with documents and queries, calculate TFIDF vectors.

doc=VectorSpaceModel_pos("TIME.ALL","TIME.QUE")
doc.import_dataset("doc")
doc.add_query("query")
doc.remove_stop_words()
doc.documents_as_vectors()
doc.query_as_vectors()

Define the Similarity class and call one of the available similarity measures.

sim=Similarity(doc)
qID=2
sim.calling(qID)

Define the Relevance feedback class and get the results for Rocchio and pseudo-relevance feedback.

rel=Relevance_feedback("TIME.REL",doc)
rel.get_Pseudo_res(2,10,100)
rel.get_ROCCHIO_res(2,10)
