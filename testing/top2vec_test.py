from top2vec import Top2Vec

# Sample corpus
documents = ["This is the first document.", 
             "This document is about machine learning.",
             "Deep learning is a subset of machine learning."]

# Train the model
model = Top2Vec(
    documents,
    min_count=1, 
    umap_args={'n_neighbors': 2, 'n_components': 2}  # Reduce UMAP parameters
)

# Get embedding for a specific document (e.g., first document)
doc_embedding = model.document_vectors[0]

print(doc_embedding)
