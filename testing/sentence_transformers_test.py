from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
descriptions = [
    "Jeg elsker at løbe om morgenen.", 
    "Jeg hader at løbe om morgenen."  
]
embeddings = model.encode(descriptions)

# Compute cosine similarity between embeddings
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        similarity = 1 - cosine(embeddings[i], embeddings[j])
        print(f"Similarity between sentence {i + 1} and sentence {j + 1}: {similarity:.4f}")