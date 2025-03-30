from sentence_transformers import SentenceTransformer

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, and effective

# Encode a single sentence
sentence = "Deep learning is a subset of machine learning."
embedding = model.encode(sentence)

print(embedding) 
