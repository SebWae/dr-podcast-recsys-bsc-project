from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
descriptions = ["Denne episode handler om kunstig intelligens og dens indflydelse på samfundet.", 
                "En dybdegående analyse af rumforskning."]
embeddings = model.encode(descriptions)

print(embeddings)