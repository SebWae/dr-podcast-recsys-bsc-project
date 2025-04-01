from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import spacy

# Load SpaCy's Danish tokenizer
nlp = spacy.load("da_core_news_sm")

# Example corpus (replace with your own Danish text corpus)
corpus = [
    "Dette er en dansk sætning.",
    "Jeg elsker at læse danske bøger.",
    "Maskinlæring er et spændende felt."
]

# Preprocess and tokenize the corpus
tagged_data = [TaggedDocument(words=[token.text for token in nlp(doc)], tags=[str(i)]) for i, doc in enumerate(corpus)]

# Train the Doc2Vec model
model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Save the model
model.save("testing/danish_doc2vec.model")
print("Model saved!")

model = Doc2Vec.load("testing/danish_doc2vec.model")

# Example text
text = "Dette er et eksempel på en dansk nationalret."

# Tokenize the input using SpaCy
tokens = [token.text for token in nlp(text)]

# Generating the embedding
embedding = model.infer_vector(tokens)
print(embedding)

