from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('stopwords')

embedding_dict = {}

# sentence transformer model used to obtain embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Danish stopwords
stop_words = set(stopwords.words('danish'))

# Danish stemmer
stemmer = SnowballStemmer("danish")



texts = [" ", ""]

for i, text in enumerate(texts):
    # Tokenize the text (split it into words)
    words = text.split()

    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Join the words back into a sentence
    filtered_text = " ".join(filtered_words)

    print(filtered_text)

    # Stem the filtered words
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Join the words back into a sentence
    stemmed_text = " ".join(stemmed_words)
    print(stemmed_text)
    embedding = model.encode(stemmed_text).tolist()

    embedding_dict[i] = embedding

print(embedding_dict)


