from collections import defaultdict
import os
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import (
    METADATA_PATH,
    EMBEDDINGS_TITLE_PATH,
    EMBEDDINGS_DESCR_PATH,
    LAMBDA_CB,
    EMBEDDINGS_COMBI_PATH,
)
import utils.utils as utils

# downloading stopwords
nltk.download('stopwords')

# dictionaries to hold embeddings
embedding_dicts = defaultdict(dict)

# sentence transformer model used to obtain embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Danish stopwords
stop_words = set(stopwords.words('danish'))

# Danish stemmer
stemmer = SnowballStemmer("danish")

# loading metadata
meta_df = pd.read_parquet(METADATA_PATH)

# computing embeddings for each episode
print("Computing embeddings based on episode titles and descriptions.")
for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
    # extracting prd_number and textual features
    prd_number = row["prd_number"]
    episode_title = row.get("episode_title", "") or ""
    episode_description = row.get("episode_description", "") or ""
    texts = [episode_title, episode_description]

    for i, text in enumerate(texts):
        # tokenizing the text 
        words = text.split()

        # removing stopwords
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # joining the words back into a sentence
        filtered_text = " ".join(filtered_words)

        # stemming the filtered words
        stemmed_words = [stemmer.stem(word) for word in filtered_words]

        # joining the words back into a sentence
        stemmed_text = " ".join(stemmed_words)
        
        # generating embedding
        embedding = model.encode(stemmed_text).tolist()

        # normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm != 0:  
            embedding = (np.array(embedding) / norm).tolist()

        embedding_dicts[i][prd_number] = embedding

# paths to embedding locations
emb_paths = [EMBEDDINGS_TITLE_PATH, EMBEDDINGS_DESCR_PATH]

# saving embeddings
print("Saving embeddings.")
for i, emb_dict in enumerate(embedding_dicts.values()):
    emb_dict_formatted = utils.format_embedding_dict(emb_dict)
    emb_df = pd.DataFrame(emb_dict_formatted)
    path = emb_paths[i]
    emb_df.to_parquet(path, index=False)

# computing and saving combi embeddings
print("Computing combi embeddings.")
items = meta_df["prd_number"].unique()
combi_embeddings = {}

for item in tqdm(items):
    title_emb = embedding_dicts[0][item]
    descr_emb = embedding_dicts[1][item]
    combi_emb = LAMBDA_CB * np.array(title_emb) + (1 - LAMBDA_CB) * np.array(descr_emb)
    combi_emb_list = list(combi_emb)
    combi_embeddings[item] = combi_emb_list

print("Saving combi embeddings.")
combi_embeddings_formatted = utils.format_embedding_dict(combi_embeddings)
combi_emb_df = pd.DataFrame(combi_embeddings_formatted)
combi_emb_df.to_parquet(EMBEDDINGS_COMBI_PATH, index=False)

print("Done! All embeddings have been saved to embeddings folder.")