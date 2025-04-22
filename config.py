# for filtering and transforming data
END_DATE = "2024-12-01 23:59:59"
FILTERED_DATA_PATH = "data\podcast_data_filtered.parquet"  
MIN_CONTENT_TIME_SPENT = 60  
RAW_DATA_PATH = "data\podcast_data_raw.parquet"
START_DATE = "2024-09-02 00:00:00"
TRANSFORMED_DATA_PATH = "data\podcast_data_transformed.parquet"
VAR_RENAME_DICT = {"Date:Time (evar16)":           "date_time",
                   "UserID Visit (evar95)":        "user_id",
                   "Production Number (evar35)":   "prd_number",
                   "Serietitel":                   "series_title",
                   "Unik titel":                   "unique_title",
                   "Platform (evar13)":            "platform",
                   "Mobile Device Type":           "device_type",
                   "Content Type (revar3)":        "content_type",
                   "Første Streaming Dato":        "pub_date",
                   "Video Length":                 "episode_duration",
                   "Genre":                        "genre",
                   "Branding Kanal":               "branding_channel",
                   "Moder Kanal":                  "mother_channel",
                   "Kategori":                     "category",
                   "Content Time Spent (revent1)": "content_time_spent",
                   }


# for extracting metadata
DESCRIPTION_VAR_RENAME_DICT = {"PRD_NUMBER":       "prd_number",
                               "PRD_SHORTDESCR":   "episode_description",
                               }
EPISODE_DESCRIPTION_PATH = "data\episode_descriptions.parquet"
METADATA_COLUMNS = {"series_title":         "first",
                    "unique_title":         "first",
                    "pub_date":             "first",
                    "episode_duration":     "first",
                    "genre":                "first",
                    "branding_channel":     "first",
                    "mother_channel":       "first",
                    "category":             "first",
                    "episode_description":  "first",
                    }
METADATA_PATH = "data\episode_metadata.parquet"


# for train-test splitting
COLUMNS_TO_KEEP = ["user_id", "prd_number", "completion_rate"]
MIN_PLAYS_PER_EPISODE = 10
MIN_PLAYS_PER_USER = 2
SPLIT_DATE = "2024-11-11"
TEST_DATA_PATH = "data\podcast_data_test.parquet"
TRAIN_DATA_PATH = "data\podcast_data_train.parquet"


# for implementing recommenders
EMBEDDINGS_COMBI_PATH = "embeddings/combined_embeddings.parquet"
EMBEDDINGS_DESCR_PATH = "embeddings/descr_embeddings.parquet"
EMBEDDINGS_TITLE_PATH = "embeddings/title_embeddings.parquet"
EPSILON = 0.001
N_COMPONENTS = 40   # inspired by Funk SVD
N_EPOCHS = 1000
N_RECOMMENDATIONS = 10
RANDOM_STATE = 250500
RECOMMENDATIONS_KEY_CB_COMBI = "cb_recommendations_combi"
RECOMMENDATIONS_KEY_CB_DESCR = "cb_recommendations_descr"
RECOMMENDATIONS_KEY_CB_TITLE = "cb_recommendations_title"
RECOMMENDATIONS_KEY_CF = "cf_recommendations"
RECOMMENDATIONS_KEY_HYBRID = "hybrid_recommendations"
RECOMMENDATIONS_PATH = "results/recommendations.json"
RECOMMENDERS = ["cf_recommendations", "hybrid_recommendations"]


# for evaluation
EVALUATION_METRICS = ["hit_rate", "ndcg"]
RECOMMENDER_EVAL_PATH = "results/recommender_eval.json"
USER_EVAL_PATH = "results/user_eval.json"