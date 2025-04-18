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


# for train-test splitting
COLUMNS_TO_KEEP = ["user_id", "prd_number", "completion_rate"]
MIN_PLAYS_PER_EPISODE = 10
MIN_PLAYS_PER_USER = 2
SPLIT_DATE = "2024-11-11"
TEST_DATA_PATH = "data\podcast_data_test.parquet"
TRAIN_DATA_PATH = "data\podcast_data_train.parquet"


# for implementing recommenders



# for evaluation