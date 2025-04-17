# for filtering and transforming data
END_DATE = "2024-12-01 23:59:59"
FILTERED_DATA_PATH = "data\podcast_data_filtered.parquet"  
MIN_CONTENT_TIME_SPENT = 60  
RAW_DATA_PATH = "data\podcast_data_raw.parquet"
START_DATE = "2024-09-02 00:00:00"
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
TRANSFORMED_DATA_PATH = "data\podcast_data_transformed.parquet"


# for extracting metadata


# for train-test splitting



# for implementing recommenders



# for evaluation