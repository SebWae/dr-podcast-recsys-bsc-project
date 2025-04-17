# Podcast Episode Recommendations in a Public Service Setting
This is the repository for my BSc project in Data Science @ IT University of Copenhagen.  
In collaboration with DR (Danmark's Radio). 

Supervisor: Toine Bogers


## Project Structure
```
├── data                                    <- data folder
│   │
│   ├── podcast_data_filtered.parquet       <- parquet file containing filtered data
│   │
│   ├── podcast_data_raw.parquet            <- parquet file containing raw data
│   │
│   └── podcast_data_transformed.parquet    <- parquet file containing transformed data
│
├── eda                                     <- folder for exploratory data analysis
│   │
│   └── eda.ipynb                           <- notebook for exploratory data analysis (eda)
│
├── scripts                                 <- Source code for the project
│   │
│   ├── 01_filter.py                        <- Marks the directory as a Python package
│   │
│   ├── 02_transform.py                     <- Script for data preprocessing and transformation
│   │
│   ├── 03_extract_metadata.py              <- Script for training the models
│   │
│   ├── 04_train_test.py                    <- Script for selecting the best perfoming model
│   │
│   ├── 05a_cf_recommender.py               <- Script for comparing new best model and production model
│   │
│   ├── 05b_cb_recommender.py               <- Script for deploying model
│   │
│   ├── 05c_hybrid_recommender.py           <- Constants and paths used in the pipeline's scripts
│   │
│   └── 06_evaluation.py                    <- Encapsulated code from the .py monolith.
│  
├── .gitattributes                          <- for handling large file storage of parquet files
│
├── config.py                               <- configuration file storing variables used in the scripts
│  
├── README.md                               <- project description and how to run the code
│
└── utils.py                                <- utility functions used in the scripts
```
