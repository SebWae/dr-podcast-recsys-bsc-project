# Podcast Episode Recommendations in a Public Service Setting
This is the repository for my BSc project in Data Science @ IT University of Copenhagen.  
In collaboration with DR (Danmarks Radio). 

Supervisor: Toine Bogers


## Project Structure
```
├── data                                  
│   │
│   ├── podcast_data_filtered.parquet       <- parquet file containing filtered data
│   │
│   ├── podcast_data_raw.parquet            <- parquet file containing raw data
│   │
│   └── podcast_data_transformed.parquet    <- parquet file containing transformed data
│
├── eda                                   
│   │
│   └── eda.ipynb                           <- notebook for exploratory data analysis (eda)
│
├── scripts                             
│   │
│   ├── 01_filter.py                        <- script for filtering the raw data
│   │
│   ├── 02_transform.py                     <- script for applying data transformations
│   │
│   ├── 03_extract_metadata.py              <- script for extracting podcast episode metadata
│   │
│   ├── 04_train_test.py                    <- script for splitting the data into a train and test set
│   │
│   ├── 05a_cf_recommender.py               <- script for implementing collaborative filtering recommender
│   │
│   ├── 05b_cb_recommender.py               <- script for implementing content-based recommenders
│   │
│   ├── 05c_hybrid_recommender.py           <- script for implementing hybrid recommender
│   │
│   └── 06_evaluation.py                    <- script for evaluating recommender systems
│  
├── .gitattributes                          <- for handling large file storage of parquet files
│
├── config.py                               <- configuration file storing variables used in the scripts
│  
├── README.md                               <- project description and how to run the code
│
└── utils.py                                <- utility functions used in the scripts
```

## How to run the code?
It is recommended to run the code using the virtual environment specified by `environment.yml`. This requires `conda 24.9.2`.

Create the virtual environment by running:
```
conda env create -f environment.yml
``` 

Activate the environment by running:
```
conda activate dr-recsys-env
```