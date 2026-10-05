# Case Study: Commercial Auto Risk Scoring

Commercial auto insurance protects vehicle fleets operated by businesses. This project builds an end-to-end risk scoring pipeline to predict claim incidence (claim_count > 0) per policy holder, quantifying fleet risk.

Primary Objective: Estimate the continuous probability of claim occurrence (claim > 0) for each policy_id using binary classification.

Core Model: XGBoost Classifier

## Repository Structure

```text
├── data/
│   ├── orig/          # original files (CSV)
│   ├── cleaned/       # preprocessed, deduplicated data (Parquet)
│   └── predictions/   # model inference outputs and score tables (CSV)
├── databricks/        # databricks screenshots
├── models/            # Serialized model pipelines & metadata
├── notebooks/
│   ├── 01_eda.ipynb           # exploratory data analysis & statistical summaries
│   ├── 02_preprocessing.ipynb # cleaning, deduplication, and feature engineering
│   ├── 03_risk_scorer.ipynb   # model training, scoring, evaluation 
│   └── 04_load_model.ipynb    # loads model, predictions & export
├── case_study_tasks.md        # details to tasks described in word document
├── data_dictionary.md         # data dictionary
├── commercial_auto_risk_scoring.pptx # presentation
├── requirements.txt   # dependencies
├── README.md
└── utils.py           # shared helper functions (visualizations, deduplication, transforms)
```

Task Details: See **_case_study_tasks.md_**

Within this document can be found details to the first 4 tasks from defined in the 
'Data Scientist Applied AI ML - Case Study.docx'

1. Exploratory data analysis
2. Modeling methodology improvements
3. Production readiness improvements
4. Production readiness roadmap
5. AI usage

Original and Cleaned datasets are not part of this repository.

## Key Assumptions and Validation Approach
- claim incidents are rare events (in trainig data 12.3%)
- the classes (claim / no claim) are very imbalanced
- we are trying to predict/train the model rather to identify these rare situations
- in these cases accuracy can be misleading, as it may correctly predict rather the 'bigger' class, returning high accuracy in identifying the 'no claim' policies and we may miss the 'outliers' == possible claims


## Approach and steps

To be able to prepare prediction model, there is need to run preprocessing steps
to cleanup data and define features, which will be used during training to get 
the most accurate target metric.

Steps:
1. **Exploratory Data Analysis** (more details in ```case_study_tasks.md``` file)
   - my aim was to get know the data, which are available and identify possible issues
   - I've focused mainly on following areas:
     - **deduplication** of records
       - duplicates per all columns
       - per primary key ```policy_id```
     - **data types mismatch**
       - e.g. date colums are read from CSV files as string -> need to cast to datetime
       - no mixed data have been found in this dataset
     - **null values**
       - I've checked primary all columns with nulls to define how to handle them
     - **inconsistent data**
       - lower/upper/capital letters inconsistency -> e.g. in ```business_type``` column
         - this creates multiple unnecessary categories
     - **suspicious data**
       - rows with policy_expiration_date earlier then policy_effective_date
       - negative values in columns where not expected, e.g ```annual_premium``` (maybe cancellations)
       - ```vehicle_count``` == 0 while ```mileage_per_vehicle``` > 0
     - **categorical columns**
       - checking number and type of categories -> to be able to define strategy to encode them
       - checking if e.g. one category doesn't lay under 1 target group, which may lead to incorrect predictions (model may learn to think all those values leads to that target group)
     - **outliers**
       - ```annual_premium``` with negative values
       - ```prior_year_mileage_000``` with very huge number
         - therefore new columns have been created: 
           - ```total_vehicle_count = vehicle_count + num_heavy_vehicles```
           - ```mileage_per_vehicle = prior_year_mileage_000 / total_vehicle_count```
     - **checking domain specifics**
       - trying to identify which columns may depends on each other and calculate new colums out of them
         - e.g. ```policy_length_days``` / ```elapsed_time``` / ```total_vehicle_count``` / ```loss_per_claim``` etc
2. **Preprocessing**
   - clean up of data issues identified in EDA step
   - creation of new columns as defined
   - define feature columns, which will be used during modeling 
     - I've decided to exclude:
       - id columns like ```policy_id``` and ```insured_id```
       - dates columns have been replaced with newly calculated columns:
         - ```exposure_ratio = elapsed_days / policy_length_days```
       - columns which depends or are determined on target column or are determining target column
   - this step must be done on the 'real' dataset as well
     - to remove duplicates/unify string columns cases/to add calculated columns
3. **Training and Validating of Model**
   - XGBoost model
     - at the end I've decided to use this model because:
       - it splits features hierarchically and can discover non-linearities without manual feature combinations
       - should be more resilient to skewness and outliers
         - many numeric values shows strong right skewness of the data
         - loss events are rather rare and thus importat data could be outliers
       - learns optimal default split directions for missing values during training
   - creating of pipeline to ensure that the train_test_split is before null handling, encoding or scaling of data
     - ```prior_loss_amount``` is set to 0.0 when ```prior_apd_claim_count``` and ```prior_al_claim_count``` == 0, otherwise median is used
   - for the rest of numerical data is generally used median, as mean is sensitive to outliers
   - for categorical data:
     - ```payment_frequency``` - for null values is used "unknown", as the distribution of other values was quite equal
     - for the rest of categorical values is used "most frequent" strategy
  

## Decisions 

### Cleanup

1. to remove duplicates 
   - may mislead the model - as some data have been read more times
2. to lower case and trim string columns
   - to unify the data - to group them - so they are not considered as new data/categories
   - important for encoding of categorical data - unnecessary categories / columns may be created
3. to cast date columns to datetime
   - to be able to use those columns as date and create new columns
4. to replace value 'agriculture' within ```business_type``` with 'other'
   - there was only 1 record with target column ```claim_count``` == 0, which may maybe mislead the model that all such records should get value of 0 (no claim risk)
5. to remove annual_premium < 0
   - were only 5 records, all with claim_count == 0
6. to remove records where policy_expiration_date < policy_effective_date
   - this would need to be checked with SME, but looks such data doesn't make sense, maybe some error
   - there are 10 such records
7. to remove 2 records where ```total_vehicle_count``` == 0 but ```prior_year_mileage_000``` or ```driver_count``` > 0
   - looks there may be some error in data and as there are only 2 records they've been dropped

### Calculated columns

1. ```policy_length_days = policy_expiration_date - policy_effective_date```
2. ```elapsed_days = snapshot_date - policy_effective_date```
3. ```exposure_days = elapsed_days.clip(lower=1.0)```
4. ```exposure_ratio = elapsed_days / policy_length_days```
5. ```remaining_days = policy_expiration_date - snapshot_date```
6. ```total_vehicle_count = vehicle_count + num_heavy_vehicles```
7. ```mileage_per_vehicle = prior_year_mileage_000 / total_vehicle_count```
8. ```loss_per_claim = if total_prior_claims > 0 then prior_loss_amount / total_prior_claims else 0.0```
9.  ```has_large_prior_loss = prior_loss_amount > 50000```
10. ```risk_score_external_isna = risk_score_external.isna().astype(int)```
   

## Limitations

There would be need to:
- enhance the solution with testing of the functions
- remove print and replacing the logic with logging
- detail column analysis to define possible new columns / features, which may better fit
- to define better way how to impute null values for some selected columns or how to encode categorical columns
- modeling
  - class imbalance and threshold tuning, as risk are rare
  - to define more/other paramters
  - use cross-validation


## How to set up and run
1. installation
   - git clone <repo-url>
   - cd <repo-name>
   - create virtual environment: python -m venv .venv
   - source venv\Scripts\activate
   - pip install -r requirements.txt
2. Data Pipeline and Execution Flow:

In files 01_eda.ipynb and 02_preprocessing.ipynb there is split between "train" and "real" dataset determined by variable: REAL_DATA, under the section 'Load Data & Set Variables' which is by default set to False (indicating the run of 'train' data). Code needs to be run for both input files separately.

In the file 03_risk_scorer.ipynb the model is trained and saved.
In the file 04_load_model.ipynb model is loaded and is used for predictions

  - original csv files "data/orig/*.csv" are being read in **01_eda.ipynb**
    - ──> **02_preprocessing.ipynb**
        - original csv files "data/orig/*.csv" read again
        - here is created parquet with cleaned data "data/cleaned/*.parquet"
      - ──> **03_risk_scorer.ipynb**
        - uses created parquet to train model and to run predictions - creates file "data/predictions/*.csv"
        - ──> **04_load_model.ipynb**
  - 01_eda.ipynb: Run first to inspect feature distributions, missingness patterns, and anomalous business constraints.
  - 02_preprocessing.ipynb: Cleans raw data and writes them to Parquet. Run this for both train.csv and score.csv.
  - 03_risk_scorer.ipynb: Trains and validates the XGBoost model on train.parquet.
  - 04_load_model.ipynb: Generates final scored predictions for score.parquet saved to data/predictions/.
  

## AI tools usage
  - Gemini / Google
    - I've used them most as teacher / accelerator / technical reviewer:
      - asking questions about domain specifics (e.g. how / when is some column used)
      - searching for information which model should be selected and which parameters to use
      - code review and code improvement suggestions