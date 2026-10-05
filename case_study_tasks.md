1.  _Exploratory data analysis._

- Explore the dataset and note any patterns or surprises that inform your modeling choices.
  1. **_comparison of training vs. real dataset_**
  - **'has_safety_program'**
    - column is not in the real dataset
    - could be good to **CHECK** with e.g. SME, if not available or how to get it / if it is important or not
    - claim percentage is 14.04 without "has_safety_program"
    - claim percentage is 10.29 with "has_safety_program"
  2. **_target columns_** analysis
  - **'claim_count'**
    - 1854 (after removal of duplicates) records where claim_count > 0
    - 15000 all rows (after removal of duplicates)
    - == only 12.36% out of all records
  - **'target columns'**
    - columns either not available for predicion or determined/determining the target column
      - "first_claim_reported_date" (datetime column)
      - "claim_paid_amount_current_period"
      - "claim_status_current_period" (categorical column)
      - "days_to_first_claim_report"
      - "total_loss_amount"
    - are consistent (if 'claim_count' == 0, there are no values in target columns)
  3. **_Analyis_**
  - **duplicities**
    - **_FIX_**: removed
    - duplicates per all columns (all rows identical) >> 42 rows
    - duplicites pre policy_id >> 75 duplicates per 'policy_id'
      - caused by inconsistency in **business_type** column (mixed lower/upper/capital letters)
  - **date columns**
    - date columns are defined as str
      - **_FIX_**: updated to datetime
    - 10 rows with 'policy_expiration_date' < 'policy_effective_date'
      - would be good to **CHECK** with SME why this happened
      - as the count is small in comparison to the complete dataset:
      - **_FIX_**: removed
    - **snapshot_date** is during time when the policy is still open (all records)
      - 77 rows: **snapshot_date** == **policy_effective_date** == elapsed time == 0
        - out of those: 11 rows: where **claim_count** > 0
    - add new columns:
      - policy_length_days = policy_expiration_date - policy_effective_date     -> use
      - elapsed_days = snapshot_date - policy_effective_date
      - exposure_days = elapsed_days.clip(lower=1.0) -> if 0 put to 1
      - exposure_ratio = elapsed_days / policy_length_days                      -> use
      - remaining_days = policy_expiration_date - snapshot_date
  - **mixed data types and not available values in columns**
    - there are no mixed values columns (like numerical column with string values etc)
    - no dummy values ('N/A', 'na', '999' etc)
  - **Categorical Columns** Analysis
    - **'business_type'** column contains inconsistency in data (mixed lower/upper/capital letters)
      - **_FIX_**: identified categorical columns stripped and lower cased
      - 'business_type' == 'agriculture' update to 'other', as there are all records from 1 target category
  - **NULL Values**
    - 8 columns contain NULL values -> there is need to check and decide how to handle those
      - **_'first_claim_reported_date'_** & **_'days_to_first_claim_report'_** --> drop - determined by target
      - **_'driver_avg_age'_** -> normal distribution -> use median
      - **_'prior_year_mileage_000'_**
        - max == 1639.10 - annual fleet mileage in thousands of miles
        - looks as outlier, therefore I've decided to create new columns, as listed below
        - add columns:
          - 'total_vehicle_count' = 'vehicle_count' + 'num_heavy_vehicles'
          - 'mileage_per_vehicle' = 'prior_year_mileage_000'/'total_vehicle_count'
        - 2 records where total_vehicle_count == 0 and prior_year_mileage_000 > 0 and driver_count > 0
          - could be good to **CHECK** with SME why
            - **_FIX_**: remove
        - histogram: right-skewed curve -> median a RobustScaler
      - **_'prior_loss_amount'_**
        - there are 688 rows where prior_loss_amount is null but prior_apd_claim_count or prior_al_claim_count is > 0
        - would be good to **CHECK** why
        - right skewed curve
        - if (prior_apd/al_claim_count then prior_loss_amount == 0) -> else median
        - new columns created: 
          - loss_per_claim = if total_prior_claims > 0 then prior_loss_amount / total_prior_claims else 0.0
          - has_large_prior_loss = prior_loss_amount > 50000
      - **_'late_payment_count'_** -> median
      - **_'payment_frequency'_** -> "unknown"
        - quite evenly distributed over categories: annual/quarterly/monthly
        - there are 978 rows with annual payment frequency and late_payment_count > 1
        - would be good to **CHECK** why
      - **_risk_score_external_**
        - keep XGBoost handle null values
        - added new column: rik_score_external_na -> True / False
    - **_annual_premium_**
      - 5 rows contains negative values
        - would be good to **CHECK** why
        - as there is claim_count == 0 and I assume these as cancellations:
        - **_FIX_**: drop records
    - **_vehicle_avg_age_**
      - 357 records with 0 average age
      - would be good to **CHECK** why if it is legitimate or invalid missing data
  - **_Feature Selection_**
    - Columns to be excluded:
      - which will not be available for prediction - e.g. has_safety_program'
      - columns, which may determine target or are determined by target
      - list is included in 02_preprocessing.ipynb


2.  _Modeling methodology improvements._

- Review the prototype notebook and identify and implement at least three most impactful improvements to the modeling methodology to improve predictive performance on unseen data.
  1. within evaluate part "X_test" / "y_test" should be used >> y_pred = tmp.predict(X_test) instead of train data
  2. cleanup of the training (real) dataset
     - remove duplicates
     - fix inconsistencies in data - trim blanks / unify case handling (lower/upper/capitals)
     - correct datatypes
     - identify and then update/remove invalid records (e.g. negarive age)
     - handle outliers - decide if to keep or to remove
     - maybe merge some categories together, if there are not enough occurances
  3. Null Values Handling
     - to put 0 to all numeric columns may not make sense (e.g. driver_avg_age)
     - null handling depends on the column - how it is used
     - better maybe to use median (as mean may be influenced by outliers)
     - in some cases additional column with attribute "column_was_null" (True/False) can be beneficial
  4. Preprocessing should be done after train_test_split
     - to avoid data leakage from test(validation) data into traing data
  5. Create Pipeline
  5. Remove columns from features:
     - which may be unavailable at the time of prediction
     - which are determined by target column
     - e.g. "claim_paid_amount_current_period" & "days_to_first_claim_report"
  6. try various models to check, which performs better, e.g. XGBoost
  7. use additional metrics to check
     - precision / recall / F1-score / ROC-AUC / PR-AUC


3.  _Production readiness improvements._

- Identify and implement at least three most impactful improvements to increase the production readiness of the modeling solution.
  - separate functions into separate file(s)
  - put the code into .py file
  - separate credentials into .env file - avoid hardcoding
  - include requirements.txt file (for compatibility)
  - separate the flow:
    - file(s) for EAD
    - file(s) for preprocessing data
    - file(s) for training the model
    - file for model prediction
    - save model and all its parameters to file
  - implement schema validation


4.  _Production readiness roadmap._

- Outline further steps that would need to be completed before this model could be deployed to production. These do NOT need to be implemented; this section demonstrates your understanding of the production ML lifecycle.
  - remove unnecessary imports
  - code review
  - where necessary add tests
  - add some logging of errors / warnings
  - prepare documentation
  - commit to github
  - after deployment monitor the performance, if needed re-train
  

5. AI tools usage
  - Gemini / Google
    - I've used them most as teacher / accelerator / technical reviewer:
      - asking questions about domain specifics (e.g. how / when is some column used)
      - searching for information which model should be selected and which parameters to use
      - code review and code improvement suggestions