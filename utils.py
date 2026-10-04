import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns


#######################
##### Duplicities #####
#######################

def check_duplicates_stats(
        dataset: pd.DataFrame, 
        columns_list: list[Union[str, list[str]]]
    ) -> None:
    
    """
    Check counts of duplicated records per all and selected columns.
    """
    
    # Check all rows
    has_duplicates = dataset.duplicated().any()
    print(f'Data contains duplicates: {has_duplicates}')

    num_duplicates = dataset.duplicated().sum()
    print(f"Total duplicate rows: {num_duplicates}", "\n")

    # Check single column or subset of columns
    has_subset_dups = {}

    for idx, check_col in enumerate(columns_list):
        has_subset_dups[idx] = dataset.duplicated(subset=check_col).any()

    for i in range(len(has_subset_dups)):
        print(f'{columns_list[i]}: {has_subset_dups[i]}, index: {i}')

    print('\n')


def check_duplicates_per_columns(
        dataset: pd.DataFrame, 
        columns_list: list[Union[str, list[str]]]
    ) -> list[str]:

    """ 
    Returns list of columns containing duplicates.
    """

    # Check number of duplicates for specified columns
    total_duplicate_rows = dataset.duplicated(subset=columns_list, keep=False).sum()
    print(f'Total duplicated rows for {columns_list}: {total_duplicate_rows}')

    if total_duplicate_rows: return columns_list


#######################
#####  uniquenes  #####
#######################

def check_unique_categorical_cols(
        dataset: pd.DataFrame,
        categorical_cols: list[str]
) -> None:

    """
    Checks unique values in string columns.
    """
    
    for col in categorical_cols:
        print(f'ANALYSIS FOR: ---- {col} ----', '\n')
        print(dataset[col].unique(), '\n')
        print(dataset[col].value_counts(), '\n')


#######################
#####   Graphs    #####
#######################

def boxplot_and_histogram(
        df: pd.DataFrame, 
        col: str,
    ) -> None:

    """
    Plot Boxplot and Histogram for provided feature column
    """
    
    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, 
                                    gridspec_kw={'height_ratios': [0.25, 0.75]})

    # 1. Boxplot
    sns.boxplot(x=df[col], ax=ax_box, color='salmon')
    ax_box.set(xlabel='')

    # 2. Histogram
    sns.histplot(df[col], kde=True, bins=60, ax=ax_hist, color='steelblue')
    ax_hist.set_xlabel(f'{col}')
    ax_hist.set_ylabel('Count')

    plt.tight_layout()
    plt.show()

def countplot(
        df: pd.DataFrame, 
        col: str,
    ) -> None:

    """
    Plot Countplot for provided feature column
    """

    plt.figure(figsize=(7, 3.5))
        
    order = df[col].value_counts().index

    sns.countplot(
        data=df, 
        x=col, 
        order=order, 
        palette='crest',
        hue=col,
        legend=False
    )

    plt.title(f"Distribution of {col}", fontsize=12, pad=10)
    plt.xlabel(col)
    plt.ylabel("Count")

    # Rotate tick labels if category names overlap
    plt.xticks(rotation=45, ha='right')
        
    plt.tight_layout()
    plt.show()


##############################
#####   Preprocessing    #####
##############################

def impute_prior_loss(df, median_loss=0.0):
    X = df.copy()

    # No claims -> loss is 0
    no_claims = (X["prior_apd_claim_count"] == 0) & (
        X["prior_al_claim_count"] == 0
    )
    X.loc[no_claims & X["prior_loss_amount"].isnull(), "prior_loss_amount"] = (
        0.0
    )

    # Remaining nulls get the training median
    X["prior_loss_amount"] = X["prior_loss_amount"].fillna(median_loss)
    return X