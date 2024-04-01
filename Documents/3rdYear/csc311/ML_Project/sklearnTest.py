"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""
# numpy and pandas are also permitted
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# basic python imports are permitted
import sys
import csv
import re
import random

data = pd.read_csv("clean_dataset.csv")

# TODO: CLEAN DATA 
# SECTION CLEAN DATA
#
#
#
#

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float("nan").
    """

    if isinstance(s, str):
        s = s.replace(",", "")
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]

def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
    """

    n_list = get_number_list(s)
    n_list += [-1]*(6-len(n_list))
    return n_list

def get_number(s):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else -1


def find_area_at_rank(l, i):
    """Return the area at a certain rank in list `l`.

    Areas are indexed starting at 1 as ordered in the survey.

    If area is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

# Vectorization
#
#
#
#
#
#

def process_Q56(data):
    categories = ["Friends", "Co-worker", "Siblings", "Partner"]
    for category in categories:
        data[f"Q5_{category}"] = (data["Q5"] == category).astype(int)

    categories_q5 = ["Friends", "Co-worker", "Sibling", "Partner"]
    categories_q6 = ["Skyscrapers", "Sport", "Art and Music", "Carnival", "Cuisine", "Economic"]
    patterns = {category: f"{category}=>(\d+)" for category in categories_q6}

    for category in categories_q5:
        data[f"Q5_{category}"] = data["Q5"].str.contains(category, na=False).astype(int)

    del data["Q5"]
    
    for category, pattern in patterns.items():
        data[category] = data["Q6"].str.extract(pattern).astype(float).fillna(0)

    del data["Q6"]

def clean_up_input(data):
    data["Q7"] = data["Q7"].apply(to_numeric).fillna(0)
    data["Q8"] = data["Q8"].apply(to_numeric).fillna(0)
    data["Q9"] = data["Q9"].apply(to_numeric).fillna(0)

    # Clean for number categories
    data["Q1"] = data["Q1"].apply(get_number)
    data["Q2"] = data["Q2"].apply(get_number)
    data["Q3"] = data["Q3"].apply(get_number)
    data["Q4"] = data["Q4"].apply(get_number)

    return data

def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    data = pd.read_csv(open(filename))

    predictions = []
    for test_example in data:
        # obtain a prediction for this test example
        pred = predict(test_example)
        predictions.append(pred)

    return predictions

if __name__ == "__main__":
    #Vectorizeation
    process_Q56(data)
    data = clean_up_input(data)

    data_fets = np.stack([
        np.ones((data.shape[0])),
        (data["Q1"]),
        (data["Q2"]),
        (data["Q3"]),
        (data["Q4"]),
        data["Q5_Friends"],
        data["Q5_Co-worker"],
        data["Q5_Siblings"],
        data["Q5_Partner"],
        data["Skyscrapers"],
        data["Sport"],
        data["Art and Music"],
        data["Carnival"],
        data["Cuisine"],
        data["Economic"],
        #Regression here sort of, numbers are continous / boundless
        (data["Q7"]),
        (data["Q8"]),
        (data["Q9"]),
    ], axis=1)
    
    # Now explicitly check for any non-numeric types or NaN values in columns to be used

    if np.isnan(data_fets).any():
        print("NaN values detected in the dataset. Consider imputation.")

    numerical_value_start = 15

    #Creating target and features mapping
    X = data_fets

    label_mapping = {
        "Dubai": 0,
        "Rio de Janeiro": 1,
        "New York City": 2,
        "Paris": 3
    }
    print(data)

    t = data["Label"].map(label_mapping)

    X_train, X_valid, t_train, t_valid= train_test_split(X, t, test_size=700/1469, random_state=1)
    
    lr = LogisticRegression(multi_class='multinomial',solver ='newton-cg').fit(X_train,t_train)

    print(f"Training accuracy: {lr.score(X_train, t_train)}")
    print(f"Validation accuracy: {lr.score(X_valid, t_valid)}")