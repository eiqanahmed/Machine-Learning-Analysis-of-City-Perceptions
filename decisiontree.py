# numpy and pandas are also permitted
import numpy as np
import pandas as pd

# basic python imports are permitted
import sys
import csv
import re
import random

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# imports to visualize tree
from sklearn import tree as treeViz
import graphviz
import pydotplus
from IPython.display import display

data = pd.read_csv("clean_dataset.csv")

# TODO: CLEAN DATA
# SECTION CLEAN DATA

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

def visualize_tree(model, max_depth=10):
    dot_data = treeViz.export_graphviz(model,
                                       feature_names=feature_names,
                                       max_depth=max_depth,
                                       class_names=["Dubai", "New York City", "Paris", "Rio de Janeiro"],
                                       filled=True,
                                       rounded=True)
    return display(graphviz.Source(dot_data))

if __name__ == "__main__":
    #Deleted all the binary vectors of q1-q4 impossible to work with
    #Vectorizeation
    process_Q56(data)
    data = clean_up_input(data)
    print(list(data))
    data_fets = np.stack([
        data["Q1"] == -1,
        data["Q1"] == 1,
        data["Q1"] == 2,
        data["Q1"] == 3,
        data["Q1"] == 4,
        data["Q1"] == 5,
        data["Q2"] == -1,
        data["Q2"] == 1,
        data["Q2"] == 2,
        data["Q2"] == 3,
        data["Q2"] == 4,
        data["Q2"] == 5,
        data["Q3"] == -1,
        data["Q3"] == 1,
        data["Q3"] == 2,
        data["Q3"] == 3,
        data["Q3"] == 4,
        data["Q3"] == 5,
        data["Q4"] == -1,
        data["Q4"] == 1,
        data["Q4"] == 2,
        data["Q4"] == 3,
        data["Q4"] == 4,
        data["Q4"] == 5,
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
        data["Q7"],
        data["Q8"],
        data["Q9"],
        ], axis=1)

    feature_names = [
        "Q1_-1",
        "Q1_1",
        "Q1_2",
        "Q1_3",
        "Q1_4",
        "Q1_5",
        "Q2_-1",
        "Q2_1",
        "Q2_2",
        "Q2_3",
        "Q2_4",
        "Q2_5",
        "Q3_-1",
        "Q3_1",
        "Q3_2",
        "Q3_3",
        "Q3_4",
        "Q3_5",
        "Q4_-1",
        "Q4_1",
        "Q4_2",
        "Q4_3",
        "Q4_4",
        "Q4_5",
        "Q5_Friends",
        "Q5_Co",
        "Q5_Sib",
        "Q5_Part",
        "Skyscrapers",
        "Sport",
        "Art and Music",
        "Carnival",
        "Cuisine",
        "Economic",
        "Q7",
        "Q8",
        "Q9"]

    # Now explicitly check for any non-numeric types or NaN values in columns to be used
    if np.isnan(data_fets).any():
        print("NaN values detected in the dataset. Consider imputation.")

    # Split the data into X (dependent variables) and t (response variable)
    X = data_fets
    t = np.array(data["Label"])

    # Split 1200 into train rest into test set
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=268/1468, random_state=1)

    # Creating a DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=6)

    # TODO: fit it to our data
    tree.fit(X_train, t_train)
    print("Training Accuracy:", tree.score(X_train, t_train))
    print(tree.score(X_test, t_test))
    visualize_tree(tree)
