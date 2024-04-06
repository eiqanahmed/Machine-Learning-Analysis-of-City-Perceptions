import numpy as np
import pandas as pd
import sys
import csv
import re
import random


def dist_single(v, x):
    """
    Compute the squared Euclidean distance between vectors `v` and `x`.

    Parameters:
        `v` - a numpy array (vector) representing one student's answers, shape (17,)
        `x` - a numpy array (vector) representing another student's answers, shape (17,)

    Returns: a scalar value representing their squared Euclidean distance
    """

    diff = x - v # compute a difference vector  (x-v)

    sqdiff = diff**2

    sumval = np.sum(sqdiff)

    return sumval


def predict(v, X_train, t_train, k=1):
    """
    Returns a prediction using the k-NN

    Parameters:
        `v` - a numpy array (vector) representing a student's answers, shape (17,)
        `X_train` - a data matrix representing a set of answers from students, shape (N, 17)
        `t_train` - a vector of ground-truth labels, shape (N,)
        `k` - a positive integer 1 < k <= N, describing the number of closest set of answers
              to consider as part of the knn algorithm

    Returns:
        A prediction for the city the student was considering while answering the questions.
    """
    dists = dist_all(v, X_train)

    enum_list = list(enumerate(dists))
    sorted_enum_list = sorted(enum_list, key=lambda x: x[1])

    sorted_dists = [value for index, value in sorted_enum_list]
    sorted_indices = [index for index, value in sorted_enum_list]
    indices = []
    j = 0
    while j < k:
      indices.append(sorted_indices[j])
      j += 1

    ts = t_train[np.array(indices)]

    unique_elements, counts = np.unique(ts, return_counts=True)
    most_common_index = np.argmax(counts)
    most_common = unique_elements[most_common_index]

    prediction = most_common

    return prediction


def compute_accuracy(X_new, t_new, X_train, t_train, k=1):
    num_predictions = 0
    num_correct = 0

    for i in range(X_new.shape[0]): # iterate over each set of answers index in X_new
        v = X_new[i] # one set of answers from a student
        t = t_new[i] # prediction target

        prediction = predict(v, X_train, t_train, k=k)

        if prediction == t:
          num_correct += 1

        num_predictions += 1

    return num_correct / num_predictions


def normalize(X, continuous_indices):
    X_cpy = X.copy()

    # Normalize each continuous feature
    for index in continuous_indices:

        mean = np.mean(X_cpy[:, index])
        std = np.std(X_cpy[:, index])

        if std > 0:
            X_cpy[:, index] = (X_cpy[:, index] - mean) / std
        else:
            # If standard deviation is zero, leave the values as they are
            pass

    return X_cpy


def dist_all(v, X):
    """
    Compute the squared Euclidean distance between one data point `v` (vector) and the
    data points in the data matrix `X`.

    Parameters:
        `v` - a numpy array (vector) representing one students answers to the first 9 questions shape (17,)
        `X` - a data matrix representing a set of answers from all students
        who participated in the survey, shape (N, 17)

    Returns: a vector of squared Euclidean distances between `v` and each image in `X`,
             shape (N,)
    """

    diff = X - v

    sqdiff = diff**2

    sumval = np.sum(sqdiff, axis=1)

    return sumval


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


def predict_all(file_path):

    # get the file path create a function for it

    file_path = "/Users/eiqan/PycharmProjects/Data_Vectorization/clean_dataset.csv"

    data = pd.read_csv(file_path)

    process_Q56(data)
    data = clean_up_input(data)

    data_fets = np.stack([
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
        (data["Q7"]),
        (data["Q8"]),
        (data["Q9"]),
    ], axis=1)

    # Now explicitly check for any non-numeric types or NaN values in columns to be used
    if np.isnan(data_fets).any():
        print("NaN values detected in the dataset. Consider imputation.")

    # add the Label column to data_fets
    labels = data["Label"].values.reshape(-1, 1)
    # print(D[0][17])

    # D is the main data matrix to use, with labels
    D = np.concatenate((data_fets, labels), axis=1)
    # print(D)

    # split data set for training, testing, and validation:
    # Generate a sequence of indices and shuffle them
    indices = np.arange(D.shape[0])
    np.random.shuffle(indices)

    # D is now shuffled
    D = D[indices]

    # Indices to split after 968 for training and then 250 for testing (remaining 250 for validation)
    train_split_index = 968
    test_split_index = 968 + 250

    train_data = D[:train_split_index]
    test_data = D[train_split_index:test_split_index]
    validation_data = D[test_split_index:]

    X_train, t_train = train_data[:, :-1], train_data[:, -1]
    X_test, t_test = test_data[:, :-1], test_data[:, -1]
    X_valid, t_valid = validation_data[:, :-1], validation_data[:, -1]

    # print(t_test.shape)
    # print(X_test.shape)
    # print(D.shape) , should be (1468, 18)

    continuous_indices = [0, 1, 2, 3, 14, 15, 16]
    X_train_norm = normalize(X_train, continuous_indices)
    X_valid_norm = normalize(X_valid, continuous_indices)
    X_test_norm = normalize(X_test, continuous_indices)

    valid_acc_normalized_data = []

    for choosing_k in range(1, 38):
        acc = compute_accuracy(X_valid_norm, t_valid, X_train_norm, t_train, choosing_k)
        valid_acc_normalized_data.append((acc, choosing_k))

    highest_accuracy, k = max(valid_acc_normalized_data, key=lambda x: x[0])

    predictions = []
    for test_example in data_fets:
        pred = predict(test_example, X_train_norm, t_train, k)
        predictions.append(pred)

    print(compute_accuracy(X_test_norm, t_test, X_train_norm, t_train, k))

    return predictions











