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

#  SECTION
#  LINEAR REGRESSION FUNCTIONS
#
#
#
#
#

def softmax(z):
    """
    Compute the softmax of vector z, or row-wise for a matrix z.
    For numerical stability, subtract the maximum logit value from each
    row prior to exponentiation (see above).

    Parameters:
        `z` - a numpy array of shape (K,) or (N, K)

    Returns: a numpy array with the same shape as `z`, with the softmax
        activation applied to each row of `z`
    """
    # print("softmax", z.shape)
    m = np.max(z, axis=1, keepdims=True)
    exp_elements = np.exp(z - m)
    sum_exp_elements = np.sum(exp_elements, axis=1, keepdims=True)
    y_k = exp_elements / sum_exp_elements

    return y_k

def pred(X, w):
    # print(X.shape, w.shape)
    z = np.matmul(X, w)  
    return softmax(z) 

def make_onehot(indicies):
    I = np.eye(4)
    return I[indicies]

def grad(w, X, t):
    """
    Return the gradient of the cost function at `w`. The cost function
    is the average cross-entropy loss across the data set `X` and the
    target vector `t`.

    Parameters:
        `w` - a current "guess" of what our weights should be,
                   a numpy array of shape (D+1)
        `X` - matrix of shape (N,D+1) of input features
        `t` - target y values of shape (N)

    Returns: gradient vector of shape (D+1)
    """

    y = pred(X, w)
    t_copy = t.copy()
    t_copy = make_onehot(t_copy)

    term1 = (y - t_copy) # (50, 4) (50, ) convert t to one hot vector should fix TIP: remind group members not to do global changes to da data

    n = len(t)

    return np.matmul(np.transpose(X), term1)/n

def solve_via_sgd(X_train, t_train, alpha=0.0025, n_epochs=0, batch_size=50):
    """
    Given `alpha` - the learning rate
          `n_epochs` - the number of **epochs** of gradient descent to run
          `batch_size` - the size of ecach mini batch
          `X_train` - the data matrix to use for training
          `t_train` - the target vector to use for training
          `X_valid` - the data matrix to use for validation
          `t_valid` - the target vector to use for validation
          `w_init` - the initial `w` vector (if `None`, use a vector of all zeros)
          `plot` - whether to track statistics and plot the training curve

    Solves for logistic regression weights via stochastic gradient descent,
    using the provided batch_size.

    Return weights after `niter` iterations.
    """
    # as before, initialize all the weights to zeros
    w = np.zeros((X_train.shape[1], 4))

    # we will use these indices to help shuffle X_train
    N = X_train.shape[0] # number of training data points
    indices = list(range(N))

    for e in range(n_epochs):
        # Each epoch will iterate over the training data set exactly once.
        # At the beginning of each epoch, we need to shuffle the order of
        # data points in X_train. Since we do not want to modify the input
        # argument `X_train`, we will instead randomly shuffle the `indices`,
        # and we will use `indices` to iterate over the training data
        random.shuffle(indices)

        for i in range(0, N, batch_size):
            if (i + batch_size) >= N:
                continue

            # TODO: complete the below code to compute the gradient
            # only across the minibatch:
            indices_in_batch = indices[i: i+batch_size]
            X_minibatch = np.take(X_train, indices_in_batch, 0) # TODO: subset of "X_train" containing only the
                                                                #       rows in indices_in_batch
            #print(X_minibatch.shape)
            t_minibatch = np.take(t_train, indices_in_batch, 0) # TODO: corresponding targets to X_minibatch

            dw = grad(w, X_minibatch, t_minibatch) # TODO: gradient of the avg loss over the minibatch
            w = w - alpha * dw

    return w

def accuracy(w, X, t):
    y_pred = pred(X, w)  
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracy = np.mean(y_pred_classes == t)
    return accuracy

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
    
    X_train_norm = X_train.copy()
    X_valid_norm = X_valid.copy()
    
    w = solve_via_sgd(alpha=0.05, X_train=X_train_norm, t_train=t_train, n_epochs=40, batch_size=50)
    y_pred = pred(X_valid_norm, w)

    validation_accuracy = accuracy(w, X_valid_norm, t_valid)

    print(f"Validation Accuracy: {validation_accuracy:.2f}")
