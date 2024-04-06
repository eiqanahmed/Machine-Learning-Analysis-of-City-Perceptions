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

# basic python imports are permitted
import sys
import csv
import random

# numpy and pandas are also permitted
import numpy as np
import pandas as pd
import re


def ReLU(Z):
    for z in Z:
        if isinstance(z, np.ndarray):
            return ReLU(z)
        else:
            return np.maximum(0, z)


def deriv_ReLU(Z):
    for z in Z:
        if isinstance(z, np.ndarray):
            return deriv_ReLU(z)
        else:
            return z > 0


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def one_hot(Y):
    # Initialize an empty list to store one-hot encoded arrays
    encoded_arrays = []

    # Define the options
    all_options = ['Dubai', 'Rio de Janeiro', 'New York City', 'Paris']

    # Iterate over each row in self.X_train[4] and one-hot encode the options
    for x in Y:
        encoded_array = np.array(
            [int(option in x) for option in all_options])
        encoded_arrays.append(encoded_array)

    # Convert the list of arrays to a NumPy array
    encoded_matrix = np.array(encoded_arrays)

    # Replace the original column with the encoded matrix
    i = 0
    while i != Y.size:
        Y = encoded_matrix[i]
        i += 1

    return encoded_matrix

def preprocess(matrix):

    # helper function for function vector_q10

    return matrix.lower().replace('.', '').replace(',', '').replace('!',
                                                                    '').replace(
        '?', '')

def flatten(X):
    for x in X:
        return x.reshape(x.shape[0], -1)
def build_vocab(matrix):

    # helper function for function vector_q10

    vocab = []
    for i in matrix:
        if type(i) == float:
            i = '.'
        x = i.split()
        for k in x:
            if k not in vocab:
                vocab.append(k.lower())
    return vocab

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
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



class NeuralNetwork:

    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.data["Q7"] = self.data["Q7"].apply(to_numeric).fillna(0)
        self.data["Q9"] = self.data["Q9"].apply(to_numeric).fillna(0)
        self.data["Q3"] = self.data["Q3"].apply(to_numeric).fillna(0)
        self.data["Q1"] = self.data["Q1"].apply(get_number)
        self.data = np.array(self.data)
        self.m, self.n = self.data.shape
        np.random.shuffle(self.data)

        self.data_train = (self.data[0:round(self.m * 0.85)]).T
        self.Y_train = self.data_train[-1]
        self.X_train = self.data_train[1:self.n - 1]

        self.data_test = (self.data[round(self.m * 0.85):self.m]).T
        self.Y_test = self.data_test[-1]
        self.X_test = self.data_test[1:self.n - 1]

        self.W1 = np.random.rand(10, 10) - 0.5
        self.b1 = np.random.rand(10, 1) - 0.5
        self.W2 = np.random.rand(10, 4) - 0.5
        self.b2 = np.random.rand(1, 4) - 0.5

    def vector_q5(self):

        # Initialize an empty list to store one-hot encoded arrays
        encoded_arrays = []

        # Define the options
        all_options = ['Friends', 'Partner', 'Siblings', 'Co-worker']

        # Iterate over each row in self.X_train[4] and one-hot encode the options
        for x in self.X_train[4]:
            encoded_array = np.array(
                [int(option in x) for option in all_options])
            encoded_arrays.append(encoded_array)

        # Convert the list of arrays to a NumPy array
        encoded_matrix = np.array(encoded_arrays)

        # Replace the original column with the encoded matrix
        i = 0
        while i != self.X_train[4].size:
            self.X_train[4][i] = encoded_matrix[i]
            i += 1


    def vector_q6(self):

        # used to vectorize Q6 to make it usable for neural networks

        data_matrix = []
        for row in self.X_train[5]:
            if row == "Skyscrapers=>,Sport=>,Art and Music=>,Carnival=>,Cuisine=>,Economic=>":
                row = "Skyscrapers=>0,Sport=>0,Art and Music=>0,Carnival=>0,Cuisine=>0,Economic=>0"
            row_values = [int(entry.split('=>')[1]) for entry in row.split(',')]
            data_matrix.append(row_values)
        data_matrix = np.array(data_matrix)

        i = 0
        while i != self.X_train[5].size:
            self.X_train[5][i] = data_matrix[i]
            i += 1

    def vector_q10(self):

        # used to vectorize Q10 and make it usable for neural network

        vocabulary = build_vocab(self.X_train[9])
        word_to_index = {word: i for i, word in enumerate(vocabulary)}

        num_quotes = len(self.X_train[9])
        num_features = len(vocabulary)
        feature_matrix = np.zeros((num_quotes, num_features))

        tokenized_quotes = []
        for quote in self.X_train[9]:
            if type(quote) == float:
                quote = '.'
            preprocessed_quote = preprocess(quote)
            tokenized_quote = preprocessed_quote.split()
            tokenized_quotes.append(tokenized_quote)

        for i, quote in enumerate(tokenized_quotes):
            for word in quote:
                if word in word_to_index:
                    feature_matrix[i, word_to_index[word]] += 1

        i = 0
        while i != self.X_train[9].size:
            self.X_train[9][i] = feature_matrix[i]
            i += 1


    def forward_prop(self, X):

        Z1 = self.W1.dot(X) + self.b1
        A1 = ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = softmax(A1)

        return Z1, A1, Z2, A2

    def backward_prop(self, Z1, A1, Z2, A2, X, Y):
        m = Y.size
        encoded_y = one_hot(Y)

        dZ2 = A2 - encoded_y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)

        dZ1 = self.W2.dot(dZ2.T) * deriv_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)

        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, learning_rate):
        print(dW1.shape, self.W1.shape, db1.shape, self.b1.shape, dW2.shape, self.W2.shape, db2.shape, self.b2.shape)
        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1
        self.W2 = self.W2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2

    def get_predictions(self, A2):
        return np.argmax(A2, 0)

    def get_accuracy(self, predictions, Y):
        Y = one_hot(Y)
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, Y, iterations, learning_rate):
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop(X)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, X, Y)
            self.update_params(dW1, db1, dW2, db2, learning_rate)
            if (i % 10 == 0):
                print("Iterations ", i)
                print("Accuracy: ",
                      self.get_accuracy(self.get_predictions(A2), Y))

        return self.W1, self.b1, self.W2, self.b2


def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """

    # randomly choose between the four choices: 'Dubai', 'Rio de Janeiro', 'New York City' and 'Paris'.
    # NOTE: make sure to be *very* careful of the spelling/capitalization of the cities!!

    y = random.choice(['Dubai', 'Rio de Janeiro', 'New York City', 'Paris'])

    # return the prediction
    return y


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    data = csv.DictReader(open(filename))

    predictions = []
    for test_example in data:
        # obtain a prediction for this test example
        pred = predict(test_example)
        predictions.append(pred)

    return predictions
