import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data():
    # read csv
    data = pd.read_csv('./SpotifyFeatures.csv')
    print(f"csv loaded, {data.shape[0]} elements in total, {data.shape[1]} features in total")
    # reduce dataset to genres pop and classical, and liveness/loudness
    data = data.filter(items=['genre', 'liveness', 'loudness'])[(data['genre'] == 'Pop') | (data['genre'] == 'Classical')]
    # create input matrix with liveness/loudness scores
    features = data[['liveness', 'loudness']].values
    # classifier for genres
    target_mapping = {'Classical': 0, 'Pop': 1}
    # create target array with genre data, also map genre to class
    target = np.array([target_mapping[genre] for genre in data['genre'].values])
    print(f"loaded {(target == 0).sum()} classical songs, {(target == 1).sum()} pop songs")
    # note: at this point songs are identified by their array index
    # use sklearn to split data into training/test pools
    return train_test_split(
        features, target, test_size = 0.2, random_state = 1)


def visualize_data(features, target, show_boundary = False, liveness_boundary = None, loudness_boundary = None):
    plt.figure(figsize=(8, 6))
    # x,0 plots liveness, x,1 plots loudness
    # 0,x = classical, 1,x = pop (based on classifier values)
    plt.scatter(features[target == 0, 0], features[target == 0, 1], color = 'red', label = 'Classical')
    plt.scatter(features[target == 1, 0], features[target == 1, 1], color = 'blue', label = 'Pop')
    # optionally plot the trained functions decision boundary
    # checking for None with numpy arrays is funky, using a boolean instead
    if show_boundary:
        print("visualizing data with decision boundary")
        plt.plot(liveness_boundary, loudness_boundary, color='green', label='Decision Boundary')
    else:
        print("visualizing data without decision boundary")
    plt.xlabel('Liveness')
    plt.ylabel('Loudness')
    plt.title('Training data visualization')
    plt.legend()
    plt.show()


def visualize_errors(epochs, errors):
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), errors)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss visualization')
    plt.legend()
    plt.show()


def sigmoid(x):
    # sigmoid function for logistic regression
    # np.exp calculates e^x, when a matrix is used
    # as input (aka an array) it calculates it for each element
    return 1 / (1 + np.exp(-x))
    # reformulated function according to lecture slides
    # return np.exp(x) / 1 + np.exp(x)


def logistic_regression_fit(features, target, epochs, learning_rate, weights = None, bias = 0):
    # gather size (# of elements) and feature count (features per element)
    # with numpys shape feature
    size, feature_count = features.shape
    # array of error rates, for us to plot later
    errors = []
    # initialize weights as all zeroes (if not passed as param)
    if weights == None:
        weights = np.zeros(feature_count)
    for epoch in range(epochs):
        # shuffle data each epoch to reduce chance of overfitting
        shuffle(features, target)
        # sum of all errors this epoch
        error_sum = 0
        for entry in range(size):
            # dot multiplication to add weights to inputs
            weighted_features = np.dot(features[entry], weights) + bias
            # apply sigmoid function to get the prediction
            prediction = sigmoid(weighted_features)
            # calculate error based on target
            error = prediction - target[entry]
            error_sum += abs(error)
            # adjust weight based on learning rate and how much error each feature contribues
            weights -= learning_rate * error * features[entry]
            bias -= learning_rate * error
        average_error = error_sum / size
        errors.append(average_error)
        # print stats every now and then
        if epoch % 50 == 0:
            print(f"epoch {epoch}: average error {average_error} | sum of errors {error_sum} | weights {weights} | bias {bias}")
    return weights, bias, errors


def predict(features, weights, bias = 0):
    # same as in training
    weighted_features = np.dot(features, weights) + bias
    # use sigmoid function to predict all elements, make them  all 1 or 0
    return [1 if prediction > 0.5 else 0 for prediction in sigmoid(weighted_features)]
        

# prepare datasets
print("loading data...")
training_features, test_features, training_target, test_target = load_data()
print(f"loaded {len(training_features)} training elements, {len(test_features)} test elements")
visualize_data(training_features, training_target)

# train model
print("training...")
weights, bias, errors = logistic_regression_fit(training_features, training_target, 1000, 0.005)
print(f"finished training, final weights: {weights}, bias: {bias}")
visualize_errors(1000, errors)

# test model
print("testing model...")
test_predictions = predict(test_features, weights, bias)
test_accuracy = accuracy_score(test_target, test_predictions)
print(f"model accuracy on test set: {test_accuracy * 100:.2f}%")

# compare test data with training data
training_predictions = predict(training_features, weights, bias)
training_accuracy = accuracy_score(training_target, training_predictions)
print(f"model accuracy on training set: {training_accuracy * 100:.2f}%")

# create a confusion matrix
confusion_matrix = confusion_matrix(test_target, test_predictions)
# [[TN: classical as classical, FP: classical as pop], [FN: pop as classical, TP: pop as pop]]
print(f"confusion matrix: {confusion_matrix}")

# visualize the decision boundary
liveness_min = training_features[:, 0].min()
liveness_max = training_features[:, 0].max()
liveness_plot = np.linspace(liveness_min, liveness_max)
# 0 = liveness, 1 = loudness - calculate loudness points from weights
loudness_plot = -(weights[0] / weights[1]) * liveness_plot - (bias / weights[1])
visualize_data(training_features, training_target, True, liveness_plot, loudness_plot)
