from matplotlib import cm, pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import math
pd.options.display.float_format = '{:.1f}'.format
RSMEs = []
XYs = []


def print_stats(predictions, label, detailed=True):
    # Format predictions as a NumPy array, so we can calculate error metrics.
    predictions = np.array([item['predictions'][0] for item in predictions])

    # Print Mean Squared Error and Root Mean Squared Error.
    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(predictions, label)
    )

    min_value = label.min()
    max_value = label.max()
    min_max_difference = max_value - min_value

    if detailed:
        print("\nMin. Label Value: {:.3f}".format(min_value))
        print("Max. Label Value: {:.3f}".format(max_value))
        print("Difference between Min. and Max.: {:.3f}".format(min_max_difference))

    print("Root Mean Squared Error: {:.3f}".format(root_mean_squared_error))
    RSMEs.append(root_mean_squared_error)

    if detailed:
        calibration_data = pd.DataFrame()
        calibration_data["predictions"] = pd.Series(predictions)
        calibration_data["label"] = pd.Series(label)
        print("\n", calibration_data.describe())


def plot_results(regressor, feature, label):
    feature_name, feature_values = feature.copy().popitem()

    # Retrieve the final weight and bias generated during training.
    weight = regressor.get_variable_value('linear/linear_model/{}/weights'.format(feature_name))[0]
    bias = regressor.get_variable_value('linear/linear_model/bias_weights')

    y_extents = np.array([0, label.max()])
    x_extents = np.maximum(
        np.minimum(
            (y_extents - bias) / weight,
            feature_values.max()
        ), feature_values.min()
    )
    y_extents = weight * x_extents + bias

    plt.plot(x_extents, y_extents, c='r')

    # Label the graph axes.
    plt.ylabel("Label")
    plt.xlabel(feature_name)

    # Plot a scatter plot from our data sample.
    plt.scatter(feature_values[:300], label[:300])

    # Display graph.
    plt.show()


def plot_data(feature, label, periods):
    feature_name, feature_values = feature.copy().popitem()
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel("Label")
    plt.xlabel(feature_name)
    plt.scatter(feature_values[:300], label[:300])
    RSMEs.clear()
    XYs.clear()


def plot_model(regressor, feature, label):
    feature_name, feature_values = feature.copy().popitem()

    # Retrieve the final weight and bias generated during training.
    weight = regressor.get_variable_value('linear/linear_model/{}/weights'.format(feature_name))[0]
    bias = regressor.get_variable_value('linear/linear_model/bias_weights')

    y_extents = np.array([0, label.max()])
    x_extents = np.maximum(
        np.minimum(
            (y_extents - bias) / weight,
            feature_values.max()
        ), feature_values.min()
    )
    y_extents = weight * x_extents + bias
    XYs.append((x_extents, y_extents))


def show_all():
    results = sorted(zip(RSMEs, XYs), reverse=True)
    print("Minimum Root Mean Squared Error: {:.3f}".format(results[-1][0]))
    for index, result in enumerate(results):
        xy = result[1]
        plt.plot(xy[0], xy[1], color=cm.coolwarm(index/len(XYs)))
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(RSMEs)
    plt.show()
