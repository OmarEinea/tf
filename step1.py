from tools import *
import tensorflow as tf
import pandas as pd

# Load data from file
data = pd.read_csv("./california_housing.csv")

# Define the label
my_label = data["house_value"]

# Define the input feature: total_rooms as rooms
my_feature = {"rooms": data["total_rooms"]}

# Configure a numeric feature column for rooms
feature_columns = [tf.feature_column.numeric_column("rooms")]

# Configure the linear regression model with our feature columns
linear_regressor = tf.estimator.LinearRegressor(feature_columns)


# Declare data input method
def get_examples():
    # Construct a TensorFlow dataset
    ds = tf.data.Dataset.from_tensor_slices((my_feature, my_label))
    return ds.batch(1)


# Train the model
linear_regressor.train(input_fn=get_examples, steps=len(my_label))

# Make predictions
predictions = linear_regressor.predict(input_fn=get_examples)

# Print statistics of predictions vs actual labels
print_stats(predictions, my_label)

# Plot trained model prediction line and a scatter plot of actual data
plot_results(linear_regressor, my_feature, my_label)
