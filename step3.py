from tools import *
import tensorflow as tf
import pandas as pd

# Load data from file
data = pd.read_csv("./california_housing.csv")

# Define the label.
my_label = data["house_value"]

# Define the input feature: total_rooms as rooms
my_feature = {"rooms": data["total_rooms"]}

# Configure a numeric feature column for rooms.
feature_columns = [tf.feature_column.numeric_column("rooms")]

# Declare model training configs
batch_size = 1
num_steps = 1000
learning_rate = 0.000025

# STEP 3: Declare period variables
periods = 10
steps_per_period = num_steps / periods

# Use gradient descent as the optimizer for training the model
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)


# Declare data input method
def get_examples():
    # Construct a TensorFlow dataset
    ds = tf.data.Dataset.from_tensor_slices((my_feature, my_label))
    # Configure batching, repeating as shuffling
    ds = ds.batch(batch_size)
    ds = ds.shuffle(len(my_label))
    return ds


# STEP 3: Draw scatter plot of data once
plot_data(my_feature, my_label, periods)

# STEP 3: Do training on periods
for period in range(periods):
    # Train the model
    linear_regressor.train(input_fn=get_examples, steps=steps_per_period)

    # Make predictions
    predictions = linear_regressor.predict(input_fn=get_examples)

    # Print statistics of predictions vs actual labels
    print_stats(predictions, my_label, detailed=False)

    # Plot trained model prediction line and a scatter plot of actual data
    plot_model_by_period(linear_regressor, my_feature, my_label, period)

show_rmses()
