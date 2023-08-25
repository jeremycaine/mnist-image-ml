import os
import tempfile

import tensorflow as tf
import pandas as pd
import numpy as np

import ibm_cos_utils as cos
import pathlib

def log(e):
    print("{0}\n".format(e))

# start
log("start")

# file vars
bucket_name = os.getenv('BUCKET_NAME', 'mnist-model')
model_file_name = os.getenv('MODEL_FILE_NAME', 'mnist-model.keras')
train_csv = os.getenv('TRAIN_CSV', 'mnist_train.csv')
test_csv = os.getenv('TEST_CSV', 'mnist_test.csv')

# -- Get the dataset
test_df: pd.DataFrame = pd.read_csv(cos.get_file(test_csv), header=None)
test_features: np.ndarray = test_df.loc[:, 1:].values
test_features = test_features.reshape((test_features.shape[0], 28, 28, 1))
test_features = test_features / 255.0
test_labels: np.ndarray = test_df[0].values
log("created test labels")

# -- Build the model
train_df: pd.DataFrame = pd.read_csv(cos.get_file(train_csv), header=None)
log("loaded train csv")
test_df: pd.DataFrame = pd.read_csv(cos.get_file(test_csv), header=None)
log("loaded test csv")

# split data set, taking off the labels e.g. '7' (first element), from the others 
train_features: np.ndarray = train_df.loc[:, 1:].values
log("split data set taking off the labels")

# shape the dataset into 60,000 28x28x1
# 60k data items, 28x28 pixels, could be 1 to 3 element, but we only need 1 element (grayscale)
train_features = train_features.reshape((train_features.shape[0], 28, 28, 1))
log("shape the dataset")

# each data item has a value between 0 and 255, so to normalise and make value betwee 0 and 1
train_features = train_features / 255.0

# get the labels
train_labels: np.ndarray = train_df[0].values
log("get the labels")

# manipulate and normailse test data set and get the labels
test_features: np.ndarray = test_df.loc[:, 1:].values
test_features = test_features.reshape((test_features.shape[0], 28, 28, 1))
test_features = test_features / 255.0
test_labels: np.ndarray = test_df[0].values
log("manipulate and normalise test data set")

# create the model
log("create the model")
# Sequential - outputs are sent to input of next layer - aka forward progression
model: tf.keras.models.Sequential = tf.keras.models.Sequential()

# add layer performing Convolution over a 2d grid
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1)))

# add layer pooling the max value in the 2x2 grid
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

# add layer to flatten 2d grid into a single array
model.add(tf.keras.layers.Flatten())

# add a dense layer which classifies the data using an activation function
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))

# add dropout layer that randomly takes out a % of the layers so training does not become too specilaised
# aka regularisation
model.add(tf.keras.layers.Dropout(rate=0.2))

# add layer that outputs a max value
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# complie the model
log("compile the model")
# using a given optimiser and loss function
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# output summary of layers
log(model.summary())

# -- Train the model
model.fit(train_features, train_labels, epochs=3, verbose=0)
log("model training complete")

# test the model
# check the accuracy
model.evaluate(test_features, test_labels)
log("model test complete")


MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version), model_file_name)
print('export_path = {}\n'.format(export_path))

# save to temporary dir
model.save(export_path)
# save to COS
cos.save_model(export_path, model_file_name)
log("saved to IBM Cloud Object Storage")

log("finish")



