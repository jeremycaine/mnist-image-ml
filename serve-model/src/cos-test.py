import os

import ibm_boto3
from ibm_botocore.client import Config, ClientError

import tensorflow as tf
import keras
import pandas as pd
import numpy as np

import pathlib
import tempfile
import h5py

# https://cloud.ibm.com/docs/cloud-object-storage/libraries?topic=cloud-object-storage-python#python-examples-list-objects

# Constants for IBM COS values
#COS_ENDPOINT = "<endpoint>" # Current list avaiable at https://control.cloud-object-storage.cloud.ibm.com/v2/endpoints
#COS_API_KEY_ID = "<api-key>" # eg "W00YixxxxxxxxxxMB-odB-2ySfTrFBIQQWanc--P3byk"
#COS_INSTANCE_CRN = "<service-instance-id>" # eg "crn:v1:bluemix:public:cloud-object-storage:global:a/3bf0d9003xxxxxxxxxx1c3e97696b71c:d6f04d83-6c4f-4a62-a165-696756d63903::"

COS_API_KEY_ID = os.getenv('COS_API_KEY_ID')
COS_ENDPOINT = os.getenv('COS_ENDPOINT', 'https://s3.eu-gb.cloud-object-storage.appdomain.cloud')
COS_AUTH_ENDPOINT = os.getenv('COS_AUTH_ENDPOINT', 'https://iam.cloud.ibm.com/identity/token')
#COS_SERVICE_CRN = os.getenv('COS_SERVICE_CRN', 'crn:v1:bluemix:public:cloud-object-storage:global:a/b71ac2564ef0b98f1032d189795994dc:875e3790-53c1-40b0-9943-33b010521174::')
COS_INSTANCE_CRN = os.getenv('COS_INSTANCE_CRN', 'crn:v1:bluemix:public:cloud-object-storage:global:a/b71ac2564ef0b98f1032d189795994dc:875e3790-53c1-40b0-9943-33b010521174::')
COS_STORAGE_CLASS = os.getenv('COS_STORAGE_CLASS','eu-gb-smart')

# file vars
bucket_name = os.getenv('BUCKET_NAME', 'mnist-model')
#model_name = os.getenv('MODEL_NAME', 'mnist-model')
model_file_name = os.getenv('MODEL_FILE_NAME', 'mnist-model.keras')

# Create resource
cos = ibm_boto3.resource("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_INSTANCE_CRN,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

def get_buckets():
    print("Retrieving list of buckets")
    try:
        buckets = cos.buckets.all()
        for bucket in buckets:
            print("Bucket Name: {0}".format(bucket.name))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to retrieve list buckets: {0}".format(e))

def get_bucket_contents(bucket_name):
    print("Retrieving bucket contents from: {0}".format(bucket_name))
    try:
        files = cos.Bucket(bucket_name).objects.all()
        for file in files:
            print("Item: {0} ({1} bytes).".format(file.key, file.size))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to retrieve bucket contents: {0}".format(e))

def get_item(bucket_name, item_name):
    print("Retrieving item from bucket: {0}, key: {1}".format(bucket_name, item_name))
    try:
        file = cos.Object(bucket_name, item_name).get()
        print("File Contents length is not zero: {0}".format(file["ContentLength"]))
        return file
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to retrieve file contents: {0}".format(e))

def create_file(bucket_name, item_name, file_text):
    print("Creating new item: {0}".format(item_name))
    try:
        cos.Object(bucket_name, item_name).put(
            Body=file_text
        )
        print("Item: {0} created!".format(item_name))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to create text file: {0}".format(e))


def get_model(bucket_name, item_name):
    print("Get a model: {0}".format(item_name))
    try:
        # Get the object
        cos_object = cos.Object(bucket_name, item_name)

        # Download the object to a file
        with open(item_name, 'wb') as f:
            cos_object.download_fileobj(f)
            path = pathlib.Path(f.name)

        with h5py.File(path.name, 'r') as hdf_file:
            model = keras.models.load_model(hdf_file)
   
        os.remove(f.name)
        print("Loaded Model from COS")
        return model
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to return mode: {0}".format(e))

def save_model(model_inst, model_file_name):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
        model_inst.save(f.name)
        path_object = pathlib.Path(f.name)
        with open(f.name, "rb") as file:
            file_contents = file.read()
            create_file(bucket_name, model_file_name, file_contents)
    os.remove(f.name)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(3,)),
    tf.keras.layers.Softmax()])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

save_model(model, 'test.keras')

get_bucket_contents(bucket_name)

loaded_model = get_model(bucket_name, 'test.keras')
print(loaded_model)
