import os
import sys

import ibm_boto3
from ibm_botocore.client import Config, ClientError

import tensorflow as tf

import pathlib
import h5py
import numpy as np
import tempfile


def log(e):
    print("{0}\n".format(e))

# start
log("start")

# Constants for IBM COS values
COS_API_KEY_ID = os.getenv('COS_API_KEY_ID')
COS_ENDPOINT = os.getenv('COS_ENDPOINT', 'https://s3.eu-gb.cloud-object-storage.appdomain.cloud')
COS_INSTANCE_CRN = os.getenv('COS_INSTANCE_CRN', 'crn:v1:bluemix:public:cloud-object-storage:global:a/b71ac2564ef0b98f1032d189795994dc:875e3790-53c1-40b0-9943-33b010521174::')
COS_AUTH_ENDPOINT = os.getenv('COS_AUTH_ENDPOINT', 'https://iam.cloud.ibm.com/identity/token')

# file vars
bucket_name = os.getenv('BUCKET_NAME', 'mnist-model')
model_file_name = os.getenv('MODEL_FILE_NAME', 'mnist-model.keras')

# create cloud object storage connection
cos_cli = ibm_boto3.client("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_INSTANCE_CRN,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)
try:
    response = cos_cli.list_buckets()
    log("Connection successful. List of buckets:")
    log(response['Buckets'])
except Exception as e:
    log("Connection failed")
    sys.exit(1)

def get_model_instance(bucket_name, item_name): 
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile() as f:
            # Get the pathlib.Path object
            path = pathlib.Path(f.name)

        # and file name
        fn = path.name
        log(path)
        log(fn)
        log(item_name)

        cos_cli.download_file(bucket_name, item_name, fn)   
        log("downloaded file from IBM COS") 
        with h5py.File(fn, 'r') as hdf_file:
            model = tf.keras.models.load_model(hdf_file)
    
        log("Loaded Model Instance from COS")
        return model
    except ClientError as be:
        log(be)
        sys.exit(1)
    except Exception as e:
        log("get_model_instance: Unable to get file: {0}".format(e))
        sys.exit(1)

class MNISTImageModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        log("Initializing - getting model instance")
        self.model = get_model_instance(bucket_name, model_file_name)
        log("Model instance is: " + self.model)

    def predict(self,data):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        log("Predict called - will run identity function")

        image_array = np.array(data)

        return self.model.predict(image_array) 
    
    def health_status(self):
        image_data = np.random.rand(28, 28).tolist() 
        response = self.predict(image_data)

        #assert len(response) == 2, "health check returning bad predictions" # or some other simple validation
        return response

#model = get_model_instance(bucket_name, 'test.keras')
#model = get_model_instance(bucket_name, model_file_name)

log("serving MNISTImageModel")
