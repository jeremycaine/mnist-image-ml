import os

import ibm_boto3
from ibm_botocore.client import Config, ClientError

import tensorflow as tf

import pathlib
import tempfile

def log(e):
    print("{0}\n".format(e))

# start
log("start")

# Constants for IBM COS values
COS_API_KEY_ID = os.getenv('COS_API_KEY_ID')
COS_ENDPOINT = os.getenv('COS_ENDPOINT', 'https://s3.eu-gb.cloud-object-storage.appdomain.cloud')
COS_INSTANCE_CRN = os.getenv('COS_INSTANCE_CRN', 'crn:v1:bluemix:public:cloud-object-storage:global:a/b71ac2564ef0b98f1032d189795994dc:875e3790-53c1-40b0-9943-33b010521174::')

# file vars
bucket_name = os.getenv('BUCKET_NAME', 'mnist-model')
model_name = os.getenv('MODEL_NAME', 'mnist-model')
model_file_name = os.getenv('MODEL_FILE_NAME', 'mnist-model.keras')

# Create resource
cos = ibm_boto3.resource("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_INSTANCE_CRN,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

def get_item(bucket_name, item_name):
    print("Retrieving item from bucket: {0}, key: {1}".format(bucket_name, item_name))
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile() as f:
            # Get the pathlib.Path object
            path = pathlib.Path(f.name)
        # and file name
        fn = path.name
        print ("path.name = ", path.name)
        cos.Object(bucket_name, item_name).download_file(fn)
        print ("in get_item fn = ", fn)
        return fn
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to retrieve file contents: {0}".format(e))


class MNISTImageModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        log("Initializing")

        fn = get_item(bucket_name, model_file_name)
        print("fn = ", fn)
        self.model = tf.keras.models.load_model(fn)

    def predict(self,data):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        log("Predict called - will run identity function")
        return self.model.predict(data) 

log("finished MNISTImageModel")
