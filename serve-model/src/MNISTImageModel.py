import os

import ibm_boto3
from ibm_botocore.client import Config, ClientError

import tensorflow as tf

import pathlib
import h5py

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
model_file_name = os.getenv('MODEL_FILE_NAME', 'mnist-model.keras')

# Create resource
cos = ibm_boto3.resource("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_INSTANCE_CRN,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

def get_model(bucket_name, item_name):
    print("Get a model: {0}".format(item_name))
    try:
        # Get the object
        cos_object = cos.Object(bucket_name, item_name)
        print("cos_object ", cos_object)

        # Download the object to a file
        with open(item_name, 'wb') as f:
            cos_object.download_fileobj(f)
            path = pathlib.Path(f.name)

        print("path name", path.name)
        with h5py.File(path.name, 'r') as hdf_file:
            model = tf.keras.models.load_model(hdf_file)
   
        os.remove(f.name)
        print("Loaded Model from COS")
        return model
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to return model: {0}".format(e))


class MNISTImageModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        log("Initializing")
        self.model = get_model(bucket_name, model_file_name)

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

#model = get_model(bucket_name, 'test.keras')
#model = get_model(bucket_name, model_file_name)
#log(model)
log("finished MNISTImageModel")
