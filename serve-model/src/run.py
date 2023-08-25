import os
import sys
import tensorflow as tf
import ibm_cos_utils as cos
import tempfile
import pathlib
import ibm_boto3
from ibm_botocore.client import Config
from ibm_botocore.exceptions import ClientError
import h5py


# credentials
COS_API_KEY_ID = os.getenv('COS_API_KEY_ID')

COS_ENDPOINT = os.getenv('COS_ENDPOINT', 'https://s3.eu-gb.cloud-object-storage.appdomain.cloud')
COS_AUTH_ENDPOINT = os.getenv('COS_AUTH_ENDPOINT', 'https://iam.cloud.ibm.com/identity/token')
COS_SERVICE_CRN = os.getenv('COS_SERVICE_CRN', 'crn:v1:bluemix:public:cloud-object-storage:global:a/b71ac2564ef0b98f1032d189795994dc:875e3790-53c1-40b0-9943-33b010521174::')
COS_STORAGE_CLASS = os.getenv('COS_STORAGE_CLASS','eu-gb-smart')

# file vars
bucket_name = os.getenv('BUCKET_NAME', 'mnist-model')
keras_file_name = os.getenv('H5_FILE_NAME', 'mnist-model.keras')

# create cloud objec storage connection
cos_cli = ibm_boto3.client("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_SERVICE_CRN,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

def log(e):
    print("{0}\n".format(e))

# Create a temporary file
with tempfile.NamedTemporaryFile() as f:
    # Get the pathlib.Path object
    path = pathlib.Path(f.name)

# and file name
fn = path.name

cos_cli.download_file(bucket_name, keras_file_name, fn)  
model = tf.keras.models.load_model(fn)

#with h5py.File(fn, 'r') as hdf_file:
#    model = tf.keras.models.load_model(hdf_file)

log(model)

