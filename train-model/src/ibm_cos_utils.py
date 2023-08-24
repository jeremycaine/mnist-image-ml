import os
import sys

import ibm_boto3
from ibm_botocore.client import Config
from ibm_botocore.exceptions import ClientError

import tempfile
import pathlib

# credentials
COS_API_KEY_ID = os.getenv('COS_API_KEY_ID')
COS_ENDPOINT = os.getenv('COS_ENDPOINT', 'https://s3.eu-gb.cloud-object-storage.appdomain.cloud')
COS_AUTH_ENDPOINT = os.getenv('COS_AUTH_ENDPOINT', 'https://iam.cloud.ibm.com/identity/token')
COS_SERVICE_CRN = os.getenv('COS_SERVICE_CRN', 'crn:v1:bluemix:public:cloud-object-storage:global:a/b71ac2564ef0b98f1032d189795994dc:875e3790-53c1-40b0-9943-33b010521174::')
COS_STORAGE_CLASS = os.getenv('COS_STORAGE_CLASS','eu-gb-smart')

# file vars
bucket_name = os.getenv('BUCKET_NAME', 'mnist-model')
tf_file_name = os.getenv('MODEL_FILE_NAME', 'mnist-model')
train_csv = os.getenv('TRAIN_CSV', 'mnist_train.csv')
test_csv = os.getenv('TEST_CSV', 'mnist_test.csv')

def log(e):
    print("{0}\n".format(e))

def get_file(file_name): 
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile() as f:
            # Get the pathlib.Path object
            path = pathlib.Path(f.name)

        # and file name
        fn = path.name

        cos_cli.download_file(bucket_name, file_name, fn) 
        return fn
    except ClientError as be:
        log(be)
        sys.exit(1)
    except Exception as e:
        log("Unable to get file: {0}".format(e))
        sys.exit(1)

# create cloud object storage connection
cos_cli = ibm_boto3.client("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_SERVICE_CRN,
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


def save_model(path_object, file_name): 
    try:
        cos_cli.upload_file(
            Filename=path_object, 
            Bucket=bucket_name, 
            Key=file_name)
    
        return 0
    except ClientError as be:
        log(be)
        sys.exit(1)
    except Exception as e:
        log("Unable to put file: {0}".format(e))
        sys.exit(1)

def save_file(path_object, file_name): 
    try:
        with open(path_object, 'rb') as file:
            # Use the file-like object here
            # For example, you can read its contents
            contents = file.read()            
            cos_cli.put_object(
                Bucket=bucket_name,
                Key=file_name,
                Body=contents) 

        return 0
    except ClientError as be:
        log(be)
        sys.exit(1)
    except Exception as e:
        log("Unable to put file: {0}".format(e))
        sys.exit(1)