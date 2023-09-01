 #!/usr/bin/env python3

import flask
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint

import os
import io
import pathlib 
import json

import PIL
import sys
import PIL.ImageOps
import numpy as np

#import tensorflow as tf
from tensorflow import keras
import h5py
import tempfile

#import pandas as pd
#import matplotlib.pyplot as plt

import ibm_boto3
from ibm_botocore.client import Config
from ibm_botocore.exceptions import ClientError

app = flask.Flask(__name__)
CORS(app)

# credentials
COS_API_KEY_ID = os.getenv('COS_API_KEY_ID')

COS_ENDPOINT = os.getenv('COS_ENDPOINT', 'https://s3.eu-gb.cloud-object-storage.appdomain.cloud')
COS_AUTH_ENDPOINT = os.getenv('COS_AUTH_ENDPOINT', 'https://iam.cloud.ibm.com/identity/token')
COS_SERVICE_CRN = os.getenv('COS_SERVICE_CRN', 'crn:v1:bluemix:public:cloud-object-storage:global:a/b71ac2564ef0b98f1032d189795994dc:875e3790-53c1-40b0-9943-33b010521174::')
COS_STORAGE_CLASS = os.getenv('COS_STORAGE_CLASS','eu-gb-smart')

# file vars
bucket_name = os.getenv('BUCKET_NAME', 'mnist-model')
h5_file_name = os.getenv('H5_FILE_NAME', 'mnist-model.h5')
train_csv = os.getenv('TRAIN_CSV', 'mnist_train.csv')

def log(e):
    print("{0}\n".format(e))

import requests

def predict(data):
    """Makes a REST call to the model server and returns the predictions."""

    url = "https://serve-model.16qg6j0mog3v.us-south.codeengine.appdomain.cloud/predict"
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"data": data.tolist()})
    print(type(data))
    
    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        raise ValueError("Error calling model server: {}".format(response.status_code))


@app.route('/', methods=['GET'])
def index():
    """
    This is the home page to serve hand draw canvas.
    ---
    responses:
      200:
        description: Home page canvas
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
    """
    return flask.render_template("mnist.html")

@app.route('/about', methods=['GET'])
def about_content():
    return {"message": "Hello, this is About content!"}

@app.route('/image', methods=['POST'])
def image():
    # Note, even though we do a little prep, we don't clean the image nearly
    # as well as the MNIST dataset expects, so there will be some issues with
    # generalizing it to handwritten digits

    # Start by taking the image into pillow so we can modify it to fit the
    # right size
    pilimage: PIL.Image.Image = PIL.Image.frombytes(
        mode="RGBA", size=(200, 200), data=flask.request.data)

    # Resize it to the right internal size
    pilimage = pilimage.resize((20, 20))

    # Need to replace the Alpha channel, since 0=black in PIL
    newimg = PIL.Image.new(mode="RGBA", size=(20, 20), color="WHITE")
    newimg.paste(im=pilimage, box=(0, 0), mask=pilimage)

    # Turn it from RGB down to grayscale
    grayscaled_image = PIL.ImageOps.grayscale(newimg)

    # Add the padding so we have it at 28x28, with the 4px padding on all sides
    padded_image = PIL.ImageOps.expand(
        image=grayscaled_image, border=4, fill=255)

    # Call Invert here, since Pillow assumes 0=black 255=white, and our neural
    # net assumes the opposite
    inverted_image = PIL.ImageOps.invert(padded_image)

    # Finally, convert our image to the (28, 28, 1) format expected by the
    # model. Tensorflow expects an array of inputs, so we end up with
    reshaped_image = np.array(
        list(inverted_image.tobytes())).reshape((1, 28, 28, 1))

    scaled_image_array = reshaped_image / 255.0

    # now call the model to predict what the digit the image is
    out = predict(scaled_image_array)
    log("out : " + out)
    log("Predicted Image is : " + str(np.argmax(out,axis=1)))
    
    response = np.array_str(np.argmax(out,axis=1))
    return response	

# Create Swagger UI blueprint
SWAGGER_URL = '/swagger'
API_URL = 'swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "MNIST Digit Image App"
    }
)

# Register the blueprint
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Get the PORT from environment
port = os.getenv('PORT', '8080')
debug = os.getenv('DEBUG', 'false')
if __name__ == "__main__":
    log("application ready - Debug is " + str(debug))
    app.run(host='0.0.0.0', port=int(port), debug=debug)
