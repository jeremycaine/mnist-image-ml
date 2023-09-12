# mnist-image-ml
MNIST Image machine learning model project


# IBM Cloud Engine
Region - WDC
Project - mnist-model
COS - caine-cos
Logging - 

## Create a project
```
ibmcloud login --sso ...

ibmcloud target -c b71ac2564ef0b98f1032d189795994dc
ibmcloud target -r us-south
ibmcloud target -g ceh-group

ibmcloud ce project create --name mnist-model
```

## Set-up the project
```
ibmcloud ce project list
ibmcloud ce project select --name mnist-model

# need secret
ibmcloud ce secret create --name caine-cos-api-key --from-literal COS_API_KEY_ID=xxx

# config map for variables
ibmcloud ce configmap create --name mnist-model-cm \
    --from-literal COS_ENDPOINT=https://s3.eu-gb.cloud-object-storage.appdomain.cloud  \
    --from-literal COS_AUTH_ENDPOINT=https://iam.cloud.ibm.com/identity/token  \
    --from-literal COS_SERVICE_CRN=crn:v1:bluemix:public:cloud-object-storage:global:a/b71ac2564ef0b98f1032d189795994dc:875e3790-53c1-40b0-9943-33b010521174::  \
    --from-literal COS_STORAGE_CLASS=eu-gb-smart  \
    --from-literal TF_FILE_NAME=mnist-model  \
    --from-literal TRAIN_CSV=mnist_train.csv  \
    --from-literal TEST_CSV=mnist_test.csv
```

# Model Development

## Train Model
Create and train a MNIST image machine learning model in Tensorflow format and store.

Job to train image prediction model to label its image
```
# create app first time
ibmcloud ce job create --name train-model --src https://github.com/jeremycaine/mnist-image-ml --bcdr train-model --str dockerfile --env-from-secret caine-cos-api-key --env-from-configmap mnist-model-cm --cpu 2 --memory 16G

# rebuild after git commit
ibmcloud ce job update --name train-model --rebuild
```

## Serve Model
App to serve prediction model 
```
# create app first time
ibmcloud ce app create --name serve-model --src https://github.com/jeremycaine/mnist-image-ml --bcdr serve-model --str dockerfile --env-from-secret caine-cos-api-key --env-from-configmap mnist-model-cm --cpu 2 --memory 16G --port 9000

# rebuild after git commit
ibmcloud ce app update --name serve-model --rebuild
```

### podman local
```
podman machine start
podman build . -t mnist-model:v1
podman run -i -u root --rm -p 9000:9000 -v "var:/tmp" -e COS_API_KEY_ID=$COS_API_KEY_ID mnist-model:v1
curl -v http://0.0.0.0:9000/health/status

```



# NOTES 
### test
curl https://serve-model.16qg6j0mog3v.us-south.codeengine.appdomain.cloud/health/status
{"status":{"code":-1,"info":"predict() takes 2 positional arguments but 3 were given","reason":"MICROSERVICE_INTERNAL_ERROR","status":1}}
(base) ➜  mnist-image-ml git:(main) curl https://serve-model.16qg6j0mog3v.us-south.codeengine.appdomain.cloud/health/ping
pong%

curl -v http://0.0.0.0:9000/v2/models/iris/infer \
        -H "Content-Type: application/json" \
        -d '{"inputs": [{"name": "predict", "shape": [1, 4], "datatype": "FP32", "data": [[1, 2, 3, 4]]}]}'

https://serve-model.16qg6j0mog3v.us-south.codeengine.appdomain.cloud

curl -v https://serve-model.16qg6j0mog3v.us-south.codeengine.appdomain.cloud:9000/healthz

curl -v https://serve-model.16qg6j0mog3v.us-south.codeengine.appdomain.cloud:9000/v2/models/MNISTImageModel/predict \
        -H "Content-Type: application/json" \
        -d '{"inputs": [{"name": "predict", "shape": [1, 4], "datatype": "FP32", "data": [[1, 2, 3, 4]]}]}'


#####
Code to setup serving of the trained model

https://github.com/InfuseAI/showcase/blob/showcase/sc-25173/huggingface-transformers-deployment/huggingface-transformers-deployment/huggingface/Model.py

https://ruivieira.dev/serving-models-with-seldon.html

https://docs.seldon.io/projects/seldon-core/en/latest/python/python_wrapping_docker.html
https://docs.seldon.io/projects/seldon-core/en/latest/python/python_wrapping_docker.html 


Sure. Here is an example of how to write a Python wrapper class and initialize a TensorFlow model so that it can be served:

```python
import tensorflow as tf

class Model:

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, data):
        return self.model.predict(data)

```

This class has two methods:

* `__init__()`: This method is called when the class is instantiated. It takes the path to the TensorFlow model as an argument and loads the model.
* `predict()`: This method takes a data array as an argument and returns the predictions of the model.

To serve this model, you can create a Flask app and add the following code to it:

```python
from flask import Flask
from model import Model

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    model = Model("model.h5")
    predictions = model.predict(data)
    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(debug=True)
```

This code creates a Flask app with a single endpoint, `/predict`. This endpoint takes a JSON object as an argument, where the `data` key is the data to be predicted. The endpoint then loads the TensorFlow model, makes predictions, and returns the predictions in JSON format.

You can then deploy this Flask app to a web server. Once the app is deployed, you can send predictions by sending a POST request to the `/predict` endpoint.

I hope this helps! Let me know if you have any other questions.

### server model in container
https://docs.seldon.io/projects/seldon-core/en/latest/workflow/serving.html#ambassador-rest 

### client web app
import requests

def predict(data):
    """Makes a REST call to the model server and returns the predictions."""

    url = "http://localhost:5000/predict"
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"data": data})

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        return response.json()["predictions"]
    else:
        raise ValueError("Error calling model server: {}".format(response.status_code))

def main():
    """Makes a prediction using the model server."""

    data = [1, 2, 3]
    predictions = predict(data)

    print(predictions)

if __name__ == "__main__":
    main()


## fine tune training 
think about how Seldon does Transformers that can be used like this

Sure, here is an example of code for fine-tuning a LLM hosted on Hugging Face:

```python
import transformers

model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Load the dataset
train_dataset = torch.load("train_dataset.pt")
test_dataset = torch.load("test_dataset.pt")

# Fine-tune the model
model.fine_tune(train_dataset, epochs=10)

# Evaluate the model on the test dataset
model.evaluate(test_dataset)
```

This code first imports the `transformers` library, which provides the necessary classes and functions for fine-tuning LLMs. Next, it creates a model object from the `bert-base-uncased` pre-trained model. This model is then loaded with the training dataset, and fine-tuned for 10 epochs. Finally, the model is evaluated on the test dataset.

Here is a more detailed explanation of the code:

* The `from_pretrained()` method of the `AutoModelForSequenceClassification` class loads a pre-trained LLM model from Hugging Face. The `bert-base-uncased` model is a BERT model that has been pre-trained on a large corpus of text data.
* The `torch.load()` function loads the training and test datasets from disk. These datasets are in PyTorch format, and they contain the text and labels for the data points.
* The `fine_tune()` method of the `AutoModelForSequenceClassification` class fine-tunes the model on the training dataset. The `epochs` parameter specifies the number of times the model will be trained on the data.
* The `evaluate()` method of the `AutoModelForSequenceClassification` class evaluates the model on the test dataset. This method returns the accuracy of the model on the test data.

This is just a basic example of how to fine-tune a LLM hosted on Hugging Face. There are many other ways to fine-tune a model, and the specific approach you take will depend on the specific model and dataset you are using.



