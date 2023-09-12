# mnist-image-ml
MNIST Image machine learning model project

## Local Development

### Setup
Log in to IBM Cloud and go to the Service Credentials of COS bucket `mnist-model` and get `apikey`. Use this to set local envvar so that local apps can connect to the COS bucket.
```
export COS_API_KEY_ID=nnnn
```

### Serve Model
From the base folder `serve-model` build a local Docker image and run the Serve Model (Seldon) application. This runs as root and maps directory so that the app can use temporary file system to retrieve the model file from IBM COS and then load an instance into memory.

```
podman machine start
podman build . -t mnist-model:v1
podman run -i -u root --rm -p 9000:9000 -v "var:/tmp" -e COS_API_KEY_ID=$COS_API_KEY_ID mnist-model:v1

# test
curl -v http://0.0.0.0:9000/health/status

curl -v http://0.0.0.0:9000/health/ping

curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"data": {"ndarray":[34.0, 100.0, 1, 2]} }' \
    http://0.0.0.0:9000/predict

```

## IBM Cloud Engine
Region - WDC
Project - mnist-model
COS - caine-cos
Logging - 

### Create a project
```
ibmcloud login --sso ...

ibmcloud target -c b71ac2564ef0b98f1032d189795994dc
ibmcloud target -r us-south
ibmcloud target -g ceh-group

ibmcloud ce project create --name mnist-model
```

### Set-up the project
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

## Model Development

### Train Model
Create and train a MNIST image machine learning model in Tensorflow format and store.

Job to train image prediction model to label its image
```
# create app first time
ibmcloud ce job create --name train-model --src https://github.com/jeremycaine/mnist-image-ml --bcdr train-model --str dockerfile --env-from-secret caine-cos-api-key --env-from-configmap mnist-model-cm --cpu 2 --memory 16G

# rebuild after git commit
ibmcloud ce job update --name train-model --rebuild
```

### Serve Model
App to serve prediction model 
```
# create app first time
ibmcloud ce app create --name serve-model --src https://github.com/jeremycaine/mnist-image-ml --bcdr serve-model --str dockerfile --env-from-secret caine-cos-api-key --env-from-configmap mnist-model-cm --cpu 2 --memory 16G --port 9000

# rebuild after git commit
ibmcloud ce app update --name serve-model --rebuild
```

## Reference

### Seldon
https://docs.seldon.io/projects/seldon-core/en/latest/python/python_wrapping_docker.html 
https://ruivieira.dev/serving-models-with-seldon.html




