# mnist-image-ml
MNIST Image machine learning model project


# IBM Cloud Engine
```
ibmcloud login --sso ...

ibmcloud target -r eu-gb
ibmcloud target -g ceh-group

ibmcloud ce project list
ibmcloud ce project select --name mnist-image-ml

# need secret
ibmcloud ce secret create --name caine-cos-api-key --from-literal COS_API_KEY_ID=xxxxxxx

# config map for variables
ibmcloud ce configmap create --name mnist-image-ml-cm \
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
ibmcloud ce job create --name train-model --src https://github.com/jeremycaine/mnist-image-ml --bcdr train-model --str dockerfile --env-from-secret caine-cos-api-key --env-from-configmap mnist-image-ml-cm --cpu 2 --memory 16G

# rebuild after git commit
ibmcloud ce job update --name train-model --rebuild
```

## Serve Model
Code to setup serving of the trained model
