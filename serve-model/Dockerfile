FROM registry.access.redhat.com/ubi8/python-39 

## Add application sources
WORKDIR /app
COPY src/requirements.txt ./

# Install the dependencies
RUN pip install -U "pip>=19.3.1" && \
    pip install tensorflow \
    pip install --no-cache-dir -r requirements.txt

## 2. Non-root, arbitrary user IDs
USER 1001

## 3. Image identification
LABEL name="jeremycaine/mnist-image-ml/serve-model" \
      vendor="Acme, Inc." \
      version="1.2.3" \
      release="45" \
      summary="Digital image recognition" \
      description="Serve model to match input image to MNIST image dataset"

USER root

## 4. Image license
## Red Hat requires that the image store the license file(s) in the /licenses directory. 
## Store the license in the image to self-document
COPY ./licenses /licenses

## 5. Latest security updates
RUN dnf clean all

## 6. Group ownership and file permission
RUN chgrp -R 0 $HOME && \
    chmod -R g=u $HOME

USER 1001
RUN chown -R 1001:0 $HOME

## 7. Application source
## Copy the application source and build artifacts from the builder image to this one
COPY src/ ./

# Port for GRPC
EXPOSE 5000
# Port for REST
EXPOSE 9000

# Define environment variables
ENV MODEL_NAME MNISTImageModel
ENV SERVICE_TYPE MODEL

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE