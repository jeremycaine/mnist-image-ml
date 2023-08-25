import os
import tensorflow as tf
import ibm_cos_utils as cos

def log(e):
    print("{0}\n".format(e))


class MNISTImageModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        log("Initializing")
        model_loc = "model"
        model_path = os.path.join(cos.get_model_endpoint(), model_loc)
        log(model_path)

        self.model = tf.keras.models.load_model(model_path)

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


log("finished MyModel")
