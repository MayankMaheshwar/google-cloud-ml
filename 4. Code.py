import os
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.contrib.learn.python.learn import learn_runner
import argparse
import pandas as pd
import tensorflow as tf

print(tf.__version__)

"""
Tensorflow is a hierarchical framework. The further down the hierarchy you go, the more flexibility you have, but that more code you have to write. A best practice is to start at the highest level of abstraction. Then if you need additional flexibility for some reason drop down one layer.

For this tutorial we will be operating at the highest level of Tensorflow abstraction, using the Estimator API.
"""

"""
Steps
Load raw data

Write Tensorflow Code

Define Feature Columns

Define Estimator

Define Input Function

Define Serving Function

Define Train and Eval Function

Package Code

Train

Deploy Model

Get Predictions
    """
"""
1) Load Raw Data
This is a publically available dataset on housing prices in Boston area suburbs circa 1978. It is hosted in a Google Cloud Storage bucket.

For datasets too large to fit in memory you would read the data in batches. Tensorflow provides a queueing mechanism for this which is documented here.

In our case the dataset is small enough to fit in memory so we will simply read it into a pandas dataframe.
"""

# downlad data from GCS and store as pandas dataframe
data_train = pd.read_csv(
    filepath_or_buffer='https://storage.googleapis.com/spls/gsp418/housing_train.csv',
    names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "MEDV"])

data_test = pd.read_csv(
    filepath_or_buffer='https://storage.googleapis.com/spls/gsp418/housing_test.csv',
    names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "MEDV"])


data_train.head()
"""
Column Descriptions:
CRIM: per capita crime rate by town
ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS: proportion of non-retail business acres per town
CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX: nitric oxides concentration (parts per 10 million)
RM: average number of rooms per dwelling
AGE: proportion of owner-occupied units built prior to 1940
DIS: weighted distances to five Boston employment centers
RAD: index of accessibility to radial highways
TAX: full-value property-tax rate per $10,000
PTRATIO: pupil-teacher ratio by town
MEDV: Median value of owner-occupied homes

"""


"""
2) Write Tensorflow Code
2.A Define Feature Columns
Feature columns are your Estimator's data "interface." They tell the estimator in what format they should expect data and how to interpret it (is it one-hot? sparse? dense? continuous?). https://www.tensorflow.org/api_docs/python/tf/feature_column
"""

FEATURES = ["CRIM", "ZN", "INDUS", "NOX", "RM",
            "AGE", "DIS", "TAX", "PTRATIO"]
LABEL = "MEDV"

feature_cols = [tf.feature_column.numeric_column(k)
                for k in FEATURES]  # list of Feature Columns

"""
2.B Define Estimator
An Estimator is what actually implements your training, eval and prediction loops. Every estimator has the following methods:

fit() for training
eval() for evaluation
predict() for prediction
export_savedmodel() for writing model state to disk
Tensorflow has several canned estimator that already implement these methods (DNNClassifier, LogisticClassifier etc..) or you can implement a custom estimator. Instructions on how to implement a custom estimator here and see an example here.

For simplicity we will use a canned estimator. To instantiate an estimator simply pass it what Feature Columns to expect and specify a directory for it to output to.

Notice we wrap the estimator with a function. This is to allow us to specify the 'output_dir' at runtime, instead of having to hardcode it here

"""


def generate_estimator(output_dir):
    return tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                     hidden_units=[10, 10],
                                     model_dir=output_dir)


"""
2.C Define Input Function
Now that you have an estimator and it knows what type of data to expect and how to interpret, you need to actually pass the data to it! This is the job of the input function.

The input function returns a (features, label) tuple

features: A python dictionary. Each key is a feature column name and its value is the tensor containing the data for that Feature
label: A Tensor containing the label column
"""


def generate_input_fn(data_set):
    def input_fn():
        features = {k: tf.constant(data_set[k].values) for k in FEATURES}
        labels = tf.constant(data_set[LABEL].values)
        return features, labels
    return input_fn


"""
2.D Define Serving Input Function
To predict with the model, we need to define a serving input function which will be used to read inputs from a user at prediction time.

Why do we need a separate serving function? Don't we input the same features during training as in serving?

Yes, but we may be receiving data in a different format during serving. The serving input function performs transformations necessary to get the data provided at prediction time into the format compatible with the Estimator API.

returns a (features, inputs) tuple

features: A dict of features to be passed to the Estimator
inputs: A dictionary of inputs the predictions server should expect from the user

"""


def serving_input_fn():
    # feature_placeholders are what the caller of the predict() method will have to provide
    feature_placeholders = {
        column.name: tf.placeholder(column.dtype, [None])
        for column in feature_cols
    }

    # features are what we actually pass to the estimator
    features = {
        # Inputs are rank 1 so that we can provide scalars to the server
        # but Estimator expects rank 2, so we expand dimension
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(
        features, feature_placeholders
    )


"""
3) Package Code
You've now written all the tensoflow code you need!

To make it compatible with Cloud AI Platform we'll combine the above tensorflow code into a single python file with two simple changes

Add some boilerplate code to parse the command line arguments required for gcloud.
Use the learn_runner.run() function to run the experiment
We also add an empty __init__.py file to the folder. This is just the python convention for identifying modules.


"""

% % bash
mkdir trainer
touch trainer/__init__.py

% % writefile trainer/task.py


print(tf.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

data_train = pd.read_csv(
    filepath_or_buffer='https://storage.googleapis.com/spls/gsp418/housing_train.csv',
    names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "MEDV"])

data_test = pd.read_csv(
    filepath_or_buffer='https://storage.googleapis.com/spls/gsp418/housing_test.csv',
    names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "MEDV"])

FEATURES = ["CRIM", "ZN", "INDUS", "NOX", "RM",
            "AGE", "DIS", "TAX", "PTRATIO"]
LABEL = "MEDV"

feature_cols = [tf.feature_column.numeric_column(k)
                for k in FEATURES]  # list of Feature Columns


def generate_estimator(output_dir):
    return tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                     hidden_units=[10, 10],
                                     model_dir=output_dir)


def generate_input_fn(data_set):
    def input_fn():
        features = {k: tf.constant(data_set[k].values) for k in FEATURES}
        labels = tf.constant(data_set[LABEL].values)
        return features, labels
    return input_fn


def serving_input_fn():
    # feature_placeholders are what the caller of the predict() method will have to provide
    feature_placeholders = {
        column.name: tf.placeholder(column.dtype, [None])
        for column in feature_cols
    }

    # features are what we actually pass to the estimator
    features = {
        # Inputs are rank 1 so that we can provide scalars to the server
        # but Estimator expects rank 2, so we expand dimension
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(
        features, feature_placeholders
    )


train_spec = tf.estimator.TrainSpec(
    input_fn=generate_input_fn(data_train),
    max_steps=3000)

exporter = tf.estimator.LatestExporter('Servo', serving_input_fn)

eval_spec = tf.estimator.EvalSpec(
    input_fn=generate_input_fn(data_test),
    steps=1,
    exporters=exporter)

######START CLOUD ML ENGINE BOILERPLATE######
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )
    args = parser.parse_args()
    arguments = args.__dict__
    output_dir = arguments.pop('output_dir')
######END CLOUD ML ENGINE BOILERPLATE######

    # initiate training job
    tf.estimator.train_and_evaluate(
        generate_estimator(output_dir), train_spec, eval_spec)

"""
4) Train
Now that our code is packaged we can invoke it using the gcloud command line tool to run the training.

Note: Since our dataset is so small and our model is simply the overhead of provisioning the cluster is longer than the actual training time. Accordingly you'll notice the single VM cloud training takes longer than the local training, and the distributed cloud training takes longer than single VM cloud. For larger datasets and more complex models this will reverse

Set Environment Vars
We'll create environment variables for our project name GCS Bucket and reference this in future commands.

If you do not have a GCS bucket, you can create one using these instructions.

"""

GCS_BUCKET = 'gs://BUCKET_NAME'  # CHANGE THIS TO YOUR BUCKET
PROJECT = 'PROJECT_ID'  # CHANGE THIS TO YOUR PROJECT ID
REGION = 'us-central1'  # OPTIONALLY CHANGE THIS

os.environ['GCS_BUCKET'] = GCS_BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION

"""
Run local
It's a best practice to first run locally on a small dataset to check for errors. Note you can ignore the warnings in this case, as long as there are no errors.
"""

% % bash
gcloud ai-platform local train \
    - -module-name = trainer.task \
    - -package-path = trainer \
    - - \
    --output_dir = './output'


"""
Run on cloud (1 cloud ML unit)
Here we specify which GCS bucket to write to and a job name. Job names submitted to the Cloud AI Platform must be project unique, so we append the system date/time. Update the cell below to point to a GCS bucket you own.
"""

% % bash
JOBNAME = housing_$(date - u + %y % m % d_ % H % M % S)

gcloud ai-platform jobs submit training $JOBNAME \
    - -region =$REGION \
    - -module-name = trainer.task \
    - -package-path = ./trainer \
    - -job-dir =$GCS_BUCKET /$JOBNAME / \
    --runtime-version 1.15 \
    - - \
    --output_dir =$GCS_BUCKET /$JOBNAME/output

"""
Run on cloud (10 cloud ML units)
Because we are using the TF Estimators interface, distributed computing just works! The only change we need to make to run in a distributed fashion is to add the --scale-tier argument. Cloud AI Platform then takes care of distributing the training across devices for you!"""

% % bash
JOBNAME = housing_$(date - u + %y % m % d_ % H % M % S)

gcloud ai-platform jobs submit training $JOBNAME \
    - -region =$REGION \
    - -module-name = trainer.task \
    - -package-path = ./trainer \
    - -job-dir =$GCS_BUCKET /$JOBNAME \
    - -runtime-version 1.15 \
    - -scale-tier = STANDARD_1 \
    - - \
    --output_dir =$GCS_BUCKET /$JOBNAME/output

"""
Run on cloud GPU (3 cloud ML units)
Also works with GPUs!

"BASIC_GPU" corresponds to one Tesla K80 at the time of this writing, hardware subject to change. 1 GPU is charged as 3 cloud ML units.

"""

% % bash
JOBNAME = housing_$(date - u + %y % m % d_ % H % M % S)

gcloud ai-platform jobs submit training $JOBNAME \
    - -region =$REGION \
    - -module-name = trainer.task \
    - -package-path = ./trainer \
    - -job-dir =$GCS_BUCKET /$JOBNAME \
    - -runtime-version 1.15 \
    - -scale-tier = BASIC_GPU \
    - - \
    --output_dir =$GCS_BUCKET /$JOBNAME/output

"""
   Run on 8 cloud GPUs (24 cloud ML units)
To train across multiple GPUs you use a custom scale tier.

You specify the number and types of machines you want to run on in a config.yaml, then reference that config.yaml via the --config config.yaml command line argument.

Here I am specifying a master node with machine type complex_model_m_gpu and one worker node of the same type. Each complex_model_m_gpu has 4 GPUs so this job will run on 2x4=8 GPUs total.

WARNING: The default project quota is 10 cloud ML units, so unless you have requested a quota increase you will get a quota exceeded error. This command is just for illustrative purposes."""

% % writefile config.yaml
trainingInput:
    scaleTier: CUSTOM
    masterType: complex_model_m_gpu
    workerType: complex_model_m_gpu
    workerCount: 1

% % bash
JOBNAME = housing_$(date - u + %y % m % d_ % H % M % S)

gcloud ai-platform jobs submit training $JOBNAME \
    - -region =$REGION \
    - -module-name = trainer.task \
    - -package-path = ./trainer \
    - -job-dir =$GCS_BUCKET /$JOBNAME \
    - -runtime-version 1.15 \
    - -config config.yaml \
    - - \
    --output_dir =$GCS_BUCKET /$JOBNAME/output

"""
5) Deploy Model For Predictions
Cloud AI Platform has a prediction service that will wrap our tensorflow model with a REST API and allow remote clients to get predictions.

You can deploy the model from the Google Cloud Console GUI, or you can use the gcloud command line tool. We will use the latter method. Note this will take up to 5 minutes."""

% % bash
gcloud config set ai_platform/region global

% % bash
MODEL_NAME = "housing_prices"
MODEL_VERSION = "v1"
MODEL_LOCATION = output/export/Servo /$(ls output/export/Servo | tail - 1)

# gcloud ai-platform versions delete ${MODEL_VERSION} --model ${MODEL_NAME} #Uncomment to overwrite existing version
# gcloud ai-platform models delete ${MODEL_NAME} #Uncomment to overwrite existing model
gcloud ai-platform models create ${MODEL_NAME} - -regions $REGION
gcloud ai-platform versions create ${MODEL_VERSION} - -model ${MODEL_NAME} - -origin ${MODEL_LOCATION} - -staging-bucket =$GCS_BUCKET - -runtime-version = 1.15

"""
6) Get Predictions
There are two flavors of the AI Platform Prediction Service: Batch and online.

Online prediction is more appropriate for latency sensitive requests as results are returned quickly and synchronously.

Batch prediction is more appropriate for large prediction requests that you only need to run a few times a day.

The prediction services expect prediction requests in standard JSON format so first we will create a JSON file with a couple of housing records.

"""

% % writefile records.json
{"CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31, "NOX": 0.538, "RM": 6.575,
    "AGE": 65.2, "DIS": 4.0900, "TAX": 296.0, "PTRATIO": 15.3}
{"CRIM": 0.00332, "ZN": 0.0, "INDUS": 2.31, "NOX": 0.437, "RM": 7.7,
    "AGE": 40.0, "DIS": 5.0900, "TAX": 250.0, "PTRATIO": 17.3}

"Now we will pass this file to the prediction service using the gcloud command line tool. Results are returned immediately!"

!gcloud ai-platform predict - -model housing_prices - -json-instances records.json
