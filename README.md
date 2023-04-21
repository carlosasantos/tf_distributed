# Example of Tensorflow / Keras distributed training on Azure ML

- The file scr/train.py corresponds to a time series forecasting LSTM model, extracted from https://www.tensorflow.org/tutorials/structured_data/time_series
    - mlflow.autolog() is added to ensure all metrics and models are logged in AzureML

- run_job.ipynb includes the execution steps to train the above model, in a distributed way, in Azure ML. This is based on AzureML documentation:
    - https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-distributed-gpu?view=azureml-api-2#tensorflow.

Distributed training is enabled in the following section:
```py
job = command(
    code="./src",  # local path where the code is stored
    command="python train.py",
    environment="AzureML-tensorflow-2.7-ubuntu20.04-py38-cuda11-gpu@latest",
    compute="gpu-cluster",
    instance_count=2,
    distribution={
        "type": "tensorflow",
        "parameter_server_count": 1,
        "worker_count": 2,
        "added_property": 7,
    },
    display_name="tensorflow_lstm_2workers",
    experiment_name="tensorflow-distributed-test"
)
```