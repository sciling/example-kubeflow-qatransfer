# Question Answering through Transfer Learning

In this repository, we adapt the code provided by the github [Question Answering through Transfer Learning](https://github.com/shmsw25/qa-transfer) project.
Our aim is to convert this code into a kubeflow pipeline.

# Pipeline parameters
| Pipeline parameter | Description |
| ------ | ------ |
|dataset_url | url of the dataset (e.g https://zenodo.org/record/3678171/files/dev_data_fan.zip)|

Some options for the dataset url are:

* Fan dataset: https://zenodo.org/record/3678171/files/dev_data_fan.zip

# Pipeline stages #

![pipeline.png](./data/images/pipeline.png)

##### 1. Download dataset ([code](./src/download.py))
This component, given the dataset url, downloads all its contents inside an OutputPath Artifact.

##### 2. Train ([code](./src/train.py))
This component performs the following operations:

    1. Given an InputPath containing the previously downloaded dataset, extracts all the training files (audio), converting them into numeric arrays.
    2. Uses those arrays, trains a model with the specified parameters.
    3. Save the model in an OutputPath Artifact.
    4. Generate a loss plot, saves it in an OutputArtifact and embed its visualization inside a web-app component.

##### 3. Test ([code](./src/test.py))
This component performs the following operations:

    1. Loads the previously saved model through an InputPath Artifact.
    2. Given an InputPath containing the previously downloaded dataset, extracts all the testing files (audio), converting them into numeric arrays.
    3. Uses those arrays to test the model.
    4. Saves the  inside a file generated as an OutputPath Artifact(results_path).
    5. Saves true labels and predicted scores to pass it later to the ROC curve.
    6. Saves the name, AUC and pAUC for each subgroup of the test into a results OutputPath Artifact.
    7. Saves the scores for the anomalies files of the test into a anomaly_dir OutputPath Artifact.
    6. Saves accuracy as metrics that will later be passed to the Metrics component.

##### 4.1. ROC Curve ([code](./src/roc_curve.py))
This component is passed the labels directory, which contains true labels and predicted scores, and generates a roc curve that the kubeflow UI can understand. This function can be reused in other pipelines if given the appropiate parameters.

##### 4.2. Metrics ([code](./src/metrics.py))
This component is passed the mlpipelinemetrics which contains metrics and generates a visualization of them that the kubeflow UI can understand.


# File generation #
To generate the pipeline from the python file, execute the following command:

```python3 pipeline.py```

pipeline.py is located inside src folder. The pipeline will be created at the same directory that the command is executed.

Also, if you want to run all tests locally, execute:
```python3 -m unittest tests/*_test.py```

Once the pipeline has been created, we can upload the generated zip file in kubeflow UI and create runs of it.

# Experimental results #

In this section we will replicate the results for the pump dataset in the [DCASE 2020 Challenge Task 2 "Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring"](https://github.com/y-kawagu/dcase2020_task2_baseline/README.md).
The pipeline outputs are a loss plot, a roc curve, and different metrics, from which metrics can be directly compared.
We can see them in the visualizations of the pipeline or in the Run Output Tab of the Run.

In order to check the validity of the pipeline, we are going to execute a run with the same parameters as the original experiment and compare the outputs with the ones obtained in [the original code](https://github.com/y-kawagu/dcase2020_task2_baseline).

### Input parameters ###
| Pipeline parameter | Value |
| ------ | ------ |
|dataset_url |
https://zenodo.org/record/3678171/files/dev_data_pump.zip|

### Loss plot ###

![lossplot.png](./data/images/lossplot.png)

### ROC Curve ###

![roccurve.png](./data/images/roccurve.png)

### Metrics ###
The original results are shown in . In particular, the results for the this task are:

| id | AUC | pAUC
| ------ | ------ | ------ |

In our replication, we get similar results (our results are in percentage format):

| id | AUC | pAUC
| ------ | ------ | ------ |

If we increase the number of epochs to 150, and the validation split to 0.15, the results improve a little:

| id | AUC | pAUC
| ------ | ------ | ------ |
