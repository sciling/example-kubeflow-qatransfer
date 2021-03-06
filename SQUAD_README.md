# Squad Modeling for Question Answering

The aim of this pipeline is to train an expert model of question answering with the squad dataset.

# Pipeline parameters
| Pipeline parameter | Description |
| ------ | ------ |
|prepro_train_ratio| Float representing the ratio of the data to be used to train  (e.g 0.7)|
|prepro_glove_vec_size| Integer representing the glove vec size to be used(e.g 100)|
|prepro_mode| String representing the preprocess mode ("full", "all" or "single")|
|prepro_tokenizer| String representing the preprocess tokenizer ("PTB" or "Stanford")|
|prepro_url| String representing the url of the server (e.g "vision-server2.corp.ai2")|
|prepro_port| Integer represeting the port of the server (e.g 8000)|
|train_sent_size_th| String representing the maximum length (# of words) of each sentence (e.g '150')|
|train_ques_size_th| String representing the maximum number of words composing the question (e.g '100')|
|train_num_epochs| Number of epochs to train the model (e.g '12')|
|train_num_steps| Number of steps to train the model (e.g '55')|
|train_eval_period| Period to perform evaluation on train (e.g '50')|
|train_save_period| Period to perform save on train (e.g '10').|
|train_learning_rate| Float representing the learning rate (e.g 0.5)|
|train_batch_size| Integer representing the size of the bacth (e.g 60)|
|train_hidden_size| Integer representing the hidden size (e.g 100)|
|train_var_decay| Integer representing the exponential decay (e.g 0.999)|
|training_mode|String representig the preprocess mode ("span" or "class")|
|device| String representing the default device for summing gradients. ("/cpu:0")|
|device_type| String representing the device for computing gradients (parallelization) ("cpu" or "gpu" )|
|num_gpus| Integer representing the number of gpus or cpus for computing gradients (1)|

# Default parameters #
Taking into account the following text extracted from [BI-DIRECTIONAL ATTENTION FLOW
FOR MACHINE COMPREHENSION]( https://arxiv.org/pdf/1611.01603.pdf#page=6),

![pipeline.png](./data/images/paper_snippet.png)

we have defined default parameters as optimal ones:
| Pipeline parameter | Description |
| ------ | ------ |
|prepro_train_ratio| 0.9 |
|prepro_glove_vec_size| 100 |
|prepro_mode| "full"|
|prepro_tokenizer| "PTB" |
|prepro_url| "vision-server2.corp.ai2"|
|prepro_port| 8000 |
|train_sent_size_th| 500 |
|train_ques_size_th| 30 |
|train_num_epochs| 12|
|train_num_steps| 20000 |
|train_eval_period| 500 |
|train_save_period| 500 |
|train_learning_rate| 0.5|
|train_batch_size| 60|
|train_hidden_size| 100|
|train_var_decay| 0.999|
|training_mode|span|
|device| "/cpu:0"|
|device_type| "gpu"|
|num_gpus| 1 |

# Pipeline stages #

![pipeline.png](./data/images/squad.png)

##### 1. Download dataset ([code](./src/squad/download.py))
This component downloads the squad and glove dataset inside an OutputPath Artifact

##### 2.1. Prepro basic ([code](./src/squad/prepro.py))
This component preprocess the squad dataset and save generated files inside an OutputPath Artifact. Only executed if training mode equals span

##### 2.2. Convert2class and Prepro class ([code](./src/squad/prepro.py))
This components preprocess the squad dataset, converting it in classes and save generated files inside an OutputPath Artifact.

##### 3. Train ([code](./src/squad/train.py))
This component trains the squad dataset taken into account squad preprocess generated files and save generated model inside an OutputPath Artifact.

##### 4. Test ([code](./src/squad/test.py))
This component tests the model and creates a different metrics that the kubeflow UI can understand, in order to visualize the accuracy and f1-score (if training mode equals span) or the accuracy and loss (if training mode equals class) of the trained model.

# File generation #
To generate the pipeline from the python file, execute the following command:

```python3 pipeline.py```

pipeline.py is located inside src/squad folder. The pipeline will be created at the same directory that the command is executed.

Also, if you want to run all tests locally, execute:
``` ./run_squad_tests.sh ```

Once the pipeline has been created, we can upload the generated zip file in kubeflow UI and create runs of it.

# Upload generated model

In the semeval pipeline, you need a link that downloads the squad model as a zip. In order to achieve that, one alternative could be:

1.  Download the generated model from the minio server.
2.  Create a Github release and upload there the zipped model.
3.  Check the request made by your browser when you click on the released model. In the request, there will be a link similar to http://github.com/sciling/qatransfer/releases/download/v0.1/save.zip that will work as squad_url in the semeval pipeline.

# Experimental results #

In this section we will replicate the results for the squad dataset in the [Question Answering through Transfer Learning from Large Fine-grained Supervision Data](https://github.com/sciling/qatransfer/blob/master/run.md).
The pipeline outputs different metrics from which can be directly compared.
In order to check the validity of the pipeline, we are going to execute a run. As we do not dispose of a capable machine, the obtained results may be a bit worse than the original ones.

### Input parameters ###
| Pipeline parameter | Value |
| ------ | ------ |
|prepro_train_ratio|0.005|
|prepro_glove_vec_size|100|
|prepro_mode|"full"|
|prepro_tokenizer|"PTB"|
|prepro_url|"vision-server2.corp.ai2"|
|prepro_port|8000|
|train_sent_size_th|10|
|train_ques_size_th|10|
|train_num_epochs|1|
|train_num_steps|1|
|train_eval_period|1|
|train_save_period|1|
|train_learning_rate| 0.5|
|train_batch_size| 60|
|train_hidden_size| 100|
|train_var_decay| 0.999|
|training_mode|span|
|device| "/cpu:0"|
|device_type| "gpu"|
|num_gpus| 1 |


### Metrics ###
Using the predefined parameters, we obtain the following results:

| Accuracy | f1-score |
| ------ | ------ |
| 0.008	 | 0.030 |

If instead of span, we put class as training:mode, leaving the rest of the parameters as defined, we obtain:

|Accuracy| Loss |
|-----|------|
|0.549|0.692|

The original results are shown in. In particular, we show the accuracy achived in ["Question Answering through Transfer Learning from Large Fine-grained Supervision Data](https://www.aclweb.org/anthology/P17-2081.pdf) and the F1 score achieved in [Bi-directional Attention Flow](https://arxiv.org/pdf/1611.01603.pdf):

| Accuracy | f1-score |
| ------ | ------ |
| 82.86 | 81.1 |

In our replication, we get way worse results as expected because of the machine and the poor parameters.
