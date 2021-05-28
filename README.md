# Question Answering through Transfer Learning

In this repository, we adapt the code provided by the github [Question Answering through Transfer Learning](https://github.com/shmsw25/qa-transfer) project.
Our aim is to convert this code into a kubeflow pipeline. For that, we provide two different pipelines:

## Squad pipeline ([README](SQUAD_README.md))
The aim of this pipeline is to train an expert model of question answering with the squad dataset.
This model is used as a pre-trained model for fine-tuning models in other domains.
You should run this pipeline if you wish to recreate the pre-trained models.
Otherwise, you can proceed with the other pipelines.
Bare in mind that, for the pre-trained models to be used in other pipelines, these models need to be accessible via HTTPS from the other pipelines.

## Semeval pipeline ([README](SEMEVAL_README.md))
The aim of this pipeline is to train a model of question answering with the semeval dataset with a technique called Transfer Learning.
This technique will use the squad model previously trained in order to get better semeval model results.
