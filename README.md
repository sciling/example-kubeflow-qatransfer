# Question Answering through Transfer Learning

In this repository, we adapt the code provided by the github [Question Answering through Transfer Learning](https://github.com/shmsw25/qa-transfer) project.
Our aim is to convert this code into a kubeflow pipeline. For that, we provide two different pipelines:

## Squad pipeline ([README](SQUAD_README.md))
The aim of this pipeline is to train an expert model of question answering with the squad dataset.

## Semeval pipeline ([README](SEMEVAL_README.md))
The aim of this pipeline is to train a model of question answering with the semeval dataset with a technique called Transfer Learning. This technique will use the squad model previously trained in order to get better semeval model results.
