# ML Ops pipeline for content classification model
## About 
The purpose of this pipeline is to automate the process of training, testing and deployment of multi-class classification model, which is used by transcoder nodes to identify unwanted content in videos. After initial discussion, it has been decided to start with simpler GitHub-based approach and a dedicated on-premise training server, instead of employing more sophisticated cloud MLOps systems, such as TFX, Azure ML or AWS SageMaker.

## Dataset management
Image dataset is managed manually and located in a dedicated GCP bucket with following structure:
```
└───data
│   └───class1
│   ...
│   └───classN
└───errors
│   └───false_positives
│           └───class1
│           ...
│           └───classN
│   └───false_negatives
│           └───class1
│           ...
│           └───classN
```
`data` directory contains images for model training and testing. `errors` directory is used to store mislabeled samples received from production for manual analysis and dataset improvement. 

## ML repository
All ML code, deployment scripts and models are stored in GitHub repository. Large files are stored on GIT LFS.

## Training and testing
Training is managed by a job in GitHub workflow, which is executed on self-hosted runner located on a dedicated GPU server. The workflow is triggered by:
* pull request to master branch of ML repository - in case there are changes to the code
* manual webhook invocation  

After training, new model is tested on test data, and, if it meets minimal quality criteria, new model file is committed to the repository (either pull request branch or master branch). Testing results are attached in the form of log files, and as a PR comment.

## Deployment
Model files on `master` branch are referenced directly by DevOps scripts in other repositories, e.g.:
```

```