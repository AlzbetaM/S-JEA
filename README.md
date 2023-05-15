# S-JEA
This project was created as part of computing science project at the University of Aberdeen. The aim is to create an architecture that can successfully perform hierarchical image classification without need for excessive amount of labelled data.
This code is based on VICReg model and is extended to encode the input twice.

## Run Code
The code contains two run files - run.sh and run_slurm.sh (the latter being for execution of the model on linux supercomputing cluster).
The ideal way to run small-scale experiments is to load the colab_run.ipynb file to Google Colaboratory.
This file provides additional instructions.

For large-scale experiment, access to some supercomputing resource is necessary.

## MODEL
The model folder contains a pretrained model (latest_checkpoint.ckpt) 
and fine-tuned model for second level of S-JEA.
First level is not included but can be easily obtained by finetuning  pretrained model at the first level.
Given the size of models, decision was made to present only the novel one (as it was shown during many experiment
that the first level behaves as usual VICReg).Thus inference and finetuning can be run for this model.
## Requirements
These can be found in requirements.txt file. Most important are Pytorch, Pytorch Lightning, configargparse, and many other ones.
The model requires access to GPU to be pre-trained and finetuned. That is why the Jupyter Notebook is provided so that one can utilise access to GPU on Google Colaboratory.
Inference can be run on CPU, requiring Python 3.7 or above.