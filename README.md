# Atrial Fibrillation Detection

This repository contains python scripts used in the study at The University of Utah Department of Cardiovascular Medicine to detect atrial fibrillation among patients using natural language processing. Two methods were evaluated in the study, a rules based method and a machine learning method.

## Directory Structure

atrial-fibrillation-detection
|
+-- src
|	|
|	+-- data 										# folder containing metadata, training data, validation data
|	|
|	+-- machine-learning 							# folder containing python scripts for machine learning method
|	|	|
|	|	+-- models 									# folder containing all trained machine learning models
|	|	|
|	|	+-- results									# folder containing training and validation performances for all the models
|	|	|	|
|	|	|	+-- ml_training_performance.csv 		# csv file with training performances
|	|	|	+-- ml_validation_performance.csv 		# csv file with validation performances
|	|	|
|	|	+-- vectorizers 							# folder containing all vectorizers
|	|	|
|	|	+-- extract.py 								# script to extract clinical notes and append to training and validation data
|	|	+-- train.py 								# script to train machine learning models
|	|	+-- validate.py 							# script to validate machine learning models
|	|
|	+-- rules-based 								# folder containing python scripts for rules based method
|	|
|	+-- README.md 									# readme file


## Machine Learning Method

### Usage

To extract clinical notes without noise into the training (-t) or validation (-v) data, run the following command on CLI:

```
python extract.py [-t | -v] <phrase_length>
```

To train the machine learning models, run the following command on CLI:

```
python train.py
```

To validate the machine learning models, run the following command on CLI:
```
python validate.py
```