# Atrial Fibrillation Detection

This repository contains python scripts used in the study at The University of Utah Department of Cardiovascular Medicine to detect atrial fibrillation among patients using natural language processing. Two methods were evaluated in the study, a rules based method and a machine learning method.

## Directory Structure

```
atrial-fibrillation-detection
|
+-- src
|	|
|	+-- data
|	|
|	+-- machine-learning
|	|	|
|	|	+-- models
|	|	|
|	|	+-- results
|	|	|	|
|	|	|	+-- ml_training_performance.csv
|	|	|	+-- ml_validation_performance.csv
|	|	|
|	|	+-- vectorizers
|	|	|
|	|	+-- extract.py
|	|	+-- train.py
|	|	+-- validate.py
|	|
|	+-- rules-based
|		|
|		+-- results
|		|
|		+-- predict.py
|		+-- compare.py
|
+-- README.md
```

## Rules Based Method

### Usage

To get the predictions of the rules-based NLP model on the training (-t) or validation (-v) data, run the following command on CLI:

```
python predict.py [-t | -v]
```

To compare the rules-based model performance with that of traditional models, run the following command on CLI:

```
python compare.py
```

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