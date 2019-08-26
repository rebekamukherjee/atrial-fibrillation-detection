from joblib import load
import numpy as np
import os
import pandas as pd
import pickle
from sklearn import metrics

VALIDATION_FILE = '../data/ml_validation_3phrases.csv'


def get_vector(num_features, num_stop_words, vectorizer_type, validation_notes):
	"""Returns a tfidf vector or a count vector.

	Args:
		num_features: The number of features to create.
		num_stop_words: The number of customized stop words.
		vectorizer_type: The type of vectorizer (tfidf or count).
		validation_notes: The list of clinical notes in the validation data.

	Returns:
		vector: The feature vector created using tfidf vectorizer or count
			vectorizer.

	Raises:
		None.
	"""
	vectorizer_file = './vectorizers/' + '_'.join([str(num_features) + 'f', str(num_stop_words) + 'sw', vectorizer_type, 'vectorizer.pkl'])
	if os.path.isfile(vectorizer_file):
		vectorizer = pickle.load(open(vectorizer_file, 'rb'))
		vector = vectorizer.transform(validation_notes)
		return vector
	else:
		print 'Vectorizer not found.'
		return None


def get_performance(y, y_pred):
	"""Returns the performance of a machine learning model.

	Args:
		y: The true class labels for the target variable.
		y_pred: The predicted class labels for the target variable.

	Returns:
		accuracy: The accuracy of the model.
		ppv: The positive predictive value of the model.
		npv: The negative predictive value of the model.
		sensitivity: The sensitivity of the model.
		specitivity: The specitivity of the model.
		fscore: The f1 score of the model.

	Raises:
		None.
	"""

	CM = metrics.confusion_matrix(y, y_pred)
	TN=CM[0][0]
	FN=CM[1][0]
	FP=CM[0][1]
	TP=CM[1][1]
	accuracy = round((TP+TN)*1.0/(TP+TN+FP+FN), 3)
	ppv = round(TP*1.0/(TP+FP), 3)
	npv = round(TN*1.0/(TN+FN), 3)
	sensitivity = round(TP*1.0/(TP+FN), 3)
	specificity = round(TN*1.0/(FP+TN), 3)
	fscore = round(2.0*((sensitivity*ppv)/(sensitivity+ppv)), 3)
	return [accuracy, ppv, npv, sensitivity, specificity, fscore]


def validate_notes(validation_frame, model_name, num_features, num_stop_words, vectorizer_type):
	"""Validates a machine learning model with specified hyperparameters with
	the validation data.

	Args:
		validation_frame: The validation data frame.
		model_name: The name of the machine learning model.
		num_features: The number of features to create.
		num_stop_words: The number of customized stop words.
		vectorizer_type: The type of vectorizer (tfidf or count).

	Returns:
		performance: The performance of the machine learning model with the
			specified hyperparameters. It lists the model's accuracy, ppv, npv,
			sensitivity, specificity and f1 score.

	Raises:
		None
	"""

	parameter_name = '_'.join([model_name, str(num_features) + 'f', str(num_stop_words) + 'sw', vectorizer_type])
	print 'Validating {}...'.format(parameter_name)
	validation_notes = list(validation_frame['notes'])
	vector = get_vector(validation_notes=validation_notes, 
							num_features=num_features, 
							num_stop_words=num_stop_words, 
							vectorizer_type=vectorizer_type)
	if vector == None:
		return None
	X = vector.toarray()
	y = np.array(validation_frame['binary_adj_goldstd'])
	model_file = './models/' + parameter_name + '.joblib'
	if os.path.isfile(model_file):
		model = load(model_file)
	else:
		print 'Model not found.'
		return None
	y_pred = model.predict(X)
	performance = [parameter_name] + get_performance(y=y, y_pred=y_pred)
	return performance


def main():
	"""Main function.

	Reads validation data with extracted notes. Validates several machine
	learning models with different hyperparameters. Writes the model
	performances into a dataframe for comparisons.

	Usage:
	python validate.py
	"""

	# read and process validation data
	print 'Reading validation file...'
	validation_frame = pd.read_csv(VALIDATION_FILE)
	validation_frame = validation_frame.dropna()

	# specify machine learning models
	models = ['logistic', 'extratrees', 'naive']

	# validate models
	performances = []
	for model_name in models:
		for num_features in [500, 1000, 1500]:
			for num_stop_words in [100, 250, 500]:
				for vectorizer_type in ['tfidf', 'count']:
					performance = validate_notes(validation_frame=validation_frame, 
													model_name=model_name, 
													num_features=num_features, 
													num_stop_words=num_stop_words, 
													vectorizer_type=vectorizer_type)
					if performance == None:
						continue
					performances.append(performance)

	# write performances
	performance_frame = pd.DataFrame(performances, columns=['parameters', 
																'validation accuracy', 
																'validation ppv', 
																'validation npv', 
																'validation sensitivity', 
																'validation specificity', 
																'validation fscore'])
	performance_frame.to_csv('./results/ml_validation_performance.csv')


if __name__ == '__main__':
	main()