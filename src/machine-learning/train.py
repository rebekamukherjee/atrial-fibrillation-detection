from joblib import dump
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

TRAINING_FILE = '../data/ml_training_3phrases.csv'
STOP_WORDS_FILE = '../data/custom_stop_words.pkl'

logistic = LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced')
forest = ExtraTreesClassifier(n_estimators=1000, max_features=500, n_jobs=1, random_state=0, class_weight='balanced')
naive = GaussianNB()


def get_stop_words(num_stop_words):
	"""Returns a list of customized stop words for the vectorizer.

	Args:
		num_stop_words: The number of customized stop words.

	Returns:
		all_stop_words: The list of all stop words, including a subset of a
			custom list of stop words specified by num_stop_words, english stop
			words from scikit-learn, all numbers in the range 1 to 3000, and
			some words specific to The University of Utah.

	Raises:
		None.
	"""
	custom_stop_words = pickle.load(open(STOP_WORDS_FILE, 'rb'))[:num_stop_words]
	sklearn_stop_words = list(stop_words.ENGLISH_STOP_WORDS)
	numeric_stop_words = list(map(str,range(0, 3000)))
	utah_stop_words = ['utah', 'hcc', 'marrouche', 'nassir']
	all_stop_words = list(dict.fromkeys(custom_stop_words + sklearn_stop_words + numeric_stop_words + utah_stop_words))
	return all_stop_words


def get_vector(num_features, num_stop_words, vectorizer_type, training_notes):
	"""Returns a tfidf vectorizer or a count vectorizer.

	Args:
		num_features: The number of features to create.
		num_stop_words: The number of customized stop words.
		vectorizer_type: The type of vectorizer (tfidf or count).
		training_notes: The list of clinical notes in the training data.

	Returns:
		vector: The feature vector created using tfidf vectorizer or count
			vectorizer.

	Raises:
		None.
	"""
	vectorizer_file = './vectorizers/' + '_'.join([str(num_features) + 'f', str(num_stop_words) + 'sw', vectorizer_type, 'vectorizer.pkl'])
	if os.path.isfile(vectorizer_file):
		vectorizer = pickle.load(open(vectorizer_file, 'rb'))
		vector = vectorizer.transform(training_notes)
	else:
		stop_words = get_stop_words(num_stop_words)
		if vectorizer_type == 'tfidf':
			vectorizer = TfidfVectorizer(max_features=num_features, ngram_range=(1,3), stop_words=stop_words)
		elif vectorizer_type == 'count':
			vectorizer = CountVectorizer(max_features=num_features, ngram_range=(1,3), stop_words=stop_words)
		vector = vectorizer.fit_transform(training_notes)
		pickle.dump(vectorizer, open(vectorizer_file, 'wb'))
	return vector


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
	accuracy = (TP+TN)*1.0/(TP+TN+FP+FN)
	ppv = TP*1.0/(TP+FP)
	npv = TN*1.0/(TN+FN)
	sensitivity = TP*1.0/(TP+FN)
	specificity = TN*1.0/(FP+TN)
	fscore = 2.0*((sensitivity*ppv)/(sensitivity+ppv))
	return [accuracy, ppv, npv, sensitivity, specificity, fscore]


def train_notes(training_frame, model_name, model, num_features, num_stop_words, vectorizer_type):
	"""Trains a machine learning model with speicified hyperparameters with the
	training data.

	Args:
		training_frame: The training data frame.
		model_name: The name of the machine learning model.
		model: The machine learning model.
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
	print 'Training {}...'.format(parameter_name)
	#training_notes = list(training_frame['notes'])
	training_notes = list(training_frame['phrases'])
	vector = get_vector(training_notes=training_notes, 
							num_features=num_features, 
							num_stop_words=num_stop_words, 
							vectorizer_type=vectorizer_type)
	X = vector.toarray()
	y = np.array(training_frame['binary_adj_goldstd'])
	model.fit(X, y)
	dump(model, './models/' + parameter_name + '.joblib')
	y_pred = model.predict(X)
	performance = [parameter_name] + get_performance(y=y, y_pred=y_pred)
	return performance


def main():
	"""Main function.

	Reads training data with extracted notes. Trains several machine learning
	models with different hyperparameters. Saves the models for validation.
	Writes the model performances into a dataframe for comparisons.

	Usage:
	python train.py
	"""

	# read and process training data
	print 'Reading training file...'
	training_frame = pd.read_csv(TRAINING_FILE)
	training_frame = training_frame.dropna()

	# define machine learning models
	models = {'logistic': logistic, 'extratrees': forest, 'naive': naive}

	# train models
	performances = []
	for model_name in models:
		model = models[model_name]
		for num_features in [500, 1000, 1500]:
			for num_stop_words in [100, 250, 500]:
				for vectorizer_type in ['tfidf', 'count']:
					performance = train_notes(training_frame=training_frame, 
												model_name=model_name, 
												model=model, 
												num_features=num_features, 
												num_stop_words=num_stop_words, 
												vectorizer_type=vectorizer_type)
					performances.append(performance)

	# write performances
	performance_frame = pd.DataFrame(performances, columns=['parameters', 
																'train accuracy', 
																'train ppv', 
																'train npv', 
																'train sensitivity', 
																'train specificity', 
																'train fscore'])
	performance_frame.to_csv('./ml_training_performance.csv')


if __name__ == '__main__':
	main()