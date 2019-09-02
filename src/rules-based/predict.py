import csv
import os
import pandas as pd
import re
import spacy
import sys
import time

from eHostess.PyConTextInterface.SentenceSplitters import SpacySplitter
from eHostess.PyConTextInterface import PyConText
import targetsandmodifiers as tm

nlp = spacy.load('en')

METADATA_FILE = '../data/notes_metadata_trainingset.csv'
TRAINING_FILE = '../data/trainingset_round2.csv'
VALIDATION_FILE = ''
TARGETS_MODIFIERS = './results/'
TARGETS_FILE = './results/targets.tsv'
MODIFIERS_FILE = './results/modifiers.tsv'
PATIENT_OBJECTS_FILE = '../data/patient_objects.pkl'


def process_notes(notes, positive_list):
	"""Identifies note ids with a positive mention of atrial fibrillation.

	Args:
		notes: The list of notes for an MRN.
		positive_list: The list of note ids with a positive mention of atrial
			fibrillation.

	Returns:
		None.

	Raises:
		None.
	"""

	for note in notes:
		note_id = note[0]
		note_text = note[1]
		# ignore empty notes
		if note_text == None:
			continue
		document = PyConText.PyConTextInterface.PerformAnnotation(SpacySplitter.splitSentencesRawString(note_text, note_id),
																	targetFilePath=TARGETS_FILE,
																	modifiersFilePath=MODIFIERS_FILE,
																	modifierToClassMap={'NEGATED_EXISTENCE': 'negative',
																						'AFFIRMED_EXISTENCE': 'positive'})
		for annotation in document.annotations:
			if annotation.annotationClass == 'positive':
				positive_list.append(note_id)
				break

def annotate_patients(metadata_frame):
	"""Returns patient objects with MRNs and list of note ids with a postive
	mention of atrial fibrillation.

	Args:
		metadata_frame: The expanded metadata frame containing clinical notes
			for all mrns.

	Returns:
		patient_objs: A dictionary of all MRNs and a corresponding list of note
			ids that contain a positive mention of atrial fibrillation.

	Raises:
		None.
	"""

	if os.path.isfile(PATIENT_OBJECTS_FILE):
		print 'Reading patient annotations...'
		patient_objs = pickle.load(open(PATIENT_OBJECTS_FILE, 'rb'))
	else:
		print 'Annotating patients...'
		mrns = metadata_frame['mrn'].unique()
		patient_objs = []
		for mrn in mrns:
			records = metadata_frame[metadata_frame['mrn'] == mrn]
			notes = []
			for row in records.itertuples():
				# remove empty notes
				if not isinstance(row.text, str):
					continue
				notes.append((row.noteid, row.text))
			obj = {'mrn': mrn, 
					'positive_notes': [], 
					'notes': notes}
			patient_objs.append(obj)
		print 'Starting annotation at ', time.ctime()
		num_patients = len(patient_objs)
		count = 0
		for patient_obj in patient_objs:
			process_notes(patient_obj['notes'], patient_obj['positive_notes'])
			count += 1
			sys.stdout.write('\rCompleted %d of %d, %d%%' % (count, len(patient_objs), count*100/len(patient_objs)))
			sys.stdout.flush()
		print('\nEnding annotation at ', time.ctime())
		trimmed_objs = []
		for patient_obj in patient_objs:
			trimmed_objs.append({'mrn': patient_obj['mrn'], 'positive_notes' : patient_obj['positive_notes']})
		patient_objs = trimmed_objs
		pickle.dump(patient_objs, open(PATIENT_OBJECTS_FILE, 'wb'))
	return patient_objs

def get_predictions(patient_objs):
	"""Returns the predicted class for each MRN.

	Args:
		patient_objs: A dictionary of all MRNs and a corresponding list of note
			ids that contain a positive mention of atrial fibrillation.

	Returns:
		predictions_frame: A dataframe containing the predicted classes for
			each MRN.

	Raises:
		None.
	"""

	mrns = []
	predictions = []
	for patient_obj in patient_objs:
		mrns.append(patient_obj['mrn'])
		if len(patient_obj['positive_notes']) > 0:
			predictions.append(1)
		else:
			predictions.append(0)
	predictions_frame = pd.DataFrame({'mrn' : mrns, 'predicted_class': predictions})
	return predictions_frame

def get_performance(y, y_pred):
	"""Returns the performance of a model.

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

def main():
	"""Main function.

	Reads training or validation data, and the metadata. Annonates patient
	notes using eHostess. Classifies patients as positive for atrial
	fibrillation is there is one or more positive mentions.

	Usage:
	python predict.py [-t | -v]

	Args:
		-t: Training.
		-v: Validation.

	Example:
	python predict.py -t
	"""

	# read CLI arguments
	if len(sys.argv) < 2:
		print 'Insufficient input.'
		exit()

	# read training or validation data
	if sys.argv[1] == '-t':
		print 'Reading training file...'
		data_frame = pd.read_csv(TRAINING_FILE)
		OUTPUT_FILE = './results/training_predictions'
	elif sys.argv[1] == '-v':
		print 'Reading validation file...'
		data_frame = pd.read_csv(VALIDATION_FILE)
		OUTPUT_FILE = './results/validation_predictions'
	else:
		print 'Incorrect input. Enter -t for training data, or -v for validation data.'
		exit()

	# read metadata
	print 'Reading metadata file...'
	metadata_frame = pd.read_csv(METADATA_FILE)

	# load atrial fibrillation targets and modifiers
	afib_targets_and_mods = tm.ModifiersAndTargets()
	afib_targets_and_mods.addTarget("afib", r"(?i)\bafib\b|\batrial\sfib|a-fib|a\.\sfib|a\.fib|\ba\sfib\b")
	afib_targets_and_mods.addModifier("no", r"(?i)\bno(?!\sfurther)\b")
	afib_targets_and_mods.addModifier("not", r"(?i)\bnot\b")
	afib_targets_and_mods.addModifier("none", r"(?i)\bnone\b", direction='backwards')
	afib_targets_and_mods.addModifier("negative", r"(?i)\bnegative\b")
	afib_targets_and_mods.addModifier("denies", r"(?i)denies|denied|denying")
	afib_targets_and_mods.addModifier("family", r"(?i)\bmother\b|\bfather\b|\bsister\b|\bbrother\b|\bdaughter\b|\bson\b|\baunt\b|\buncle\b|\bgranddaughter\b|\bgrandson\b", direction='bidirectional')
	afib_targets_and_mods.addModifier("rule_out", r"(?i)r/o|r\\o|\brule\s+out\b|\brules\s+out\b|\bruled\s+out\b")
	afib_targets_and_mods.addModifier("unlikely", r"(?i)\bunlikely\b", direction='bidirectional')
	afib_targets_and_mods.addModifier("without", r"(?i)\bunlikely\b")
	afib_targets_and_mods.addModifier("investigate", r"(?i)\binvestigate\b|\binvestigating\b")
	afib_targets_and_mods.addModifier("look_for", r"(?i)\blook\s+for\b\b")
	afib_targets_and_mods.addModifier("differential", r"(?i)\bdifferential\b\b|ddx", direction='bidirectional')
	afib_targets_and_mods.addModifier("possible", r"(?i)\bpossible\b", direction='bidirectional')
	afib_targets_and_mods.addModifier("holter", r"(?i)\b(holter|event)\s+(monitor(ing)?\s+)?ordered\s+for\b\b|ddx")
	afib_targets_and_mods.addModifier("etc", r"(?i)\betc\b", direction='backwards')
	afib_targets_and_mods.addModifier("screen_for", r"(?i)\bscreen\s+for\b")
	afib_targets_and_mods.addModifier("risk_of", r"(?i)\brisk\s+(of|for)\b")
	afib_targets_and_mods.addModifier("suspicious", r"(?i)\bsuspicious\b")
	afib_targets_and_mods.addModifier("question_of", r"(?i)\bquestion\s+of\b")
	afib_targets_and_mods.writeTargetsAndModifiers(TARGETS_MODIFIERS, targets_name='targets.tsv', modifiers_name='modifiers.tsv')

	# annotate patients
	patient_objs = annotate_patients(metadata_frame=metadata_frame)

	# classify patients as having atrial fibrillation if there are more than one mentions.
	predictions_frame = predict(patient_objs=patient_objs)

	# get performance
	combined_frame = predictions_frame.merge(data_frame, 'left', on='mrn')
	performance = get_performance(y=combined_frame['binary_adj_goldstd'], y_pred=combined_frame['predicted_class'])
	print 'Accuracy:', performance[0]
	print 'Positive predictive value:', performance[1]
	print 'Negative predictive value::', performance[2]
	print 'Sensitivity:', performance[3]
	print 'Specificity:', performance[4]
	print 'F1 score:', performance[5]

	# write results
	combined_frame.to_csv(OUTPUT_FILE)


if __name__ == '__main__':
	main()