import csv
import pandas as pd
import re
import spacy
import sys

nlp = spacy.load('en')

METADATA_FILE = '../data/notes_metadata.csv'
TRAINING_FILE = '../data/trainingset_2019_0605.xls'
VALIDATION_FILE = ['../data/group300.xlsx', '../data/group301.xlsx', '../data/group302.xlsx']


def extract_phrases(data_frame, metadata_frame, phrase_length, target_pattern, output_file):
	"""Extracts phrases around target terms indicating atrial fibrillation.

	Extracts clinical notes for each unique mrn in the training or validation
	data. Uses spacy sentence splitter to extract sentences among these notes.
	If a term indicating atrial fibrilaltion is identified in any of these
	sentences, extracts phrases from the clinical notes before and after the
	term, specified by phrase_length.

	Args:
		data_frame: The training or validatation data frame that contains the
			mrns and target labels (binary_adj_goldstd).
		metadata_frame: The expanded metadata frame containing clinical notes
			for all mrns.
		phrase_length: The number of phrases to extract when a term indicating
			atrial fibrillation is found in the clinical notes.
			(e.g. if phrase_length = 5, this function extracts 2 phrases before
			the phrase containing the term, the phrase containting the term,
			and 2 phrases after the phrase containing the term)
		targets_pattern: The regex for extracting terms indicating atrial
			fibrillation.
		output_file: The name of the file where the training or validation data
			will be written, with the extracted notes appended to it.

	Returns:
		None.

	Raises:
		None.
	"""

	output_fieldnames = ['mrn', 'notes', 'binary_adj_goldstd']
	with open(output_file, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(output_fieldnames)
		count = 0
		mrns = data_frame['mrn'].unique()
		for mrn in mrns:
			binary_adj_goldstd = int(data_frame[data_frame['mrn'] == mrn]['binary_adj_goldstd'])
			notes = list(metadata_frame[metadata_frame['mrn'] == mrn]['text'])
			extracted_notes = []
			for note in notes:
				if isinstance(note, str):
					doc = nlp(unicode(note, 'utf-8'))
					sentences = [str(i) for i in list(doc.sents)]
					for i in range(len(sentences)):
						targets = re.findall(target_pattern, sentences[i])
						if len(targets) > 0:
							# beginning index
							beg = i - int(phrase_length/2)
							if beg < 0:
								beg = 0
							# ending index
							end = i + phrase_length - int(phrase_length/2)
							if end > len(sentences):
								end = len(sentences)
							extracted_notes += sentences[beg:end]
			extracted_notes = ' '.join(extracted_notes)
			line = [mrn, extracted_notes, binary_adj_goldstd]
			writer.writerow(line)
			count += 1
			sys.stdout.write('\rCompleted %d of %d, %d%%' % (count, len(mrns), count*100/len(mrns)))
			sys.stdout.flush()


def main():
	"""Main function.

	Reads training or validation data, and number of phrases to extract when a
	term indicating atrial fibrillation is encountered. Finds the correct
	columns for mrn and binary_adj_goldstd. Calls the function to extract
	notes from the metadata.

	Usage:
	python extract.py [-t | -v] <phrase_length>

	Args:
		-t: Training.
		-v: Validation.
		phrase_length: The number of phrases to extract.

	Example:
	python extract.py -t 3
	"""

	# read CLI arguments
	if len(sys.argv) < 3:
		print 'Insufficient input.'
		exit()

	# read training or validation data
	if sys.argv[1] == '-t':
		print 'Reading training file...'
		data_frame = pd.read_excel(TRAINING_FILE)
		OUTPUT_FILE = ['../data/ml_training']
	elif sys.argv[1] == '-v':
		print 'Reading validation file...'
		data_frame = pd.concat([pd.read_excel(x) for x in VALIDATION_FILE])
		OUTPUT_FILE = ['../data/ml_validation']
	else:
		print 'Incorrect input. Enter -t for training data, or -v for validation data.'
		exit()

	# read phrase length
	phrase_length = int(sys.argv[2])
	if phrase_length < 1:
		print 'Phrase length too small. Enter a number between 0 and 5.'
		exit()
	elif phrase_length > 5:
		print 'Phrase length too big. Enter a number between 0 and 5.'
		exit()
	OUTPUT_FILE.append(str(phrase_length) + 'phrases.csv')

	# get correct column name for mrn
	if 'mrn' not in data_frame.columns:
		mrn_column = raw_input('Enter name of column with MRNs: ')
		if mrn_column not in data_frame.columns:
			print 'Column not in data.'
			exit()
		data_frame['mrn'] = data_frame[mrn_column]

	# get correct column name for binary_adj_goldstd
	if 'binary_adj_goldstd' not in data_frame.columns:
		if 'adj_goldstd' in data_frame.columns:
			data_frame['binary_adj_goldstd'] = [1 if x > 0 else 0 for x in data_frame['adj_goldstd']]
		else:
			binary_adj_goldstd_column = raw_input('Enter name of column with binary adjudicated goldstandard: ')
			if binary_adj_goldstd_column not in data_frame.columns:
				print 'Column not in data.'
				exit()
			data_frame['binary_adj_goldstd'] = data_frame[binary_adj_goldstd_column]

	# read metadata
	print 'Reading metadata file...'
	metadata_frame = pd.read_csv(METADATA_FILE)

	# define regex for atrial fibrillation target terms
	afib_targets = r'(?i)\bafib\b|\batrial\sfib|a-fib|a\.\sfib|a\.fib|\ba\sfib\b'

	# extract phrases from notes
	OUTPUT_FILE = '_'.join(OUTPUT_FILE)
	print 'Writing extracted phrases...'
	extract_phrases(data_frame = data_frame, 
		metadata_frame = metadata_frame, 
		phrase_length = phrase_length, 
		target_pattern = afib_targets, 
		output_file = OUTPUT_FILE)


if __name__ == '__main__':
	main()