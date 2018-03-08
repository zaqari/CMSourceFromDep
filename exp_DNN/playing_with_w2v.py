import gensim
import codecs
from gensim import corpora, models, similarities
import nltk
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

###INPUTS
#The following are the inputs from the already sorted and organized data. Up to
# the next note, the following only creates data inputs.
train_f = '/Users/ZaqRosen/Documents/Corpora/mohler/train_data6.csv'
test_f = '/Users/ZaqRosen/Documents/Corpora/mohler/test_data6.csv'

COLUMNS = ['tref', 'subj', 'dobj', 'syn', 'obl1', 'obl2', 'verb', 'label']

df_train = pd.read_csv(train_f, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_f, names=COLUMNS, skipinitialspace=True)

###SETTING UP XML PARSER
tree = ET.parse('/Users/ZaqRosen/Documents/Corpora/mohler/en_small_noDTR.xml')
root = tree.getroot()

originallines = []
def xml_txt(listin=originallines, root=root):
	for child in root.findall('LmInstance'):
		childlines = child.find('TextContent')
		for child in childlines:
			listin.append(str(child.text))
	print(len(listin))

###TOKENIZING DATA COLLECTED FROM XML CORPUS
tok_corpus =[]
def prep_tokens(corpuslines=originallines, output_tokens=tok_corpus):
        print('Preparing tokens for w2v conversion . . .')
        for item in corpuslines:
                readable = item.replace('\\', '').replace('}', '').replace('uc0u8232', '').replace('\'92', '\'').replace('a0', '').replace('\'93', '\"').replace('\'94', '\"').replace('\'96', ',').replace('\'97', ',').replace('f0fs24 ', '').replace('cf0 ', '').replace('< ', '').replace(' >', '').replace('\r\n', '')
                output_tokens.append(nltk.word_tokenize(readable))
        print('Tokens made!')

###CREATE W2V's
xml_txt()
prep_tokens(originallines)

model = gensim.models.Word2Vec(tok_corpus, min_count=1, size=300)


###TRANSLATE INPUTS FROM TRAIN AND TEST DATA INTO W2V FORMAT
COLS = ['subj', 'dobj', 'obl1', 'obl2', 'verb']

for itm in COLS:
	appendables=[]
	for its in df_train[itm].values.tolist():
		try:
			appendables.append(0) if its == 0 else appendables.append(model[its])
		except KeyError:
			appendables.append(its)
	train_cols.append(appendables)

for itm in COLS:
	appendables=[]
	for its in df_test[itm].values.tolist():
		try:
			appendables.append(0) if its == 0 else appendables.append(model[its])
		except KeyError:
			appendables.append(its)
	test_cols.append(appendables)

###REBUILD TRAIN AND TEST DOCUMENTS WITH W2Vs###
df_testing = pd.DataFrame(np.array(test_cols).reshape(-1, 5), columns=COLS)
df_training = pd.DataFrame(np.array(train_cols).reshape(-1, 5), columns=COLS)

df_train_out = pd.concat([df_train['tref'],
                          df_training['subj'],
                          df_training['dobj'],
                          df_train['syn'],
                          df_training['obl1'],
                          df_training['obl2'],
                          df_training['verb'],
                          df_train['label'].astype(int)], axis=1, join='inner')

df_test_out = pd.concat([df_test['tref'],
			 df_testing['subj'],
			 df_testing['dobj'],
			 df_test['syn'],
			 df_testing['obl1'],
			 df_testing['obl2'],
			 df_testing['verb'],
			 df_test['label'].astype(int)], axis=1, join='inner')

###EXPORT TO NEW FILE
#Here's the tricky bitch: how the hell do I export these with the W2V in tact???
# This needs to be worked out additionally, and ASAP for additional experiments
# to be run.
export_train_f = '/Users/ZaqRosen/Documents/Corpora/mohler/train_data_embeds.csv'
export_test_f = '/Users/ZaqRosen/Documents/Corpora/mohler/test_data_embeds.csv'

def Training_Data_Builder(array, fileOut):
	with codecs.open(fileOut, 'a', 'utf-8') as csvfile:
		databuilder = csv.writer(csvfile, delimiter=',',
				quotechar='|',
				quoting=csv.QUOTE_MINIMAL)
		databuilder.writerow(array)
	csvfile.close()


