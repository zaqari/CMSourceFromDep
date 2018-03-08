from random import randint
import tempfile
import tensorflow as tf
import pandas as pd
import numpy as np
import codecs
import gensim
from sklearn.cluster import KMeans
import csv

COLUMNS = ['LmSource', 'LmTarget', 'Metaphoricity', 'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2', 'sent1']

df_in = pd.read_csv('/Users/ZaqRosen/Desktop/ML_Meta4/mohler_withTags.csv', names=COLUMNS, skipinitialspace=True)

###MOHLER MODEL VECTOR REPRESENTATIONS###
#The following 'opens up' the package for the Glove2Vec models. This will then
# allow us to feed items from df_in['LmSource'] into the system in order to read
# and cluster the variables according to their vectors.
w2v_model = '/Users/ZaqRosen/Documents/Corpora/mohler/w2v_model_mohler.bin'

model = gensim.models.Word2Vec.load(w2v_model)

###BUILDING CLUSTER REPRESENTATIONS USING W2Vs###
#At some point, this will need to be edited such that word_vector is
# representative of only the w2vs gained from the labels column selected. This
# can and should be accomplished by simply following these two steps to est.
# a feed-dictionary syle loop to the following variable . . .
df_source = df_in['LmSource'].values.tolist() #Creates a set of values and then turns them into a list.

df_source_w2v =[]
df_source_items = []
key_error = []

def createClusters():
	print('Building a list of clusterfucks . . .')
	key_counter = int(0)
	att_counter = int(0)
	for item in df_source:
		if ' ' in str(item):
			NN = item.replace(' ', '-')
		else:
			NN = str(item)
		try:
			df_source_w2v.append(model[NN])
			df_source_items.append(item)
		except AttributeError:
			att_counter +=1
		except KeyError:
			key_error.append(df_source.index(item))
	print('Finished!')
	print('Key Errors: ', str(len(list(set(key_error)))))
	print('Attribute errors: ', str(att_counter))
	print('Remove the following indeces from df_data: key_set returned')

createClusters()

key_set = list(set(key_error))

word_vector = np.asarray(df_source_w2v)

kmeans_clustering = KMeans(n_clusters=int(108)) #108=length of source domains posited by Mohler & co.
idx = kmeans_clustering.fit_predict(word_vector)

word_map = dict(zip(df_source_items, idx))


###OUTPUT FROM WHICH TO BUILD OUR LABELER FROM###
#The following will give us an output of lists which I'm leaving right now in
# a format ripe to be manually input into the real system. Why, you ask? Simply
# because until we have a successful model, I don't see the utility in allowing
# the system itself to make it's own mistakes. Let's instead maintain some
# control over sanitized inputs into our grouping system.

dic_pic=[]
def freeza(dic):
	print()
	print()
	print()
	print('==INITIATING BRIDGES PROTOCOL==')
	print('=============&&&===============')
	print('Running========================')
	print('============running============')
	print('========================running')
	print(' . . . . . . . . . . . . . . . ')
	ke = dic.keys()
	val = list(set(dic.values()))
	item = dic.items()
	for item in val:
		v =[]
		for key, value in dic.items():
			if item == value:
				v.append(key)
		dic_pic.append(v)
		print(' . . . . . . . . . . . . . . . ')
	print('JEFF BRIDGES!!!')

freeza(word_map)


###REPLACING WORDS WITH THEIR CENTROIDS
#Now that we've got the centroid values for all the words that we care about for
# our clustering/semantic recognition activity, we can change the values out in
# df_in['LmSource'] for the measurement of that centroid. This gives us the numer-
# ical labels necessary to run the classification task.
                      
print()
print()
print('Creating LmSource cluster labels, cap\'n!')
SDID = df_in['LmSource'].values.tolist()
SDID_len = len(SDID)
indeces_repl = []

for k in range(0, len(dic_pic)):
        repl_tier = dic_pic[k]
        for item in repl_tier:
                repl = item
                for item in SDID:
                        if str(item) == repl:
                                indeces_repl.append(SDID.index(item))
                                SDID[SDID.index(item)] = int(k)
                                
for item in list(range(0, SDID_len)):
        if item not in indeces_repl:
                SDID[item] = int(109)
print('Finished, cap\'n!')
print()

###FINAL PRODUCT
#Everything calculated to this point leads us to this column, which needs be
# concatenated into the final data somehow. Cheers.

dfLabels = pd.DataFrame(np.array(SDID).reshape(-1, 1), columns=['Labels'])
#print(df_labels)

###CREATING USEFUL FEATURES###
#The following section is useful in a couple of crucial ways. First and fore-
# most, it normalizes the data input by making the target referent in an array
# a nonce-word--'XREF'--which has no semantic representation outside of its
# construct-dependent role. This means that only the context is used to make
# a class-category decision with respect to source domain. Second, the PRESENCE
# of 'XREF' can be a trigger to analyze the array as being of interest at all
# via an if-clause which can be inserted later into the input function (I've
# not created an if-gate in the input function, yet).
##IDEA FOR A FUTURE UPDATE
#There is a very real possibility that ALL of the arrays collected could be used
# as features/grouped in a recurrent NN to boost the evidence of a given
# source domain. How, you ask? Because the target referent should typically by
# its frame-based constraints (Sullivan) only be capable of being construed
# naturally with a select few other possible constructional units. These could
# be used as evidence for one metaphorical source domain versus another.

listy = df_in['LmTarget'].values.tolist()

Values = ['subj', 'dobj', 'syn', 'obl1', 'obl2', 'verb', 'Labels']
ReplaceValues = ['subj', 'dobj', 'syn', 'obl1', 'obl2']

print('Concatenating data, sir.')

df_XREF = df_in[ReplaceValues].replace(listy, 'XREF')

LmTarget_only_indeces = []
for item in ReplaceValues:
    a = item
    edited_loci = df_XREF.index[df_XREF[a].isin(['XREF'])].values.tolist()
    for item in edited_loci:
        LmTarget_only_indeces.append(item)

#Creates a column of untouched verbs.
dfVerb = pd.DataFrame(np.array(df_in['verb'].values.tolist()).reshape(-1,1), columns=['verb'])
#Creates a data frame with only values at loci indicated in LmTarget_only_indeces.
df_XREF = pd.concat([df_in[ReplaceValues].loc[LmTarget_only_indeces], dfVerb.loc[LmTarget_only_indeces], dfLabels.loc[LmTarget_only_indeces].astype(int) ], axis=1, join='inner')

###UPDATE: When running this, I ran into several errors in which 'NaN' was
# included in the dataframe. Tried as I could, I couldn't kill all these
# examples. Something corrupted the file on input. So I put in the following
# protocol to circumvent the issue (which, honestly, is just data cleanliness
# given large dependency data-sets . . . 
print('Giving the data a final scrub, cap!')

fuckers =[]
for item in Values:
  damn = df_XREF.index[df_XREF[item].isin(['nan', 'NaN', 'NAN'])].values.tolist()
  for item in damn:
    fuckers.append(item)

df_XREF = df_XREF.drop(df_XREF.index[[fuckers]]) if len(fuckers) > int(0) else None
df_XREF = df_XREF.drop(df_XREF.index[[key_set]]) if len(key_set) > int(0) else None

print()

#############################THE FOLLOWING MUST BE UPDATED######################

###NOTE: ONLY PICKING UP 'XREF's
#The system works as is, now. The only thing that needs to be fixed is limiting
# collecting data from rows in which 'XREF' occurs. We might be able to do
# that by using the following line in some way:
#           df_XREF.loc[df_XREF[ReplaceValues].isin(['XREF'])
# Note that this line gives us the line loci, but does not give us any
# additional information. What we might do, is use this to create an actual
# list of loci with these numbers, and can then define a variable that is
# only made up of these values.
##UPDATE: System to pick them up works if you run it individually per column.

print('Building training and test data from data collected!')

########################## DEPRECATED ##########################################
df_data_list = []
df_data_loci = []
list_to_train_data = []

for item in ReplaceValues:
        df_data_list.append(df_XREF.loc[df_XREF[item].isin(['XREF'])].values.tolist())

for item in df_data_list:
        index = df_data_list.index(item)
        for item in df_data_list[index]:
                df_data_loci.append(item)

df_data = pd.DataFrame(np.array(df_data_loci).reshape(-1, len(Values)), columns=Values)

df_train_loci  = []
df_test_loci  = []

############# NEEDS BE RETOOLED TO ONLY USE df_XREF ############################
for item in list(set(df_data['Labels'].values.tolist())):
	alist = df_data[Values].index[df_data['Labels'].isin([item])].tolist()
	for k in range(0, int(len(alist)*.8)):
		randIDX = randint(0, int(len(alist)-1))
		df_train_loci.append(alist[randIDX])

#OLD VERSION . . . works 20% of the time . . .
#for item in list(set(df_data['Labels'].values.tolist())):
#        for k in range(0, int(len(df_data)*.8)):
#                df_train_loci.append(randint(1, len(df_data)))

df_train = df_data[Values].loc[df_train_loci]

for k in range(1, len(df_data)):
	if k not in df_train_loci:
		  df_test_loci.append(k)

df_test = df_data[Values].loc[df_test_loci]

print()
print('Now it\'s up to you, DNN man. Or rather, after all that, RNN man.')

###DNN SETUP###
#Herein lies the powerhouse of this system. The following establishes a DNN
# in which the edited information above is passed into a network classifier
# and is then classified to what its 'LmSource' value ought to be.
##NOTE ON GENERATING SOURCE DOMAIN CLASSES
#It would be worthwhile to /group/ the source domain values in some meaningful
# way. I recommend the following technique to do such:
# (1) Generate a list of all the values for source tags via: df_in['LmSource']
#     .values.tolist() and print said list to a different document.
# (2) Using their vector representations to group them geometrically, or
#     manually grouping them all together.
# (3) Using these as a list to replace the values in the column with a numerical
#     token/classifier.
##NOTE: CHANGING CATEGORICAL_COLUMNS
#As of right now, the CATEGORICAL_COLUMNS will only call out to columns in df[].
# The problem is, the data is now in df_XREF. The solution, then, might lie in
# putting everything into df_XREF except the label column, and then running
# /those/ values through the feed function. The only difference, then, is
# including 'verb' and 'syn' in df_XREF.
##UPDATE:
#The system may need to be changes such that the labels generated are not the
# hashbucket values listed here, but are instead their own tf.constant(),
# referring thus to their unicode makeup as the label. This shouldn't, in
# theory, mess up the classifer, so long as the number of possible labels
# remains constant. A variable for such could be forged out of the set of labels
# present in the system:
#  (1) labels = list(set(df_data['LmSource'].values.tolist()))
#  (in input_fn) tf.constant(labels)

#DNN_COLUMNS = ['subj', 'dobj', 'syn', 'obl1', 'obl2', 'verb', 'Labels']

#CATEGORICAL_COLUMNS = ['subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2']
#LABELS_COLUMN = ['Labels']

#def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  #continuous_cols = {k: tf.constant(df[k].values)
				   #  for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
#  categorical_cols = {k: tf.SparseTensor(
#	  indices=[[i, 0] for i in range(df[k].size)],
#	  values=df[k].values,
#	  dense_shape=[df[k].size, 1])
#					  for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
#  feature_cols = dict(categorical_cols.items())
  # Converts the label column into a constant Tensor.
#  label = tf.constant(df['Labels'].values.astype(int))
  # Returns the feature columns and the label.
#  return feature_cols, label

#def train_input_fn():
#  return input_fn(df_train)

#def eval_input_fn():
#  return input_fn(df_test)

#subj = tf.contrib.layers.sparse_column_with_hash_bucket("subj", hash_bucket_size=int(1e9))

#syn = tf.contrib.layers.sparse_column_with_hash_bucket("syn", hash_bucket_size=int(1e9))

#verb = tf.contrib.layers.sparse_column_with_hash_bucket("verb", hash_bucket_size=int(1e9))

#obl1 = tf.contrib.layers.sparse_column_with_hash_bucket("obl1", hash_bucket_size=int(1e9))

#obl2 = tf.contrib.layers.sparse_column_with_hash_bucket("obl2", hash_bucket_size=int(1e9))

#dobj = tf.contrib.layers.sparse_column_with_hash_bucket("dobj", hash_bucket_size=int(1e9))

#Ultimately needs to be changed to represent an actual model path:
# /tmp/meta4Model, or somesuch.
#model_dir = tempfile.mkdtemp()

##DNN##
#The following are the components necessary to build a hybrid DNN classifier
# from the components being fed into the system thus far and through
# our chosen feed dictionary function.

#subjxverb = tf.contrib.layers.crossed_column(
#	[subj, verb],
#	hash_bucket_size=int(1e10),
#	combiner='sum')

#dobjxverb = tf.contrib.layers.crossed_column(
#	[dobj, verb],
#	hash_bucket_size=int(1e10),
#	combiner='sum')

#dobjxverbxobl1xobl2 = tf.contrib.layers.crossed_column(
#	[dobj, verb, obl1, obl2],
#	hash_bucket_size=int(1e10),
#	combiner='sum')

#wide_collumns = [subj, dobj, obl2]

#deep_columns = [
#	tf.contrib.layers.embedding_column(subjxverb, dimension=5),
#	tf.contrib.layers.embedding_column(dobjxverb, dimension=5),
#	tf.contrib.layers.embedding_column(dobjxverbxobl1xobl2, dimension=5),
#	tf.contrib.layers.embedding_column(verb, dimension=5)]

#The number of classes must always be N+1.
#m = tf.contrib.learn.DNNLinearCombinedClassifier(
#    model_dir=model_dir,
#    linear_feature_columns=wide_collumns,
#    dnn_feature_columns=deep_columns,
#    dnn_hidden_units=[100, 50],
#    n_classes=6)

#m.fit(input_fn=train_input_fn, steps=2000)

#results = m.evaluate(input_fn=eval_input_fn, steps=20)

#print(results)

Train_Data = '/Users/ZaqRosen/Documents/Corpora/mohler/train_data2.csv'
Test_Data = '/Users/ZaqRosen/Documents/Corpora/mohler/test_data2.csv'

def Training_Data_Builder(array, fileOut):
	with codecs.open(fileOut, 'a', 'utf-8') as csvfile:
		databuilder = csv.writer(csvfile, delimiter=',',
				quotechar='|',
				quoting=csv.QUOTE_MINIMAL)
		databuilder.writerow(array)
	csvfile.close()

