import tempfile
import tensorflow as tf
import pandas as pd
import numpy as np

###LOGGING SET-UP###
tf.logging.set_verbosity(tf.logging.INFO)

###SMALL DATASETS, NO INCLUSION OF ARG=XREF###
train_data = '/Users/ZaqRosen/Documents/Corpora/mohler/train_data9v4.csv'
test_data = '/Users/ZaqRosen/Documents/Corpora/mohler/test_data9v4.csv'

feature_columns = ['LmTarget',
        'subj', 'dobj', 'syn', 'obl1', 'obl2', 'verb']
DNN_COLUMNS = ['LmTarget',
        'subj', 'dobj', 'syn', 'obl1', 'obl2', 'verb', 'Labels']

###DATA INPUT BUILDER###
df_tr = pd.read_csv(train_data, names=DNN_COLUMNS, skipinitialspace=True)
df_te = pd.read_csv(test_data, names=DNN_COLUMNS, skipinitialspace=True)

df_train = pd.concat([df_tr[feature_columns], df_tr['Labels'].astype(int)], axis=1, join='inner')
df_test = pd.concat([df_te[feature_columns], df_te['Labels'].astype(int)], axis=1, join='inner')

###PRINT LEN OF TEST RUNS###
print(len(df_train)*50, ' iterations to run \n')

###SETTING CLASSES AND THE NUMBER OF DNN VARIABLES###
#It's not good enough to simply set the number of classes you might have. We
# really want to have some sort of automated and accurate way of rendering the
# number of classes we're looking at so we can avoid the mistake earlier of
# SAYING we have x-classes, and really having less than that (and thus having
# a loss-value of +5.4 . . . ).
##NOTE: the number of classes must always be n+1
nClasses= int(len(set(df_train['Labels'].values.tolist())))
#all_classes = list(set(df_train['Labels'].values.tolist()))
#S_V_hash_size = int(len(df_train))

###DNN SETUP###
#Herein lies the powerhouse of this system. The following establishes a DNN
# in which the edited information above is passed into a network classifier
# and is then classified to what its 'LmSource' value ought to be.

##NOTE ON GENERATING SOURCE DOMAIN CLASSES (9.19.17)
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

CATEGORICAL_COLUMNS = [#'LmTarget',
                       'subj', 'dobj', 'syn', 'obl1', 'obl2', 'verb']
LABELS_COLUMN = ['Labels']

def input_fn(df):
        # Creates a dictionary mapping from each continuous feature column name (k) to
        # the values of that column stored in a constant Tensor.
        #continuous_cols = {k: tf.constant(df[k].values)
                                #  for k in CONTINUOUS_COLUMNS}
        # Creates a dictionary mapping from each categorical feature column name (k)
        # to the values of that column stored in a tf.SparseTensor.
        categorical_cols = {k: tf.SparseTensor(
                indices=[[i, 0] for i in range(df[k].size)],
                values=df[k].values,
                dense_shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
        # Merges the two dictionaries into one.
        feature_cols = dict(categorical_cols.items())
        # Converts the label column into a constant Tensor.
        label = tf.constant(df['Labels'].values.astype(int))
        # Returns the feature columns and the label.
        return feature_cols, label

def train_input_fn():
        return input_fn(df_train)

def eval_input_fn():
        return input_fn(df_test)

#tref = tf.contrib.layers.sparse_column_with_hash_bucket("LmTarget", hash_bucket_size=int(15000))

subj = tf.contrib.layers.sparse_column_with_hash_bucket("subj", hash_bucket_size=int(15000))

syn = tf.contrib.layers.sparse_column_with_hash_bucket("syn", hash_bucket_size=int(15000))

verb = tf.contrib.layers.sparse_column_with_hash_bucket("verb", hash_bucket_size=int(15000))

obl1 = tf.contrib.layers.sparse_column_with_hash_bucket("obl1", hash_bucket_size=int(15000))

obl2 = tf.contrib.layers.sparse_column_with_hash_bucket("obl2", hash_bucket_size=int(15000))

dobj = tf.contrib.layers.sparse_column_with_hash_bucket("dobj", hash_bucket_size=int(15000))

#This is set up to be realized as either the tempfile doc, or
# /tmp/KellyGEN
model_dir = '/tmp/klstm'  #tempfile.mkdtemp()


#####
###CREATE CROSSED COLUMNS
#####

#0.501
subjxverb = tf.contrib.layers.crossed_column(
	[subj, verb],
	hash_bucket_size=int(1e6),
	combiner='sum')

subjxdobj = tf.contrib.layers.crossed_column(
	[subj, dobj],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.489
dobjxverb = tf.contrib.layers.crossed_column(
	[dobj, verb],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.579
dobjxobl1xobl2 = tf.contrib.layers.crossed_column(
	[dobj, obl1, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

###MOVEMENT FROM VERB FOCAL TO THETA-ROLE FOCAL MODEL
#The following are new features that could be used in the DNN in order to make
# decisions pertaining to categorization of examples. The percentage above each
# one indicates the total variance across the entiretyof the text. Ideally, DNN
# featues will be around 40-50%, with wide features being low (~10%). To date,
# subj and obj are the only features with that low a variance. The real paradigm
# shift lies in the realization that verb semantics are in fact less useful for
# classification than previoulsy thought with the temporal, toy-data set. From
# here on, we'll be focusing on what would traditionally be the theta-roles
# for categorical differentiation.

#0.442
obl1xobl2 = tf.contrib.layers.crossed_column(
	[obl1, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.449
synxsubj = tf.contrib.layers.crossed_column(
	[syn, subj],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.424
synxdobj = tf.contrib.layers.crossed_column(
	[syn, dobj],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.587
synxobl1 = tf.contrib.layers.crossed_column(
	[syn, obl1],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.739, but very low categorical intersection.
#trefxsyn = tf.contrib.layers.crossed_column(
#	[syn, tref],
#	hash_bucket_size=int(1e6),
#	combiner='sum')

#0.595
synxobl2 = tf.contrib.layers.crossed_column(
	[syn, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.245
subjxobl1 = tf.contrib.layers.crossed_column(
	[subj, obl1],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.331
subjxobl2 = tf.contrib.layers.crossed_column(
	[subj, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.204
dobjxobl1 = tf.contrib.layers.crossed_column(
	[dobj, obl1],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.307
dobjxobl2 = tf.contrib.layers.crossed_column(
	[dobj, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

#0.369
#trefxdobj = tf.contrib.layers.crossed_column(
#	[dobj, tref],
#	hash_bucket_size=int(1e6),
#	combiner='sum')

#0.416
#trefxsubj = tf.contrib.layers.crossed_column(
#	[subj, tref],
#	hash_bucket_size=int(1e6),
#	combiner='sum')

#0.496
#trefxobl1 = tf.contrib.layers.crossed_column(
#	[obl1, tref],
#	hash_bucket_size=int(1e6),
#	combiner='sum')

#0.479
#trefxobl2 = tf.contrib.layers.crossed_column(
#	[obl2, tref],
#	hash_bucket_size=int(1e6),
#	combiner='sum')

#0.531
#trefxverb = tf.contrib.layers.crossed_column(
#	[verb, tref],
#	hash_bucket_size=int(1e6),
#	combiner='sum')

#####
###CREATE EMBEDDING COLUMNS FROM CROSSED COLS
#####
ssubjxverb = tf.contrib.layers.embedding_column(subjxverb, dimension=5)

###VALIDATION MONITORING:
#The following are the metrics and set-up for the validation monitor such that
# we can track the progress of the system overtime using Tensorboard.

###OPENING TENSORBOARD:
#Now that we have a way of monitoring everything, we can use TENSORBOARD to
# visually track the progress of KellYGEN as it iterates over the training data.
# To do this, type the following into the terminal:
#             tensorboard --logdir=/tmp/KellyGEN/
# after doing this, it should prompt you with a port number in yellow-ish.
# Open up any browser and type in http://0.0.0.0:<port number> and it'll open
# tensorboard.

###PAST HIDDEN UNIT LAYER MODELS
#Just in case, here are some prior models. I'm confident that, given enough
# epochs, the one directly below this note would prove the most accurate, as
# it should replicate the number of possible connections between each piece of
# data. However, this WOULD take weeks of training to finish, since the number
# of training periods is an exponential function times the number of examples,
# ergo N^N * len(df_train).
#               #HU1# [100, 50]                         .72 after mods
#               #HU2# [54, 54, 54, 54, 54, 54, 54]      untested
#               #HU4# [80, 50]                          .63
#               #HU4# [61, 100]                         .63xV1, .72xEX3

##PREVIOUS wide_columns VALUES
#Not every iteration was better than the others . . . the wide columns indicated
# here coincide with the highest accuracy yield thus far. It's worth noting that
# using V1 as wide_columns is much faster than EX1 & EX2.
#       #V1#    [subj, dobj, obl2]
#       #EX1#   [dobjxverb, subjxverb]
#       #EX2#   [dobjxverb, subjxverb, trefxsyn]
#       #EX3#   []

context_columns = [
                tf.contrib.layers.embedding_column(subjxverb, dimension=5),
                tf.contrib.layers.embedding_column(dobjxverb, dimension=5),
                ]

##PREVIOUS deep_columns VALUES
#Similar to the wide_columns, deep_columns are integral to making things work.
# Here are a couple of previous iterations of this set-up.
#       #V1#    [100, 50]
#               tf.contrib.layers.embedding_column(dobjxobl1xobl2, dimension=5), tf.contrib.layers.embedding_column(subjxverb, dimension=5), tf.contrib.layers.embedding_column(dobjxverb, dimension=5), tf.contrib.layers.embedding_column(dobjxobl1, dimension=5), tf.contrib.layers.embedding_column(subjxobl1, dimension=5), tf.contrib.layers.embedding_column(obl1xobl2, dimension=5)
#       #EX1#   tf.contrib.layers.embedding_column(dobjxobl1, dimension=5), tf.contrib.layers.embedding_column(subjxobl1, dimension=5), tf.contrib.layers.embedding_column(obl1xobl2, dimension=5)
#       #EX2#   WITH V1 wide_columns#   .62 acc
#               tf.contrib.layers.embedding_column(dobjxobl1xobl2, dimension=5), tf.contrib.layers.embedding_column(subjxverb, dimension=5), tf.contrib.layers.embedding_column(dobjxverb, dimension=5), tf.contrib.layers.embedding_column(dobjxobl1, dimension=5), tf.contrib.layers.embedding_column(subjxobl1, dimension=5), tf.contrib.layers.embedding_column(obl1xobl2, dimension=5), tf.contrib.layers.embedding_column(trefxdobj, dimension=5), tf.contrib.layers.embedding_column(trefxsubj, dimension=5), tf.contrib.layers.embedding_column(trefxobl2, dimension=5)
#       #EX3#   WITH [] wide_columns    .7219 acc
#               tf.contrib.layers.embedding_column(dobjxobl1xobl2, dimension=5), tf.contrib.layers.embedding_column(subjxverb, dimension=5), tf.contrib.layers.embedding_column(dobjxverb, dimension=5), tf.contrib.layers.embedding_column(dobjxobl1, dimension=5), tf.contrib.layers.embedding_column(subjxobl1, dimension=5), tf.contrib.layers.embedding_column(obl1xobl2, dimension=5), tf.contrib.layers.embedding_column(trefxdobj, dimension=5), tf.contrib.layers.embedding_column(trefxsubj, dimension=5), tf.contrib.layers.embedding_column(trefxsyn, dimension=5), tf.contrib.layers.embedding_column(trefxobl2, dimension=5)
#       #EX4#   WITH [] wide_columns    .510 acc
#               tf.contrib.layers.embedding_column(subj, dimension=5), tf.contrib.layers.embedding_column(dobj, dimension=5), tf.contrib.layers.embedding_column(verb, dimension=5), tf.contrib.layers.embedding_column(obl1, dimension=5), tf.contrib.layers.embedding_column(obl2, dimension=5)

seq_columns = [
        #first three are possibly omittable . . . need to try one last time with
        # [100, 50] HUnits.
        tf.contrib.layers.embedding_column(verb, dimension=12),
        tf.contrib.layers.embedding_column(subj, dimension=12),
        tf.contrib.layers.embedding_column(dobj, dimension=12),
        tf.contrib.layers.embedding_column(obl1, dimension=12),
        tf.contrib.layers.embedding_column(obl2, dimension=12),
        tf.contrib.layers.embedding_column(syn, dimension=12),
        ]


validation_metrics = {
        #The below is the best bet to run accuracy in here, but we need to
        # somehow run labels as a full-blown tensor of some sort.
#        'acc': m.evaluate(input_fn=eval_input_fn, steps=2)
#        'accuracy': tf.contrib.learn.MetricSpec(
#                metric_fn=tf.contrib.metrics.streaming_accuracy,
#                prediction_key=tf.contrib.learn.PredictionKey.
#                CLASSES),
#        'precision':tf.contrib.learn.MetricSpec(
#                metric_fn=tf.contrib.metrics.streaming_precision,
#                prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
#        'recall': tf.contrib.learn.MetricSpec(
#                metric_fn=tf.contrib.metrics.streaming_recall,
#                prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
        }

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        df_test[feature_columns].values,
        df_test['Labels'].values,
        every_n_steps=50,
        #metrics=validation_metrics
        )

m = tf.contrib.learn.DynamicRnnEstimator(
        problem_type=tf.contrib.learn.ProblemType.CLASSIFICATION,
        prediction_type=1,
        sequence_feature_columns=seq_columns,
        num_classes=nClasses,
        num_units=[77],
        cell_type='lstm',
        predict_probabilities=True,
        dropout_keep_probabilities=[.4, .4],
        model_dir=model_dir
        )

pred_m = tf.contrib.learn.DynamicRnnEstimator(
        problem_type=tf.contrib.learn.ProblemType.CLASSIFICATION,
        prediction_type=1,
        sequence_feature_columns=seq_columns,
        num_classes=nClasses,
        num_units=[77],
        cell_type='lstm',
        predict_probabilities=True,
        model_dir=model_dir
        )

m.fit(input_fn=train_input_fn, steps=int(len(df_train)*20))

results = pred_m.evaluate(input_fn=eval_input_fn, steps=10)

#print(results)
