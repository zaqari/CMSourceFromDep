import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer

#####
##DOCUMENT DIRECTORIES
####
Train_Data = '/Users/ZaqRosen/Documents/Corpora/mohler/train_data11-9v3.csv'
Test_Data = '/Users/ZaqRosen/Documents/Corpora/mohler/test_data11-9v3.csv'
train_data_save='/Users/ZaqRosen/Documents/Corpora/mohler/train_data11-9v4.csv'
test_data_save='/Users/ZaqRosen/Documents/Corpora/mohler/test_data11-9v4.csv'


#####
##IMPORTING DOCUMENTS
#####
DNN_COLUMNS = ['LmTarget', 'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2', 'Labels']

df_tr = pd.read_csv(Train_Data, names=DNN_COLUMNS, skipinitialspace=True)
df_te = pd.read_csv(Test_Data, names=DNN_COLUMNS, skipinitialspace=True)

#####
##ESTABLISH LEMMATIZER
#####
lem=WordNetLemmatizer()


#####
##FUNCTIONS
#####
def create_lemma_data(dfk, outlist):
        for it in dfk.values.tolist():
                a=[]
                a.append([it[0],
                          lem.lemmatize(it[1]),
                          lem.lemmatize(it[2]),
                          it[3],
                          lem.lemmatize(it[4]),
                          lem.lemmatize(it[5]),
                          lem.lemmatize(it[6]),
                          it[7]
                          ])
                outlist.append(a)

#####
##IMPLEMENTATION
#####
train_out=[]
test_out=[]

create_lemma_data(df_tr, train_out)
create_lemma_data(df_te, test_out)

df_train=pd.DataFrame(np.array(train_out).reshape(-1, 8), columns=DNN_COLUMNS)
df_test=pd.DataFrame(np.array(test_out).reshape(-1, 8), columns=DNN_COLUMNS)

#####
##DATA OUT
#####
df_train.to_csv(train_data_save, sep=',', header=False, index=False, encoding='utf-8')
df_test.to_csv(test_data_save, sep=',', header=False, index=False, encoding='utf-8')
