import pandas as pd
import numpy as np

DNN_COLUMNS = ['LmTarget', 'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2', 'Labels']

Train_Data2 = '/Users/ZaqRosen/Documents/Corpora/mohler/train_data9.csv'
Test_Data2 = '/Users/ZaqRosen/Documents/Corpora/mohler/test_data9.csv'

df_train_in = pd.read_csv(Train_Data2, names=DNN_COLUMNS, skipinitialspace=True)
df_test_in = pd.read_csv(Test_Data2, names=DNN_COLUMNS, skipinitialspace=True)

def build_tref_data(dfk, outlist):
        for lista in dfk.values.tolist():
                out_tensor=[]
                a=lista[0]
                for word in lista[1:]:
                        if word != a:
                                out_tensor.append(word)
                        else:
                                out_tensor.append('TREF')
                outlist.append(out_tensor)

out_train=[]
build_tref_data(df_train_in, out_train)

out_test=[]
build_tref_data(df_test_in, out_test)

out_var=out_train+out_test

df_train_out=pd.DataFrame(np.array(out_train).reshape(-1, 7), columns=DNN_COLUMNS[1:])
df_test_out=pd.DataFrame(np.array(out_test).reshape(-1, 7), columns=DNN_COLUMNS[1:])
df_variance_out=pd.DataFrame(np.array(out_var).reshape(-1, 7), columns=DNN_COLUMNS[1:])


df_train_out.to_csv('/Users/ZaqRosen/Documents/Corpora/mohler/train_data9v2.csv', sep=',', encoding='utf-8', header=False, index=False)
df_test_out.to_csv('/Users/ZaqRosen/Documents/Corpora/mohler/test_data9v2.csv', sep=',', encoding='utf-8', header=False, index=False)
#df_variance_out.to_csv('/Users/ZaqRosen/Documents/Corpora/mohler/rev_sref4.csv', sep=',', encoding='utf-8', header=False, index=False)
