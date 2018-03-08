from random import randint
import pandas as pd
import numpy as np


#####
###INPUT DOCUMENT
#####

cols=['tref', 'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2', 'Labels']
df_in = pd.read_csv('/Users/ZaqRosen/Documents/Corpora/mohler/all_data11-9.csv', names=cols, skipinitialspace=True)

#####
###FUNCTIONS
#####

def create_train_test(dfk, outlist_train, outlist_test):
	for item in list(set(dfk['Labels'].values.tolist())):
		a = dfk.index[dfk['Labels'].isin([item])].tolist()
		alist = list(set(a))
		print('Initial length of ', str(item), str(len(alist)))
		for k in range(0, int(len(alist)*.85)):
			randIDX = randint(0, int(len(alist)-1))
			outlist_train.append(alist[randIDX])
			alist.remove(alist[randIDX])
		for itm in alist:
			outlist_test.append(itm)
		print('Number of items sent to test_data from ', str(item), ': ', str(len(alist))), ': ', str(len(alist))
	print('\n', len(list(set(dfk['Labels'].values.tolist()))), ' tags found')


def omit_notref(dfi, outlist):
	counter=0
	for array in dfi.values.tolist():
		a=array[0]
		a2=a.split()
		for word in a2:
			if word in array[1:]:
				outlist.append(array)
		else:
			counter+=1
		#The following is an edit. Prior it had just been
		#else:
		#        outlist.append(array)                              
	print('percent omitted: ', counter/len(dfi)*100, '\n')


def omit_notref_deprecated(dfi, outlist):
	counter=0
	for array in dfi.values.tolist():
		if array[0] not in array[1:]:
			counter+=1
		#The following is an edit. Prior it had just been
		#else:
		#        outlist.append(array)
		elif array[0] in array[1:3]:
			if array[5]=='0':
				outlist.append(array)
		else:
			outlist.append(array)                               
	print('percent omitted: ', counter/len(dfi)*100, '\n')


def filt_singular_examples(dftr, dfte, train_out, test_out):
	counter=0
	a= dftr['Labels'].values.tolist()
	b= dfte['Labels'].values.tolist()
	c= list(set(a) - set(b))
	a2= dftr.values.tolist()
	b2= dfte.values.tolist()
	for array in a2:
		if array[len(array)-1] not in c:
			train_out.append(array)
		else:
			counter+=1
	for array in b2:
		test_out.append(array)
	print('\n Number of unusable tags: ', len(c))
	print('Number of unusable examples: ', counter)


def enumerate_labels(dftr, dfte, train_out, test_out):
	dic=[]
	a=list(set(dftr['Labels'].values.tolist()))
	train=dftr.values.tolist()
	test=dfte.values.tolist()
	for it in a:
		dic.append((it, a.index(it)))
	for dec in dic:
		for array in train:
			if array[len(array)-1] == dec[0]:
				out=array[:len(array)-1]
				out.append(dec[1])
				train_out.append(out)
		for array in test:
			if array[len(array)-1] == dec[0]:
				out=array[:len(array)-1]
				out.append(dec[1])
				test_out.append(out)
	print('\n Completed enumerating values!')

#####
###IMPLEMENTATION
#####

#PHASE I
#In phase two, we take the train and test data and omit all arrays that do not
# contain the TREF argument--which is crucial to the SOURCE domain ID, since
# the SOURCE tag is for all accounts and purposes in the LCC dataset the
# thematic-role for TREF.
train_a=[]
omit_notref(df_in, train_a)
df_in2=pd.DataFrame(np.array(train_a).reshape(-1, 8), columns=cols)


#PHASE II
#This takes all the data that has been cleaned from PHASE I and splits it into
# train and test data.
df_train_idx=[]
df_test_idx=[]
create_train_test(df_in2, df_train_idx, df_test_idx)
df_tr1 = df_in.loc[df_train_idx]
df_te1 = df_in.loc[df_test_idx]



#PHASE III
#Gets rid of labels that aren't in both datasets.
train_b=[]
test_b=[]
filt_singular_examples(df_tr1, df_te1, train_b, test_b)



#PHASE IV
#Enumerates labels so that they can be used in the DNN implementation
df_tr2=pd.DataFrame(np.array(train_b).reshape(-1, 8), columns=cols)
df_te2=pd.DataFrame(np.array(test_b).reshape(-1, 8), columns=cols)

train_c=[]
test_c=[]
enumerate_labels(df_tr2, df_te2, train_c, test_c)

#PHASE V
#Outputs all data into train and test data
df_train=pd.DataFrame(np.array(train_c).reshape(-1, 8), columns=cols)
df_test=pd.DataFrame(np.array(test_c).reshape(-1, 8), columns=cols)

df_train.to_csv('/Users/ZaqRosen/Documents/Corpora/mohler/train_data11-9v3.csv', sep=',', header=False, index=False, encoding='utf-8')
df_test.to_csv('/Users/ZaqRosen/Documents/Corpora/mohler/test_data11-9v3.csv', sep=',', header=False, index=False, encoding='utf-8')

print('Length of train data doc: ', len(df_train))
print('Length of test data doc: ', len(df_test))


