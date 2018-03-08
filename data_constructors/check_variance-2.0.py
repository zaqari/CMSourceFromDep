import pandas as pd
import numpy as np

#####
###IMPORT DATA
#####
DNN_COLUMNS = ['tref',
	       'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2', 'labels']
DNN_COLUMNS6 = ['tref',
		'subj', 'dobj', 'syn', 'obl1', 'obl2', 'verb', 'labels']

#####
##TR & TE
#####

Train_Data = '/Users/ZaqRosen/Documents/Corpora/mohler/train_data11-9v4.csv'
Test_Data = '/Users/ZaqRosen/Documents/Corpora/mohler/test_data11-9v4.csv'


#####
##CREATE INPUTS
#####
df_train = pd.read_csv(Train_Data, names=DNN_COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(Test_Data, names=DNN_COLUMNS, skipinitialspace=True)
foo=df_train.values.tolist()+df_test.values.tolist()
df_all = pd.DataFrame(
	np.array(foo).reshape(-1, len(DNN_COLUMNS)),
	columns=DNN_COLUMNS
	)



#####
###FUNCTIONS
#####
def check(listvar, dfset=[df_train, df_test]):
	varis_train=[]
	varis_test=[]
	counter=0
	for it in listvar:
		varis_train.append(dfset[0][it].values.tolist())
		varis_test.append(dfset[1][it].values.tolist())
	b=list(zip(zip(*varis_train), dfset[0]['labels'].values.tolist()))
	c=list(zip(zip(*varis_test), dfset[1]['labels'].values.tolist()))
	for it in c:
		if it in b:
			counter+=1
	percent=counter/len(c)
	print(percent*100, '% overlap train & test')

def zero(listcheck, dfset=[df_train, df_test]):
	counter=0
	a=[]
	b=[]
	for it in listcheck:
		a.append(dfset[0][it].values.tolist())
		b.append(dfset[1][it].values.tolist())
	a2=list(zip(*a))
	b2=list(zip(*b))
	c=a2+b2
	for it in c:
		if all([ k =='0' for k in it]):
			counter+=1
	print(counter/len(c)*100, '% is nothing but zeroes.')



def check2_0(listcheck, dfk=df_all):
	check_values=[]
	label_overlap=[]
	maxlabels=int(0)
	minlabels=int(10000)
	for lab in listcheck:
		check_values.append(dfk[lab].values.tolist())
	a=list(zip(*check_values))
	a2=set(a)
	a3=list(zip(a, dfk['labels'].values.tolist()))
	for it in set(dfk['labels'].values.tolist()):
		b=[]
		for val in a3:
			if it in val[1]:
				for beluga in val[0]:
					b.append(beluga)
		#bc we're not interested in the item, but the
		# amount of overlap of unique items
		label_overlap.append(len(set(b)))
	for it in label_overlap:
		if it > maxlabels:
			maxlabels=it
		if it < minlabels:
			minlabels=it
	over_mean=[]
	for it in label_overlap:
		if it > int(sum(label_overlap)/len(label_overlap)):
			over_mean.append(it)
	print('=====')
	print(len(a2), ' number of total items in the category.')
	print(sum(label_overlap)/len(label_overlap), ' mean number of items per label.')
	#print(len(over_mean)/len(label_overlap)*100, '% of labels contain more items than the mean.')
	print('=====')


def stats(listin):
	check(listin)
	zero(listin)
	check2_0(listin)


























#####
###CALC INTER ANNOTATOR AGREEMENT
#####
import statistics as st
import xml.etree.ElementTree as ET

tree = ET.parse('/Users/ZaqRosen/Documents/Corpora/mohler/en_small.xml')
root = tree.getroot()

def calc_agreement(root=root):
	mem=[]
	for child in root.findall('LmInstance'):
		break_and_save=0
		metaphoricity=[]
		a = child.find('TextContent')
		annots = child.find('Annotations')
		for chi in annots:
			CMSOURCE = chi.find('CMSourceAnnotation')
			if CMSOURCE is None:
				continue
			else:
				collect_annots(annots, mem)
	annot_agreement=[]
	for it in mem:
		amia= int(it[0]/it[1])
		annot_agreement.append(amia)
	fin_score= sum(annot_agreement)/len(annot_agreement)
	print('The average annotator agreement is ', fin_score, '.')
	print('The variance present in the data is ', st.variance(annot_agreement))
				


def collect_annots(inlist, outlist, root=root):
	sourceTag=''
	num_win=0
	sortlist=[]
	by_score=[]
	#recount_by_score=0
	for chi in inlist:
		CMSOURCE = chi.findall('CMSourceAnnotation')
		for kid in CMSOURCE:
			sortlist.append(str(kid.attrib['sourceConcept']))
	for itm in sortlist:
		num=sortlist.count(itm)
		if num > num_win:
			sourceTag=itm
			num_win=num
	outlist.append([num_win, len(sortlist), sourceTag])
       
