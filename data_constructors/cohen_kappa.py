import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.metrics import cohen_kappa_score as kappa

tree = ET.parse('/Users/ZaqRosen/Documents/Corpora/mohler/en_small.xml')
root = tree.getroot()

def collect_annotations_for_kappa_score(outlist, root=root):
	counter=0
	for child in root.findall('LmInstance'):
		counter+=1
		tags=[]
		annots = child.find('Annotations')
		for chi in annots:
			CMSOURCE = chi.findall('CMSourceAnnotation')
			if CMSOURCE is None:
				continue
			else:
				for kid in CMSOURCE:
					outlist.append([str(kid.attrib['sourceConcept']),
							str(kid.attrib['annotatorID']),
							counter])
	return counter

def tags_to_ints(dfk, label='tag'):
	outlist=[]
	dic=list(set(dfk[label]))
	for itm in dic:
		for array in dfk.values.tolist():
			if array[0] == itm:
				outlist.append([dic.index(itm), array[1], array[2]])
	return outlist

def split_annotators(dfk, outlist, rangy):
	for name in set(dfk['ID'].values.tolist()):
		fin_list=[]
		notes=[]
		indeces=[]
		for itm in dfk.values.tolist():
			if name == itm[1]:
				notes.append(itm[0])
				indeces.append(int(itm[2]))
		for k in range(rangy):
			if k not in indeces:
				fin_list.append('nan')
			else:
				fin_list.append(notes[indeces.index(k)])
		outlist.append(fin_list)
	labels=[]
	for it in outlist:
		labels.append(str(outlist.index(it)))
	return labels


def kappas(inlist):
	kaps=[]
	for itm in inlist:
		if inlist.index(itm) != int(len(inlist)-1):
			for itk in inlist[inlist.index(itm)+1:]:
				a_values=[]
				b_values=[]
				checklist=list(zip(itm, itk))
				for it in checklist:
					if 'nan' not in it:
						a_values.append(it[0])
						b_values.append(it[1])
				kaps.append([kappa(a_values, b_values), len(a_values)])
	return kaps
					


collection=[]
x=collect_annotations_for_kappa_score(collection)
df_collect=pd.DataFrame(np.array(collection).reshape(-1, 3), columns=['tag', 'ID', 'sent'])
y=tags_to_ints(df_collect)

df_collect2=pd.DataFrame(np.array(y).reshape(-1, 3), columns=['tag', 'ID', 'sent'])


annots_rectified=[]
z=split_annotators(df_collect2, annots_rectified, x)


z=kappas(annots_rectified)
