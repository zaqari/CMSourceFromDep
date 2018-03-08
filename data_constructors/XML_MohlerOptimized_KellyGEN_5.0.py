#Current system is operational. There was an issue with clausal recursions that
# had a 'WDT' unit as the head without additional verb or adjectival units. The
# problem led to a malformation of information packaged the lmtc, and has since
# been rectified via the addition of an if-gate dictating whether or not
# corpus_callosum() is called depending on the richness of data in lmtc (if not
# information was collected into the list, then corpus_callosum() is not called
# and the system, per usual, prints out an insult in a pseudo-british vernacular.
# Also: The current edits and focusing on the granular, target referent specific
# dependency geometry has finally made the [pfc] useful in terms of classification
# tasks--an exciting development.
###
#When running pvc() you need to manually enter in the target referents you're
# looking for in the list target_referents. You can use media_list to either
# capture the media source as indicated in your coprus for the line item, or
# to adhoc add a label column of some sort.

#imports Stanford’s Dependency Parser and sets up environment.
import pandas as pd
import numpy as np
from nltk.parse.stanford import StanfordDependencyParser as sparse
pathmodelsjar = '/Users/ZaqRosen/nltk_data/stanford-english-corenlp-2016-01-10-models.jar'
pathjar = '/Users/ZaqRosen/nltk_data/stanford-parser/stanford-parser.jar'
depparse = sparse(path_to_jar=pathjar, path_to_models_jar=pathmodelsjar)

#Update progress note:
print('Start from idea for the corpus all for media_type.start(written to be spoken).')
print('')
#corpus_file_selection = '/Users/ZaqRosen/Documents/Corpora/' + input('Which project, sir? ') + '/'

#
##
##Dependency to Constructions pipeline {Using Stanford's Dependency Parser}##
##
#
#This will be autodefined to 'nsubj' or 'dobj', but the point is to find
# any local deps that correlate to the sentence headers, essentially--those
# bits that more or less establish the relationship of the verb. The POS#
# .append protocol is effectively a fail-safe in case you end up analyzing
# a sub-structure lacking the 'nsubj' and 'dobj' components.
def litc(POS1, POS2, listy):
	for tuple in listy:
		POS1.append(tuple[2][0]) if 'nsubj' in tuple[1] else 0
		POS2.append(tuple[2][0]) if 'dobj' in tuple[1] else 0
	POS1.append('0')
	POS2.append('0')

#We know that WHAT the dep tree looks like is important--is it just an ADJ.P?
# or is the search item part of the entire clausal unit? This'll get to the
# bottom of these questions.
def sylvan(lmtc, pfc):
	for tuple in lmtc:
		pfc.append(tuple[1])

#Oblique elements appear to be vitally important in metaphor and constructional
# analyses. This block will thus derive the oblique elements' structure in
# simplified terms, or establish an adjective relationship of some sort to
# analyze.
def obl(oblique, lmtc, ventral_stream):
	oblique.append((0,0))
	for tuple in lmtc:
		oblique.append((tuple[2][0], tuple[0][0])) if 'JJ' in tuple[2][1] else 0
		if 'mod' in tuple[1]:
			ph_head = tuple[2]
			for tuple in ventral_stream:
				oblique.append((tuple[2][0], ph_head[0])) if tuple[1]=='case' and tuple[0]==ph_head else 0

def print_protocol( a, b, c, d):
	e = [a, b, c, d]
	for item in e:
		print('')
		print(item)
		print('=========')
			
#Where the magic happens, this links everything up into a coherent chunk that
# can then be passed to another function later in order to utilize the const.
# components, or print everything to a .csv via brocas().
###
#When running it for Thesis data collection, kill LmTarget by noting it out, as
# in corpus_callosum1 = [media, #LmTarget . . .
def corpus_callosum(ventral_stream, v1, lmtc, sentence, sourcetag, LmTarget,
		    #meta_tag,
		    outlistb, TEST):
	oblique=[]
	pfc=[]
	NSUBJ=[]
	DOBJ=[]
	sylvan(lmtc, pfc)
	litc(NSUBJ, DOBJ, lmtc)
	obl(oblique, lmtc, ventral_stream)
	#print_protocol(oblique, pfc, NSUBJ, DOBJ)
	for pp_phrase in oblique:
		corpus_callosum1 = [
			LmTarget,
			#meta_tag,
			NSUBJ[0],
			DOBJ[0],
			str(pfc).replace(',', ' ').replace('\"', '').replace('\'', ''),
			v1[0],
			pp_phrase[0],
			pp_phrase[1],
			sourcetag,
			#sentence
			]
		brocas(sentence, pfc, lmtc, corpus_callosum1, outlistb, TEST)
		corpus_callosum1 = []
		
#This little function simply packages and presents all the data collected in
# an easily interpretable chunk. If TEST=='build', it'll generate a .csv of
# the data the rest of the script collects.
def brocas(sentence, pfc, lmtc, arraya, outlistc, TEST='build'):
	print('==============')
	if TEST == 'test':
		print('Grammatical Structure:')
		print(pfc)
		print(' ')
		print('Lexical Items:')
		print(lmtc)
		print(' ')
	#Builds the training data sheet for you, if selected.
	if TEST == 'build':
		Training_Data_Builder(arraya, outlistc)
	print('Array: \n', arraya, '\n')
	print(sentence)
	print('==============')

#Occipital(), named after the occipital lobe, is the integrative tissue
# that triggers the whole process, and links all the components together.
def occipital(sentence, search1, mediaitem, mediaitem2,
	      #metatag,
	      outlista, TEST='non'):
	#Resets triggers and data failsafes
	v1 = ''
	lmtc = []
	#components from Stanford's Dependency parser to create dep. tree.
	try:
		res = depparse.raw_parse(sentence)
		dep = res.__next__()
		ventral_stream = list(dep.triples())
		for tuple in ventral_stream:
			if search1 in tuple[2][0]:
				v1=tuple[0]
			elif search1 in tuple[0][0] and 'cop' in tuple[1]:
				v1=tuple[0]
			elif search1 in tuple[0][0] and 'neg' in tuple[1]:
				v1=tuple[0]
		for tuple in ventral_stream:
			if tuple[0]==v1:
				lmtc.append(tuple)
		corpus_callosum(ventral_stream, v1, lmtc, sentence, mediaitem, mediaitem2,
				#metatag,
				outlista, TEST) if len(lmtc)>0 else print('Bloody hell, you dolt-minded crayon!')
	except OSError:
		print('Bloody hell, you dolt-minded crayon!')
	except AssertionError:
		print('Is it that hard to learn how to write a .csv file???')
	except UnicodeEncodeError:
		print('A pu ouela-ba n angre?')


#
##
##Data input and output##
##
#
#The following are .csv reader and output components, including the training
# data builder function (for DNN work later), and the input function for data
# collected in the form of a corpus.
import codecs
import csv
#export_file = '/Users/ZaqRosen/Desktop/' + input('What are you naming the output .csv?  ') + '.csv'

#takes data and saves it to a CSV to build training file.

def Training_Data_Builder(array1, outlistd):
	outlistd.append(array1)

def Training_Data_Builder_old(array):
	with codecs.open(export_file, 'a', 'utf-8') as csvfile:
		databuilder = csv.writer(csvfile, delimiter=',',
				quotechar='|',
				quoting=csv.QUOTE_MINIMAL)
		databuilder.writerow(array)
	csvfile.close()


#The following was added in the event that a problem arose concerning the
# dependency structure of a given collected piece. Test does not run any
# additional parsing tasks after raw text is passed to it, but instead
# prints the entire dep-tree for analysis and trouble shooting.
def test(sentence):
	res = depparse.raw_parse(sentence)
	dep = res.__next__()
	ventral_stream = list(dep.triples())
	for tuple in ventral_stream:
		print(tuple)

##
### XML Data Input and Output
##
#Component iterates through and recompiles the sentence into something that is
# actually parse-able. The next step is to set everything up in such a way as
# to pass all information to occipital(), with 'LmTarget' migrating to
# ('search'), some label (either if/not metaphor or the source) migrating to
# ('media'), and of course 'passable_sentence' migrating to ('sentence'). This
# should all occur where print(passable_sentence) is currently located.
###
###
#After some digging, I found that source domain annotations are embedded in
# the root <CMSourceAnnotations> as a tag in <CMSourceAnnotation>. This'll
# need to be pulled for Source Domain tagging.
# This can be accomplished via:
# root.find('CMSourceAnnotation').attrib['SourceConcept'] /
# for <CMSourceAnnotation> in <LmInstance>
# It's worth noting that the annotators put multiple, POSSIBLE source domains
# in some instances.

import xml.etree.ElementTree as ET

tree = ET.parse('/Users/ZaqRosen/Documents/Corpora/mohler/en_small.xml')
root = tree.getroot()

def LmTarget(tuply, searchy):
	if ' ' in tuply:
		head = tuply.split(' ')
		searchy.append(head[len(head)-1])
	else:
		searchy.append(tuply)

###DEPRECATED###
def eyes_deprecated(root=root):
	mem=[]
	former_mem_length=0
	for child in root.findall('LmInstance'):
		c_elements = []
		search = []
		media = []
		CMSource=''
		CMSourceCount=0
		LmTargetItem = ''
		passable_sentence =''
		metaphoricity = []
		a = child.find('TextContent')
		annots = child.find('Annotations')
		for child in a:
			if child.tag == 'Current':
				c = child
				for child in c:
					#print(child.tag)
					c_elements.append((child.tag, str(child.text)))
		for tuple in c_elements:
			passable_sentence+=tuple[1] if tuple[1] != 'None' else ''
			passable_sentence = passable_sentence.replace('\\', '').replace('}', '').replace('uc0u8232', '').replace('\'92', '\'').replace('a0', '').replace('\'93', '\"').replace('\'94', '\"').replace('\'96', ',').replace('\'97', ',').replace('f0fs24 ', '').replace('cf0 ', '').replace('< ', '').replace(' >', '').replace('\n', '').replace('\r', '').replace('\t', '').replace('0xc5', '').replace('0xc3', 'e').replace('0xc4', 'a')
			if tuple[0] == 'LmTarget':
			    LmTargetItem = tuple[1]
			    LmTarget(tuple[1], search)
		if child.tag == 'CMSourceAnnotations':
			media.append(str(root.find('CMSourceAnnotation').attrib['SourceConcept']))
		for it in media:
			a = media.count(it)
			if a > CMSourceCount:
				CMSource=it
		#get_meta_tags(annots, metaphoricity)
		occipital(passable_sentence,
			  search[len(search)-1],
			  CMSource,
			  LmTargetItem,
			  #metaphoricity[0],
			  mem,
			  'build')
		if len(mem) > int(former_mem_length +50):
			df_out=pd.DataFrame(np.array(mem).reshape(-1, len(mem[0])), columns=['tref', 'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2', 'source'])
			df_out.to_csv('/Users/ZaqRosen/Documents/Corpora/mohler/better_source_anots.csv', sep=',', header=False, index=False, encoding='utf-8')
			former_mem_length=len(mem)
		print(len(mem))


###REPAIRED XML DATA PULL
def sent_capture(text_layer, text_output_list):
	for child in text_layer:
		if child.tag == 'Current':
			c = child
			for child in c:
				#print(child.tag)
				text_output_list.append((child.tag, str(child.text)))

		
def eyes(root=root):
	mem=[]
	former_mem_length=0
	for child in root.findall('LmInstance'):
		break_and_save=0
		c_elements=[]
		search=[]
		media = []
		CMSource=''
		CMSourceCount=0
		LmTargetItem=''
		passable_sentence=''
		metaphoricity=[]
		a = child.find('TextContent')
		annots = child.find('Annotations')
		for chi in annots:
			CMSOURCE = chi.find('CMSourceAnnotation')
			if CMSOURCE is None:
				continue
			else:
				best_cmsource(annots, media)
				#print(media)
				sent_capture(a, c_elements)
				for tuple in c_elements:
					passable_sentence+=tuple[1] if tuple[1] != 'None' else ''
					passable_sentence = passable_sentence.replace('\\', '').replace('}', '').replace('uc0u8232', '').replace('\'92', '\'').replace('a0', '').replace('\'93', '\"').replace('\'94', '\"').replace('\'96', ',').replace('\'97', ',').replace('f0fs24 ', '').replace('cf0 ', '').replace('< ', '').replace(' >', '').replace('\n', '').replace('\r', '').replace('\t', '').replace('0xc5', '').replace('0xc3', 'e').replace('0xc4', 'a')
					if tuple[0] == 'LmTarget':
						LmTargetItem = tuple[1]
						LmTarget(tuple[1], search)
				if tuple[0] == 'LmTarget':
					LmTargetItem = tuple[1]
					LmTarget(tuple[1], search)
				#get_meta_tags(annots, metaphoricity)
				occipital(passable_sentence,
					  search[len(search)-1],
					  media[len(media)-1],
					  LmTargetItem,
					  #metaphoricity[0],
					  mem, 'build')
				print(len(mem))
		if len(mem) > int(former_mem_length +50):
			df_out=pd.DataFrame(np.array(mem).reshape(-1, len(mem[0])), columns=['tref', 'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2', 'source'])
			df_out.to_csv('/Users/ZaqRosen/Documents/Corpora/mohler/all_data9.csv', sep=',', header=False, index=False, encoding='utf-8')
			former_mem_length=len(mem)
	df_out=pd.DataFrame(np.array(mem).reshape(-1, len(mem[0])), columns=['tref', 'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2', 'source'])
	df_out.to_csv('/Users/ZaqRosen/Documents/Corpora/mohler/all_data9.csv', sep=',', header=False, index=False, encoding='utf-8')
			
			
		
				

#The following two contain the xml reading components necessary to complete the
# the task of retroactively obtaining the metaphoricity tags from the xml corpus
# document. find_tags checks if the sentence is correct, and get_meta_tags is
# the tag acquisition component.
def get_meta_tags(child_context, meta_score):
	for child in child_context:
		if child.tag == 'MetaphoricityAnnotations':
			metatags = child
			for child in metatags:
				if child.tag == 'MetaphoricityAnnotation':
					meta_score.append(child.attrib['score'])


def best_cmsource(inlist, outlist, root=root):
	sourceTag=''
	num_win=0
	sortlist=[]
	by_score=[]
	#recount_by_score=0
	for chi in inlist:
		CMSOURCE = chi.findall('CMSourceAnnotation')
		for kid in CMSOURCE:
			sortlist.append(str(kid.attrib['sourceConcept']))
			by_score.append(str(kid.attrib['score']))
	for itm in sortlist:
		num=sortlist.count(itm)
		if num > num_win:
			sourceTag=itm
			num_win=num
	#if num_win == 1:
	#TO FIX, HIT TAB BY 1 FOR THE FOLLOWING
        #        for it in by_score:
        #                if int(it[0]) > recount_by_score:
        #                        sourceTag=sortlist[by_score.index(it)]
        #                        recount_by_score=int(it[0])
	outlist.append(sourceTag)

def collect_annotations_for_kappa_score(outlist, root=root):
	counter=0
	for child in root.findall('LmInstance'):
		counter+=1
		tags=[]
		annots = child.find('Annotations')
		for chi in annots:
			CMSOURCE = chi.find('CMSourceAnnotations')
			if CMSOURCE is None:
				continue
			else:
				for kid in CMSOURCE:
					outlist.append([str(kid.attrib['sourceConcept']),
                                                        str(kid.attrib['annotatorID']),
                                                        counter])
