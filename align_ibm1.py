from lm_train import *
from log_prob import *
from preprocess import *
from math import log2
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
    AM = {}
    
    # Read training data
    eng_sents, fre_sents = read_hansard(train_dir, num_sentences) 

    print("Read sents")
    
    # Initialize AM uniformly
    AM = initialize(eng_sents, fre_sents)
    
    print("FINDING AM")

    # Iterate between E and M steps
    for _ in range(max_iter):
        print("ITERATION OF AM")
        AM = em_step(AM, eng_sents, fre_sents)

    # Dump AM

    AM['SENTSTART'] = {}
    AM['SENTSTART']['SENTSTART'] = 1
    AM['SENTEND'] = {}
    AM['SENTEND']['SENTEND'] = 1


    with open(fn_AM+'.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""
    eng_sentences = []
    fre_sentences = []
    num_parsed = 0
    for (dirpath, dirnames, filenames) in os.walk(train_dir):
        for filename in filenames:
            if(filename.endswith('e')):
                file_prefix = filename[0:len(filename)-1]
                with open(train_dir + "/" + file_prefix + "e") as train_eng, open(train_dir + "/" + file_prefix + "f") as train_fre:
                     for eng_line, fre_line in zip(train_eng, train_fre):
                         eng_sent = preprocess(eng_line, 'e')
                         fre_sent = preprocess(fre_line, 'f')
                         eng_sentences.append(eng_sent)
                         fre_sentences.append(fre_sent)
                         num_parsed = num_parsed + 1
                         if(num_parsed == num_sentences):
                             print("Breaking, reached number of sents")
                             break
                if(num_parsed == num_sentences):
                    print("Breaking two")
                    break
    return(eng_sentences, fre_sentences)
                    

def initialize(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
    AM = {}

    for eng_line, fre_line in zip(eng, fre):
        for eng_word in eng_line.split():
            for fre_word in fre_line.split():
                if(eng_word not in AM):
                    AM[eng_word] = {}
                #print("Init: " + eng_word + " with: " + fre_word)
                AM[eng_word][fre_word] = 1

    for eng_key in AM.keys():
        for fre_key in AM[eng_key].keys():
            AM[eng_key][fre_key] = 1 / len(AM[eng_key])

    return(AM)
    
def em_step(t, eng, fre):
    """
    One step in the EM algorithm.
    Follows the pseudo-code given in the tutorial slides.
    """
    AM = t
    t_count = {}
    total = {}
    for eng_sent, fre_sent in zip(eng, fre):
        fre_words_done = []
        for fre_word in fre_sent.split():
            if(fre_word not in fre_words_done):
                fre_words_done.append(fre_word)
                denom_c = 0
                eng_words_done = []
                eng_words_done_2 = []
                for eng_word in eng_sent.split():
                    if(eng_word not in eng_words_done):
                        eng_words_done.append(eng_word)
                        denom_c = denom_c + AM[eng_word][fre_word] * fre_sent.count(fre_word)
                    if(eng_word not in eng_words_done_2):
                        eng_words_done_2.append(eng_word)
                        if(eng_word not in t_count):
                            t_count[eng_word] = {}
                        if(fre_word not in t_count[eng_word]):
                            t_count[eng_word][fre_word] = 0
                        partial_term = AM[eng_word][fre_word] * fre_sent.count(fre_word) * eng_sent.count(eng_word) / denom_c
                        t_count[eng_word][fre_word] = t_count[eng_word][fre_word] + partial_term 
                        if(eng_word not in total):
                            total[eng_word] = 0
                        total[eng_word] = total[eng_word] + partial_term
    
    for eng_key in total.keys():
        for fre_key in t_count[eng_key].keys():
            AM[eng_key][fre_key] = t_count[eng_key][fre_key] / total[eng_key]

    return(AM)


#AM = align_ibm1("/u/cs401/A2_SMT/data/Hansard/Training/", 1000, 40, "/h/u6/c7/05/solomahy/CSC401A2/CSC401_Assign2/am")
