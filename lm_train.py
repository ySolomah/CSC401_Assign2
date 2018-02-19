from preprocess import *
import pickle
import os

def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM
	
	INPUTS:
	
    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
	
	LM			: (dictionary) a specialized language model
	
	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts
	
	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
	
	# TODO: Implement Function

    LM = {}
    LM['uni'] = {}
    LM['bi'] = {}

    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for filename in filenames:
            if(filename.endswith(language)):
                with open(data_dir + "/" + filename) as train_data:
                    for line in train_data:
                        line = preprocess(line, language)
                        words = line.split()
                        for word in line.split():
                            if word in LM['uni']:
                                LM['uni'][word] = LM['uni'][word] + 1
                            else:
                                LM['uni'][word] = 1

                        for i in range(len(words)-1):
                            if not words[i] in LM['bi']:
                                LM['bi'][words[i]] = {}
                            if words[i+1] in LM['bi'][words[i]]:
                                LM['bi'][words[i]][words[i+1]] = LM['bi'][words[i]][words[i+1]] + 1
                            else:
                                LM['bi'][words[i]][words[i+1]] = 1
                            prevWord = word
                        


    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(LM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''        
    for key, val in LM['uni'].items():
        print("Key: " + key + " Val: " + str(val))
    for key, val in LM['bi'].items():
        print("First word: " + key)
        for key2, val2 in val.items():
            print("     Second word: " + key2 + " with total count: " + str(val2))
    '''
    return LM

