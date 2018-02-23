from preprocess import *
from lm_train import *
from math import log2

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0, verbose=False):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary
	
	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
	
	#TODO: Implement by student.



    likelihood = []
    splitSent = sentence.split()
    if(smoothing):
        for i in range(len(splitSent)-1):
            if(splitSent[i] in LM['uni'] and splitSent[i+1] in LM['bi'][splitSent[i]]):
                likelihood.append( 
                    log2(
                        (LM['bi'][splitSent[i]][splitSent[i+1]] + delta) /
                        (LM['uni'][splitSent[i]] + (delta * vocabSize))
                        )
                    )
            elif(splitSent[i] in LM['uni']):
                likelihood.append(
                    log2(
                        delta / (LM['uni'][splitSent[i]] + (delta * vocabSize))
                        )
                    )
            else:
                likelihood.append(
                    log2( 1 / vocabSize )
                    )
                        
    else:
        for i in range(len(splitSent)-1):
            if(splitSent[i] not in LM['uni'] or splitSent[i+1] not in LM['bi'][splitSent[i]]):
                if(splitSent[i] not in LM['uni']):
                    if(verbose):
                        print("Could not find: " + splitSent[i])
                elif(splitSent[i+1] not in LM['bi'][splitSent[i]]):
                    if(verbose):
                        print("Could not find bigram: " + splitSent[i] + " with: " + splitSent[i+1])
                likelihood.append(float('-inf'))
            else:
                likelihood.append(
                    log2(
                        (LM['bi'][splitSent[i]][splitSent[i+1]] + delta) /
                        (LM['uni'][splitSent[i]] + delta * vocabSize)
                    )
                )
                if(verbose):
                    print("\n")
                    print("Count of bigram: " + splitSent[i] + " " + splitSent[i+1])
                    print(LM['bi'][splitSent[i]][splitSent[i+1]])
                    print("Count of uni: " + splitSent[i])
                    print(LM['uni'][splitSent[i]])

    log_probability = sum(likelihood)
    return log_probability

    
def preplexity(LM, test_dir, language, smoothing = False, delta = 0):
    """
	Computes the preplexity of language model given a test corpus
	
	INPUT:
	
	LM : 		(dictionary) the language model trained by lm_train
	test_dir : 	(string) The top-level directory name containing data
				e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	language : `(string) either 'e' (English) or 'f' (French)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	"""
	
    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])
    
    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue
        
        opened_file = open(test_dir+ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language, add_null=True)
            tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)
            
            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
        opened_file.close()
    pp = 2**(-pp/N)
    return pp


'''
LM = [lm_train("/u/cs401/A2_SMT/data/Hansard/Training/", 'e', "langModelEnglish"), lm_train("/u/cs401/A2_SMT/data/Hansard/Training/", 'f', "langModelFrench")]

for delta, smoothing in zip([0.33, 0.66, 1.0, 0.0], [True, True, True, False]):
    for sent in ["The salad James said?", "The general said."]:
        prob = log_prob(preprocess(sent, 'e', add_null=True), LM[0], smoothing, delta, vocabSize=len(LM[0]['uni']))
        if(prob > float("-inf")):
            perplex = 2 ** (-prob/len(sent.split()))
        else:
            perplex = "INF"
        print("\nSentence: ", sent, " with delta: ", delta)
        print("Perplexity: ", perplex)
    for i, lang in enumerate(['english', 'french']):
        print("\n ", lang, " : ", " with delta ", delta)
        data_preplexity = preplexity(LM[i], "/u/cs401/A2_SMT/data/Hansard/Testing/", lang[0], smoothing, delta)
        print("Test preplexity: ", data_preplexity)

        data_preplexity = preplexity(LM[i], "/u/cs401/A2_SMT/data/Hansard/Training/", lang[0], smoothing, delta)
        print("Train preplexity: ", data_preplexity)
'''
