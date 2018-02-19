import re
import string


punct = string.punctuation
punct = punct.replace("'", "");
print(punct)

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 

    INPUTS:
    in_sentence : (string) the original sentence to be processed
    language	: (string) either 'e' (English) or 'f' (French)
                   Language of in_sentence
                   
    OUTPUT:
    out_sentence: (string) the modified sentence
    """
    # TODO: Implement Function
    out_sentence = in_sentence
    out_sentence = re.sub(r'([' + re.escape(punct) + r']+)', r' \1 ', out_sentence)
    if(language == 'f'):
        out_sentence = out_sentence.replace("l'", "l' ")
        out_sentence = out_sentence.replace("qu'", "qu' ")
        out_sentence = out_sentence.replace("puisqu'on", "puisqu' on")
        out_sentence = out_sentence.replace("puisqu'il", "puisqu' il")
        out_sentence = out_sentence.replace("lorsqu'on", "lorsqu' on")
        out_sentence = out_sentence.replace("lorsqu'il", "lorsqu' il")
        out_sentence = re.sub(r'([b-df-hj-np-tv-xz]){1}(\')', r'\1\2 ', out_sentence)
    out_sentence = re.sub(r'([ ]+)', " ", out_sentence)

    out_sentence = "NULL_START " + out_sentence + " NULL_END"
    return out_sentence

