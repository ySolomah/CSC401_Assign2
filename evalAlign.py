import random
import numpy as np
from math import log
from preprocess import *
from lm_train import *
from log_prob import *
from align_ibm1 import *
from decode import * 
import math

results = open("/h/u6/c7/05/solomahy/CSC401A2/CSC401_Assign2/Task5.txt", 'w')

smoothBlue = True

LM = lm_train("/u/cs401/A2_SMT/data/Hansard/Training/", "e", "~")
for num_sent in [1000, 10000, 15000, 30000]:
    AM = align_ibm1("/u/cs401/A2_SMT/data/Hansard/Training/", num_sent, 40, "~") 
    for n in [1, 2, 3]:
        print("\n\n-----------------")
        print("EVALUATING WITH: ", num_sent, " TOTAL TRAINING SENTENCES")
        print("VALUE OF BLEU SCORE N: ", n)

        results.write("\n\n\n-----------------")
        results.write("\nEVALUATING WITH: " + str(num_sent) + " TOTAL TRAINING SENTENCES")
        results.write("\nVALUE OF BLEU SCORE N: " + str(n) + "\n")

        with open("/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f") as freFile, \
        open("/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e") as groundTruth, \
        open("/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e") as groundTruthHansard:
            accum = 0.0
            for freLine, engTruth1, engTruth2 in zip(freFile, groundTruth, groundTruthHansard):
                engSentence = decode(preprocess(freLine, 'f', add_null=True), LM, AM)
                engTruth1 = preprocess(engTruth1, 'e', add_null=True)
                engTruth2 = preprocess(engTruth2, 'e', add_null=True)
                dist1 = abs(len(engTruth1.split()) - len(engSentence.split()))
                dist2 = abs(len(engTruth2.split()) - len(engSentence.split()))
                if( dist1 > dist2 ):
                    bleuSent = engTruth2
                    dist = dist2
                else:
                    bleuSent = engTruth1
                    dist = dist1
                BP = min(1, math.exp(1 - (len(bleuSent.split())/len(engSentence.split()))))

                engSentence = engSentence.split()
                engTruth1 = engTruth1.split()
                engTruth2 = engTruth2.split()

                engSentenceTemp = ""
                for word in range(len(engSentence)-2):
                    engSentenceTemp = engSentenceTemp + engSentence[word+1] + " "
                engSentence = engSentenceTemp
                engSentence = engSentence.split()

                engTruthTemp = ""
                for word in range(len(engTruth1)-2):
                    engTruthTemp = engTruthTemp + engTruth1[word+1] + " "
                engTruth1 = engTruthTemp

                engTruthTemp = ""
                for word in range(len(engTruth2)-2):
                    engTruthTemp = engTruthTemp + engTruth2[word+1] + " "
                engTruth2 = engTruthTemp

                allP = []
                if(n == 1):
                    i_range = [1]
                if(n == 2):
                    i_range = [1, 2]
                if(n == 3):
                    i_range = [1, 2, 3]

                for i in i_range:
                    #print("i: ", i)
                    p = 0.0
                    for j in range(len(engSentence) - i):
                        if(i == 1):
                            checkSent = engSentence[j]
                        if(i == 2):
                            checkSent = engSentence[j] + " " + engSentence[j+1]
                        if(i == 3):
                            checkSent = engSentence[j] + " " + engSentence[j+1] + " " + engSentence[j+2]
    
                        if(checkSent in engTruth1 or checkSent in engTruth2):
                            p = p + 1
                    if(smoothBlue):
                        allP.append( (p + (0.02 * (len(engSentence) - i))) / (1.02 * (len(engSentence) - i)) ) 
                    else:
                        allP.append( p / (len(engSentence) - i))

                bleuScore = BP
                for p_score in allP:
                    bleuScore = bleuScore * (p_score ** (1/n))
                print("\n\n\nInput french: '" + freLine.strip() + "'")
                print("Output english: '" + engSentenceTemp + "'")
                print("Ground truth (google): '" + engTruth1 + "'")
                print("Ground truth (hansard): '" + engTruth2 + "'")
                print("BLEU score: ", bleuScore)
                results.write("\nBLEU score: " + str(bleuScore))
                accum = accum + bleuScore
            results.write("\n Average score: " + str(accum / 25))
