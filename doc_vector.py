import string
import pandas as pd
import pickle
import math



def compute_Weight(doc_Count,dictinarys,index_terms):
    dummy_List = []
    # list for performing some operations and clearing them
    term_Freq = {} 
    # dictionary to store the name of the document and the weight as list
    idf = {}
    # dictionary to store the term and the inverse document frequency

    for i in index_terms:   #initialize value
        idf.update({i: 0})
    for key in dictinarys:
        term_Freq[key]={}
        for i in dictinarys[key]:
            term_Freq[key][i]=0
    
    for key in dictinarys:
        doc_words = len(dictinarys[key])
        for k in dictinarys[key]:
            if k not in dummy_List:
                dummy_List.append(k)
                idf[k] +=1  # NUM OF DUCUMENT CONTAIN THE TERM
            term_Freq[key][k] += 1    # value incremented by one if the term is found in the documents 
        dummy_List.clear()
        for i in term_Freq[key]:
            term_Freq[key][i] = math.log2(1+(term_Freq[key][i]/doc_words))   # term_frequency  
    
    
    for term in index_terms:              #IDF
        idf.update({term: math.log2(1+(doc_Count/idf[term]))})
    with open("newantique_all_idf.pkl", "wb") as f:
        pickle.dump(idf, f) 
    

    for i in dictinarys:
        for j in dictinarys[i]:
            term_Freq[i][j]=idf[j]*term_Freq[i][j]    #  CALC Tf_idf
    return term_Freq    #RETURN TF_IDF

