import string
import pandas as pd
import pickle
import math

def get_Weight_For_Query(query,index_terms,idf):
    query_Freq = {} 
    for i in index_terms:
        if i not in query_Freq:
            query_Freq.update({i: 0})

    for val in query:
        if val in query_Freq:
            query_Freq[val] += 1
            
    for i in query_Freq:
        if len(query) >0:                 
            query_Freq[i] = math.log2(1+(query_Freq[i] / len(query)))*idf[i]
    return query_Freq 
