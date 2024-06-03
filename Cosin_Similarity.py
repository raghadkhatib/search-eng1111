import string
import pandas as pd
import pickle
import math



def similarity_Computation(denomi2,query_Weight,vector_Dic):
    numerator = 0 
    denomi1 = 0 
    similarity = {}
    for document in vector_Dic:
        similarity.update({document: 0})
        for term in vector_Dic[document]:
            numerator += vector_Dic[document][term] * query_Weight[term]
        
        if numerator!=0:    # there is common term/s between query and document
            for term in vector_Dic[document]:
                denomi1 += vector_Dic[document][term] * vector_Dic[document][term]
            simi = numerator / (math.sqrt(denomi1) * math.sqrt(denomi2))
            similarity.update({document: simi})
            numerator = 0
            denomi1 = 0
    return (similarity)


def prediction(similarity, doc_count):
    dummy_List2= []
    count=0
    with open('output.txt', 'w') as f:
        ans = max(similarity, key=similarity.get)
        print(ans, "is the most relevant document")
        #print("ranking of the documents")
        for i in range(doc_count):
            ans = max(similarity, key=lambda x: similarity[x])
            #if similarity[ans] <=0.9:    ######  لازم نعدلو ل  0.5 او اعلى 
            if count >20:
                break
            dummy_List2.append(ans)
            count+=1
            #print(ans, "rank is", i+1)
            f.write(f"{ans},,")
            similarity.pop(ans)
    return dummy_List2