import string
import pandas as pd
import pickle
import math
import TextPreprocess_indexTerm
import doc_vector
import Query_vector
import Cosin_Similarity  

def textproc_and_wight_document_antique():
    ds1 = pd.read_csv(r'C:/Users/ragha/.ir_datasets/antique/test/antique-collection.txt', sep='\t', names=['number','doc'])
    pd.set_option('display.max_colwidth',None)
    dictin = {}
    terms=[]
    for i in ds1.index:
        doc_proces = TextPreprocess_indexTerm.text_processing(ds1.iloc[i]['doc'])
        dictin.update({ds1.iloc[i]['number']: doc_proces})
        terms = TextPreprocess_indexTerm.remove_dublicated_words(doc_proces,terms)
    with open("antique_processed_data.pkl", "wb") as f:
        pickle.dump(dictin, f)
    with open("antique_term_index.pkl", "wb") as f:
        pickle.dump(terms, f)
    doc_Count = len(dictin)     # to get the number of docu
    vec_Dic1=doc_vector.compute_Weight(doc_Count,dictin,terms)
    with open("antique_docs_vector.pkl", "wb") as f:
        pickle.dump(vec_Dic1, f)


def textproc_and_wight_document_lifestyle():
    ds2 = pd.read_csv(r'C:/Users/ragha/.ir_datasets/lotte/lifestyle/dev/collection.tsv', sep='\t', names=['number','doc'])
    pd.set_option('display.max_colwidth',None)
    dictin2 = {}
    terms2=[]
    for i in ds2.index:
        doc_proces = TextPreprocess_indexTerm.text_processing(ds2.iloc[i]['doc']) #ارسال دوكمن واحدة ل تتفلتر 
        dictin2.update({ds2.iloc[i]['number']: doc_proces})  #  تخزين الدوكمنت الجديدة بعد الفلترة  كتيرم 
        terms2 = TextPreprocess_indexTerm.remove_dublicated_words(doc_proces,terms2)    #لحساب و تخزين ال اندكس تيرم هي بدون تكرار 
    with open("life_processed_data.pkl", "wb") as f:  #تخزين الدوكمنتات ك تيرم 
        pickle.dump(dictin2, f)
    with open("life_term_index.pkl", "wb") as f:   #تخزن الاندكس تيرم 
        pickle.dump(terms2, f)
    doc_Count = len(dictin2)     # to get the number of docu
    vec_Dic2=doc_vector.compute_Weight(doc_Count,dictin2,terms2)   #تابع التوزين تحويل ل فيكتور 
    with open("life_docs_vector.pkl", "wb") as f:  #تخزين الدوكمنتز ك فيكتور
        pickle.dump(vec_Dic2, f)


def all_queries_wight(queries,index_terms,idf,result_file_name):
    query_Weight = {}
    for i in queries.index:
        query_before = queries.iloc[i]['quer']
        proce_query = TextPreprocess_indexTerm.text_processing(query_before)    #تيكس بروسيس للكويري 
        query_vect = Query_vector.get_Weight_For_Query(proce_query,index_terms,idf) #توزين الكويري تحويلها الى فيكتور
        query_Weight.update({i: query_vect})
    with open(result_file_name, "wb") as f:  
        pickle.dump(query_Weight, f)

def all_queries_similarety(queries,vector_Dic,r_query_vector_fname,result_file_name):
    similarity = {}
    with open(r_query_vector_fname, "rb") as f: 
        query_Weight = pickle.load(f)
    for i in queries.index:
        denomi2=0
        print(i)
        for term in query_Weight[i]:
            denomi2 += query_Weight[i][term] * query_Weight[i][term]
        cos_similarity = Cosin_Similarity.similarity_Computation(denomi2,query_Weight[i],vector_Dic)   #مقارنة فيكتور الكويري مع فيكتورز الدوكمنتس
        similarity.update({i: cos_similarity})
    with open(result_file_name, "wb") as f: 
        pickle.dump(similarity, f)


def all_queries_prediction(queries,doc_Count,r_cosin_fname,result_file_name):
    doc_preduction={}
    with open(r_cosin_fname, "rb") as f:
        similarity = pickle.load(f)
    for i in queries.index:
        query_predict=Cosin_Similarity.prediction(similarity[i], doc_Count)       #ترتيب النتائج حسب درجة التشابه  
        doc_preduction.update({queries.iloc[i]['number']: query_predict})  #تخزين كل كويري مع النتائج يلي طلعت ببرنامجنا مشان التقييم
    with open(result_file_name, "wb") as f:  #تخزين الدوكمنتات ك تيرم 
        pickle.dump(doc_preduction, f)