from fastapi import FastAPI
import json
import string
import pandas as pd
import pickle
import math
import TextPreprocess_indexTerm
import OffLine_ProcesData
import Evaluation
import Query_vector
import Cosin_Similarity
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

origins = [
    "http://localhost:5173",  # Frontend development server
    "http://192.168.192.107:8000"  # Add any other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#####response_model=result
@app.get("/search")  # 0 to life / >=1 to antique
def search(query: str, dataset: int):
    if dataset >=1:#dataset antique
        with open("antique_term_index.pkl", "rb") as f:   #read index term
            index_terms = pickle.load(f)
        
        with open("antique_docs_vector.pkl", "rb") as f:   #read document vectors
            vector_Dic = pickle.load(f)
        doc_Count = len(vector_Dic)

        with open("newantique_all_idf.pkl", "rb") as f:
            idf = pickle.load(f)
        
        data = pd.read_csv(r'C:/Users/USER/.ir_datasets/antique/collection.tsv', sep='\t', names=['number','doc'])
        pd.set_option('display.max_colwidth',None)
    
    else:   #dataset life
        with open("life00_alll_term_index.pkl", "rb") as f:   #read index term
            index_terms = pickle.load(f)
        
        with open("life00_alll_docs_vector.pkl", "rb") as f:   #read document vectors
            vector_Dic = pickle.load(f)
        doc_Count = len(vector_Dic)

        with open("newlife00_all_idf.pkl", "rb") as f:
            idf = pickle.load(f)
        
        data= pd.read_csv(r'C:/Users/USER/.ir_datasets/lotte/lotte_extracted/lotte/lifestyle/dev/collection.tsv', sep='\t', names=['number','doc'])
        pd.set_option('display.max_colwidth',None)

    result = {}
    global unique_term_proces
    with open("term_processed_text_processed_data.pkl", "rb") as f:
        unique_term_proces = pickle.load(f)
    query_before = query
    proce_query = TextPreprocess_indexTerm.text_processing(query_before)
    query_Weight = Query_vector.get_Weight_For_Query(proce_query,index_terms,idf)
    denomi2=0
    for term in query_Weight:
        denomi2 += query_Weight[term] * query_Weight[term]
    cos_similarity = Cosin_Similarity.similarity_Computation(denomi2,query_Weight,vector_Dic)
    query_predict=Cosin_Similarity.prediction(cos_similarity, doc_Count)
    for doc_number in query_predict:
        inde=data.isin([doc_number]).any(axis=1).idxmax()
        result.update({doc_number: data.iloc[inde]['doc']})
    print(data.iloc[inde]['doc'])
    return{json.dumps(result,separators=(",",":"))}

@app.get("/eva/antique/{evaluation_type}")
def eva_antique(evaluation_type:int):
    with open("newantique00_alll_prediction7.pkl", "rb") as f:   #read  result/ Retrived document for all query
        doc_preduction = pickle.load(f)
    qrel_data=pd.DataFrame(columns=['qid','answer_pids'])
    qrel = pd.read_csv(r'C:/Users/ragha/.ir_datasets/antique/antique-test.qrel', sep=' ', names=['qid','iter','docid','relev'])
    pd.set_option('display.max_colwidth',None)
    for i in qrel.index:      ## dataFormat to qrel to suit the function
        dummy_List3=[]
        if qrel.iloc[i]['relev']==1:   ## doc_id that has relevace=1 dont answer the question
            continue
        include=qrel_data.loc[qrel_data['qid']==qrel.iloc[i]['qid']]
        if(include.empty):
            dummy_List3.append(qrel.iloc[i]['docid'])
            qrel_data.loc[len(qrel_data.index)]=[qrel.iloc[i]['qid'], dummy_List3]
        else:
            dummy_List3=include.iloc[0]['answer_pids']
            dummy_List3.append(qrel.iloc[i]['docid'])
            qrel_data.at[include.index[0],'answer_pids']=dummy_List3
    if evaluation_type ==1:
        Evaluation.evaluation_MAP(doc_preduction,qrel_data,'antique_maptest.txt')
    elif evaluation_type==2:
        Evaluation.evaluation_Mrr(doc_preduction,qrel_data,'antique_mrr.txt')
    elif evaluation_type==3:
        Evaluation.evaluation_recall(doc_preduction,qrel_data,'antique_recall.txt')
    elif evaluation_type==4:
        Evaluation.evaluation_percision(doc_preduction,qrel_data,'antique_percision.txt')



@app.get("/eva/life/{evaluation_type}")
def eva_life(evaluation_type:int):
    qrel = pd.read_json('C:/Users/ragha/.ir_datasets/lotte/lifestyle/dev/qas.search.jsonl',lines=True)
    pd.set_option('display.max_colwidth',None)
    with open("newlife00_alll_prediction7.pkl", "rb") as f: #read  result/ Retrived document for all queries
        doc_preduction = pickle.load(f)
    if evaluation_type ==1:
        Evaluation.evaluation_MAP(doc_preduction,qrel,'lifestyle_maptest.txt')
    elif evaluation_type==2:
        Evaluation.evaluation_Mrr(doc_preduction,qrel,'lifestyle_mrr.txt')
    elif evaluation_type==3:
        Evaluation.evaluation_recall(doc_preduction,qrel,'lifestyle_recall.txt')
    elif evaluation_type==4:
        Evaluation.evaluation_percision(doc_preduction,qrel,'lifestyle_percision.txt')
    return{it}



@app.get("/offline/antique/{process_type}")
def offline_antique(process_type:int):

    if process_type ==1:#proces docu
        #step 1:applying text preproces for all document
        #step 2: get term index
        #step 3:transform document (from text preproces result as terms) to vector
        OffLine_ProcesData.textproc_and_wight_document_antique()
    
    elif process_type==2:#Query proces
        queries = pd.read_csv(r'C:/Users/ragha/.ir_datasets/antique/test/queries.txt', sep='\t', names=['number','quer']) 
        
        with open("antique_term_index.pkl", "rb") as f:   #read index term
            index_terms = pickle.load(f)
        
        with open("antique_docs_vector.pkl", "rb") as f:   #read document vectors
            vector_Dic = pickle.load(f)
        doc_Count = len(vector_Dic)

        with open("newantique_all_idf.pkl", "rb") as f:
            idf = pickle.load(f)
        #step 1: applying text preproces for all Query and transform them to vectors
        #step 2 :Calc Cosin similarity between query and document foe all queries
        #step 3: rank the result from similarity and return 20 document with max similarity
        OffLine_ProcesData.all_queries_wight(queries,index_terms,idf,"antique_query_wight.pkl")
        OffLine_ProcesData.all_queries_similarety(queries,vector_Dic,"antique_query_wight.pkl","antique_cosin_simi.pkl")
        OffLine_ProcesData.all_queries_prediction(queries,doc_Count,"antique_cosin_simi.pkl","antique_prediction7.pkl")



@app.get("/offline/life/{process_type}")
def offline_life(process_type:int):

    if process_type ==1:#proces docu
        OffLine_ProcesData.textproc_and_wight_document_lifestyle()
    
    elif process_type==2:#Query proces
        queries = pd.read_csv(r'C:/Users/ragha/.ir_datasets/lotte/lifestyle/dev/questions.search.tsv', sep='\t', names=['number','quer'])
        
        with open("life00_alll_term_index.pkl", "rb") as f:   #read index term
            index_terms = pickle.load(f)
        
        with open("life00_alll_docs_vector.pkl", "rb") as f:   #read document vectors
            vector_Dic = pickle.load(f)
        doc_Count = len(vector_Dic)

        with open("newlife00_all_idf.pkl", "rb") as f:
            idf = pickle.load(f)
        OffLine_ProcesData.all_queries_wight(queries,index_terms,idf,"newlife00_alll_query_wight.pkl")
        OffLine_ProcesData.all_queries_similarety(queries,vector_Dic,"newlife00_alll_query_wight.pkl","newlife00_allquer_cosin_simi.pkl")
        OffLine_ProcesData.all_queries_prediction(queries,doc_Count,"newlife00_allquer_cosin_simi.pkl","newlife00_alll_prediction7.pkl")
