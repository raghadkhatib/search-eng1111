import string
import pandas as pd
import pickle
import math

def evaluation_recall(docum_preduction,qrel,result_file_name):
    query_num = len(docum_preduction)
    with open(result_file_name, 'w') as f:
        for q in range(query_num):
            relevant=0
            act_set = set(qrel.iloc[q]['answer_pids'])
            predec_documents=docum_preduction[qrel.iloc[q]['qid']]
            for i in range(10):
                if predec_documents[i] in act_set:
                    relevant+=1
            f.write(f"recall@{i+1}_{q+1}={round(relevant/len(act_set),2)},,\n")

def evaluation_percision(docum_preduction,qrel,result_file_name):
    query_num = len(docum_preduction)
    with open(result_file_name, 'w') as f:
        for q in range(query_num):
            act_set = set(qrel.iloc[q]['answer_pids'])
            predec_documents=docum_preduction[qrel.iloc[q]['qid']]
            pred_set = set(predec_documents[:10])
            precision_at_10 = len(act_set & pred_set) /10       #truePosition/(truePosition+falsePosition)
            f.write(f"precision@{10}_{q+1}={round(precision_at_10,2)},,\n")


def evaluation_MAP(docum_preduction,qrel,result_file_name):
    query_num = len(docum_preduction)
    print(query_num)
    ap = []
    with open(result_file_name, 'w') as f:
        for q in range(query_num):
            ap_num = 0
            k=10
            num=0
            act_set = set(qrel.iloc[q]['answer_pids'])
            predec_documents=docum_preduction[qrel.iloc[q]['qid']]
            for x in range(k):
                pred_set = set(predec_documents[:x+1])
                precision_at_k = len(act_set & pred_set) / (x+1)   # calculate precision@k
                if predec_documents[x] in act_set:
                    num+=1
                    ap_num += precision_at_k
            if num==0:
                ap_q=0
            else:
                ap_q = ap_num / num        #avarege percision
            f.write(f"AP@{k}_{q+1}={round(ap_q,3)},,\n")
            ap.append(ap_q)
        map_at_k = sum(ap) / query_num
        f.write(f"mAP@{k}={round(map_at_k,3)},,\n")



def evaluation_Mrr(docum_preduction,qrel,result_file_name):
    query_num = len(docum_preduction)
    sum_rank=0
    with open(result_file_name, 'w') as f:
        for q in range(query_num):
            act_set = set(qrel.iloc[q]['answer_pids'])
            predec_documents=docum_preduction[qrel.iloc[q]['qid']]
            for x in range(len(predec_documents)):
                if predec_documents[x] in act_set:
                    sum_rank+=1/(x+1)
                    break
        mrr=(1/query_num)*sum_rank
        f.write(f"Mrr ={round(mrr,3)},,\n")