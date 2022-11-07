import faiss,os
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import argparse


parser = argparse.ArgumentParser()
model_path = parser.add_argument('--model_path',required=True)
collection_folder = parser.add_argument('--collection_path',required=True)
collection_name = parser.add_argument('--collection_filename',required=True)

index= faiss.read_index(collection_folder+'/'+ collection_name+ 'faiss')
print('done')
data_folder = collection_folder
top_k = 1000
model = SentenceTransformer('output/' + model_path)
print('model loaded')

model.max_seq_length=350
queries_filepath = os.path.join(data_folder, 'queries.dev.small.tsv')

c=0
out=open('run_file','w')
qids=[]
queries=[]
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        qids.append(qid)
        queries.append(query)
xq = model.encode(queries)
print('queries encoded')
D, I = index.search(xq, top_k)  # search
rank=1
for q_id in range(len(I)):
    for rank in range(1,1001):
        out.write(qids[q_id]+'\t'+str( I[q_id][rank-1])+'\t'+str(rank)+ '\t' +str(D[q_id][rank-1])+'\n')

out.close()