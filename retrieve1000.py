import faiss,os
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util

index= faiss.read_index('Only_5q/Only_5q_lm_faiss')
print('done')
data_folder='Only_5q'
top_k = 1000
model = SentenceTransformer('output-s/Q2Q_5q_train_bi-encoder-mnrl-sentence-transformers-all-MiniLM-L6-v2-margin_3.0-2022-10-21_20-26-43')
print('model loaded')

model.max_seq_length=350
queries_filepath = os.path.join(data_folder, 'queries.dev.small.tsv')

c=0
out=open('run/out_5q_lm_1000_w_score','w')
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