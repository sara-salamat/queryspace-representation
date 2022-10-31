import json, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch

if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")

#We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
bi_encoder = SentenceTransformer('output-s/Q2Q_30q_train_bi-encoder-mnrl-sentence-transformers-all-MiniLM-L6-v2-margin_3.0-2022-10-30_14-53-30')
bi_encoder.max_seq_length = 350     #Truncate long passages to 256 tokens

collection_filepath = 'Only_30q/Only_30q.tsv'
passages=[]

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        passages.append(passage)


print("Passages:", len(passages))


for x in [0,1,2,3,4,5,6,7]:
    corpus_embeddings = bi_encoder.encode(passages[x*1000000:(x+1)*1000000], convert_to_tensor=True, show_progress_bar=True,batch_size=128)
    torch.save(corpus_embeddings, 'Only_30q/lm_corpus_tensor_'+str(x+1)+'.pt')

corpus_embeddings = bi_encoder.encode(passages[8000000:], convert_to_tensor=True, show_progress_bar=True,batch_size=128)
torch.save(corpus_embeddings, 'Only_30q/lm_corpus_tensor_9.pt')


d = 384

index = faiss.IndexFlatL2(d)
print(index.is_trained)

for i in range(1,10):

    all_corpus=torch.load('Only_30q/lm_corpus_tensor_'+str(i)+'.pt', map_location=torch.device('cuda')).detach().cpu().numpy()
    index.add(all_corpus)
    print(i)

print(index.ntotal)
faiss.write_index(index, 'Only_30q/Only_30q_lm_faiss')
#os.system('python retrieve.py')