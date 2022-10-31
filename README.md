# Learning Query-Space Document Representations for High-Recall Retrieval
Neural rankers have shown good performance on retrieval tasks recently but they do not necessarily work well on diverse range of queries. In this research, we propose a method that can improve the prformance of rankers significantly on MS-MARCO hard queries subsets and MS MARCO dev set. Our proposed method suggests using a novel document representation using query space. We represent each document with k appended queries. We tested our approach with k=5, 10, 20, 30. The results are shown in [the Results section](##Results)


## Train
We trained rankers on document representations with 5 queries, 10 queries, 20 queries and 30 queries. 
To train a bi-encoder on original MS MARCO data, run the following command:
```
python train_bi.py
```
To train a bi-encoder on query-to-query (q2q) approach you first need to download our constructed corpus in which documents are replaced with a number of queries. To do so, download each from [here]([./Data/](https://www.dropbox.com/sh/ewori42b9e5d71b/AABM2IN9kveNlByY9zeRt_lRa?dl=0)) and the run the following command:
```
python train_bi_q2q.py
```

## Results
the following tables show the results of our approach. Our method has improves the performnce on MS MARCO dev set and chameleons.

