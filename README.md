# Learning Query-Space Document Representations for High-Recall Retrieval
Neural rankers have shown good performance on retrieval tasks recently but they do not necessarily work well on diverse range of queries. In this research, we propose a method that can improve the prformance of rankers significantly on MS-MARCO hard queries subsets. We present documents in query space and train rankers based on that.


## Train
We trained rankers on document representations with 5 queries, 10 queries, 20 queries and 30 queries. 

```
python train_bi.py -model model_name
```

## Results


