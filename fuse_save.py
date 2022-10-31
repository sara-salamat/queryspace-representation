import pandas as pd
from trectools import TrecEval, TrecQrel, TrecRun, fusion

## build trec format and save
qrels = TrecQrel('qrels.dev.small.tsv')


def build_trec_run(run_file, trec_dataframe):
    trec_dataframe['qid'] = run_file[0]
    trec_dataframe['Q0'] = ['Q0']*len(run_file)
    trec_dataframe['docid'] = run_file[1]
    trec_dataframe['rank'] = run_file[2]
    trec_dataframe['score'] = [1000 - score for score in run_file[3].tolist()]
    trec_dataframe['tag'] = ['STANDARD']*len(run_file)




## fuse with original
original = TrecRun('run_original_trec')
fuse_with = TrecRun('run_only30q_trec') ## change

fused_run = fusion.reciprocal_rank_fusion([original,fuse_with]) 
fused_run.print_subset('q30_original', topics=fused_run.topics()) ## change

