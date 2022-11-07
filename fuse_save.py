import pandas as pd
import argparse
from trectools import TrecEval, TrecQrel, TrecRun, fusion



parser = argparse.ArgumentParser()
qrels_path = parser.add_argument('--qrels',required=True)
run1_path = parser.add_argument('--run1',required=True)
run2_path = parser.add_argument('--run2',required=True)

## build trec format and save
qrels = TrecQrel(qrels_path)

def build_trec_run(run_file, trec_dataframe):
    trec_dataframe['qid'] = run_file[0]
    trec_dataframe['Q0'] = ['Q0']*len(run_file)
    trec_dataframe['docid'] = run_file[1]
    trec_dataframe['rank'] = run_file[2]
    trec_dataframe['score'] = [1000 - score for score in run_file[3].tolist()]
    trec_dataframe['tag'] = ['STANDARD']*len(run_file)




## fuse with original
original = TrecRun(run1_path)
fuse_with = TrecRun(run2_path) ## change

fused_run = fusion.reciprocal_rank_fusion([original,fuse_with]) 
fused_run.print_subset('fused_run', topics=fused_run.topics())

