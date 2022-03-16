# coding: utf-8
from train_and_evaluate import *
from models import *
import time
import torch.optim
from expressions_transfer import *
from torch.optim import lr_scheduler
import os
from nlgeval import NLGEval,compute_metrics
import nltk
import argparse
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
batch_size = 64
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2

dataset_name="Math_23K"
data = load_raw_data("data/Math_23K.json")
parser = argparse.ArgumentParser(description="Demo of argparse")
parser.add_argument('--test', default='MWPGen')
args = parser.parse_args()
print(args)

project_name=args.test
#project_list=["217_pointer","219_iq","210_basic","211_basic","211_CNN","211_transformer"]
#"215_basic","215_CNN","215_transformer",
pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs

fold_size = int(len(pairs) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * 4):])


for fold in range(1):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]
    file_ground=open("output/ground_"+str(fold),"w") 
    for pair in pairs_tested:
        file_ground.write(" ".join(pair[0])+"\n")
    file_ground.close()

    project_list=project_name.split(",")
    for project_name in project_list:
        if os.path.getsize("../"+project_name+"/output/generate_"+str(fold))!=0:
            #print("fold:", fold + 1)
            print("***********************")
            print(project_name)
            metrics_dict = compute_metrics(hypothesis="../"+project_name+"/output/generate_"+str(fold),references=["output/ground_"+str(fold)],no_skipthoughts=True, no_glove=True)
    