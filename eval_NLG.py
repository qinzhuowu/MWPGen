# coding: utf-8
from train_and_evaluate import *
from models import *
import time
import torch.optim
from expressions_transfer import *
from torch.optim import lr_scheduler
import os
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

best_acc_fold = []
def indexes_from_sentence_NLG(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word.startswith("N"):
            res.append(lang.word2index["NUM"])
        elif word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res
for fold in range(1):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

    input_lang, output_lang, output_exp_lang,train_pairs, test_pairs,train_cate_problem_dict = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                    copy_nums, tree=False)#input_cell, len(input_cell), output_cell, len(output_cell),pair[2], pair[3]
    # Initialize models

    backward_encoder=MWPEncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                         n_layers=n_layers)
    backward_predict=MWPPrediction(hidden_size=hidden_size, op_nums=output_exp_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums))
    generate = MWPGenerateNode(hidden_size=hidden_size, op_nums=output_exp_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size)
    merge = MWPMerge(hidden_size=hidden_size, embedding_size=embedding_size)
    backward_encoder_optimizer = torch.optim.Adam(backward_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    backward_predict_optimizer = torch.optim.Adam(backward_predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

    backward_encoder_scheduler = lr_scheduler.StepLR(backward_encoder_optimizer, step_size=20, gamma=0.5)
    backward_predict_scheduler = lr_scheduler.StepLR(backward_predict_optimizer, step_size=20, gamma=0.5)
    generate_scheduler = lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
    merge_scheduler = lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)


    if os.path.exists("evaluate/encoder"+str(fold)+dataset_name):
        backward_encoder.load_state_dict(torch.load("evaluate/encoder"+str(fold)+dataset_name))
        backward_predict.load_state_dict(torch.load("evaluate/predict"+str(fold)+dataset_name))
        generate.load_state_dict(torch.load("evaluate/generate"+str(fold)+dataset_name))
        merge.load_state_dict(torch.load("evaluate/merge"+str(fold)+dataset_name))
    # Move models to GPU
    if USE_CUDA:

        backward_encoder.cuda()
        backward_predict.cuda()
        generate.cuda()
        merge.cuda()


    generate_num_ids = []
    need_to_print=[]
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])
    project_list=project_name.split(",")
    for project_name in project_list:
        if os.path.getsize("../"+project_name+"/output/generate_"+str(fold))!=0:
            #print("fold:", fold + 1)
            hypothesis1=[]
            with open("../"+project_name+"/output/generate_"+str(fold), 'r') as f:
                for line in f.readlines():
                    hypothesis1.append(line.strip().split())
            ground1=[]
            list_file=[]
            with open("../"+project_name+"/output/ground_"+str(fold), 'r') as f:
                for line in f.readlines():
                    ground1.append(line.strip().split())

            #print("fold:", fold + 1)
            #print("epoch:", epoch + 1)
            #print("--------------------------------")
            #
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            start = time.time()
            eval_idx=0
            for test_batch in test_pairs:
                if eval_idx%20==0:
                    test_res = MWP_evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, backward_encoder,
                        backward_predict,generate,merge, output_exp_lang, test_batch[5],  beam_size=beam_size)
                    val_ac, equ_ac, test_list, tar_list = compute_prefix_tree_result(test_res, test_batch[2], output_exp_lang, test_batch[4], test_batch[7])
                    if eval_idx%400==0:
                        print(test_res)
                        print(" ".join(indexes_to_sentence(output_exp_lang,test_res)))
                        print(" ".join(indexes_to_sentence(output_exp_lang,test_batch[2])))
                    
                    if val_ac:
                        value_ac += 1
                    if equ_ac:
                        equation_ac += 1
                    eval_total += 1
                eval_idx+=1
            print(equation_ac, value_ac, eval_total)
            print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            print("testing time", time_since(time.time() - start))
            print("------------------------------------------------------")

            out_filename="output/evaluate_test_result"+str(fold)
            out_filename1="output/evaluate_test_wrong"+str(fold)
            file_out=open(out_filename,"w")
            file_wrong=open(out_filename1,"w") 
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            start = time.time()

            for idx in range(len(test_pairs)):
                test_batch =test_pairs[idx]

                ground_=indexes_from_sentence(input_lang,ground1[idx])
                hypothesis_=indexes_from_sentence(input_lang,hypothesis1[idx])
                #print("---------------------------")
                #print(idx)
                #print(" ".join(ground1[idx]))
                #print(" ".join(hypothesis1[idx]))
                #print(ground_)
                #print(hypothesis_)
                #print(test_batch[5])
                '''
                if test_batch[0]!=ground_:
                    print("***********")
                    print(test_batch[0])
                    print(ground_)
                    print(hypothesis_)
                    print(" ".join(indexes_to_sentence(input_lang,test_batch[0])))
                    print(" ".join(indexes_to_sentence(input_lang,ground_)))
                    print(" ".join(indexes_to_sentence(input_lang,hypothesis_)))
                else:
            ''' 
                hypothesis_len=len(hypothesis_)
                num_pos = []
                for i, j in enumerate(hypothesis_):
                    word=input_lang.index2word[j]
                    if word.startswith("N"):
                        num_pos.append(i)
                if len(num_pos)==0 or len(num_pos)>10:
                    eval_total+=1
                else:

                    test_res = MWP_evaluate_tree(hypothesis_, hypothesis_len, generate_num_ids, backward_encoder, backward_predict, generate,
                                             merge, output_exp_lang, num_pos,  beam_size=beam_size)
                    val_ac, equ_ac, test_list, tar_list = compute_prefix_tree_result(test_res, test_batch[2], output_exp_lang, test_batch[4],test_batch[7])
                    #print(test_list)
                    #print(tar_list)
                    #print(" ".join(indexes_to_sentence(input_lang,test_batch[0])))
                    #print(" ".join(indexes_to_sentence(input_lang,hypothesis_)))
                    if test_list is None:
                        file_out.write("None"+"###"+" ".join([str(x) for x in tar_list])+"###"+" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
                    else:
                        file_out.write(" ".join([str(x) for x in test_list])+"###"+" ".join([str(x) for x in tar_list])+"###"+" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")

                    if val_ac:
                        value_ac += 1
                    else:
                        if test_list is None:
                            file_out.write("None"+"###"+" ".join([str(x) for x in tar_list])+"###"+" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
                        else:
                            file_out.write(" ".join([str(x) for x in test_list])+"###"+" ".join([str(x) for x in tar_list])+"###"+" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
                    if equ_ac:
                        equation_ac += 1
                    eval_total += 1
            print(equation_ac, value_ac, eval_total)
            print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            print("testing time", time_since(time.time() - start))
            print("------------------------------------------------------")
            #print(project_name)
            #print("fold:", fold + 1)
            #print(round(float(equation_ac) / eval_total, 4))
            #print(round(float(value_ac) / eval_total, 4))
            need_to_print.append(project_name)
            need_to_print.append("fold:"+str(fold + 1))
            need_to_print.append(str(round(float(equation_ac) / eval_total, 4)))
            need_to_print.append(str(round(float(value_ac) / eval_total, 4)))
            for str_need in need_to_print:
                print(str_need)
            '''
            best_acc_fold.append((equation_ac, value_ac, eval_total))
            a, b, c = 0, 0, 0
            for bl in range(len(best_acc_fold)):
                print(best_acc_fold[bl][0] / float(best_acc_fold[bl][2]), best_acc_fold[bl][1] / float(best_acc_fold[bl][2]))
                a += best_acc_fold[bl][0]
                b += best_acc_fold[bl][1]
                c += best_acc_fold[bl][2]
            '''

