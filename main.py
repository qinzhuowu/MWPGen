# coding: utf-8
from train_and_evaluate import *
from models import *
import time
import torch.optim
from expressions_transfer import *
from torch.optim import lr_scheduler
import os
from nlgeval import NLGEval,compute_metrics
import shutil

batch_size = 64
embedding_size = 128
hidden_size = 512
n_epochs = 200
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2

dataset_name="Math_23K"
data = load_raw_data("data/Math_23K.json")

pairs, generate_nums, copy_nums = transfer_num(data)#input_seq, out_seq, nums, num_pos

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


for fold in range(1):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]
    #input_lang problem  output_lang equation
    input_lang, output_lang, output_exp_lang,train_pairs, test_pairs,train_cate_problem_dict = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                    copy_nums, tree=False)#input_cell, len(input_cell), output_cell, len(output_cell),pair[2], pair[3]
    print(test_pairs[0])
    print(" ".join(indexes_to_sentence(input_lang,test_pairs[0][0])))
    print(" ".join(indexes_to_sentence(output_lang,test_pairs[0][2])))
    #print(" ".join(output_lang.index2word))
    print(" ".join(output_exp_lang.index2word))
    # Initialize models
    encoder = EncoderSeq(problem_size=input_lang.n_words,expression_size=output_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                         n_layers=n_layers)
    predict = Prediction(hidden_size=hidden_size, problem_size=input_lang.n_words,
                         input_size=len(generate_nums))

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)

    encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    predict_scheduler = lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)

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

    if not os.path.exists("models"):
          os.makedirs("models")    
    if not os.path.exists("output"):
          os.makedirs("output")
    start_epoch=0

    if os.path.exists("models/forward_encoder"+str(fold)+dataset_name):
        encoder.load_state_dict(torch.load("models/forward_encoder"+str(fold)+dataset_name))
        predict.load_state_dict(torch.load("models/forward_predict"+str(fold)+dataset_name))
        file_epoch_out=open( "models/epoch_num"+str(fold)).readlines()
        word=file_epoch_out[0]
        start_epoch=int(word.strip())
        print("start_from_epoch:"+str(start_epoch))
        if os.path.exists("output/generate_"+str(fold)):
            print("fold:", fold + 1)
            print("epoch:", start_epoch)
            metrics_dict=compute_metrics(hypothesis="output/generate_"+str(fold),references=["output/ground_"+str(fold)],no_skipthoughts=True, no_glove=True)

    start_MWP_epoch=0
    if os.path.exists("models/encoder"+str(fold)+dataset_name):
        backward_encoder.load_state_dict(torch.load("models/encoder"+str(fold)+dataset_name))
        backward_predict.load_state_dict(torch.load("models/predict"+str(fold)+dataset_name))
        generate.load_state_dict(torch.load("models/generate"+str(fold)+dataset_name))
        merge.load_state_dict(torch.load("models/merge"+str(fold)+dataset_name))
        file_epoch_out=open( "models/epoch_num_MWP"+str(fold)+dataset_name).readlines()
        word=file_epoch_out[0]
        start_MWP_epoch=int(word.strip())+1
        print("MWP_start_from_epoch:"+str(start_MWP_epoch))
    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        backward_encoder.cuda()
        backward_predict.cuda()
        generate.cuda()
        merge.cuda()


    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    for epoch in range(start_MWP_epoch,200):
        backward_encoder_scheduler.step()
        backward_predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches,num_pos_batches,keyword_batches,exp_batches,exp_length_batches,num_stack_batches = prepare_train_batch(train_pairs, batch_size)
        print("fold:", fold + 1)
        print("epoch:", epoch + 1)
        start = time.time()
        for idx in range(len(input_lengths)):
            num_size=[]
            for num_ in nums_batches[idx]: 
                num_size.append(len(num_))
            loss = MWP_train_tree(
                input_batches[idx], input_lengths[idx], exp_batches[idx], exp_length_batches[idx],
                num_stack_batches[idx], num_size, generate_num_ids, backward_encoder, backward_predict, generate, merge,
                backward_encoder_optimizer, backward_predict_optimizer, generate_optimizer, merge_optimizer, 
                output_exp_lang, num_pos_batches[idx])
            loss_total += loss
        print("loss:", loss_total / len(input_lengths))
        print("training time", time_since(time.time() - start))
        print("--------------------------------")
        out_filename="output/test_result"+str(fold)
        out_filename1="output/test_wrong"+str(fold)
        file_out=open(out_filename,"w")
        file_wrong=open(out_filename1,"w") 
        if epoch % 50 == 0 or epoch > n_epochs - 5:
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
                    if eval_idx%1800==0:
                        print(test_res)
                        print(" ".join(indexes_to_sentence(output_exp_lang,test_res)))
                        print(" ".join(indexes_to_sentence(output_exp_lang,test_batch[2])))
                    file_out.write(" ".join([str(x) for x in test_list])+"###"+" ".join([str(x) for x in tar_list])+"###"+" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
                    if val_ac:
                        value_ac += 1
                    else:
                        file_wrong.write(" ".join([str(x) for x in test_list])+"###"+" ".join([str(x) for x in tar_list])+"###"+" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
                    if equ_ac:
                        equation_ac += 1
                    eval_total += 1
                eval_idx+=1
            print(equation_ac, value_ac, eval_total)
            print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            print("testing time", time_since(time.time() - start))
            print("------------------------------------------------------")
            torch.save(backward_encoder.state_dict(), "models/encoder"+str(fold)+dataset_name)
            torch.save(backward_predict.state_dict(), "models/predict"+str(fold)+dataset_name)
            torch.save(generate.state_dict(), "models/generate"+str(fold)+dataset_name)
            torch.save(merge.state_dict(), "models/merge"+str(fold)+dataset_name)

            file_epoch_out=open( "models/epoch_num_MWP"+str(fold)+dataset_name,"w")
            file_epoch_out.write(str(epoch))
            file_epoch_out.close()
            
            if epoch == n_epochs - 1:
                best_acc_fold.append((equation_ac, value_ac, eval_total))
                a, b, c = 0, 0, 0
                for bl in range(len(best_acc_fold)):
                    print(best_acc_fold[bl][0] / float(best_acc_fold[bl][2]), best_acc_fold[bl][1] / float(best_acc_fold[bl][2]))
                    a += best_acc_fold[bl][0]
                    b += best_acc_fold[bl][1]
                    c += best_acc_fold[bl][2]
                print(a / float(c), b / float(c))

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
            if eval_idx%1800==0:
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

    for epoch in range(start_epoch,n_epochs):
        encoder_scheduler.step()
        predict_scheduler.step()
        backward_encoder_scheduler.step()
        backward_predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        loss_total = 0
        #input:problem output:keyword10+exp
        input_batches, input_lengths, output_batches, output_lengths, nums_batches,num_pos_batches,keyword_batches,exp_batches,exp_length_batches,num_stack_batches = prepare_train_batch(train_pairs, batch_size)
        print("fold:", fold + 1)
        print("epoch:", epoch + 1)
        start = time.time()
        for idx in range(len(input_lengths)):
            #if idx%10==0:
            #    print(idx)
            forward_loss,forward_len,response = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                encoder, predict,encoder_optimizer, predict_optimizer, input_lang,output_lang)
            reward=calculate_rewards(input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                exp_batches[idx], exp_length_batches[idx],encoder, predict, backward_encoder, backward_predict,
                encoder_optimizer, predict_optimizer,backward_encoder_optimizer, backward_predict_optimizer,
                generate_num_ids,generate,merge, generate_optimizer,merge_optimizer,
                input_lang,output_lang,output_exp_lang,nums_batches[idx], num_stack_batches[idx],num_pos_batches[idx],forward_loss,forward_len,response)
            
            loss=reward
            loss.backward()
            encoder_optimizer.step()
            predict_optimizer.step()
            #backward_encoder_optimizer.step()
            #backward_predict_scheduler.step()

            loss_total += loss.item()

        print("loss:", loss_total / len(input_lengths))
        print("training time", time_since(time.time() - start))
        print("--------------------------------")
        out_filename="output/test_result"+str(fold)
        out_filename1="output/test_wrong"+str(fold)
        file_out=open(out_filename,"w")
        file_wrong=open(out_filename1,"w") 
        if epoch % 10 == 0 or epoch > n_epochs - 2 or (epoch ==2 or epoch==4):
            if os.path.exists("output/ground_"+str(fold)):
                shutil.copyfile("output/ground_"+str(fold),"output/ground_"+str(fold)+"_old")
                shutil.copyfile("output/generate_"+str(fold),"output/generate_"+str(fold)+"_old")
            file_ground=open("output/ground_"+str(fold),"w") 
            file_gene=open("output/generate_"+str(fold),"w") 
            ground_list=[]
            generate_list=[]
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            start = time.time()
            for test_batch in test_pairs:
                if epoch<n_epochs-2 :
                    if eval_total%10==0 :
                        test_keyword_exp=test_batch[6]+[0 for _ in range(keyword_num - len(test_batch[6]))]
                        test_keyword_exp=test_keyword_exp[0:keyword_num]+test_batch[2]
                        test_res = evaluate_tree(test_keyword_exp,len(test_keyword_exp), encoder, predict,
                            input_lang,  beam_size=beam_size)
                        
                        if eval_total==0:
                            print(test_batch[0])
                            print(" ".join(indexes_to_sentence(input_lang,test_batch[0])))
                            print(test_keyword_exp)
                            print(" ".join(indexes_to_sentence(output_lang,test_keyword_exp)))
                        #val_ac, equ_ac, test_list, tar_list = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
                        if eval_total%200==0:
                            print(test_res)
                            print(" ".join(indexes_to_sentence(input_lang,test_res)))
                            print(" ".join(indexes_to_sentence(input_lang,test_batch[0])))
                        generate_list.append(" ".join(indexes_to_sentence(input_lang,test_res)))
                        ground_list.append(" ".join(indexes_to_sentence(input_lang,test_batch[0])))
                        file_gene.write(" ".join(indexes_to_sentence(input_lang,test_res))+"\n")
                        file_ground.write(" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
                        file_out.write(" ".join([str(test_res) for x in test_res])+"###"+" ".join(indexes_to_sentence(input_lang,test_res))+"\n")
                else:
                    test_keyword_exp=test_batch[6]+[0 for _ in range(keyword_num - len(test_batch[6]))]
                    test_keyword_exp=test_keyword_exp[0:keyword_num]+test_batch[2]
                    test_res = evaluate_tree(test_keyword_exp,len(test_keyword_exp), encoder, predict,
                        input_lang,  beam_size=beam_size)
                    
                    if eval_total%200==0:
                        print(test_res)
                        print(" ".join(indexes_to_sentence(input_lang,test_res)))
                        print(" ".join(indexes_to_sentence(input_lang,test_batch[0])))
                    generate_list.append(" ".join(indexes_to_sentence(input_lang,test_res)))
                    ground_list.append(" ".join(indexes_to_sentence(input_lang,test_batch[0])))
                    file_gene.write(" ".join(indexes_to_sentence(input_lang,test_res))+"\n")
                    file_ground.write(" ".join(indexes_to_sentence(input_lang,test_batch[0]))+"\n")
                    file_out.write(" ".join([str(test_res) for x in test_res])+"###"+" ".join(indexes_to_sentence(input_lang,test_res))+"\n")

                #val_ac, equ_ac, test_list, tar_list = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
                
                eval_total += 1
            print("testing time", time_since(time.time() - start))
            print("------------------------------------------------------")
            torch.save(encoder.state_dict(), "models/forward_encoder"+str(fold)+dataset_name)
            torch.save(predict.state_dict(), "models/forward_predict"+str(fold)+dataset_name)

            file_epoch_out=open( "models/epoch_num"+str(fold),"w")
            file_epoch_out.write(str(epoch))
            file_epoch_out.close()
            file_gene.close()
            file_ground.close()
            file_out.close()
            metrics_dict=compute_metrics(hypothesis="output/generate_"+str(fold),references=["output/ground_"+str(fold)],no_skipthoughts=True, no_glove=True)
            #nlgeval = NLGEval(no_skipthoughts=True, no_glove=True,metrics_to_omit=["METEOR"])  # loads the models
    

    for curr_fold in range(fold+1):
        if os.path.getsize("output/generate_"+str(curr_fold))!=0:
            print("fold:", curr_fold + 1)
            print("epoch:", n_epochs)
            metrics_dict = compute_metrics(hypothesis="output/generate_"+str(curr_fold),references=["output/ground_"+str(curr_fold)],no_skipthoughts=True, no_glove=True)
    