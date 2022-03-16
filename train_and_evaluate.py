#coding:utf-8
from masked_cross_entropy import *
from pre_data import *
from expressions_transfer import *
from models import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import time

from torch.autograd import Variable
from evaluation.eval import QGEvalCap, WMD
MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)



def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            if len(nums_stack_batch[i])>0:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float("1e12")
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target[i] == unk:
                target[i] = num_start
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, out_expression_list(test_res, output_lang, num_list), out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False
    if test == tar:
        return True, True
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False
        else:
            return False, False
    except:
        return False, False


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [True for _ in range(hidden_size)]
    temp_0 = [False for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.BoolTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous() #S*B*H,B*S*H
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0),indices,masked_index#(B*num*H)


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = node_stack
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal

def MWP_train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos,english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.BoolTensor(seq_mask)

    num_mask = []
    #print(num_size_batch)
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder
    #print(input_batch)
    #print(input_length)
    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs,indices,masked_index = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    #for idx in range(len(target_batch)):
    #    print(target_batch[idx])
    #    print(" ".join(indexes_to_sentence(output_lang,target_batch[idx])))
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask,indices,masked_index)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        #print(t)
        #print(target_t)
        #print(" ".join(indexes_to_sentence(output_lang,target_t)))
        #print(num_start)
        #print(current_nums_embeddings.size())
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue
            #print(i)
            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()

def MWP_evaluate_tree_batch(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
    encoder, predict, generate, merge, output_lang, num_pos,forward_loss,english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.BoolTensor(seq_mask)

    num_mask = []
    #print(num_size_batch)
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs,indices,masked_index = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask,indices,masked_index)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue
            #print(i)
            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    loss = masked_cross_entropy_with_forwardloss(all_node_outputs, target, target_length,forward_loss)
    #loss.backward()

    return loss.item()  # , loss_0.item(), loss_1.item()

def MWP_evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos,beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    
    input_batch=input_batch[0:input_length]
    seq_mask = torch.BoolTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.BoolTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs,indices,masked_index = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask,indices,masked_index)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out


#def evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos,
#                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):
# train_tree有 traget信息，stack信息 optimizer信息
#evaluate有 max_length=MAX_OUTPUT_LENGTH  45

#单位 无生物 心怀 材料 sequence|次序 ordinal|第 数量值 amount|多少 undesired|莠 物质
#[832, 387, 160, 503, 1278, 1293, 715, 804, 270, 203]
#[[1, 9, 16, 21, 28], [1, 24], [2, 26], [4, 24, 25], [11, 13], [11, 13], [12, 13, 26, 27], [12, 26, 27], [18, 19], [19, 24]]
#里 吨 吨 吨 吨 ###里 批 ###存有 有 ###面粉 批 面粉 ###第 次 ###第 次 ###二 次 有 多少 ###二 有 多少 ###还 剩 ###剩 批 

#树 数量值 amount|多少 many|多 question|疑问
#[610, 715, 804, 1181, 1173]
#[[0, 3, 7, 9, 13], [2, 10, 14, 15], [2, 10, 14, 15], [10, 15], [10, 15]]
#果园 苹果树 苹果树 梨树 梨树 ###有 多 有 多少 ###有 多 有 多少 ###多 多少 ###多 多少
# train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
#                     pair[2], pair[3], num_stack, pair[4]))
def train_tree(target_batch, target_length,input_batch, input_length,encoder, predict, encoder_optimizer, predict_optimizer, problem_lang,keywordexp_lang,english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)

    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.BoolTensor(seq_mask)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1) # S*B*h
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)
    sos=problem_lang.word2index["SOS"]
    decoder_input = torch.LongTensor([sos for _ in range(batch_size)])#B
    #print(" ".join(indexes_to_sentence(keywordexp_lang,input_batch[0])))

    #print(input_batch)
    #print(input_length)
    encoder.train()
    predict.train()
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        decoder_input=decoder_input.cuda()
        target = target.cuda()
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()




    A_matrix=[[[0 for _ in range(max_len-keyword_num)] for _ in range(max_len-keyword_num)] for _ in range(batch_size)]

    parent_exp=[[[0 for _ in range(max_len-keyword_num)] for _ in range(max_len-keyword_num)] for _ in range(batch_size)]
    lchild_exp=[[[0 for _ in range(max_len-keyword_num)] for _ in range(max_len-keyword_num)] for _ in range(batch_size)]
    rchild_exp=[[[0 for _ in range(max_len-keyword_num)] for _ in range(max_len-keyword_num)] for _ in range(batch_size)]

    parent_var=[[0 for _ in range(max_len-keyword_num)] for _ in range(batch_size)]
    lchild_var=[[0 for _ in range(max_len-keyword_num)] for _ in range(batch_size)]
    rchild_var=[[0 for _ in range(max_len-keyword_num)] for _ in range(batch_size)]    

    pad=keywordexp_lang.word2index["PAD"]
    add=keywordexp_lang.word2index["+"]
    sub=keywordexp_lang.word2index["-"]
    mul=keywordexp_lang.word2index["*"]
    div=keywordexp_lang.word2index["/"]
    exp=keywordexp_lang.word2index["^"]
    for idx in range(batch_size):
        exp_line=input_batch[idx][keyword_num:]
        exp_stack=[]
        lr_stack=[]
        for idx_i in range(len(exp_line)):
            if len(exp_stack)>0:
                idx_j=exp_stack.pop()
                lr=lr_stack.pop()
                A_matrix[idx][idx_j][idx_i]=1
                A_matrix[idx][idx_i][idx_j]=1
                if lr=="left":
                    parent_exp[idx][idx_i][idx_j]=1
                    lchild_exp[idx][idx_j][idx_i]=1
                    parent_var[idx][idx_i]=input_batch[idx][keyword_num+idx_j]
                    lchild_var[idx][idx_j]=input_batch[idx][keyword_num+idx_i]
                elif lr=="right":
                    parent_exp[idx][idx_i][idx_j]=1
                    rchild_exp[idx][idx_j][idx_i]=1
                    parent_var[idx][idx_i]=input_batch[idx][keyword_num+idx_j]
                    rchild_var[idx][idx_j]=input_batch[idx][keyword_num+idx_i]


            if exp_line[idx_i] in [add,sub,mul,div,exp]:
                exp_stack.append(idx_i)
                exp_stack.append(idx_i)
                lr_stack.append("right")
                lr_stack.append("left")

            if exp_line[idx_i] != pad:
                A_matrix[idx][idx_i][idx_i]=1

    #print("*******************")
    #print(input_batch[1])
    #print(" ".join(indexes_to_sentence(keywordexp_lang,input_batch[1])))
    #print(parent_var[1])
    #print(lchild_var[1])
    #print(rchild_var[1])

    #encoder_outputs, problem_output = encoder(input_var, input_length,parent_exp,lchild_exp,rchild_exp)

    encoder_outputs, problem_output = encoder(input_var, input_length,parent_var,lchild_var,rchild_var)

    max_target_length = max(target_length)
    all_node_outputs = []
    node_stacks = problem_output#B*N

    response = []
    sample_response=[]
    base_response=[]
    for t in range(max_target_length):
        #print(decoder_input)
        outputs,hidden = predict(decoder_input,node_stacks, encoder_outputs,seq_mask,input_var)
        all_node_outputs.append(outputs)#B*N
        node_stacks=hidden
        decoder_input=target[t]

        prob_distribution = torch.softmax(outputs, dim=1)

        top_idx = torch.multinomial(prob_distribution, 1)
        top_idx = top_idx.squeeze(1).detach()  # detach from history as input
        sample_response.append(top_idx)
        _, top_i = outputs.topk(1)
        top_i=top_i.squeeze(1).detach()
        base_response.append(top_i)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    sample_response=torch.stack(sample_response, dim=1)
    base_response=torch.stack(base_response, dim=1)
    response.append(sample_response)
    response.append(base_response)
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        
    #loss = masked_cross_entropy(all_node_outputs, target, target_length)
    loss = masked_cross_entropy_forwardloss(all_node_outputs, target, target_length)
    return loss,target_length,response  # , loss_0.item(), loss_1.item()

class NewBeam:  # the class save the beam node
    def __init__(self, score, node_stack,embedding_stack,cur_mix_context, precompute,topic_precompute, out):
        self.score = score
        self.node_stack = node_stack
        self.embedding_stack = copy_list(embedding_stack)
        self.cur_mix_context =cur_mix_context
        self.precompute = precompute
        self.topic_precompute = topic_precompute
        self.out = copy.deepcopy(out)


def evaluate_tree(input_batch, input_length, encoder, predict, problem_lang,beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):
    seq_mask = torch.BoolTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    eos=problem_lang.word2index["EOS"]
    encoder.eval()
    predict.eval()
    batch_size = 1

    sos=problem_lang.word2index["SOS"]
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()


    A_matrix=[[[0 for _ in range(input_length-keyword_num)] for _ in range(input_length-keyword_num)] for _ in range(batch_size)]
    parent_exp=[[[0 for _ in range(input_length-keyword_num)] for _ in range(input_length-keyword_num)] for _ in range(batch_size)]
    lchild_exp=[[[0 for _ in range(input_length-keyword_num)] for _ in range(input_length-keyword_num)] for _ in range(batch_size)]
    rchild_exp=[[[0 for _ in range(input_length-keyword_num)] for _ in range(input_length-keyword_num)] for _ in range(batch_size)]

    parent_var=[[0 for _ in range(input_length-keyword_num)] for _ in range(batch_size)]
    lchild_var=[[0 for _ in range(input_length-keyword_num)] for _ in range(batch_size)]
    rchild_var=[[0 for _ in range(input_length-keyword_num)] for _ in range(batch_size)]    
    pad=problem_lang.word2index["PAD"]
    add=problem_lang.word2index["+"]
    sub=problem_lang.word2index["-"]
    mul=problem_lang.word2index["*"]
    div=problem_lang.word2index["/"]
    exp=problem_lang.word2index["^"]
    exp_line=input_batch[keyword_num:]
    exp_stack=[]
    lr_stack=[]
    for idx_i in range(len(exp_line)):
        if len(exp_stack)>0:
            idx_j=exp_stack.pop()
            lr=lr_stack.pop()
            A_matrix[0][idx_j][idx_i]=1
            A_matrix[0][idx_i][idx_j]=1
            if lr=="left":
                parent_exp[0][idx_i][idx_j]=1
                lchild_exp[0][idx_j][idx_i]=1
                parent_var[0][idx_i]=input_batch[keyword_num+idx_j]
                lchild_var[0][idx_j]=input_batch[keyword_num+idx_i]
            elif lr=="right":
                parent_exp[0][idx_i][idx_j]=1
                rchild_exp[0][idx_j][idx_i]=1
                parent_var[0][idx_i]=input_batch[keyword_num+idx_j]
                rchild_var[0][idx_j]=input_batch[keyword_num+idx_i]

        if exp_line[idx_i] in [add,sub,mul,div,exp]:
            exp_stack.append(idx_i)
            exp_stack.append(idx_i)
            lr_stack.append("right")
            lr_stack.append("left")
        if exp_line[idx_i] != pad:
            A_matrix[0][idx_i][idx_i]=1


    #encoder_outputs, problem_output = encoder(input_var, [input_length],parent_exp,lchild_exp,rchild_exp)
    encoder_outputs, problem_output = encoder(input_var, [input_length],parent_var,lchild_var,rchild_var)

    node_stacks = problem_output#B*N
    #cur_mix_context = init_att
    precompute = None
    topic_precompute = None
    beams = [NewBeam(0.0, node_stacks, [sos],None, None,None, [])]
    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.out)>0 and b.out[-1] == eos:
                current_beams.append(b)
                continue

            decoder_input = torch.LongTensor(b.embedding_stack).cuda()#B
            outputs,next_node = predict(decoder_input,b.node_stack, encoder_outputs,seq_mask,input_var)

            out_score = nn.functional.log_softmax(outputs, dim=1)

            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = next_node
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)


                current_beams.append(NewBeam(b.score+float(tv), current_node_stack, [out_token],None,None,None, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out
def evaluate_predictions(target_src, decoded_text):
    assert len(target_src) == len(decoded_text)
    eval_targets = {}
    eval_predictions = {}
    for idx in range(len(target_src)):
        eval_targets[idx] = [target_src[idx]]
        eval_predictions[idx] = [decoded_text[idx]]

    QGEval = QGEvalCap(eval_targets, eval_predictions)
    scores = QGEval.evaluate_Bleu4()
    return scores

def calculate_rewards(input_batches, input_lengths, output_batches, output_lengths,exp_batches, exp_length_batches,
    encoder, predict, backward_encoder, backward_predict,encoder_optimizer, predict_optimizer, backward_encoder_optimizer, backward_predict_optimizer,
    generate_num_ids,generate,merge, generate_optimizer,merge_optimizer,input_lang,output_lang,output_exp_lang,
    num_list, num_stack,num_pos,forward_loss,forward_len,forward_response):
    ep_rewards = []
    # ep_num are used to bound the number of episodes
    # MAXIMUM ep = 10
    beam_size=5
    batch_size=len(input_batches)
    ep_num = 1
    responses = []
    num_size=[]
    for num_ in num_list: 
        num_size.append(len(num_))
    batch_size=len(input_batches)
    #for idx in range(len(input_batches)):
    #    print(" ".join(indexes_to_sentence(output_lang,exp_batches[idx])))
    '''
    backward_loss = MWP_train_tree(input_batches, input_lengths,exp_batches, exp_length_batches,
        num_stack,num_size,generate_num_ids,backward_encoder, backward_predict,generate,merge,
        backward_encoder_optimizer, backward_predict_optimizer, generate_optimizer, merge_optimizer, 
        output_exp_lang, num_pos)
    '''

    #r1=forward_loss+backward_loss

    sample_out=[]#B*S
    baseline_out=[]
    input_batch=[]
    neg_reward = []
    for idx in range(len(input_batches)):
        #baseline_res=evaluate_tree(output_batches[idx][0: output_lengths[idx]], output_lengths[idx], encoder, predict,
        #            input_lang,  beam_size=beam_size)
        #print(sample_out.size())
        input_line=" ".join(indexes_to_sentence(input_lang,input_batches[idx]))
        sample_out_decoded=" ".join(indexes_to_sentence(input_lang,forward_response[0][idx].tolist()))
        baseline_out_decoded =" ".join(indexes_to_sentence(input_lang,forward_response[1][idx].tolist()))

        sample_score = evaluate_predictions([input_line], [sample_out_decoded])["Bleu_4"]
        greedy_score = evaluate_predictions([input_line], [baseline_out_decoded])["Bleu_4"]
        reward_=sample_score- greedy_score
        neg_reward.append(reward_)
    
    #print("training time", time_since(time.time() - start))
    neg_reward=torch.FloatTensor(neg_reward).cuda()

    r_bleu_loss = torch.sum(neg_reward * forward_loss) / batch_size
    


    #start = time.time()
    sample_response=[]
    sample_length=[]
    sample_num_pos=[]
    sample_num_size=[]
    baseline_response=[]
    baseline_length=[]
    baseline_num_pos=[]
    baseline_num_size=[]
    for idx in range(len(input_batches)):
        resp_num_pos=[]
        flag=0
        sample_id=forward_response[0][idx].tolist()
        sample_length.append(len(sample_id))
        #print(sample_id)
        response=[]
        for i in range(len(sample_id)):
            j=sample_id[i]
            if flag==0:
                if j < input_lang.n_words:
                    word=input_lang.index2word[j]
                    if word.startswith("N"):
                        resp_num_pos.append(i)
                        response.append(input_lang.word2index["NUM"])
                    elif word=="EOS":
                        sample_length[idx]=i+1
                        response.append(input_lang.word2index["EOS"])
                        flag=1
                    else:
                        response.append(j)
                else:
                    response.append(input_lang.word2index["UNK"])
            else:
                response.append(0)
        #print(response)
        sample_response.append(response)
        while len(resp_num_pos)<num_size[idx]:
            resp_num_pos.append(0)
        if len(resp_num_pos)>num_size[idx]:
            resp_num_pos=resp_num_pos[:num_size[idx]]
        sample_num_pos.append(resp_num_pos)

        resp_num_pos=[]
        flag=0
        baseline_id=forward_response[1][idx].tolist()
        baseline_length.append(len(baseline_id))
        response=[]
        #for i,j in enumerate(baseline_id):
        for i in range(len(baseline_id)):
            j=baseline_id[i]
            if flag==0:
                if j < input_lang.n_words:
                    word=input_lang.index2word[j]
                    if word.startswith("N"):
                        resp_num_pos.append(i)
                        response.append(input_lang.word2index["NUM"])
                    elif word=="EOS":
                        baseline_length[idx]=i+1
                        response.append(input_lang.word2index["EOS"])
                        flag=1
                    else:
                        response.append(j)
                else:
                    response.append(input_lang.word2index["UNK"])
            else:
                response.append(0)
        baseline_response.append(response)
        while len(resp_num_pos)<num_size[idx]:
            resp_num_pos.append(0)
        if len(resp_num_pos)>num_size[idx]:
            resp_num_pos=resp_num_pos[:num_size[idx]]
        baseline_num_pos.append(resp_num_pos)
    sample_reward=MWP_evaluate_tree_batch(sample_response,sample_length,exp_batches, exp_length_batches,
        num_stack,num_size,generate_num_ids,backward_encoder, backward_predict,generate,merge,
        output_exp_lang,sample_num_pos,forward_loss)
    baseline_reward=MWP_evaluate_tree_batch(baseline_response,baseline_length,exp_batches, exp_length_batches,
        num_stack,num_size,generate_num_ids,backward_encoder, backward_predict,generate,merge,
        output_exp_lang,baseline_num_pos,forward_loss)    
    r_scst_loss=sample_reward- baseline_reward

    #print(input_batches[idx])
    #print(input_lengths[idx])
    
    #print(forward_response[:,0])
    #print(" ".join(indexes_to_sentence(input_lang,test_res)))
    #response = evaluate_tree(output_batches[idx][0:output_lengths[idx]],output_lengths[idx], encoder, predict,input_lang,  beam_size=5)
    #print(" ".join(indexes_to_sentence(input_lang,response)))
    #
    '''
    resp_num_pos=[]
    flag=0
    response=evaluate_tree(output_batches[idx][0: output_lengths[idx]], output_lengths[idx], encoder, predict,
                    input_lang,  beam_size=beam_size)
    #response=forward_response[idx].tolist()
    resp_input_length=len(response)
    for i, j in enumerate(response):
        if j < input_lang.n_words:
            word=input_lang.index2word[j]
            if word.startswith("N"):
                resp_num_pos.append(i)
                response[i]=input_lang.word2index["NUM"]
            if word=="EOS":
                resp_input_length=i+1
        else:
            response[i]=input_lang.word2index["UNK"]
    #response_id=indexes_from_sentence(input_lang, response)
    test_res = MWP_evaluate_tree(response, resp_input_length, generate_num_ids, backward_encoder, backward_predict, generate,
                                         merge, output_exp_lang,resp_num_pos,  beam_size=5)
    

    #print(test_res)
    #print(" ".join(indexes_to_sentence(output_exp_lang,test_res)))
    #print(" ".join(indexes_to_sentence(output_exp_lang,exp_batches[idx])))

    val_ac, equ_ac, test_list, tar_list = compute_prefix_tree_result(test_res, exp_batches[idx], output_exp_lang, num_list[idx], num_stack[idx])
    '''
    
    #r2=0.5
    #if val_ac:
    #    r2 = 0.0
    #r=1
    
    loss = 0.5 * forward_loss.sum()/batch_size+0.25*r_bleu_loss+0.25*r_scst_loss
    return loss