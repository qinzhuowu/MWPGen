import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import copy
from pre_data import *
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

import math
print_dims = False

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask , -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)



class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask , -1e12)
        return score
class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        #hidden 1*B*H  encoder_outputs:S*B*H
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)# S x B x H

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask , -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)



class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, problem_size,layers,dropout):
        super(Encoder, self).__init__()
        self.layers = layers
        self.num_directions = 2 
        self.hidden_size = hidden_size
        input_size = embedding_size

        self.word_lut = nn.Embedding(problem_size,embedding_size,padding_idx=0)
        self.rnn = nn.GRU(embedding_size, hidden_size/2,num_layers=self.layers,dropout=dropout,bidirectional=True)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input_idx,wordEmb,input_lengths, hidden=None):
        """
        input: (wrap(srcBatch), wrap(srcBioBatch), lengths)
        """
        #lengths = input[-1].data.view(-1).tolist()  # lengths data is wrapped inside a Variable
        #wordEmb = self.word_lut(input_idx)
        emb = pack(wordEmb, input_lengths)
        outputs, hidden_t = self.rnn(emb, hidden)
        #if isinstance(input, tuple):
        outputs = unpack(outputs)[0]
        return hidden_t, outputs

class TopicEncoder(nn.Module):
    def __init__(self, embedding_size,problem_size):
        super(TopicEncoder, self).__init__()
        self.word_lut = nn.Embedding(problem_size,embedding_size,padding_idx=0)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input_idx, hidden=None):
        """
        input: (wrap(ldaBatch), lda_length)
        """
        mask = input_idx.eq(0).float().transpose(0, 1).contiguous()  # (batch, seq)
        wordEmb = self.word_lut(input_idx)
        return wordEmb, mask
class EncoderSeq(nn.Module):
    def __init__(self, problem_size,expression_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.problem_size = problem_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.exp_embedding = nn.Embedding(expression_size, embedding_size, padding_idx=0)
        self.prob_embedding = nn.Embedding(problem_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.LSTM(embedding_size*4, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.gru_keyword = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        self.exp_attn=ExpAttn(hidden_size,hidden_size)
        self.dense_exp = nn.Linear(hidden_size*2, hidden_size)
        self.dense_topic = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, exp_seqs, exp_lengths,parent_var,lchild_var,rchild_var, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        #cate_word_edge,cate_index_input)#B*cate+seq*cate+seq #B*cate_num
        keyword_seq=exp_seqs[0:keyword_num,:]
        exp_seq=exp_seqs[keyword_num:,:]

        keyword_embedded=self.prob_embedding(keyword_seq)  # S x B x E
        exp_embedded = self.exp_embedding(exp_seq)  # S x B x E
        keyword_embedded= self.em_dropout(keyword_embedded)#S+10,B,E
        exp_embedded = self.em_dropout(exp_embedded)#S+10,B,E

        par_embedded= self.exp_embedding(torch.LongTensor(parent_var).transpose(0,1).cuda())
        par_embedded = self.em_dropout(par_embedded)#S+10,B,E
        lchild_embedded= self.exp_embedding(torch.LongTensor(lchild_var).transpose(0,1).cuda())
        lchild_embedded = self.em_dropout(lchild_embedded)#S+10,B,E
        rchild_embedded= self.exp_embedding(torch.LongTensor(rchild_var).transpose(0,1).cuda())
        rchild_embedded = self.em_dropout(rchild_embedded)#S+10,B,E

        exp_embedded_cat=torch.cat((exp_embedded,par_embedded,lchild_embedded,rchild_embedded),2)

        keyword_hidden=None
        keyword_outputs,keyword_hidden= self.gru_keyword(keyword_embedded, keyword_hidden)
        keyword_outputs=keyword_outputs[:, :, :self.hidden_size] +keyword_outputs[:, :, self.hidden_size:]
        
        for idx_ in range(len(exp_lengths)):
            exp_lengths[idx_]=exp_lengths[idx_]-keyword_num
        
        exp_lengths=torch.LongTensor(exp_lengths)
        lens, indices = torch.sort(exp_lengths, 0, True)
        len_list = lens.tolist()
        packed = torch.nn.utils.rnn.pack_padded_sequence(exp_embedded_cat[:,indices,:], len_list)#S+10*B*H
        _, _indices = torch.sort(indices, 0)
        pade_hidden = hidden
        pade_outputs, (hidden_c,hidden_t) = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)#S*B*N
        pade_outputs=pade_outputs[:,_indices,:]

        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        
        seq_mask = exp_seq.eq(0).bool().contiguous()
        new_topic,new_exp=self.exp_attn(keyword_outputs, pade_outputs, None,seq_mask)## B x S1 x N
        concat_topic=self.dense_topic(torch.cat((keyword_outputs,new_topic.transpose(0,1)),2))
        concat_exp=self.dense_exp(torch.cat((pade_outputs,new_exp.transpose(0,1)),2))
        
        concat_outputs=torch.cat((concat_topic,concat_exp),0)
        problem_output=F.max_pool1d(concat_outputs.permute(1,2,0), concat_outputs.shape[0]).squeeze(-1) #(batch_size, d_model,S) (batch_size, d_model)
        #print(exp_seqs.size())
        #print(keyword_outputs.size())
        #print(pade_outputs.size())
        #print(concat_outputs.size())
        return concat_outputs, problem_output.unsqueeze(0)


class ExpAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ExpAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, topic, exp_output, topic_mask=None,seq_mask=None):
        #topic S1*B*H  encoder_outputs:S2*B*H
        
        topic_len = topic.size(0)#S1
        exp_len=exp_output.size(0)#S2
        this_batch_size = exp_output.size(1)# S x B x H

        topic_=topic.transpose(0,1).unsqueeze(2)
        exp=exp_output.transpose(0,1).unsqueeze(1)
        repeat_dims= [1,1,exp_len,1]
        topic_hidden = topic_.repeat(*repeat_dims)  #B x S1*S2 x  H
        repeat_exp=[1,topic_len,1,1]
        exp_hidden = exp.repeat(*repeat_exp)  # B x S1*S2 x  H
        #print(topic_hidden.size())
        #print(exp_hidden.size())

        energy_in = torch.cat((topic_hidden, exp_hidden), 3)
        score_feature = torch.tanh(self.attn(energy_in))#B*S1*S2*H
        attn_energies = self.score(score_feature)  # B*S1*S2 x 1
        attn_energies = attn_energies.squeeze(3)

        if seq_mask is not None:
            seq_mask_dims=[1,topic_len,1]
            new_seq_mask=seq_mask.transpose(0,1).unsqueeze(1).repeat(*seq_mask_dims)
            attn_energies = attn_energies.masked_fill_(new_seq_mask , -1e12)
        
        attn_s1= nn.functional.softmax(attn_energies.transpose(1,2), dim=2)  # B x S2*S1
        attn_s2 = nn.functional.softmax(attn_energies, dim=2)  # B x S1*S2

        new_exp=attn_s1.bmm(topic.transpose(0, 1))  # B x S2 x N
        new_topic = attn_s2.bmm(exp_output.transpose(0, 1))  # B x S1 x N
        return new_topic,new_exp
class Prediction(nn.Module):
    # no copy,just attention
    def __init__(self, hidden_size, problem_size, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.problem_size = problem_size

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.generate_out = nn.Linear(hidden_size, problem_size)
        self.generate_node= nn.Linear(hidden_size * 3, hidden_size)

        self.attn = TreeAttn(hidden_size* 2, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.embedding = nn.Embedding(self.problem_size, self.hidden_size)
        # weights
        self.Ws = nn.Linear(hidden_size*2, hidden_size) # only used at initial stage
        self.Wo = nn.Linear(hidden_size, problem_size) # generate mode
        self.Wc = nn.Linear(hidden_size, hidden_size) # copy mode

    def forward(self, dec_input,dec_hidden, encoder_outputs,seq_mask,input_concat):
        #self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums,indices,masked_index,num_score_constraints

        embedded = self.embedding(dec_input).unsqueeze(1)#B*1*N
        embedded = self.dropout(embedded)

        current_h=dec_hidden.squeeze(0).unsqueeze(1)#B*N B*1*N
        #print(current_h.size())
        #print(current_c.size())
        #current_embeddings_h = self.dropout(current_h)
        #current_embeddings_c = self.dropout(current_c)
        current_embeddings = self.dropout(current_h)
        current_attn = self.attn(torch.cat((embedded, current_embeddings), 2).transpose(0, 1), encoder_outputs, seq_mask)#1*B*2H,S*B*H
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE
        leaf_input = torch.cat((embedded,current_h, current_context), 2) #B*1*3H
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)
        #print(leaf_input.size())
        leaf_node=self.generate_node(leaf_input)#1*B*H
        #print(leaf_node.size())
        output, hidden = self.gru(leaf_node.unsqueeze(0),dec_hidden)#1*B*H,(1*B*H,1*B*H)
        output = F.log_softmax(self.generate_out(output[0]),dim=1)
        # return p_leaf, num_score, op, current_embeddings, current_attn
        return output, hidden

'''
MAGNET
class EncoderSeq(nn.Module):
    def __init__(self, problem_size,expression_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.problem_size = problem_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.exp_embedding = nn.Embedding(expression_size, embedding_size, padding_idx=0)
        self.prob_embedding = nn.Embedding(problem_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.gru_keyword = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)


        self.gcn = nn.Linear(hidden_size, hidden_size)
        self.gcn1 = nn.Linear(hidden_size, hidden_size)
        self.gcn_dense=nn.Linear(hidden_size*2, hidden_size)
        self.tree_dense=nn.Linear(hidden_size*4, hidden_size)

        self.encoder=Encoder(embedding_size, hidden_size, problem_size,1,dropout)
        self.topic_encoder=TopicEncoder(hidden_size,problem_size)

        self.initer = nn.Linear(hidden_size/2, hidden_size)
    def forward(self, exp_seqs, exp_lengths,parent_var,lchild_var,rchild_var,hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        #cate_word_edge,cate_index_input)#B*cate+seq*cate+seq #B*cate_num
        keyword_seq=exp_seqs[0:keyword_num,:]
        exp_seq=exp_seqs[keyword_num:,:]

        exp_embedded = self.exp_embedding(exp_seq)  # S x B x E
        exp_embedded = self.em_dropout(exp_embedded)#S+10,B,E

        par_embedded= self.exp_embedding(torch.LongTensor(parent_var).transpose(0,1).cuda())
        par_embedded = self.em_dropout(par_embedded)#S+10,B,E
        lchild_embedded= self.exp_embedding(torch.LongTensor(lchild_var).transpose(0,1).cuda())
        lchild_embedded = self.em_dropout(lchild_embedded)#S+10,B,E
        rchild_embedded= self.exp_embedding(torch.LongTensor(rchild_var).transpose(0,1).cuda())
        rchild_embedded = self.em_dropout(rchild_embedded)#S+10,B,E

        exp_embedded_cat=torch.cat((exp_embedded,par_embedded,lchild_embedded,rchild_embedded),2)
        
        max_len=exp_seq.size(0)
        new_exp_lengths=[]
        for idx_ in range(len(exp_lengths)):
            new_exp_lengths.append(exp_lengths[idx_]-keyword_num)

        enc_hidden, context = self.encoder(exp_seq,exp_embedded_cat,new_exp_lengths)#[num_layer,B,H],[S,B,H]
        topic_context, topic_mask = self.topic_encoder(keyword_seq)#[S,B,H],[B,S]
        
        #print(context.size())
        #init_att = self.make_init_att(context)
        batch_size = context.size(1)
        h_size = (batch_size, self.hidden_size)
        init_att= Variable(context.data.new(*h_size).zero_(), requires_grad=False)#[b,h*2]

        enc_hidden = self.initer(enc_hidden[1]).unsqueeze(0)  # [1] is the last backward hiden#[1,B,H]

        src_pad_mask = Variable(exp_seq.data.eq(0).transpose(0, 1).float(), requires_grad=False,
                                volatile=False)


        return enc_hidden, context,topic_context, topic_mask,src_pad_mask,init_att

class Prediction(nn.Module):
    # no copy,just attention
    def __init__(self, hidden_size, problem_size, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.problem_size = problem_size

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.generate_out = nn.Linear(hidden_size, problem_size)
        self.generate_node= nn.Linear(hidden_size * 3, hidden_size)

        #self.attn = TreeAttn(hidden_size* 2, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.embedding = nn.Embedding(self.problem_size, self.hidden_size)
        # weights
        self.Ws = nn.Linear(hidden_size*2, hidden_size) # only used at initial stage
        self.Wo = nn.Linear(hidden_size, problem_size) # generate mode
        self.Wc = nn.Linear(hidden_size, hidden_size) # copy mode
        self.attn = ConcatAttention(hidden_size,hidden_size, hidden_size)
        self.topic_attn = ConcatAttention(hidden_size,hidden_size, hidden_size)
        self.rnn = StackedGRU(1, hidden_size*2,hidden_size, dropout)
        self.readout = nn.Linear(hidden_size*3,hidden_size)
        self.maxout = MaxOut(2)
        self.maxout_out = nn.Linear(hidden_size/2, problem_size)
        self.mix_gate = nn.Linear(hidden_size, 1)
    def forward(self, dec_input,enc_hidden, context,src_pad_mask,topic_context, topic_mask,cur_mix_context,precompute,topic_precompute):
        #self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums,indices,masked_index,num_score_constraints
        #enc_hidden#[1,B,H]#context [S,B,H]##topic_context, topic_mask [S,B,H],[B,S] #cur_mix_context #[b,h]
        embedded = self.embedding(dec_input)#B*1*N
        #embedded = self.dropout(embedded)

        self.attn.applyMask(src_pad_mask)
        self.topic_attn.applyMask(topic_mask)

        input_emb = torch.cat([embedded, cur_mix_context], 1)#B*2N
        output, enc_hidden = self.rnn(input_emb, enc_hidden) #B*2N,1*B*N  ->B*N, 2*B*N
        mix_gate_value = F.sigmoid(self.mix_gate(output))#B*1

        #print(output.size())
        #print(context)
        cur_context, attn, precompute = self.attn(output, context.transpose(0, 1), precompute)#B*H,B*S*H,# B*S*att_dim
        cur_topic_context, topic_attn, topic_precompute = self.topic_attn(output, topic_context.transpose(0, 1),
                                                                          topic_precompute)
        cur_mix_context = mix_gate_value * cur_context + (1 - mix_gate_value) * cur_topic_context# B*H

        readout = self.readout(torch.cat((embedded, output, cur_mix_context), dim=1))#B*N,B*N,B*N
        maxout = self.maxout(readout)
        output = self.dropout(maxout)#B*N
        output=self.maxout_out(output)
        return output,enc_hidden, cur_mix_context,precompute,topic_precompute

'''
class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class ConcatAttention(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim):
        super(ConcatAttention, self).__init__()
        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=False)
        self.linear_v = nn.Linear(att_dim, 1, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, precompute=None):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        if precompute is None:
            #print(context.size())
            precompute00 = self.linear_pre(context.contiguous().view(-1, context.size(2)))
            precompute = precompute00.view(context.size(0), context.size(1), -1)  # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp10 = precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim
        tmp20 = self.tanh(tmp10)  # batch x sourceL x att_dim
        energy = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL
        if self.mask is not None:
            # energy.data.masked_fill_(self.mask, -float('inf'))
            # energy.masked_fill_(self.mask, -float('inf'))   # TODO: might be wrong
            energy = energy * (1 - self.mask) + self.mask * (-1000000)
        score = self.sm(energy)
        score_m = score.view(score.size(0), 1, score.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(score_m, context).squeeze(1)  # batch x dim

        return weightedContext, score, precompute

    def extra_repr(self):
        return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
               + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
               + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
               + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
               + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'
class MaxOut(nn.Module):
    def __init__(self, pool_size):
        super(MaxOut, self).__init__()
        self.pool_size = pool_size

    def forward(self, input):
        """
        input:
        reduce_size:
        """
        input_size = list(input.size())
        assert input_size[-1] % self.pool_size == 0
        output_size = [d for d in input_size]
        output_size[-1] = output_size[-1] // self.pool_size
        output_size.append(self.pool_size)
        last_dim = len(output_size) - 1
        input = input.view(*output_size)
        input, idx = input.max(last_dim, keepdim=True)
        output = input.squeeze(last_dim)

        return output

    def extra_repr(self):
        return self.__class__.__name__ + '({0})'.format(self.pool_size)

    def __repr__(self):
        return self.__class__.__name__ + '({0})'.format(self.pool_size)

class CopyPrediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding
    # copy
    def __init__(self, hidden_size, problem_size, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.problem_size = problem_size

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.generate_out = nn.Linear(hidden_size, problem_size)
        self.generate_node= nn.Linear(hidden_size * 3, hidden_size)

        self.attn = TreeAttn(hidden_size* 2, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.embedding = nn.Embedding(self.problem_size, self.hidden_size)
        # weights
        self.Ws = nn.Linear(hidden_size*2, hidden_size) # only used at initial stage
        self.Wo = nn.Linear(hidden_size, problem_size) # generate mode
        self.Wc = nn.Linear(hidden_size, hidden_size) # copy mode

    def forward(self, dec_input,dec_hidden, encoder_outputs,seq_mask,input_concat):
        #self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums,indices,masked_index,num_score_constraints

        embedded = self.embedding(dec_input).unsqueeze(1)#B*1*N
        embedded = self.dropout(embedded)

        current_node=dec_hidden.unsqueeze(1)#B*N B*1*N
        current_embeddings = self.dropout(current_node)
        current_attn = self.attn(torch.cat((embedded, current_embeddings), 2).transpose(0, 1), encoder_outputs, seq_mask)#1*B*2H,S*B*H
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        
        batch_size = current_embeddings.size(0)
        in_seq_size = encoder_outputs.size(0)
        leaf_input = torch.cat((embedded,current_node, current_context), 2) #B*1*3H
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)#1*B*3H
        leaf_node=self.generate_node(leaf_input)#1*B*H
        output, hidden = self.gru(leaf_node.unsqueeze(0), dec_hidden.unsqueeze(0)) #1*B*N 1*B*N
        score_g = self.generate_out(output[0]) # [b x vocab_size] 

        score_c = torch.tanh(self.Wc(encoder_outputs).transpose(0, 1)) #B*S*H 
        score_c = torch.bmm(score_c, output.squeeze(0).unsqueeze(2)).squeeze(2)#B*S*H B*N*1 # [b x seq]# 
        
        encoded_mask = ((input_concat==0).float()*(-1000)).transpose(0,1) # [b x seq] #[00000,-1000,-1000]
        score_c = score_c + encoded_mask
        score = torch.cat([score_g,score_c],1) # [b x (vocab+seq)] score_gen score_copy
        probs = F.log_softmax(score,dim=1)
        prob_g = probs[:,:self.problem_size] # [b x vocab]
        prob_c = probs[:,self.problem_size:] # [b x seq]

        en = input_concat.transpose(0,1).unsqueeze(2) # [b x in_seq*1]
        one_hot = torch.FloatTensor(batch_size,in_seq_size,self.problem_size).zero_().cuda() # [b x in_seq x vocab]
        one_hot.scatter_(2,en,1) # one hot tensor: [b x seq x vocab] input_one_hot

        prob_c_to_g = torch.bmm(prob_c.unsqueeze(1),Variable(one_hot, requires_grad=False)).squeeze(1) # [b x 1 x vocab]
        out = prob_g + prob_c_to_g#[b x vocab] 
        '''
        idx_from_input = []#b*inseq 
        for i in range(batch_size):
            idx_from_input.append([int(input_concat[k][i]==dec_input[i]) for k in range(in_seq_size)])
        idx_from_input = Variable(torch.Tensor(np.array(idx_from_input, dtype=float)).cuda())#[b x seq]
        for i in range(batch_size):
            if idx_from_input[i].sum()>1:
                idx_from_input[i] = idx_from_input[i]/idx_from_input[i].sum() #if idx[i] has several token = lask token,mean

        attn = prob_c * idx_from_input #probc *idx b*seq
        attn = attn.unsqueeze(1) # [b x 1 x seq]
        weighted = torch.bmm(attn, encoder_outputs.transpose(0, 1)) # B*1*S B*S*H: [b x 1 x hidden*2]
        '''
        #output = F.log_softmax(self.generate_out(output[0]),dim=1)#B*N
        # return p_leaf, num_score, op, current_embeddings, current_attn
        return out, hidden.squeeze(0) #[b x vocab] B*N B*1*N
        

class MWPEncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(MWPEncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)

        exp_lengths=torch.LongTensor(input_lengths)
        lens, indices = torch.sort(exp_lengths, 0, True)
        len_list = lens.tolist()
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded[:,indices,:], len_list)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        _, _indices = torch.sort(indices, 0)
        pade_outputs=pade_outputs[:,_indices,:]

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        return pade_outputs, problem_output


class MWPPrediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(MWPPrediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)
        self.concat_encoder_outputs = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums,indices,masked_index):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N
        
        #encoder_outputs_knowledge=input_edge_batch.bmm(encoder_outputs.transpose(0, 1)) # B x S*S  B x S x H B x S x H
        #concat_encoder_outputs=torch.cat((encoder_outputs, encoder_outputs_knowledge.transpose(0,1)), dim=2)
        #current_attn = self.attn(current_embeddings.transpose(0, 1), concat_encoder_outputs, seq_mask) # B x S
        #current_context = current_attn.bmm(concat_encoder_outputs.transpose(0, 1))  #B x S S*B*N  B x 1 x N
        #current_context=self.concat_encoder_outputs(current_context)
        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        op = self.ops(leaf_input)
        return num_score, op, current_node, current_context, embedding_weight

class MWPGenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(MWPGenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class MWPMerge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(MWPMerge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Linear(in_features, out_features, bias=False)
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, graph_input, adj):
        #[B*S*H] [B*S*S]
        h = self.W(graph_input)
        # [batch_size, N, out_features]
        batch_size, N,  _ = h.size()
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, N) #B*S*N - B*S*1 - B*S*S B*S*S
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, N).transpose(1, 2) ##B*S*S
        e = self.leakyrelu(middle_result1 + middle_result2)
        attention = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(attention, dim=2)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, graph_input)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, output_vocab_len,embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, hidden_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + hidden_size+hidden_size*3, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + hidden_size+hidden_size*3, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + hidden_size+hidden_size*3, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + hidden_size+hidden_size*3, hidden_size)

        #self.generate_l = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)
        #self.generate_r = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)

        self.embeddings_middle=nn.Embedding(output_vocab_len,hidden_size)
        self.embeddings_middle_left=nn.Embedding(output_vocab_len,hidden_size)
        self.embeddings_middle_right=nn.Embedding(output_vocab_len,hidden_size)

    def forward(self, node_embedding, node_label, current_context,outputs_middle_predict):
        outputs_middle_self,outputs_middle_left,outputs_middle_right=outputs_middle_predict.split(1,2)
        middle_self_label=self.em_dropout(self.embeddings_middle(torch.argmax(outputs_middle_self.squeeze(2),dim=1)))
        middle_left_label=self.em_dropout(self.embeddings_middle_left(torch.argmax(outputs_middle_self.squeeze(2),dim=1)))
        middle_right_label=self.em_dropout(self.embeddings_middle_left(torch.argmax(outputs_middle_self.squeeze(2),dim=1)))
        

        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        #l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        #r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label,middle_self_label,middle_left_label,middle_right_label), 1)))
        #l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label,middle_self_label,middle_left_label,middle_right_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label,middle_self_label,middle_left_label,middle_right_label), 1)))
        #r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label,middle_self_label,middle_left_label,middle_right_label), 1)))
        #l_child = l_child * l_child_g
        #r_child = r_child * r_child_g
        return l_child, r_child, node_label_
class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.gcn = nn.Linear(hidden_size, hidden_size)
        self.gcn1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, tree_embed_mat, A_matrix):
        ##t*1*H,t*t
        #current_embeddings_stacks=node_embedding from generate or current_nums_embeddings from predict
        A_matrix=A_matrix+torch.eye(A_matrix.size(0)).cuda()
        d=A_matrix.sum(1)
        D=torch.diag(torch.pow(d,-1))
        A=D.mm(A_matrix)
        tree_embed_mat=self.em_dropout(tree_embed_mat.squeeze(0))#1*t*H

        new_tree_embed_mat=nn.functional.relu(self.gcn(A.mm(tree_embed_mat)))
        new_tree_embed_mat=nn.functional.relu(self.gcn1(A.mm(new_tree_embed_mat)))
        return new_tree_embed_mat#t*H
'''
class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree
'''


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len):
        """ d_model hidden 512  max_seq_len largedt len
        """
        super(PositionalEncoding, self).__init__()
        # PE matrix
        position_encoding = np.array([
          [pos / pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]  for pos in range(max_seq_len)])
        # odd line use sin,even line use cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # first line is all 0 as PAD positional encoding
        # word embedding add UNK as word embedding
        # use PAD to represent PAD position
        position_encoding=torch.FloatTensor(position_encoding)
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding),0)
        
        # +1 because adding PAD
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,requires_grad=False)
    def forward(self, input_len,category_num):
        """input_len  [BATCH_SIZE]
        """
        max_len = max(input_len)+category_num
        #tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # pad position add 0  range start by 1 to avoid pad(0)
        input_pos = torch.cuda.LongTensor([list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)



def attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled dot product self attention
        query: (batch_size, h, seq_len, d_k), seq_len can be either src_seq_len or tgt_seq_len
        key: (batch_size, h, seq_len, d_k), seq_len in key, value and mask are the same
        value: (batch_size, h, seq_len, d_k)
        mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, tgt_seq_len, tgt_seq_len) (legacy)
    """
    if print_dims:
        print("{0}: query: type: {1}, shape: {2}".format("attention func", query.type(), query.shape))
        print("{0}: key: type: {1}, shape: {2}".format("attention func", key.type(), key.shape))
        print("{0}: value: type: {1}, shape: {2}".format("attention func", value.type(), value.shape))
        print("{0}: mask: type: {1}, shape: {2}".format("attention func", mask.type(), mask.shape))
    d_k = query.size(-1)

    # scores: (batch_size, h, seq_len, seq_len) for self_attn, (batch_size, h, tgt_seq_len, src_seq_len) for src_attn
    scores = torch.matmul(query, key.transpose(-2, -1)/math.sqrt(d_k)) #B,H,S,S
    # print(query.shape, key.shape, mask.shape, scores.shape)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        #scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)##B,H,S,S
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0
        self.dim_per_head = model_dim//num_heads
        self.h = num_heads
        #self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(model_dim, model_dim)) for i in range(4)])
        self.key_dim=12
        self.value_dim=32
        self.linear_k = nn.Linear(model_dim, self.key_dim * num_heads)
        self.linear_v = nn.Linear(model_dim, self.value_dim * num_heads)
        self.linear_q = nn.Linear(model_dim, self.key_dim * num_heads)

        self.linear_x=nn.Linear(self.value_dim * num_heads,model_dim)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.a = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parameter(torch.Tensor([1]))
        
    def forward(self, query, key, value, mask=None):
        """
            query: (batch_size, seq_len, d_model), seq_len can be either src_seq_len or tgt_seq_len
            key: (batch_size, seq_len, d_model), seq_len in key, value and mask are the same
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len) or (batch_size, tgt_seq_len, tgt_seq_len) (legacy)
        """
        if print_dims:
            print("{0}: query: type: {1}, shape: {2}".format(self.__class__.__name__, query.type(), query.shape))
            print("{0}: key: type: {1}, shape: {2}".format(self.__class__.__name__, key.type(), key.shape))
            print("{0}: value: type: {1}, shape: {2}".format(self.__class__.__name__, value.type(), value.shape))
            print("{0}: mask: type: {1}, shape: {2}".format(self.__class__.__name__, mask.type(), mask.shape))
        if mask is not None:
            mask = mask.unsqueeze(1)#B,1,1,S
        nbatches = query.size(0)
        
        # 1) Do all linear projections in batch from d_model to (h, d_k)
        key = self.linear_k(key).view(nbatches, -1, self.h, self.key_dim).transpose(1,2)#B*S*(dim_per_head * num_heads)
        value = self.linear_v(value).view(nbatches, -1, self.h, self.value_dim).transpose(1,2)#B,H,S,dim
        query = self.linear_q(query).view(nbatches, -1, self.h, self.key_dim).transpose(1,2)
        #query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch
        x, p_attn = attention(query, key, value, mask=mask, dropout=self.dropout) # (batch_size, h, seq_len, d_k),#B,H,S,S
        if print_dims:
            print("{0}: x (after attention): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))

        # 3) Concatenate and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.value_dim)##B,S,H,dim
        x = self.linear_x(x) # (batch_size, seq_len, d_model)
        if print_dims:
            print("{0}: x (after concatenation and linear): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        return x,p_attn

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        return self.w2(self.dropout(F.relu(self.w1(x))))

class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, n_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(n_features))
        self.b_2 = nn.Parameter(torch.zeros(n_features))
        self.eps = eps
        
    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean)/(std + self.eps) + self.b_2
class EncoderLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.norm = LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, attn_mask=None):
        """norm -> self_attn -> dropout -> add -> 
        norm -> feed_forward -> dropout -> add"""
        # self attention  ##[B,S,H] B*S*S
        norm_inputs=self.norm(inputs)
        context, attention = self.attention(norm_inputs, norm_inputs, norm_inputs, attn_mask)#[B,S,H] B*S*S
        context=self.dropout(context)
        context=inputs+context
        # feed forward network
        output=self.norm(context)
        output = self.feed_forward(output)#[B,S,H] #ff x+output
        output=self.dropout(output)
        output=context+output
        return output, attention
