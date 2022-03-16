# coding: utf-8
import random
import json
import copy
import re
import time
import math
from classes.EquationConverter import EquationConverter
from expressions_transfer import compute_prefix_expression,from_infix_to_prefix
#from textrank4zh import TextRank4Keyword, TextRank4Sentence
from collections import Counter
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

PAD_token = 0

cons_mode=[1,2,3]

keyword_num=5

class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence,flag="in"):  # add words of sentence to vocab
        for word in sentence:
            #if re.search("N\d+|NUM|\d+", word):
            if re.search("NUM", word):
                continue
            if flag=="out":
                if re.search("N\d+|NUM|\d+", word):
                    continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        #print('keep_words %s / %s = %.4f' % (
        #    len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        #))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1


    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD","EOS","SOS", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i
    def build_input_lang_justout(self, out_index2word):  # build the input lang vocab and dict
        new_index2word=[]
        for word in self.index2word:
            if word not in out_index2word:
                new_index2word.append(word)
        self.index2word=out_index2word+new_index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i
    def build_output_lang(self, generate_num, copy_nums,tfidf_list,in_index2word):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)+4
        print(self.index2word)
        self.index2word = ["PAD", "EOS","SOS","other"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)]+[ "UNK"]
        word_list=[]
        for word in tfidf_list:
            if word not in self.index2word:
                word_list.append(word)
        '''
        for word in category_keyword_dict:
            if word not in self.index2word:
                word_list.append(word)
            for keyword in category_keyword_dict[word]:
                if keyword not in self.index2word:
                    word_list.append(keyword)
        '''
        self.index2word =self.index2word +  word_list
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i
    '''
    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)
        word_list=[]
        for word in category_keyword_dict:
            if word not in self.index2word:
                word_list.append(word)
            for keyword in category_keyword_dict[word]:
                if keyword not in self.index2word:
                    word_list.append(keyword)
        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]+word_list
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i
    '''
    def build_output_lang_for_exp(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)+4

        self.index2word = ["PAD", "EOS","SOS","other"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)]  + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename,'r')
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data
def one_sentence_clean(text):
    # Clean up the data and separate everything by spaces
    text = re.sub(r"(?<!Mr|Mr|Dr|Ms)(?<!Mrs)(?<![0-9])(\s+)?\.(\s+)?", " . ",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"(\s+)?\?(\s+)?", " ? ", text)
    text = re.sub(r",", "", text)
    text = re.sub(r"^\s+", "", text)
    text = text.replace('\n', ' ')
    text = text.replace("'", " '")
    text = text.replace('%', ' percent')
    text = text.replace('$', ' $ ')
    text = re.sub(r"\.\s+", " . ", text)
    text = re.sub(r"\s+", ' ', text)

    sent = []
    for word in text.split(' '):
        try:
            print("not_w2n")
            sent.append(str(w2n.word_to_num(word)))
        except:
            sent.append(word)

    return ' '.join(sent)


def to_lower_case(text):
    # Convert strings to lowercase
    try:
        return text.lower()
    except:
        return text

def transform_MaWPS(filename):
    print("\nWorking on MaWPS data...")
    problem_list = []
    MAWPS=[]
    with open(filename) as fh:
        json_data = json.load(fh)

    has_id=[]
    for i in range(len(json_data)):
            # A MWP
        problem = []
        has_all_data = True
        data = json_data[i]
        if "sQuestion" in data and "lEquations" in data and "lSolutions" in data:
            for key, value in data.items():
                if key == "sQuestion" or key == "lEquations" or key == "lSolutions":
                    if len(value) == 0 or (len(value) > 1 and (key == "lEquations" or key == "lSolutions")):
                        has_all_data = False

                    if key == "sQuestion":
                        desired_key = "question"
                        #value = one_sentence_clean(value)
                        problem.append((desired_key,to_lower_case(value)))
                    elif key == "lEquations":
                        if len(value) > 1:
                            continue
                        desired_key = "equation"
                        value = value[0]
                        problem.append((desired_key,to_lower_case(value)))
                    elif key == "lSolutions":
                        desired_key = "answer"

                        problem.append((desired_key,to_lower_case(value[0])))
                    else:
                        problem.append((desired_key,to_lower_case(value)))

        if has_all_data == True and problem != []:
            problem_list.append(problem)
            MAWPS.append(problem)
        else:
            print(data)

    print("Retrieved",len(problem_list),len(json_data),"problems.")

    print("...done.\n")
    print(MAWPS[0])
    return MAWPS


def transform_nois(filename1,filename2):
    print("\nWorking on MaWPS data...")
    problem_list = []
    MAWPS=[]
    with open(filename1) as fh:
        json_data = json.load(fh)

    has_id=[]
    for i in range(len(json_data)):
            # A MWP
        problem = []
        has_all_data = True
        data = json_data[i]
        if "sQuestion" in data and "lEquations" in data and "lSolutions" in data:
            for key, value in data.items():
                if key == "sQuestion" or key == "lEquations" or key == "lSolutions":
                    if len(value) == 0 or (len(value) > 1 and (key == "lEquations" or key == "lSolutions")):
                        has_all_data = False

                    if key == "sQuestion":
                        desired_key = "question"
                        #value = one_sentence_clean(value)
                        problem.append((desired_key,to_lower_case(value)))
                    elif key == "lEquations":
                        if len(value) > 1:
                            continue
                        desired_key = "equation"
                        value = value[0]
                        problem.append((desired_key,to_lower_case(value)))
                    elif key == "lSolutions":
                        desired_key = "answer"

                        problem.append((desired_key,to_lower_case(value[0])))
                    else:
                        problem.append((desired_key,to_lower_case(value)))

        if has_all_data == True and problem != []:
            problem_list.append(problem)
            MAWPS.append(problem)
            has_id.append(data["iIndex"])
        else:
            print(data)

    with open(filename2) as fh:
        json_data = json.load(fh)

    for i in range(len(json_data)):
            # A MWP
        problem = []
        has_all_data = True
        data = json_data[i]
        if data["iIndex"] not in has_id:
            if "sQuestion" in data and "lEquations" in data and "lSolutions" in data:
                for key, value in data.items():
                    if key == "sQuestion" or key == "lEquations" or key == "lSolutions":
                        if len(value) == 0 or (len(value) > 1 and (key == "lEquations" or key == "lSolutions")):
                            has_all_data = False

                        if key == "sQuestion":
                            desired_key = "question"
                            #value = one_sentence_clean(value)
                            problem.append((desired_key,to_lower_case(value)))
                        elif key == "lEquations":
                            if len(value) > 1:
                                continue
                            desired_key = "equation"
                            value = value[0]
                            problem.append((desired_key,to_lower_case(value)))
                        elif key == "lSolutions":
                            desired_key = "answer"

                            problem.append((desired_key,to_lower_case(value[0])))
                        else:
                            problem.append((desired_key,to_lower_case(value)))

            if has_all_data == True and problem != []:
                problem_list.append(problem)
                MAWPS.append(problem)
                has_id.append(data["iIndex"])
            else:
                print(data)
    print(has_id)

    print("Retrieved",len(problem_list),len(json_data),"problems.")

    print("...done.\n")
    print(MAWPS[0])
    return MAWPS

WORDS_FOR_OPERATORS = False
def word_operators(text):
    if WORDS_FOR_OPERATORS:
        rtext = re.sub(r"\+", "add", text)
        rtext = re.sub(r"(-|\-)", "subtract", rtext)
        rtext = re.sub(r"\/", "divide", rtext)
        rtext = re.sub(r"\*", "multiply", rtext)
        return rtext
    return text

def convert_to(l, t):
    output = []

    for p in l:
        p_dict = dict(p)

        ol = []

        discard = False

        for k, v in p_dict.items():
            if k == "equation":
                convert = EquationConverter()
                convert.eqset(v)

                if t == "infix":
                    ov = convert.expr_as_infix()
                elif t == "prefix":
                    ov = convert.expr_as_prefix()
                elif t == "postfix":
                    ov = convert.expr_as_postfix()
                #print("###############")
                #print(v)
                #print("##########*********")
                #print(ov)
                #convert.show_expression_tree()
                #print("Infix with no parenthesis:", convert.expr_as_infix())
                #print("Prefix:", convert.expr_as_prefix())
                #print("Postfix:", convert.expr_as_postfix())


                if re.match(r"[a-z] = .*\d+.*", ov):
                    ol.append((k, word_operators(ov)))
                else:
                    ol.append((k,ov))
                    
                    #discard = True
            else:
                ol.append((k, v))

        if not discard:
            output.append(ol)
    #print(len(output))
    #print(output[0])

    return output


def transfer_num_MaWPS(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    count_empty=0

    UNK2word_vocab={}
    #input1=open("data//UNK2word_vocab","r").readlines()
    #for word in input1:
    #    UNK2word_vocab[word.strip().split("###")[0]]=word.strip().split("###")[1]
    #    
    print(len(data))
    data_postfix = convert_to(data, "infix")
    print(len(data_postfix))
    
    for i in range(len(data)):
        if i%100==0:
            print("***************")
            print(data[i])
            print(data_postfix[i])
    
    for d in data_postfix:
        d_dict = dict(d)
        nums = []
        input_seq = []
        seg_line = d_dict["question"].encode("UTF-8").strip()
        #for UNK_word in UNK2word_vocab:
        #    if UNK_word in seg_line:
        #        seg_line=seg_line.replace(UNK_word,UNK2word_vocab[UNK_word])
        seg=seg_line.split(" ")
        equations = d_dict["equation"]


        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                if len(s)>0:
                    input_seq.append(s)
                else:
                    count_empty=count_empty+1
        if copy_nums < len(nums):
            copy_nums = len(nums)

        equation_list=equations.strip().split()
        #print(equations)
        equ_line=""
        not_num=0
        for equ_ in equation_list:
            if equ_ in ["+","-","*","/","^"]:
                equ_line+=equ_
            elif equ_ =="=":
                #print("equ==")
                #print(seg_line)
                #print(equations)
                not_num=1
                break
            else:
                number_list_temp="0123456789"
                if equ_[0] not in number_list_temp:
                    #print("equ.isdigit")
                    #print(seg_line)
                    #print(equations)
                    not_num=1
                    break
                flag=0
                for num_ in nums:
                    if float(num_)==float(equ_):
                        equ_line+=num_
                        flag=1
                        break
                if flag==0:
                    equ_line+=equ_
        if not_num==1:
            continue
        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res
        out_seq = seg_and_tag(equ_line)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
                input_seq[i]=str("N"+str(len(num_pos)-1))
        assert len(nums) == len(num_pos)
        #unit_list,rule3_list=get_constraint_unit(input_seq,num_pos)
        #new_input_seq,new_num_pos =get_new_inputseq(input_seq,unit_list,rule3_list,num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        #* N0 N1 + N0 N2 * N1 N2 爱心 超市 运 来 NUM 千克 大米 ， 卖 了 NUM 天 后 ， 还 剩 NUM 千克 ， 平均 每天 卖 大米 多少 千克 ？
        #print(" ".join(new_input_seq))
        
        if len(equation_list)!=0:
            if compute_prefix_expression(from_infix_to_prefix(equation_list))-float(d_dict["answer"])> 1e-4:
                #print(seg_line)
                #print(equations)
                #print(out_seq)
                #print(d_dict["answer"])
                count_empty+=1
            else:
                pairs.append((input_seq, out_seq, nums, num_pos))
        else:
            print(seg_line)
            print(equations)
            count_empty+=1
        
        #print(input_seq)
        #print(out_seq)
        #print(nums)
        #print(num_pos)
    print("count_empty")
    print(count_empty)
    print(len(data))
    print(len(pairs))
    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    print(temp_g)
    return pairs, temp_g, copy_nums

def transfer_num_nois(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    count_empty=0

    UNK2word_vocab={}
    #input1=open("data//UNK2word_vocab","r").readlines()
    #for word in input1:
    #    UNK2word_vocab[word.strip().split("###")[0]]=word.strip().split("###")[1]
    #    
    print(len(data))
    #data_postfix = convert_to(data, "infix")
    #print(len(data_postfix))
    
    
    for d in data:
        d_dict = dict(d)
        nums = []
        input_seq = []
        seg_line = d_dict["question"].encode("UTF-8").strip()
        #for UNK_word in UNK2word_vocab:
        #    if UNK_word in seg_line:
        #        seg_line=seg_line.replace(UNK_word,UNK2word_vocab[UNK_word])
        seg=seg_line.split(" ")
        equations = d_dict["equation"].split("=")[1]


        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                if len(s)>0:
                    input_seq.append(s)
                else:
                    count_empty=count_empty+1
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    flag=0
                    for num_ in nums:
                        if float(num_)==float(st_num):
                            res.append("N"+str(nums.index(num_)))
                            flag=1
                            break
                    if flag==0:
                        res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                if ss!=" ":
                    res.append(ss)
            return res
        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
                input_seq[i]=str("N"+str(len(num_pos)-1))
        assert len(nums) == len(num_pos)
        #unit_list,rule3_list=get_constraint_unit(input_seq,num_pos)
        #new_input_seq,new_num_pos =get_new_inputseq(input_seq,unit_list,rule3_list,num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        #* N0 N1 + N0 N2 * N1 N2 爱心 超市 运 来 NUM 千克 大米 ， 卖 了 NUM 天 后 ， 还 剩 NUM 千克 ， 平均 每天 卖 大米 多少 千克 ？
        #print(" ".join(new_input_seq))
        
        if len(equations)!=0:
            #print(equations)
            #print(compute_prefix_expression(from_infix_to_prefix(equations.strip().split())))
            #print(d_dict["answer"])
            if compute_prefix_expression(from_infix_to_prefix(equations.strip().split()))-float(d_dict["answer"])> 1e-4:
                print(seg_line)
                print(equations)
                print(out_seq)
                print(d_dict["answer"])
                count_empty+=1
            else:
                pairs.append((input_seq, out_seq, nums, num_pos))
        else:
            print(seg_line)
            print(equations)
            count_empty+=1
        
        #print(input_seq)
        #print(out_seq)
        #print(nums)
        #print(num_pos)
    print("count_empty")
    print(count_empty)
    print(len(data))
    print(len(pairs))
    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    print(temp_g)
    print(pairs[0])
    print(pairs[1])
    return pairs, temp_g, copy_nums


def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    count_empty=0

    UNK2word_vocab={}
    input1=open("data//UNK2word_vocab","r").readlines()
    for word in input1:
        UNK2word_vocab[word.strip().split("###")[0]]=word.strip().split("###")[1]

    for d in data:
        nums = []
        input_seq = []
        seg_line = d["segmented_text"].encode("UTF-8").strip()
        for UNK_word in UNK2word_vocab:
            if UNK_word in seg_line:
                seg_line=seg_line.replace(UNK_word,UNK2word_vocab[UNK_word])
        seg=seg_line.split(" ")
        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                if len(s)>0:
                    input_seq.append(s)
                else:
                    count_empty=count_empty+1
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)


        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
                input_seq[i]=str("N"+str(len(num_pos)-1))
        assert len(nums) == len(num_pos)
        #unit_list,rule3_list=get_constraint_unit(input_seq,num_pos)
        #new_input_seq,new_num_pos =get_new_inputseq(input_seq,unit_list,rule3_list,num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        #* N0 N1 + N0 N2 * N1 N2 爱心 超市 运 来 NUM 千克 大米 ， 卖 了 NUM 天 后 ， 还 剩 NUM 千克 ， 平均 每天 卖 大米 多少 千克 ？
        #print(" ".join(new_input_seq))
        pairs.append((input_seq, out_seq, nums, num_pos))
    print("count_empty")
    print(count_empty)
    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)



# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    #if "NUM" in lang.index2word and not tree:
    #    res.append(lang.word2index["EOS"])#if is a question,add a EOS
    return res
def indexes_to_sentence(lang, index_list, tree=False):
    res = []
    for index in index_list:
        if index < lang.n_words:
            res.append(lang.index2word[index])
        if index==lang.word2index["EOS"]:
            return res
    return res

def save_keywords():
    out=open("data//category_keyword_dict.txt","w")
    '''
    category_keyword=["Tracing###相遇,相对,相反,相背,相向","Engineering###工程,零件,工程队,公路,修路",
    "Distance&Speed###速度,千米,路程,相距,全程","Interval###利润","SolidGeometry###体积,侧面积,横截面,表面积,圆柱,长方体",
    "Circle###半径,圆,直径,周长","PlaneGeometry###三角形,正方形,长方形,边长,四边形,底,高","Profit###间隔,隔",
    "InterestRate###利息,利率","Production###超产,减产","Discount###原价,降价,打折,成本","Page###页",
    "TakeAway###取出","Remain###第一天,第二天,还剩","Concentration###浓度,酒精,糖水,蜜,盐水,盐,硫酸","Library###图书馆,书,册,本",
    "Permutation###白球,红球,黑球","Plan###原计划","Storage###仓库,仓"
    "Animal###养,只","Plant###养,只,棵,树,公顷,蔬菜,黄瓜,西红柿,土豆","Purchase###买,单价,价格,元","Student###男生,女生,学生",
    "Fraction###分数,分子,分母,因数,甲数,乙数,一个数,被除数,商","Weight###重,千克,吨"]
    '''
    category_keyword=["Tracing###相遇,相对,相反,相背,相向,反向",
"Engineering###工程,零件,工程队,公路,修路",
"Distance&Speed###速度,路程,相距,甲地,乙地,去 时,返回 时,甲 站,乙 站,全程,甲 港,乙 港,A 城,B 城,C 城",
"Discount###利润,原价,降价,打折,成本",
"SolidGeometry###体积,侧 面积,横截面,表面积,圆柱,长方体,圆柱形,正方体,底面 直径,棱 长,底面 半径,侧面,底面",
"Circle###半径,圆,直径,周长,圆形",
"PlaneGeometry###三角形,正方 形,长方 形,边长,菱形,高度,占地面积,平 行四边形,梯形,直角 梯形,直角三角形,直角 边,上 底,下 底,面积",
"Interval###间隔,隔,每隔",
"InterestRate###利息,利率,超产,减产,存期,年利率,本金,税率,税后,利息税,定期,整存整取,贷款,到期,年 率",
"Plan###原 计划,计划,现在",
"Concentration###浓度,酒精,糖水,蜜,盐水,盐,硫酸,溶液",
"Probability###含量,出米率,出油率,出粉率",
"Permutation###白 球,红球,黑 球,黄 球,绿 球,蓝球,摸 出,摸 到",
"Storage###仓库,仓,粮仓,甲 仓,乙 仓,甲 仓库,乙 仓库",
"FeedAnimal###养,饲养,饲养 厂,养殖场,饲养 小组,畜牧场,养鸡,养鸭,养 鹅,养猪,养牛",
"Purchase###买,单价,价格,起步价",
"Fraction###分数,分子,分母,因数,甲 数,乙 数,丙 数,一 个数,被除数,除数,商,余数,余,积,两 数,这 个数,已知,计算 ：,数 =,差 =,连续 减去,加数,一个 加数,另 一个 加数,约 分,化 简分数,化简,小数,倒数,内 项,外 项,乘数",
"Weight###重",
"Remain###第一天,第 二 天,两天 共,两天 一共,第一次,第 二 次,原有,取出,还 剩,剩下,余下,第一周,第 二 周",
"Person###男生,女生,学生,菜农,管理员,小朋友,孩子,儿童,老师,父亲,爸爸,妈妈,爷爷,奶奶,姑姑,叔叔,阿姨,弟弟,妹妹,哥哥,姐姐,外公,外婆,一年级,二年级,三年级,四年级,五年级,六年级,乘客,中 年级,高年级,一班,二 班,特快 列车,火车,汽车,客车,货车,轮船,卡车,营业员,大客车,小轿车,同学",
"FoodProduct###素菜,味精,大米,饼,果品,面粉,月饼,方便面,醋,白糖,饼干,酱油,糖果,葵花籽,奶粉,馒头,奶油,食品,小米,黄花,奶糖,食盐,巧克力,蛋糕,面包,饲料,蜂蜜,花生油,冰棒,肉,油,油壶,油桶,糖,水果 糖,酥糖,红糖,什锦糖,花生 糖,果汁,苹果汁,橙汁,酸奶,菠萝汁",
"Place###图书馆,水果店,篮子,粮店,小学,养鸡场,商店,动物园,果园,牧场,农场,菜园,花园,菜地,花圃,饲养场,林场,食堂,饭店,菜市场,菜场,造纸厂,超市,剧院,夏令营,中学,邮局,分行,公司,敬老院,幼儿园,厂房,化工厂,俱乐部,奶牛场,商家,大学,印刷厂,糖厂,宾馆,旅行社,酒店,剧场,车间,博物馆,少年宫,书店,纺织厂,水泥厂,市场,工厂,批发市场,银行,小班,厂家,服装店,发电厂,制造厂,电影院,学校,医院,百货商店,商场,食品店,保险公司,加工厂,校园,停 车场",
"Commodity###书,化肥,煤,课桌椅,课桌,椅子,故事书,洗衣粉,汽油,石油,味精,电视,毛巾,马桶,农具,铁轨,棋类,分针,集装箱,筷子,金牌,石子,油漆,皮球,家电,彩带,木材,木板,录音机,胶卷,盒子,铁皮,炉灶,电器,绳子,信封,杯子,气球,茶杯,毛线,棉花,香皂,航模,白布,布料,茶具,彩电,木头,光盘,电杆,花布,空调器,手表,书包,铁丝,螺母,瓶子,丝带,原料,电冰箱,水箱,水管,电话,电视机,柴油,纸箱,塑料袋,笼子,涂料,电线,电扇,文具,手机,玻璃杯,石头,雨伞,盘子,容器,钢材,钢板,帐篷,软盘,微波炉,纸盒,彩球,瓷砖,伞,温度计,奖牌,油桶,体育用品,易拉罐,计算器,热水瓶,日光灯,木料,邮票,水壶,布袋,红领巾,冰箱,酒精,物资,象棋,煤炭,文具盒,玉,食盐,蜡烛,箱子,抹,筐,塑料,袋子,电风扇,天平,风筝,银牌,粮食,材料,肥皂,灯笼,模型,珠子,空调,水龙头,玻璃,玩具,花瓶,砝码,钢筋,珍珠,洗衣机,窗帘,花生油,钢管,水桶,水泥,乒乓球 拍,羽毛球 拍,篮球,足球,羽毛球,乒乓球,齿轮,练习本,铅笔,钢笔,圆珠笔,自动笔,水笔,尺子,圆规,橡皮,纸 杯",
"AnimalProduct###鸡蛋,鸭蛋,鹅蛋,蛋,牛奶,奶",
"Animal###鸡,鸭,鹅,鹏,燕,信鸽,鸟,肉鸡,啄木鸟,燕子,公鸡,鸵鸟,山雀,鸭子,鸟类,鸽子,母鸡,麻雀,松鼠,海豚,大象,熊猫,斑马,梅花鹿,大熊猫,猴子,骆驼,鲸鱼,熊,青蛙,长颈鹿,狮子,北极熊,鲸,野牛,羚羊,猴,象,龙,虎,野兔,牦牛,狼,老虎,奶牛,猪,兔子,马,山羊,家畜,羊,兔,绵羊,猫,牛,肉牛,奶羊,狗,黄牛,蜜蜂,灰 兔,白兔,小 猴,大 猴,红 金鱼,黄 金 鱼,黑 兔,小猪,大 猪,穿山甲,白蚁,鱼,蚕,蜻蜒,昆虫",
"ClothProduct###男装,衣服,服装,羊毛衫,裙子,裤子,袜子,皮鞋,衬衣,衬衫,草帽,棉衣,皮衣,纽扣,风衣,上衣,毛线,白布,布料,帽子,花布,运动服,围巾,童装,毛衣,口袋,套装,女装,手套,短裤,凉鞋,校服,西服,连衣裙,大衣,布鞋,羽绒服,西装",
"Plant###蔬菜,杨树,茶叶,桃树,柏树,苹果树,槐树,梨树,柳树,水杉,果树,松树,樟树,花生,小麦,茄子,大豆,大蒜,土豆,玉米,菠菜,黄豆,西红柿,白菜,黄瓜,葵花籽,油菜籽,番茄,芹菜,蘑菇,青菜,种子,大白菜,萝卜,辣椒,油菜,豆角,松果,谷子,枣,枣树,花,草,芝麻,盆花,郁金香,草地,浮萍,鲜花,兰,草坪,花朵,月季,草原,青草,黄花,月季花,兰花,萍,花瓶,花圃,花坛,玫瑰,玫瑰花,菊花,百合花",
"Fruit###橙子,桃树,桃,苹果树,柑桔,果树,水果,苹果,香蕉,菠萝,荔枝,梨子,桔子,草莓,西瓜,李,葡萄,芒果,橘子,梨,梅,桃子",
"EachUnit###每册,每本,每页,每袋,每箱,每根,每堆,每排,每人,每棵,每桶",
"Unit###千克,吨,册,本,页,袋,元,箱,根,堆,排,人,棵,公顷,平方米,桶,盏,立方分米,立方米,立方厘米,厘米,分 米,碗,平 方分米,平 方米,平 方厘米,米,千米,公里",
"Color###彩,颜色,黑色,蓝,红色,金,绿色,灰,翠,色,花,白色,绿,黑白,红,青,白,黄色,银,蓝色,黑,彩色,丹,黄"
    ]

    for word in category_keyword:
        out.write(word+"\n")

def read_keywords():
    category_keyword_dict={}
    category_keyword_order=[]
    category_lang_list=[]
    input1=open("data//category_keyword_dict.txt","r").readlines()
    for word in input1:
        word_list=word.strip().split("###")[1].split(",")
        word_set=[]
        for keyword in word_list:
            if keyword not in word_set and len(keyword)>0:
                word_set.append(keyword)
            if keyword not in category_lang_list and len(keyword)>0:
                category_lang_list.append(keyword)
            #else:
            #    print(keyword)
            #    print(word)
        category_keyword_dict[word.strip().split("###")[0]]=word_set
        category_keyword_order.append(word.strip().split("###")[0])
        category_=word.strip().split("###")[0]
        if category_ not in category_lang_list and len(category_)>0:
            category_lang_list.append(category_)
    return category_keyword_dict,category_keyword_order,category_lang_list

def save_train_cate_problem_dict(train_cate_problem_dict,train_cate_problem_order):
    train_num_list=[]
    max_num=0
    for word in train_cate_problem_dict:
        max_num+=len(train_cate_problem_dict[word])
    #print("category_num:"+str(len(train_cate_problem_dict.keys())))
    for word in train_cate_problem_order:
        if word in train_cate_problem_dict:
            filepath="data//category//"+word+".txt"
            out=open(filepath,"w")
            for problem in train_cate_problem_dict[word]:
                out.write(problem+"\n")
            out.close()
            cate_prob_num=len(train_cate_problem_dict[word])
            #print(word+" "+str(cate_prob_num)+" "+str(max_num)+" "+str(float(cate_prob_num)/(max_num)))
    #word="other"
    #cate_prob_num=len(train_cate_problem_dict[word])
    #print(word+" "+str(cate_prob_num)+" "+str(max_num)+" "+str(float(cate_prob_num)/(max_num)))

def annotate_problem_by_keyword(problem,category_keyword_dict,category_keyword_order,cate_problem_dict):
    cate_problem_keyword=[]
    problem_line=" ".join(problem)
    problem_word=" "+problem_line+" "
    all_keyword_list=[]
    for word in category_keyword_order:
        all_keyword_list.extend(category_keyword_dict[word])
    for index,word in enumerate(category_keyword_order):
        if "Unit" not in word and index<=20:
            keyword_list=category_keyword_dict[word]
            for keyword in keyword_list:
                keyword_space=" "+keyword+" "
                if keyword_space in problem_word:
                    cate_problem_keyword.append(word)
                    cate_problem_keyword.append(keyword)
                    for keyword_ in all_keyword_list:
                        keyword_space=" "+keyword_+" "
                        if keyword_space in problem_word and keyword_ not in cate_problem_keyword:
                            cate_problem_keyword.append(keyword_)
                    if word in cate_problem_dict:
                        cate_problem_dict[word].append(problem_line)
                    else:
                        cate_problem_dict[word]=[problem_line]
                    return cate_problem_dict,cate_problem_keyword
    word="other"
    cate_problem_keyword.append(word)
    for keyword_ in all_keyword_list:
        keyword_space=" "+keyword_+" "
        if keyword_space in problem_word and keyword_ not in cate_problem_keyword:
            cate_problem_keyword.append(keyword_)
    if word in cate_problem_dict:
        cate_problem_dict[word].append(problem_line)
    else:
        cate_problem_dict[word]=[problem_line]
    return cate_problem_dict,cate_problem_keyword


class IDF():

    def __init__(self,text):
        #self.base_path = os.getcwd()
        self.text=text
        #self.idf_input_path  = os.path.join(self.base_path + '/data/tf_idf_input/') 
        self.stop_word_file  = 'data/stopwords.txt' 
        self.idf_output_path = 'data/'
        self.idf()

    def idf(self):
        all_chars_dict, total = self._read_file()
        with open(self.idf_output_path + 'idf.txt', 'w') as wf:
            for char,value in all_chars_dict.items():
                #if char > u'\u4e00' and char <= u'\u9fa5':
                p = math.log(float(total) / (value + 1))
                wf.write(char + ' ' + str(p) + '\n')


    def _read_file(self):
        stop_words = self._stop_words()
        #file = open(file, 'r', encoding='utf-8',errors='ignore') .read()
        #content = file.replace("\n","").replace("\u3000","").replace("\u00A0","").replace(" ","")
        #content_chars = jieba.cut(content, cut_all= True)
        all_dict = {}
        total=0
        for sen in self.text:
            word_list=[]
            for char in sen:
                if char not in stop_words:
                    word_list.append(char)
            words = list(set(word_list))
            tmp_dict = {char: 1 for char in words}
            total+=1
            for tmp_char in tmp_dict: 
                num = all_dict.get(tmp_char, 0)
                all_dict[tmp_char] = num + 1
        return all_dict, total

    def _stop_words(self):
        stop_words = []
        with open(self.stop_word_file, 'r') as f:
            words = f.readlines()
            for word in words:
                word = word.replace("\n","").strip()
                stop_words.append(word)
        return stop_words


class TFIDF():
    def __init__(self, topK=5):
        #self.base_path = os.getcwd()
        #self.file_path = os.path.join(self.base_path, file)  # 需提取关键词的文件, 默认在根目录下
        self.stop_word_file = 'data/stopwords.txt'  # 停用词
        self.idf_file = 'data/idf.txt'  # 停用词
        #self.idf_file = os.path.join(self.base_path + '/data/idf_out/idf.txt')  # idf文件
        self.idf_freq = {}
        self._load_idf()
        self.topK = topK

        self.stop_words = self._get_stop_words()

    def key_abstract(self,sentence):
        # 获取处理后数据
        data = [char for char in sentence if char not in self.stop_words]
        total_count = data.__len__()
        list_s = Counter(data).most_common()
        keywords = {}
        for chars in list_s:
            char_tmp = {}
            char_tmp[chars[0]] = (chars[1] / total_count) * self.idf_freq.get(chars[0],self.mean_idf)  # TF * IDF(IDF不存在就取平均值)值
            keywords.update(char_tmp)
        tags = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        if self.topK:
            return [tag[0] for tag in tags[:self.topK]]
        else:
            return [tag[0] for tag in tags]


    def _load_idf(self):  # 从文件中载入idf
        cnt = 0
        with open(self.idf_file, 'r') as f:
            for line in f:
                try:
                    word, freq = line.strip().split(' ')
                    cnt += 1
                except Exception as e:
                    pass
                self.idf_freq[word] = float(freq)
        print('Vocabularies loaded: %d' % cnt)
        self.mean_idf = sum(self.idf_freq.values()) / cnt

    def _get_stop_words(self):
        stop_words = []
        with open(self.stop_word_file, 'r') as f:
            words = f.readlines()
            for word in words:
                word = word.replace("\n", "").strip()
                stop_words.append(word)
        return stop_words

def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    output_exp_lang=Lang()
    train_pairs = []
    test_pairs = []

    save_keywords()
    category_keyword_dict,category_keyword_order,category_lang_list=read_keywords()
    train_cate_problem_list=[]
    train_cate_problem_dict={}
    test_cate_problem_list=[]
    test_cate_problem_dict={}
    #print("Indexing words...")
    text=[]
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0],"in")
            output_lang.add_sen_to_vocab(pair[1],"out")
            output_exp_lang.add_sen_to_vocab(pair[1],"out")
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0],"in")
            output_lang.add_sen_to_vocab(pair[1],"out")
            output_exp_lang.add_sen_to_vocab(pair[1],"out")
        text.append(pair[0])
    '''
    idf=IDF(text)
    tfidf=TFIDF(5)
    tfidf_list=[]
    for sen in text:
        #print("**********************")
        #print(" ".join(sen))
        #print(" ".join(tfidf.key_abstract(sen)))
        for key_ in tfidf.key_abstract(sen):
            if key_ in input_lang.index2word and key_ not in tfidf_list:
                tfidf_list.append(key_)
    with open('data/tfidf_list.txt', 'w') as wf:
        for char in tfidf_list:
            wf.write(char+ '\n')
    '''
    input_lang.build_input_lang(trim_min_count)
    #output_lang.build_output_lang(generate_nums, copy_nums,tfidf_list,input_lang.index2word)
    output_lang.build_output_lang(generate_nums, copy_nums,category_lang_list,input_lang.index2word)
    output_exp_lang.build_output_lang_for_exp(generate_nums, copy_nums)

    #input_lang.build_input_lang(trim_min_count)
    
    #new_tfidf=TFIDF(keyword_num)
    input_lang.build_input_lang_justout(output_lang.index2word)
    #print(output_lang.index2word)

    dict_keyword_problem={}# this is used to make every keyword line match one problem

    count_keyword_num=0
    count_id=0
    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])
        train_cate_problem_dict,cate_problem_keyword=annotate_problem_by_keyword(pair[0],category_keyword_dict,category_keyword_order,train_cate_problem_dict)
        input_cell = indexes_from_sentence(input_lang, pair[0])
        input_cell.append(input_lang.word2index["EOS"])
        output_cell = indexes_from_sentence(output_lang, pair[1])
        #output_cell.append(output_lang.word2index["EOS"])
        #cate_problem_keyword=new_tfidf.key_abstract(pair[0])
        cate_cell=indexes_from_sentence(output_lang, cate_problem_keyword)
        train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3],cate_cell,num_stack))
        keyword_expression_line=" ".join(cate_problem_keyword)+" ".join(pair[1])
        if keyword_expression_line in dict_keyword_problem:
            dict_keyword_problem[keyword_expression_line].append(" ".join(pair[0]))
        else:
            dict_keyword_problem[keyword_expression_line]=[" ".join(pair[0])]

        if len(cate_problem_keyword)>5:
            count_keyword_num+=1
        count_id+=1
        if count_id%2000==0:
            print(" ".join(pair[0]))
            print(" ".join(pair[1]))
            print(" ".join(cate_problem_keyword))
    #print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    #print('Number of training data %d' % (len(train_pairs)))
    print('Number of keyword num more than 5 %d' % (count_keyword_num))
    print('Number of training data %d' % (len(train_pairs)))

    max_similar_key=10
    curr_max=0

    output_=open("dict_keyword_problem_similar.txt","w")
    for line in dict_keyword_problem:
        #if len(dict_keyword_problem[line])>curr_max and not line.startswith("other") and not line.startswith("Fraction"):
        #    curr_max=len(dict_keyword_problem[line])
        if len(dict_keyword_problem[line])>max_similar_key:
            #if not line.startswith("Fraction"):
            #    print(len(dict_keyword_problem[line]))
            #    print(line)
            output_.write(str(len(dict_keyword_problem[line]))+"\n")
            output_.write(str(line)+"\n")
            for problem_ in dict_keyword_problem[line]:
                output_.write(problem_+"\n")
    output_.close()

    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        test_cate_problem_dict,cate_problem_keyword=annotate_problem_by_keyword(pair[0],category_keyword_dict,category_keyword_order,test_cate_problem_dict)
        input_cell = indexes_from_sentence(input_lang, pair[0])
        input_cell.append(input_lang.word2index["EOS"])
        output_cell = indexes_from_sentence(output_lang, pair[1])
        #cate_cell=indexes_from_sentence(output_lang, cate_problem_keyword)
        
        #cate_problem_keyword=new_tfidf.key_abstract(pair[0])
        cate_cell=indexes_from_sentence(output_lang, cate_problem_keyword)
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                           pair[2], pair[3],cate_cell,num_stack))
    #print('Number of testind data %d' % (len(test_pairs)))
    category_keyword_order.append("other")
    save_train_cate_problem_dict(train_cate_problem_dict,category_keyword_order)
    return input_lang, output_lang,output_exp_lang, train_pairs, test_pairs,train_cate_problem_dict


# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq

def pad_input_edge(input_edge, seq_len, max_length):
    for i in range(len(input_edge)):
        input_edge[i]+=[PAD_token for _ in range(max_length-seq_len)]
    for i in range(max_length-seq_len):
        temp_list=[PAD_token for _ in range(max_length)]
        input_edge.append(temp_list)
    return input_edge

def pad_middle_exp(seq, seq_len, max_length):
    for _ in range(max_length - seq_len):
        pad_list=[PAD_token,PAD_token,PAD_token]
        seq.append(pad_list)
    return seq

# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_pos_batches = []
    num_size_batches = []
    num_stack_batches = []
    unit_list_batches=[]
    input_edge_batches = []
    rule3_list_batches=[]
    output_middle_batches=[]
    first_10_word_batches=[]
    keyword_batches=[]
    exp_batches=[]
    exp_length_batches=[]
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[3], reverse=True)
        input_length = []
        output_length = []
        keyword_length=[]
        for _, i, _, j, _, _,keyword_list,_ in batch:
            input_length.append(i)
            output_length.append(j)
            keyword_length.append(len(keyword_list))
        input_lengths.append(input_length)
        input_len_max = max(input_length)
        output_len_max = max(output_length)
        keyword_len_max=max(keyword_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_pos_batch = []
        first_10_word_batch=[]
        keyword_batch=[]
        output_len=[]
        exp_batch=[]
        exp_length_batch=[]
        num_stack_batch=[]
        for i, li, j, lj, num, num_pos,keyword_list,num_stack in batch:
            num_batch.append(num)
            input_batch.append(pad_seq(i, li, input_len_max))
            #output_batch.append(pad_seq(j, lj, output_len_max))
            exp_list_expand=pad_seq(j, lj, output_len_max)
            exp_batch.append(exp_list_expand)
            exp_length_batch.append(lj)
            num_pos_batch.append(num_pos)
            num_stack_batch.append(num_stack)

            if len(keyword_list)>keyword_num:
                keyword_list=keyword_list[0:keyword_num]
            keyword_list_expand=pad_seq(keyword_list, len(keyword_list), keyword_num)
            keyword_batch.append(keyword_list_expand)
            keyword_exp=keyword_list_expand+exp_list_expand
            output_batch.append(keyword_exp)
            output_len.append(keyword_num+lj)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        output_lengths.append(output_len)
        num_pos_batches.append(num_pos_batch)
        first_10_word_batches.append(first_10_word_batch)
        keyword_batches.append(keyword_batch)
        exp_batches.append(exp_batch)
        exp_length_batches.append(exp_length_batch)
        num_stack_batches.append(num_stack_batch)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_pos_batches,keyword_batches,exp_batches,exp_length_batches,num_stack_batches
def get_num_stack(eq, output_lang, num_pos):
    num_stack = []
    for word in eq:
        temp_num = []
        flag_not = True
        if word not in output_lang.index2word:
            flag_not = False
            for i, j in enumerate(num_pos):
                if j == word:
                    temp_num.append(i)
        if not flag_not and len(temp_num) != 0:
            num_stack.append(temp_num)
        if not flag_not and len(temp_num) == 0:
            num_stack.append([_ for _ in range(len(num_pos))])
    num_stack.reverse()
    return num_stack


