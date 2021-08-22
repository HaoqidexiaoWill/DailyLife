import os
import random
import copy
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import numpy as np
import time
import math
import gc
import re
# from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertModel
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
import json
from sklearn.metrics import *
import collections

CON_NUM_OP_DICT = {i: [] for i in range(1, 5)}

def num2chinese(num):
    num_dict = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '0': '零', }
    index_dict = {1: '', 2: '十', 3: '百', 4: '千', 5: '万', 6: '十', 7: '百', 8: '千', 9: '亿'}
    nums = list(num)
    nums_index = [x for x in range(1, len(nums)+1)][-1::-1]
    chinese_num = ''
    for index, item in enumerate(nums):
        chinese_num = "".join((chinese_num, num_dict[item], index_dict[nums_index[index]]))
    chinese_num = re.sub("零[十百千零]*", "零", chinese_num)
    chinese_num = re.sub("零万", "万", chinese_num)
    chinese_num = re.sub("亿万", "亿零", chinese_num)
    chinese_num = re.sub("零零", "零", chinese_num)
    chinese_num = re.sub("零\\b" , "", chinese_num)
    if chinese_num[:2] == "一十":
        chinese_num = chinese_num[1:]
    if num == "0":
        chinese_num = "零"
    return chinese_num


def process_two(chinese_num):
    final_list = []

    def recursive(chinese_num_list, index):
        if index == len(chinese_num_list): return
        if chinese_num_list[index] != "二":
            recursive(chinese_num_list, index + 1)
        else:
            new_chinese_num_list = copy.copy(chinese_num_list)
            new_chinese_num_list[index] = "两"
            final_list.append(chinese_num_list)
            final_list.append(new_chinese_num_list)
            recursive(chinese_num_list, index + 1)
            recursive(new_chinese_num_list, index + 1)

    if "二" in chinese_num:
        recursive(list(chinese_num), 0)
        chinese_nums = list(set(map(lambda x: "".join(x), final_list)))
    else:
        chinese_nums = [chinese_num]
    return chinese_nums


def float2chinese(num):
    num_dict = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '0': '零', }
    chinese_num_set = set()
    if num.count(".") == 1 and num[-1] != "." and num[0] != ".":
        index = num.index(".")
        part1 = num[: index]
        part2 = num[index + 1:]
        part1_chinese = num2chinese(part1)
        part2_chinese = "".join(list(map(lambda x: num_dict[x], list(part2))))
        chinese_num_set.add(part1_chinese + "点" + part2_chinese)
        chinese_num_set.add(part1_chinese + "块" + part2_chinese)
        chinese_num_set.add(part1_chinese + "元" + part2_chinese)
        if part1 == "0":
            chinese_num_set.add(part2_chinese)
    else:
        chinese_num_set.add(num2chinese(num.replace(".", "")))
    return chinese_num_set


def create_mix_num(num):
    num = int(num)
    if int(num % 1e12) == 0: # 万亿
        top_digit = str(int(num / 1e12))
        num = top_digit + "万亿"
    elif int(num % 1e11) == 0:
        top_digit = str(int(num / 1e11))
        num = top_digit + "千亿"
    elif int(num % 1e10) == 0:
        top_digit = str(int(num / 1e10))
        num = top_digit + "百亿"
    elif int(num % 1e9) == 0:
        top_digit = str(int(num / 1e9))
        num = top_digit + "十亿"
    elif int(num % 1e8) == 0:
        top_digit = str(int(num / 1e8))
        num = top_digit + "亿"
    elif int(num % 1e7) == 0:
        top_digit = str(int(num / 1e7))
        num = top_digit + "千万"
    elif int(num % 1e6) == 0:
        top_digit = str(int(num / 1e6))
        num = top_digit + "百万"
    elif int(num % 1e5) == 0:
        top_digit = str(int(num / 1e5))
        num = top_digit + "十万"
    elif int(num % 1e4) == 0:
        top_digit = str(int(num / 1e4))
        num = top_digit + "万"
    elif int(num % 1e3) == 0:
        top_digit = str(int(num / 1e3))
        num = top_digit + "千"
    elif int(num % 1e2) == 0:
        top_digit = str(int(num / 1e2))
        num = top_digit + "百"
    elif int(num % 1e1) == 0:
        top_digit = str(int(num / 1e1))
        num = top_digit + "十"
    else:
        num = str(num)
    return num


def nums_add_unit(nums):
    final_nums = set()
    for num in nums:
        final_nums.add(num)
        final_nums.add(num + "百")
        final_nums.add(num + "千")
        final_nums.add(num + "万")
        final_nums.add(num + "亿")
        if len(num) == 1:
            final_nums.add(num + "十万")
            final_nums.add(num + "百万")
            final_nums.add(num + "千万")
            final_nums.add(num + "十亿")
            final_nums.add(num + "百亿")
            final_nums.add(num + "千亿")
            final_nums.add(num + "万亿")
    final_nums = list(final_nums)
    return final_nums


def num2year(num):
    num_dict = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '0': '零', }
    year_list = []
    if "." not in num:
        if len(num) == 4 and 1000 < int(num) < 2100:
            year_num_list = []
            year_num_list.append(num)
            year_num_list.append(num[2] + num[3])
            year_num_list.append(num_dict[num[2]] + num_dict[num[3]])
            year_num_list.append(num_dict[num[0]] + num_dict[num[1]] + num_dict[num[2]] + num_dict[num[3]])
            for year_num in year_num_list:
                year_list.append(year_num)
                year_list.append(year_num + "年")
                year_list.append(year_num + "级")
                year_list.append(year_num + "届")
        if len(num) == 8 and 1800 < int(num[0: 4]) < 2100 and 0 < int(num[4: 6]) <= 12 and 0 < int(num[6: 8]) <= 31:
            year_list.append("%s年%s月%s日" % (num[0: 4], str(int(num[4: 6])), str(int(num[6: 8]))))
            year_list.append("%s年%s月%s日" % (num[2: 4], str(int(num[4: 6])), str(int(num[6: 8]))))
            year_list.append("%s年%s月%s号" % (num[0: 4], str(int(num[4: 6])), str(int(num[6: 8]))))
            year_list.append("%s年%s月%s号" % (num[2: 4], str(int(num[4: 6])), str(int(num[6: 8]))))
    else:
        if num.count(".") == 1:
            year, month = num.split(".")
            if len(year) >= 2:
                year_list.append("%s年%s月" % (year, str(int(month))))
                year_list.append("%s年%s月" % (year[-2: ], str(int(month))))
    return year_list


def convert_num(num):
    num = str(num)
    if "." not in num:
        chinese_num = num2chinese(num)
        chinese_nums = process_two(chinese_num)
        mix_num = create_mix_num(num)
        candidate_nums = nums_add_unit(chinese_nums + [num, mix_num])
    else:
        candidate_nums = nums_add_unit([num])
    return candidate_nums


def convert_str(value):
    candidate_substrs = set()
    candidate_substrs.add(value)
    if len(value) > 2:  # 去掉最后一个字
        candidate_substrs.add(value[: -1])
    if ("(" in value and ")" in value) or ("（" in value and "）" in value):
        tmp_value = value.replace("（", "(").replace("）", ")")
        index1 = tmp_value.index("(")
        index2 = tmp_value.index(")")
        if index1 < index2:
            candidate_substrs.add(tmp_value.replace("(", "").replace(")", ""))
            #candidate_substrs.add(tmp_value[index1 + 1: index2])   # 括号不能取出来
            candidate_substrs.add(tmp_value.replace(tmp_value[index1: index2 + 1], ""))
    candidate_substrs.add(value.replace("公司", ""))
    candidate_substrs.add(value.replace("有限", ""))
    candidate_substrs.add(value.replace("有限公司", ""))
    candidate_substrs.add(value.replace("合格", "达标"))
    candidate_substrs.add(value.replace("合格", "及格"))
    candidate_substrs.add(value.replace("不合格", "不达标"))
    candidate_substrs.add(value.replace("不合格", "不及格"))
    candidate_substrs.add(value.replace("风景名胜区", ""))
    candidate_substrs.add(value.replace("著", ""))
    candidate_substrs.add(value.replace("等", ""))
    candidate_substrs.add(value.replace("省", ""))
    candidate_substrs.add(value.replace("市", ""))
    candidate_substrs.add(value.replace("区", ""))
    candidate_substrs.add(value.replace("县", ""))
    candidate_substrs.add(value.replace("岗", "员"))
    candidate_substrs.add(value.replace("员", "岗"))
    candidate_substrs.add(value.replace("岗", "人员"))
    candidate_substrs.add(value.replace("岗", ""))
    candidate_substrs.add(value.replace("人员", "岗"))
    candidate_substrs.add(value.replace("岗位", "人员"))
    candidate_substrs.add(value.replace("人员", "岗位"))
    candidate_substrs.add(value.replace("岗位", ""))
    candidate_substrs.add(value.replace("人员", ""))
    candidate_substrs.add(value.lower())
    candidate_substrs.add(value.replace("-", ""))
    candidate_substrs.add(value.replace("-", "到"))
    candidate_substrs.add(value.replace("-", "至"))
    candidate_substrs.add(value.replace("否", "不"))
    candidate_substrs.add(value.replace("否", "没有"))
    candidate_substrs.add(value.replace("未", "没有"))
    candidate_substrs.add(value.replace("《", "").replace("》", "").replace("<", "").replace(">", ""))
    candidate_substrs.add(value.replace("免费", "免掉"))
    candidate_substrs.add(value.replace("免费", "免"))
    candidate_substrs.add(value.replace("免", "不用"))
    candidate_substrs.add(value.replace("免", "不需要"))
    candidate_substrs.add(value.replace("的", ""))
    candidate_substrs.add(value.replace("\"", "").replace("“", "").replace("”", ""))
    candidate_substrs.add(value.replace("类", ""))
    candidate_substrs.add(value.replace("级", "等"))
    candidate_substrs.add(value.replace("附属小学", "附小"))
    candidate_substrs.add(value.replace("附属中学", "附中"))
    candidate_substrs.add(value.replace("三甲", "三级甲等"))
    candidate_substrs.add(value.replace("三乙", "三级乙等"))
    candidate_substrs.add(value.replace("不限", "不要求"))
    candidate_substrs.add(value.replace("不限", "没有要求"))
    candidate_substrs.add(value.replace("全日制博士", "博士"))
    candidate_substrs.add(value.replace("本科及以上", "本科"))
    candidate_substrs.add(value.replace("硕士及以上学位", "硕士"))
    candidate_substrs.add(value.replace("主编", ""))
    candidate_substrs.add(value.replace("性", ""))
    candidate_substrs.add(value.replace("教师", "老师"))
    candidate_substrs.add(value.replace("老师", "教师"))
    candidate_substrs.add(value.replace(":", ""))
    candidate_substrs.add(value.replace("：", ""))
    candidate_substrs.add(value.replace("股份", "股"))
    candidate_substrs.add(value.replace("股份", ""))
    candidate_substrs.add(value.replace("控股", ""))
    candidate_substrs.add(value.replace("中山大学", "中大"))
    candidate_substrs.add(value.replace("重庆大学", "重大"))
    candidate_substrs.add(value.replace("中国人民大学", "人大"))
    candidate_substrs.add(value.replace("北京大学", "北大"))
    candidate_substrs.add(value.replace("南京大学", "南大"))
    candidate_substrs.add(value.replace("武汉大学", "武大"))
    candidate_substrs.add(value.replace("复旦大学", "复旦"))
    candidate_substrs.add(value.replace("清华大学", "清华"))
    candidate_substrs.add(value.replace("广州大学", "广大"))
    candidate_substrs.add(value.replace("北京体育大学", "北体"))
    candidate_substrs.add(value.replace(".00", ""))
    candidate_substrs.add(value.replace(",", ""))
    candidate_substrs.add(value.replace("，", ""))
    candidate_substrs.add(value.replace("0", "零"))
    candidate_substrs.add(value.replace("第", "").replace("学", ""))
    candidate_substrs.add(value.replace("省", "").replace("市", "").replace("第", "").replace("学", ""))
    candidate_substrs.add(value.replace("年", ""))
    candidate_substrs.add(value.replace("粤语", "广东话"))
    candidate_substrs.add(value.replace("粤语", "白话"))
    candidate_substrs.add(value.replace("市", "").replace("医院", "院"))
    candidate_substrs.add(value.replace("研究生/硕士", "硕士"))
    candidate_substrs.add(value.replace("研究生/硕士", "硕士研究生"))
    candidate_substrs.add(value.replace("中医医院", "中医院"))
    candidate_substrs.add(value.replace("医生", "医师"))
    candidate_substrs.add(value.replace("合格", "符合"))
    candidate_substrs.add(value.replace("合格", "达到标准"))
    candidate_substrs.add(value.replace("工学", "工程学"))
    candidate_substrs.add(value.replace("场", "馆"))
    candidate_substrs.add(value.replace("市保", "市级保护单位"))
    candidate_substrs.add(value.replace("市保", "保护单位"))
    candidate_substrs.add(value.replace("经理女", "女经理"))
    candidate_substrs.add(value.replace("大专及以上", "大专"))
    candidate_substrs.add(value.replace("大专及以上", "专科"))
    candidate_substrs.add(value.replace("北京青年报社", "北青报"))
    candidate_substrs.add(value.replace("不限", "没有限制"))
    candidate_substrs.add(value.replace("高级中学", "高中"))
    candidate_substrs.add(value.replace("中共党员", "党员"))
    digit = "".join(re.findall("[0-9.]", value))
    # 有3年及以上相关工作经验 你能告诉我那些要求[三]年相关工作经验，还有要求本科或者本科以上学历的是什么职位吗
    # 2014WTA广州国际女子网球公开赛 你知道在什么地方举办[2014]年的WTA广州国际女子网球公开赛吗
    if len(digit) > 0 and digit.count(".") <= 1 and is_float(digit) and float(digit) < 1e8 and len(digit) / len(value) > 0.4:
        candidate_substrs.add(digit)
        chinese_num_set = float2chinese(digit)
        candidate_substrs |= chinese_num_set
    year1 = re.match("[0-9]{4}年", value)
    if year1:
        year1 = year1.string
        candidate_substrs.add(year1)
        candidate_substrs.add(year1[2:])
    year2 = re.match("[0-9]{4}-[0-9]{2}", value)
    if year2:
        year2 = year2.string
        year = year2[0: 4]
        mongth = year2[5: 7] if year2[5] == "1" else year2[6]
        candidate_substrs.add("%s年%s月" % (year, mongth))
        candidate_substrs.add("%s年%s月" % (year[2:], mongth))
    year3 = re.match("[0-9]{4}-[0-9]{2}-[0-9]{2}", value)
    if year3:
        year3 = year3.string
        year = year3[0: 4]
        mongth = year3[5: 7] if year3[5] == "1" else year3[6]
        day = year3[8: 10] if year3[8] == "1" else year3[9]
        candidate_substrs.add("%s年%s月%s日" % (year, mongth, day))
        candidate_substrs.add("%s年%s月%s日" % (year[2:], mongth, day))
        candidate_substrs.add("%s年%s月%s号" % (year, mongth, day))
        candidate_substrs.add("%s年%s月%s号" % (year[2:], mongth, day))
    return list(candidate_substrs)


def duplicate_relative_index(conds):
    value_dict = {}
    duplicate_indices = []
    for _, _, value in conds:
        if value not in value_dict:
            duplicate_indices.append(0)
            value_dict[value] = 1
        else:
            duplicate_indices.append(value_dict[value])
            value_dict[value] += 1
    return duplicate_indices


def is_float(x):
    try:
        float(x)
        return True
    except:
        return False


def match_str(question, value, precision_limit=0.8, recall_limit=0.65, match_type="recall"):
    value_char_dict = {}
    for char in value:
        if char in value_char_dict:
            value_char_dict[char] += 1
        else:
            value_char_dict[char] = 1
    candidate_substrs = []
    matched_str = ""
    for n in range(2, min(len(question), len(value) + 5)):
        for i in range(0, len(question) - n + 1):
            substr = question[i: i + n]
            char_dict = copy.copy(value_char_dict)
            positive_num = 0
            for char in substr:
                if char in char_dict and char_dict[char] > 0:
                    positive_num += 1
                    char_dict[char] -= 1
            precision = positive_num / len(substr)
            recall = positive_num / len(value)
            if precision == 0 or recall == 0: continue
            candidate_substrs.append([substr, precision, recall])
    if match_type == "recall":
        fully_matched_substrs = list(filter(lambda x: x[2] == 1, candidate_substrs))
        sorted_list = sorted(fully_matched_substrs, key=lambda x: x[1], reverse=True)
        if len(sorted_list) > 0:
            if sorted_list[0][1] > precision_limit:
                #print(value, question, sorted_list[0])
                matched_str = sorted_list[0][0]
    if match_type == "precision":
        precise_substrs = list(filter(lambda x: x[1] == 1, candidate_substrs))
        sorted_list = sorted(precise_substrs, key=lambda x: x[2], reverse=True)
        if len(sorted_list) > 0:
            if sorted_list[0][2] > recall_limit:
                #print(value, question, sorted_list[0])
                matched_str = sorted_list[0][0]
    return matched_str


def match_value(question, value, duplicate_index):
    pre_stopchars = set("0123456789一二三四五六七八九十")
    post_stopchars = set("0123456789一二三四五六七八九十百千万亿")
    stopwords = {"一下", "一共", "一起", "一并", "一致", "一周", "一共"}
    original_value = value
    if is_float(value) and float(value) < 1e8:  # 数字转中文只能支持到亿
        if float(value) - math.floor(float(value)) == 0:
            value = str(int(float(value)))
        candidate_nums = convert_num(value) if "-" not in value else [value]   # - 是负数
        year_list = num2year(value)
        candidate_values = candidate_nums + year_list + [original_value]
    else:
        if value in question:
            candidate_values = [value]
        else:
            candidate_values = convert_str(value)
            if sum([candidate_value in question for candidate_value in candidate_values]) == 0:
                matched_str = match_str(question, value, precision_limit=0.8, recall_limit=0.65, match_type="recall")
                if len(matched_str) > 0:
                    candidate_values.append(matched_str)
    matched_value = ""
    matched_index = None
    for value in candidate_values:
        if value in question and len(value) > len(matched_value):
            indices = [i for i in range(len(question)) if question.startswith(value, i)]
            valid_indices = []
            for index in indices:
                flag = 0
                if index - 1 >= 0:
                    previsou_char = question[index - 1]
                    if previsou_char in pre_stopchars: flag = 1
                if index + len(value) < len(question):
                    post_char = question[index + len(value)]
                    if post_char in post_stopchars: flag = 1
                    if question[index] + post_char in stopwords: flag = 1
                if flag == 1: continue
                valid_indices.append(index)
            if len(valid_indices) == 1:
                matched_value = value
                matched_index = valid_indices[0]
            elif len(valid_indices) > 1 and duplicate_index < len(valid_indices):
                matched_value = value
                matched_index = valid_indices[duplicate_index]
    if matched_value != "":
        question = list(question)
        question_1 = "".join(question[: matched_index])
        question_2 = "".join(question[matched_index: matched_index + len(matched_value)])
        question_3 = "".join(question[matched_index + len(matched_value): ])
        question = question_1 + "[" + question_2 + "]" + question_3
        print(original_value, question)
    else:
        print(original_value, "不匹配", question)
    return matched_value, matched_index


class BertNeuralNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNeuralNet, self).__init__(config)
        self.num_tag_labels = 5
        self.num_agg_labels = 6
        self.num_connection_labels = 3
        self.num_con_num_labels = 4
        self.num_type_labels = 3

        op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!=", 4: "不选中"}
        agg_sql_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
        conn_sql_dict = {0: "", 1: "and", 2: "or"}
        con_num_dict = {0: 0, 1: 1, 2: 2, 3: 3}
        type_dict = {0: "sel", 1: "con", 2: "none"}
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_tag = nn.Linear(self.hidden_size, self.num_tag_labels)
        self.linear_agg = nn.Linear(self.hidden_size, self.num_agg_labels)
        self.linear_connection = nn.Linear(self.hidden_size, self.num_connection_labels)
        self.linear_con_num = nn.Linear(self.hidden_size, self.num_con_num_labels)
        self.linear_type = nn.Linear(self.hidden_size, self.num_type_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask, all_masks, cls_index_list, train_dependencies=None):
        sequence_output, _ = self.bert(input_ids, None, attention_mask, output_all_encoded_layers=False)
        if train_dependencies:
            tag_masks = train_dependencies[0].view(-1) == 1      # 必须要加 view 和 == 1
            sel_masks = train_dependencies[1].view(-1) == 1
            con_masks = train_dependencies[2].view(-1) == 1
            type_masks = all_masks.view(-1) == 1
            connection_labels = train_dependencies[3]
            agg_labels = train_dependencies[4]
            tag_labels = train_dependencies[5]
            con_num_labels = train_dependencies[6]
            type_labels = train_dependencies[7]
            # mask 后的 bert_output
            tag_output = sequence_output.view(-1, self.hidden_size)[tag_masks]
            tag_labels = tag_labels.view(-1)[tag_masks]
            agg_output = sequence_output[sel_masks, cls_index_list[sel_masks], :]
            agg_labels = agg_labels[sel_masks]
            connection_output = sequence_output[con_masks, 0, :]
            connection_labels = connection_labels[con_masks]
            con_num_output = sequence_output[con_masks, cls_index_list[con_masks], :]
            con_num_labels = con_num_labels[con_masks]
            type_output = sequence_output[type_masks, cls_index_list[type_masks], :]
            type_labels = type_labels[type_masks]
            # 全连接层
            tag_output = self.linear_tag(self.dropout(tag_output))
            agg_output = self.linear_agg(self.dropout(agg_output))
            connection_output = self.linear_connection(self.dropout(connection_output))
            con_num_output = self.linear_con_num(self.dropout(con_num_output))
            type_output = self.linear_type(self.dropout(type_output))
            # 损失函数
            loss_function = nn.CrossEntropyLoss(reduction="mean")
            tag_loss = loss_function(tag_output, tag_labels)
            agg_loss = loss_function(agg_output, agg_labels)
            connection_loss = loss_function(connection_output, connection_labels)
            con_num_loss = loss_function(con_num_output, con_num_labels)
            type_loss = loss_function(type_output, type_labels)
            loss = tag_loss + agg_loss + connection_loss + con_num_loss + type_loss
            return loss
        else:
            all_masks = all_masks.view(-1) == 1
            batch_size, seq_len, hidden_size = sequence_output.shape
            tag_output = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float32,device = 'cuda')
            for i in range(batch_size):
                for j in range(seq_len):
                    if attention_mask[i][j] == 1:
                        tag_output[i][j] = sequence_output[i][j]
            head_output = sequence_output[:, 0, :]
            cls_output = sequence_output[all_masks, cls_index_list, :]
            tag_output = self.linear_tag(self.dropout(tag_output))
            agg_output = self.linear_agg(self.dropout(cls_output))
            connection_output = self.linear_connection(self.dropout(head_output))
            con_num_output = self.linear_con_num(self.dropout(cls_output))
            type_output = self.linear_type(self.dropout(cls_output))
            tag_logits = torch.argmax(F.log_softmax(tag_output, dim=2), dim=2).detach().cpu().numpy().tolist()
            agg_logits = torch.argmax(F.log_softmax(agg_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            connection_logits = torch.argmax(F.log_softmax(connection_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            con_num_logits = torch.argmax(F.log_softmax(con_num_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            type_logits = torch.argmax(F.log_softmax(type_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            return tag_logits, agg_logits, connection_logits, con_num_logits, type_logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
        else:
            bce_loss = nn.BCELoss(reduction="none")(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        #focal_loss = (1 - pt) ** self.gamma * bce_loss
        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss


class Trainer:
    def __init__(self, data_dir, model_name, epochs=4, batch_size=64, base_batch_size=32, part=1., seed=1234, debug_mode=False):
        # TODO 这两个量是干嘛用的
        '''
        base_batch_size
        part
        '''
        self.device = torch.device('cuda')
        self.data_dir = data_dir
        self.debug_mode = debug_mode
        self.model_name = model_name
        self.seed = seed
        self.part = part
        self.seed_everything()
        self.max_len = 120
        self.epochs = epochs
        self.base_batch_size = base_batch_size
        self.batch_size = batch_size
        # TODO split ratio 是干嘛用的
        self.split_ratio = 0.80
        if os.path.exists(self.data_dir):
            self.train_data_path = os.path.join(self.data_dir, "train/train.json")
            self.train_table_path = os.path.join(self.data_dir, "train/train.tables.json")
            self.valid_data_path = os.path.join(self.data_dir, "val/val.json")
            self.valid_table_path = os.path.join(self.data_dir, "val/val.tables.json")
            self.test_data_path = os.path.join(self.data_dir, "test/test.json")
            self.test_table_path = os.path.join(self.data_dir, "test/test.tables.json")
            self.bert_model_path = os.path.join('/home1/lsy2018/NL2SQL/python5', "chinese_wwm_L-12_H-768_A-12/")
            self.pytorch_bert_path = os.path.join('/home1/lsy2018/NL2SQL/python5', "/chinese_wwm_L-12_H-768_A-12/pytorch_model.bin")
            self.bert_config = BertConfig(os.path.join('/home1/lsy2018/NL2SQL/python5', "chinese_wwm_L-12_H-768_A-12/bert_config.json"))
        else:
            input_dir = "/home1/lsy2018/NL2SQL/XSQL/data"
            self.train_data_path = os.path.join(input_dir, "train/train.json")
            self.train_table_path = os.path.join(input_dir, "train/train.tables.json")
            self.valid_data_path = os.path.join(input_dir, "val/val.json")
            self.valid_table_path = os.path.join(input_dir, "val/val.tables.json")
            self.test_data_path = os.path.join(input_dir, "test/test.json")
            self.test_table_path = os.path.join(input_dir, "test/test.tables.json")
            self.bert_model_path = os.path.join('/home1/lsy2018/NL2SQL/python5', "chinese_wwm_L-12_H-768_A-12/")
            self.pytorch_bert_path = os.path.join('/home1/lsy2018/NL2SQL/python5', "/chinese_wwm_L-12_H-768_A-12/pytorch_model.bin")
            self.bert_config = BertConfig(os.path.join('/home1/lsy2018/NL2SQL/python5', "chinese_wwm_L-12_H-768_A-12/bert_config.json"))

    def load_data(self, path, num=None):
        data_list = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if self.debug_mode and i == 100: break
                sample = json.loads(line)
                data_list.append(sample)
        if num and not self.debug_mode:
            random.seed(self.seed)
            data_list = random.sample(data_list, num)
        return data_list

    def load_table(self, path):
        table_dict = {}
        with open(path, "r") as f:
            for i, line in enumerate(f):
                table = json.loads(line)
                table_dict[table["id"]] = table
        return table_dict

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        # 用于模型来进行复现的
        torch.backends.cudnn.deterministic = True

    # 这个函数应该是没有用上？
    def convert_lines(self, text_series, max_seq_length, bert_tokenizer):
        max_seq_length -= 2
        all_tokens = []
        for text in text_series:
            tokens = bert_tokenizer.tokenize(text)
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
            one_token = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"]) + [0] * (max_seq_length - len(tokens))
            all_tokens.append(one_token)
        return np.array(all_tokens)

    def process_sample(self, sample, table_dict, bert_tokenizer):
        question = sample["question"]
        table_id = sample["table_id"]
        sel_list = sample["sql"]["sel"]
        agg_list = sample["sql"]["agg"]
        con_list = sample["sql"]["conds"]
        connection = sample["sql"]["cond_conn_op"]

        # 生成某一个query 对应的 表格的属性信息
        table_title = table_dict[table_id]["title"]
        table_header_list = table_dict[table_id]["header"]
        table_row_list = table_dict[table_id]["rows"]

        sel_dict = {sel: agg for sel, agg in zip(sel_list, agg_list)}
        # <class 'list'>: [[0, 2, '大黄蜂'], [0, 2, '密室逃生']] 一列两value 多一个任务判断where一列的value数, 
        # con_dict里面的数量要和conds匹配，否则放弃这一列（但也不能作为非con非sel训练）
        # 标注只能用多分类？有可能对应多个
        duplicate_indices = duplicate_relative_index(con_list)
        con_dict = {}
        for [con_col, op, value], duplicate_index in zip(con_list, duplicate_indices):  # duplicate index 是跟着 value 的
            value = value.strip()
            matched_value, matched_index = match_value(question, value, duplicate_index)
            if len(matched_value) > 0:
                if con_col in con_dict:
                    con_dict[con_col].append([op, matched_value, matched_index])
                else:
                    con_dict[con_col] = [[op, matched_value, matched_index]]

            # con_dict {2: [[2, '汽车电子控制技术', 0]]}
        # TODO：con_dict要看看len和conds里同一列的数量是否一致，不一致不参与训练
        # TODO：多任务加上col对应的con数量
        # TODO：需要变成训练集的就是 sel_dict、con_dict和connection
        # TODO: 只有conds的序列标注任务是valid的，其他都不valid

        conc_tokens = []
        tag_masks = []
        sel_masks = []
        con_masks = []
        type_masks = []
        attention_masks = []
        connection_labels = []
        agg_labels = []
        tag_labels = []
        con_num_labels = []
        type_labels = []
        cls_index_list = []

        op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!=", 4: "不选中"}
        agg_sql_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
        conn_sql_dict = {0: "", 1: "and", 2: "or"}
        con_num_dict = {0: 0, 1: 1, 2: 2, 3: 3}
        type_dict = {0: "sel", 1: "con", 2: "none"}

        question_tokens = bert_tokenizer.tokenize(question)
        question_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + question_tokens + ["[SEP]"])
        header_cls_index = len(question_ids)

        for col in range(len(table_header_list)):
            header = table_header_list[col]
            header_tokens = bert_tokenizer.tokenize(header)
            header_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + header_tokens + ["[SEP]"])
            conc_ids = question_ids + header_ids + [0] * (self.max_len - len(question_ids) - len(header_ids))
            # tag id 对应的就是 op_sql_dict 也就是 where_condition 里的操作符的预测
            tag_ids = [4] * len(conc_ids)  # 4 是不标注
            attention_mask = [1] * (len(question_ids) + len(header_ids)) + [0] * (self.max_len - len(question_ids) - len(header_ids))
            # TODO 为啥这里type_mask 是1？
            sel_mask, con_mask, type_mask = 0, 0, 1
            connection_id, agg_id, con_num = 0, 0, 0
            # 这里的col 对应的应该是 col名的下标索引
            if col in con_dict:
                # 如果 header 对应多个 values，values 必须全部匹配上才进入训练
                # TODO map 这个用法没有看懂
                if list(map(lambda x: x[0], con_list)).count(col) != len(con_dict[col]): continue
                header_con_list = con_dict[col]
                for [op, value, index] in header_con_list:
                    tag_ids[index + 1: index + 1 + len(value)] = [op] * len(value)
                tag_mask = [0] + [1] * len(question) + [0] * (self.max_len - len(question) - 1)
                con_mask = 1
                connection_id = connection
                con_num = len(header_con_list)
                op_list = list(map(lambda x: x[0], header_con_list))
                CON_NUM_OP_DICT[con_num].extend(op_list)
                type_id = 1
            elif col in sel_dict:
                # TODO: 是不是还有同一个个sel col，多个不同聚合方式
                tag_mask = [0] * self.max_len
                sel_mask = 1
                agg_id = sel_dict[col]
                type_id = 0
            else:
                tag_mask = [0] * self.max_len
                type_id = 2
            conc_tokens.append(conc_ids)
            tag_masks.append(tag_mask)
            sel_masks.append(sel_mask)
            con_masks.append(con_mask)
            type_masks.append(type_mask)
            attention_masks.append(attention_mask)
            connection_labels.append(connection_id)
            agg_labels.append(agg_id)
            tag_labels.append(tag_ids)
            con_num_labels.append(con_num)
            type_labels.append(type_id)
            cls_index_list.append(header_cls_index)
        return tag_masks, sel_masks, con_masks, type_masks, attention_masks, connection_labels, agg_labels, tag_labels, con_num_labels, type_labels, cls_index_list, conc_tokens

    def create_dataloader(self):
        """
        sel 列 agg类型
        where 列 逻辑符 值
        where连接符
        问题开头cls：where连接符（或者新模型，所有header拼一起，预测where连接类型？）
        列的开头cls，多任务学习：1、（不选中，sel，where） 2、agg类型（0~5：agg类型，6：不属于sel） 3、逻辑符类型：（0~3：逻辑符类型，4：不属于where）
        问题部分：序列标注，（每一个字的隐层和列开头cls拼接？再拼接列所有字符的avg？），二分类，如果列是where并且是对应value的，标注为1
        """
        # train: 41522 val: 4396 test: 4086
        train_data_list = self.load_data(self.train_data_path, num=int(41522 * self.part))
        train_table_dict = self.load_table(self.train_table_path)
        valid_data_list = self.load_data(self.valid_data_path)
        valid_table_dict = self.load_table(self.valid_table_path)
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)
        train_conc_tokens = []
        train_tag_masks = []
        train_sel_masks = []
        train_con_masks = []
        train_type_masks = []
        train_attention_masks = []
        train_connection_labels = []
        train_agg_labels = []
        train_tag_labels = []
        train_con_num_labels = []
        train_type_labels = []
        train_cls_index_list = []
        for sample in train_data_list:
            processed_result = self.process_sample(sample, train_table_dict, bert_tokenizer)
            train_tag_masks.extend(processed_result[0])
            train_sel_masks.extend(processed_result[1])
            train_con_masks.extend(processed_result[2])
            train_type_masks.extend(processed_result[3])
            train_attention_masks.extend(processed_result[4])
            train_connection_labels.extend(processed_result[5])
            train_agg_labels.extend(processed_result[6])
            train_tag_labels.extend(processed_result[7])
            train_con_num_labels.extend(processed_result[8])
            train_type_labels.extend(processed_result[9])
            train_cls_index_list.extend(processed_result[10])
            train_conc_tokens.extend(processed_result[11])
        valid_conc_tokens = []
        valid_tag_masks = []
        valid_sel_masks = []
        valid_con_masks = []
        valid_type_masks = []
        valid_attention_masks = []
        valid_connection_labels = []
        valid_agg_labels = []
        valid_tag_labels = []
        valid_con_num_labels = []
        valid_type_labels = []
        valid_cls_index_list = []
        for sample in valid_data_list:
            processed_result = self.process_sample(sample, valid_table_dict, bert_tokenizer)
            valid_tag_masks.extend(processed_result[0])
            valid_sel_masks.extend(processed_result[1])
            valid_con_masks.extend(processed_result[2])
            valid_type_masks.extend(processed_result[3])
            valid_attention_masks.extend(processed_result[4])
            valid_connection_labels.extend(processed_result[5])
            valid_agg_labels.extend(processed_result[6])
            valid_tag_labels.extend(processed_result[7])
            valid_con_num_labels.extend(processed_result[8])
            valid_type_labels.extend(processed_result[9])
            valid_cls_index_list.extend(processed_result[10])
            valid_conc_tokens.extend(processed_result[11])
        # 测试比例
        tag_labels_list = []
        for tag_labels in train_tag_labels + valid_tag_labels:
            tag_labels_list.extend(tag_labels)
        connectioin_labels_list = train_connection_labels + valid_connection_labels
        agg_labels_list = train_agg_labels + valid_agg_labels
        con_num_labels_list = train_con_num_labels + valid_con_num_labels
        type_labels_list = train_type_labels + valid_type_labels
        print("tag_labels_list", collections.Counter(tag_labels_list))
        print("connectioin_labels_list", collections.Counter(connectioin_labels_list))
        print("agg_labels_list", collections.Counter(agg_labels_list))
        print("con_num_labels_list", collections.Counter(con_num_labels_list))
        print("type_labels_list", collections.Counter(type_labels_list))
        global CON_NUM_OP_DICT
        CON_NUM_OP_DICT = {key: collections.Counter(CON_NUM_OP_DICT[key]) for key in CON_NUM_OP_DICT}
        print("CON_NUM_OP_DICT", CON_NUM_OP_DICT)

        train_dataset = data.TensorDataset(torch.tensor(train_conc_tokens, dtype=torch.long),
                                           torch.tensor(train_tag_masks, dtype=torch.long),
                                           torch.tensor(train_sel_masks, dtype=torch.long),
                                           torch.tensor(train_con_masks, dtype=torch.long),
                                           torch.tensor(train_type_masks, dtype=torch.long),
                                           torch.tensor(train_attention_masks, dtype=torch.long),
                                           torch.tensor(train_connection_labels, dtype=torch.long),
                                           torch.tensor(train_agg_labels, dtype=torch.long),
                                           torch.tensor(train_tag_labels, dtype=torch.long),
                                           torch.tensor(train_con_num_labels, dtype=torch.long),
                                           torch.tensor(train_type_labels, dtype=torch.long),
                                           torch.tensor(train_cls_index_list, dtype=torch.long)
                                           )
        valid_dataset = data.TensorDataset(torch.tensor(valid_conc_tokens, dtype=torch.long),
                                           torch.tensor(valid_tag_masks, dtype=torch.long),
                                           torch.tensor(valid_sel_masks, dtype=torch.long),
                                           torch.tensor(valid_con_masks, dtype=torch.long),
                                           torch.tensor(valid_type_masks, dtype=torch.long),
                                           torch.tensor(valid_attention_masks, dtype=torch.long),
                                           torch.tensor(valid_connection_labels, dtype=torch.long),
                                           torch.tensor(valid_agg_labels, dtype=torch.long),
                                           torch.tensor(valid_tag_labels, dtype=torch.long),
                                           torch.tensor(valid_con_num_labels, dtype=torch.long),
                                           torch.tensor(valid_type_labels, dtype=torch.long),
                                           torch.tensor(valid_cls_index_list, dtype=torch.long)
                                           )
        # 将 dataset 转成 dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.base_batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.base_batch_size, shuffle=False)
        # 返回训练数据
        return train_loader, valid_loader

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def detail_score(self, y_true, y_pred, num_labels, ignore_num=None):
        detail_y_true = [[] for _ in range(num_labels)]
        detail_y_pred = [[] for _ in range(num_labels)]
        for i in range(len(y_pred)):
            for label in range(num_labels):
                if y_true[i] == label:
                    detail_y_true[label].append(1)
                else:
                    detail_y_true[label].append(0)
                if y_pred[i] == label:
                    detail_y_pred[label].append(1)
                else:
                    detail_y_pred[label].append(0)
        pre_list = []
        rec_list = []
        f1_list = []
        detail_output_str = ""
        for label in range(num_labels):
            if label == ignore_num: continue
            pre = precision_score(detail_y_true[label], detail_y_pred[label])
            rec = recall_score(detail_y_true[label], detail_y_pred[label])
            f1 = f1_score(detail_y_true[label], detail_y_pred[label])
            detail_output_str += "[%d] pre:%.3f rec:%.3f f1:%.3f\n" % (label, pre, rec, f1)
            pre_list.append(pre)
            rec_list.append(rec)
            f1_list.append(f1)
        acc = accuracy_score(y_true, y_pred)
        output_str = "overall_acc:%.3f, avg_pre:%.3f, avg_rec:%.3f, avg_f1:%.3f \n" % (acc, np.mean(pre_list), np.mean(rec_list), np.mean(f1_list))
        output_str += detail_output_str
        return output_str

    def evaluate(self, logits_lists, cls_index_list, labels_lists):
        [tag_logits_list, agg_logits_list, connection_logits_list, con_num_logits_list, type_logits_list] = logits_lists
        [tag_labels_list, agg_labels_list, connection_labels_list, con_num_labels_list, type_labels_list] = labels_lists

        op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!=", 4: "不选中"}
        agg_sql_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
        conn_sql_dict = {0: "", 1: "and", 2: "or"}
        con_num_dict = {0: 0, 1: 1, 2: 2, 3: 3}
        type_dict = {0: "sel", 1: "con", 2: "none"}

        tag_pred = []
        tag_true = []
        tag_fully_matched = []
        agg_pred = []
        agg_true = []
        connection_pred = []
        connection_true = []
        con_num_pred = []
        con_num_true = []
        type_pred = type_logits_list
        type_true = type_labels_list

        for i, col_type in enumerate(type_true):
            if col_type == 0: # sel
                agg_pred.append(agg_logits_list[i])
                agg_true.append(agg_labels_list[i])
            elif col_type == 1: # con
                cls_index = cls_index_list[i]
                tmp_tag_pred = tag_logits_list[i][1: cls_index - 1] # 不取 cls 和 sep
                tmp_tag_true = tag_labels_list[i][1: cls_index - 1]
                matched = 1 if tmp_tag_pred == tmp_tag_true else 0
                tag_fully_matched.append(matched)
                tag_pred.extend(tmp_tag_pred)
                tag_true.extend(tmp_tag_true)
                connection_pred.append(connection_logits_list[i])
                connection_true.append(connection_labels_list[i])
                con_num_pred.append(con_num_logits_list[i])
                con_num_true.append(con_num_labels_list[i])

        eval_result = ""
        eval_result += "TYPE: 0: sel, 1: con, 2: none\n" + self.detail_score(type_true, type_pred, num_labels=3, ignore_num=None) + "\n"
        eval_result += "TAG: {0: >, 1: <, 2: ==, 3: !=, 4: 不选中\n" + self.detail_score(tag_true, tag_pred, num_labels=5, ignore_num=4) + "\n"
        eval_result += "CONNECTION: 0: , 1: and, 2: or\n" + self.detail_score(connection_true, connection_pred, num_labels=3, ignore_num=None) + "\n"
        eval_result += "CON_NUM: 0: 0, 1: 1, 2: 2, 3: 3\n" + self.detail_score(con_num_true, con_num_pred, num_labels=4, ignore_num=0) + "\n"
        eval_result += "AGG: 0: , 1: AVG, 2: MAX, 3: MIN, 4: COUNT, 5: SUM\n" + self.detail_score(agg_true, agg_pred, num_labels=6, ignore_num=None) + "\n"
        return eval_result

    def train(self):
        if self.debug_mode: self.epochs = 1
        # 加载 dataloader
        train_loader, valid_loader = self.create_dataloader()
        # 训练
        self.seed_everything()
        lr = 1e-5
        accumulation_steps = math.ceil(self.batch_size / self.base_batch_size)
        # 预训练 bert 转成 pytorch
        if os.path.exists(self.bert_model_path + "pytorch_model.bin") is False:
            convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
                self.bert_model_path + 'bert_model.ckpt',
                self.bert_model_path + 'bert_config.json',
                self.bert_model_path + 'pytorch_model.bin')
        # 加载预训练模型
        model = BertNeuralNet.from_pretrained(self.bert_model_path, cache_dir=None)
        model.zero_grad()
        model = model.to(self.device)
        # 不同的参数组设置不同的 weight_decay
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        epoch_steps = int(train_loader.sampler.num_samples / self.base_batch_size / accumulation_steps)
        num_train_optimization_steps = int(self.epochs * epoch_steps)
        valid_every = math.floor(epoch_steps * accumulation_steps / 5)
        optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.05, t_total=num_train_optimization_steps)
        # 渐变学习速率
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
        #model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        # 开始训练
        for epoch in range(self.epochs):
            train_start_time = time.time()
            model.train()
            optimizer.zero_grad()
            # 加载每个 batch 并训练
            for i, batch_data in enumerate(train_loader):
                input_ids = batch_data[0].to(self.device)
                tag_masks = batch_data[1].to(self.device)
                sel_masks = batch_data[2].to(self.device)   # 至少要有一个？ 否则 continue？
                con_masks = batch_data[3].to(self.device)  # 至少要有一个？ 否则 continue？
                type_masks = batch_data[4].to(self.device)
                attention_masks = batch_data[5].to(self.device)
                connection_labels = batch_data[6].to(self.device)
                agg_labels = batch_data[7].to(self.device)
                tag_labels = batch_data[8].to(self.device)
                con_num_labels = batch_data[9].to(self.device)
                type_labels = batch_data[10].to(self.device)
                cls_index_list = batch_data[11].to(self.device)
                train_dependencies = [tag_masks, sel_masks, con_masks, connection_labels, agg_labels, tag_labels, con_num_labels, type_labels]
                loss = model(input_ids, attention_masks, type_masks, cls_index_list, train_dependencies)
                loss.backward()
                #with amp.scale_loss(loss, optimizer) as scaled_loss:
                #    scaled_loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                # 验证
                #if (i + 1) % valid_every == 0:
                if True:
                    valid_start_time = time.time()
                    model.eval()
                    tag_logits_list = []
                    agg_logits_list = []
                    connection_logits_list = []
                    con_num_logits_list = []
                    type_logits_list = []
                    tag_labels_list = []
                    agg_labels_list = []
                    connection_labels_list = []
                    con_num_labels_list = []
                    type_labels_list = []
                    cls_index_list = []
                    for j, valid_batch_data in enumerate(valid_loader):
                        input_ids = valid_batch_data[0].to(self.device)
                        tag_masks = valid_batch_data[1].to(self.device)
                        sel_masks = valid_batch_data[2].to(self.device)
                        con_masks = valid_batch_data[3].to(self.device)
                        type_masks = valid_batch_data[4].to(self.device)
                        attention_masks = valid_batch_data[5].to(self.device)
                        connection_labels = valid_batch_data[6].to(self.device)
                        agg_labels = valid_batch_data[7].to(self.device)
                        tag_labels = valid_batch_data[8].to(self.device)
                        con_num_labels = valid_batch_data[9].to(self.device)
                        type_labels = valid_batch_data[10].to(self.device)
                        cls_indices = valid_batch_data[11].to(self.device)


                        tag_logits, agg_logits, connection_logits, con_num_logits, type_logits = model(input_ids, attention_masks, type_masks, cls_indices)
                        
                        connection_labels = connection_labels.to('cpu').numpy().tolist()
                        agg_labels = agg_labels.to('cpu').numpy().tolist()
                        tag_labels = tag_labels.to('cpu').numpy().tolist()
                        con_num_labels = con_num_labels.to('cpu').numpy().tolist()
                        type_labels = type_labels.to('cpu').numpy().tolist()
                        cls_indices = cls_indices.to('cpu').numpy().tolist()
                        
                        tag_logits_list.extend(tag_logits)
                        agg_logits_list.extend(agg_logits)
                        connection_logits_list.extend(connection_logits)
                        con_num_logits_list.extend(con_num_logits)
                        type_logits_list.extend(type_logits)
                        tag_labels_list.extend(tag_labels)
                        agg_labels_list.extend(agg_labels)
                        connection_labels_list.extend(connection_labels)
                        con_num_labels_list.extend(con_num_labels)
                        type_labels_list.extend(type_labels)
                        cls_index_list.extend(cls_indices)
                        if self.debug_mode and j == 3: break
                    logits_lists = [tag_logits_list, agg_logits_list, connection_logits_list, con_num_logits_list, type_logits_list]
                    labels_lists = [tag_labels_list, agg_labels_list, connection_labels_list, con_num_labels_list, type_labels_list]
                    eval_result = self.evaluate(logits_lists, cls_index_list, labels_lists)

                    score = 0
                    print("epoch: %d  batchIndex: %d duration: %d min \n" % (epoch,i, int((time.time() - train_start_time) / 60)))
                    print(eval_result)

                    if not self.debug_mode:
                        state_dict = model.state_dict()
                        stage = int((i + 1) / valid_every)
                        train_duration = int((time.time() - train_start_time) / 60)
                        valid_duration = int((time.time() - valid_start_time) / 60)
                        train_start_time = time.time()
                        # model[bert][seed][epoch][stage][model_name][stage_train_duration][valid_duration][score].bin
                        model_name = "./model_save/model_%s_%d_%d_%d_%dmin_%dmin_%.4f.bin" % (self.model_name, self.seed, epoch + 1, stage, train_duration, valid_duration, score)
                        torch.save(state_dict, model_name)
                    model.train()
        # del 训练相关输入和模型
        training_history = [train_loader, valid_loader, model, optimizer, param_optimizer, optimizer_grouped_parameters]
        for variable in training_history:
            del variable
        gc.collect()


if __name__ == "__main__":
    data_dir = "/home1/lsy2018/NL2SQL/XSQL/data"
    trainer = Trainer(data_dir, "model_name", debug_mode=False)
    trainer.train()