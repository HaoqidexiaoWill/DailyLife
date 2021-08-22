import pandas as pd
import re
import Levenshtein

def process(data_path):

    # reader = pd.read_table(data_path, low_memory=False,header = None,names=['question','answer','label'],sep='\t',nrows =20)
    reader = pd.read_table(data_path, low_memory=False,header = None,names=['question','answer','label'],sep='\t')

    #print(reader.head(2))
    question_new,answer_new,label_new = [],[],[]
    for index, row in reader.iterrows():
        # print(row['question'],row['answer'],row['label'])
        if len(row['answer'])> int(45):
            answer_list_ = cut_sentence(row['answer'])
            answer_list = [x for x in answer_list_ if len(x)>15]
            positive_answer_num,negative_answer_num = 0,0
            for each_subanswer in answer_list:
                if similarity(row['question'],each_subanswer) > float(0.8):
                    question_new.append(row['question'])
                    answer_new.append(each_subanswer)
                    label_new.append(1)
                    positive_answer_num = positive_answer_num + 1
                elif similarity(row['question'],each_subanswer)<float(0.8) and similarity(row['question'],each_subanswer) > float(0.5):
                    if positive_answer_num-negative_answer_num >= 0 :
                        question_new.append(row['question'])
                        answer_new.append(each_subanswer)
                        label_new.append(0)
                        negative_answer_num = negative_answer_num +1 
                    else:
                        continue
            assert len(question_new) == len(answer_new)
            assert len(question_new) == len(label_new)
            if positive_answer_num == 0 or negative_answer_num == 0:
                question_new.append(row['question'])
                # print(answer_list)
                answer_new.append(answer_list_[0])
                label_new.append(0)
        else:
            question_new.append(row['question'])
            answer_new.append(row['answer'])
            label_new.append(row['label'])
        
        assert len(question_new) == len(answer_new)
        assert len(question_new) == len(label_new)


    return question_new,answer_new,label_new
                                    
def writeFile(question_new,answer_new,label_new):
    # dic1={'name':['小明','小红','狗蛋','铁柱'],'age':[17,20,5,40],'gender':['男','女','女','男']}
    data = {
        'question':question_new,
        'answer':answer_new,
        'label':label_new
        }
    data_pandas = pd.DataFrame(data)
    data_pandas.to_csv('QA数据处理完.csv', index = False, header = False)




# dic1={'name':['小明','小红','狗蛋','铁柱'],'age':[17,20,5,40],'gender':['男','女','女','男']}
# df3=pd.DataFrame(dic1)
# sentence = '如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号'
# sentence = '驾车路线：全程约#.#公里起点：凉山火盆烧烤#.成都市内驾车方案#从起点向正西方向出发，沿金光路行驶#米，调头进入金光路#沿金光路行驶#米，直行进入金光路#沿金光路行驶#米，在第#个出口，朝g#/京昆高速/新都大道方向，左前方转弯进入新都大道#沿新都大道行驶#.#公里，朝g#/成绵高速方向，直行上匝道#沿匝道行驶#米，直行进入京昆高速#沿京昆高速行驶#.#公里，过龙佛寺高架桥，朝成都绕城高速/g#方向，稍向右转进入城北立交桥#沿城北立交桥行驶#.#公里，过白鹤林立交约#米后，直行进入成都绕城高速#沿成都绕城高速行驶#.#公里，朝三环路成南立交/十里店方向，稍向右转进入螺狮坝立交桥#.沿螺狮坝立交桥行驶#.#公里，直行进入沪蓉高速#.沿沪蓉高速行驶#.#公里，直行进入沪渝高速#.沿沪渝高速行驶#.#公里，朝当阳/荆门/武汉北/合肥方向，稍向右转上匝道#.沿匝道行驶#米，直行进入沪蓉高速#.沿沪蓉高速行驶#.#公里，朝武汉外环/上海/g#方向，稍向左转上匝道#.沿匝道行驶#.#公里，直行进入沪蓉高速#.沿沪蓉高速行驶#.#公里，过横店互通桥，直行进入武汉绕城高速#.沿武汉绕城高速行驶#.#公里，朝麻城/合肥/天兴洲大桥方向，稍向右转进入新集互通#.沿新集互通行驶，过熊家桥约#.#公里后，直行进入沪蓉高速#.沿沪蓉高速行驶#.#公里，稍向左转进入沪陕高速#.沿沪陕高速行驶#.#公里，朝蚌埠/芜湖/徐州/南京方向，直行进入路口枢纽#.沿路口枢纽行驶#.#公里，直行进入京台高速#.沿京台高速行驶#.#公里，朝芜湖/杭州/g#方向，直行进入芜合高速#.沿芜合高速行驶#.#公里，过清溪河桥，朝马鞍山/和县/南京/常熟方向，稍向右转进入马鞍山西枢纽#.沿马鞍山西枢纽行驶#.#公里，直行进入天潜高速#.沿天潜高速行驶#.#公里，直行进入巢马高速#.沿巢马高速行驶#.#公里，直行进入常合高速#.沿常合高速行驶#.#公里，直行进入沿江高速#.沿沿江高速行驶#.#公里，直行进入常合高速#.沿常合高速行驶#.#公里，朝上海/g#方向，稍向右转进入横林枢纽立交桥#.沿横林枢纽立交桥行驶#.#公里，直行进入沪蓉高速#.沿沪蓉高速行驶#.#公里，直行进入京沪高速#.沿京沪高速行驶#.#公里，直行进入京沪高速#.沿京沪高速行驶#米，直行进入武宁路#.上海市内驾车方案#沿武宁路行驶#.#公里，左前方转弯进入中山北路#沿中山北路行驶#米，稍向右转上匝道#沿匝道行驶#米，过右侧的上海物资贸易中心大厦约#米后，直行进入内环高架路#沿内环高架路行驶#.#公里，过中山北路桥，在广中路出口，稍向右转上匝道#沿匝道行驶#米，过左侧的久乐大厦约#米后，直行进入中山北一路#沿中山北一路行驶#米，朝西江湾路方向，右转进入广中路#沿广中路行驶#米，过右侧的广中大楼约#米后，右转进入西江湾路#沿西江湾路行驶#米，直行进入东江湾路#沿东江湾路行驶#米，过右侧的方舟大厦，直行进入四川北路#沿四川北路行驶#米，左前方转弯进入溧阳路#沿溧阳路行驶#米，左转#行驶#米，右前方转弯#行驶#米，到达终点(在道路左侧)终点：华盛顿咖啡厅'

def cut_sentence(para):
    para = re.sub('([,，。！？\?])([^”’])', r"\1\n\2", para)                                                                    # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)                                                                         # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)                                                                        # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def similarity(context1,context2):
    # print(Levenshtein.ratio(context1, context2))
    # print( Levenshtein.distance(context1, context2))
    # if float(Levenshtein.ratio(context1, context2)) > float(0.8):
    #     return 1
    # else:
    #     return 0
    return float(Levenshtein.ratio(context1, context2))
# a = cut_sentence(sentence)
# b = [x for x in a if len(x)>15]
# print(b)
# print(len(b))
if __name__ == '__main__':
    data_path = '../data/Docomo/QA数据集'
    question_new,answer_new,label_new = process(data_path)
    writeFile(question_new,answer_new,label_new)

