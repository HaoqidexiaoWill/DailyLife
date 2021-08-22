import requests
import json
import urllib.request
import ssl
#import jieba
import pkuseg


def pre_prosess(text):
    stop = [line.strip() for line in open('stopwords.txt', 'r').readlines()]
    #segs = jieba.cut(text, cut_all=False)
    segDefault = pkuseg.pkuseg()					            #默认分词类型
    segs = segDefault.cut(text)
    text = [seg for seg in segs if seg not in stop]
    text = ''.join(text)
    return text


def parsing_baidu(text):
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=l4mtRVYK39sdYnbmR9c6RceU&client_secret=A2o44wUYohdfrOwwQqN8DFLhBXwXOnGo'
    request = urllib.request.Request(host)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = urllib.request.urlopen(request)
    content = response.read()
    if (content):
        content = bytes.decode(content)
        content = json.loads(content)
        access_token = content['access_token']
    url = 'https://aip.baidubce.com/rpc/2.0/kg/v1/cognitive/entity_annotation?access_token=' + str(access_token)
    headers = {
        'content-type': 'application/json',
    }
    data = json.dumps({'data': text})
    response = requests.post(url, headers=headers, data=data)
    ret = json.loads(response.text)
    entity_list = []
    if ret["entity_annotation"]:
        for item in ret["entity_annotation"]:
            entity = item['mention']
            level1 = item['concept']['level1']
            level2 = item['concept']['level2']
            description = item['desc']
            entity_list.append({entity: {'level1': level1, 'level2': level2, 'description': description}})
    # print(entity_list)
    return entity_list
    # print(response.text)

if __name__ == '__main__':
    text = '月下旬要去日本了，先去神户，大约待天，请问电'
    # parsing('月下旬要去日本了，先去神户，大约待天，请问电')
    text = pre_prosess(text)
    #print(text)
    '''
    月下旬去日本先去神户待天电
    '''
    #print(parsing_baidu(text))

    '''
    [
        {'下旬': {'level1': '语言文化', 'level2': '文字词汇', 'description': '下旬'}}, 
        {'去': {'level1': '语言文化', 'level2': '文字词汇', 'description': '去'}}, 
        {'日本': {'level1': '地理', 'level2': '行政区域', 'description': '日本国'}}, 
        {'神户': {'level1': '地理', 'level2': '行政区域', 'description': '神户'}}, 
        {'天电': {'level1': '', 'level2': '', 'description': '天电'}}
        ]
    '''