import codecs
import re
import json

file_list = ['data_异核1层800.log.txt']
# file_list = ['data_train_att_all.log.txt']
final_data = {}
for eachfile in file_list:
    data = {}
    read_file = codecs.open(eachfile,'r','utf-8')
    lines = read_file.readlines()

    index = 0
    for line in lines:
        data[index] = line.strip()
        index = index + 1

    final_data['success_rate'] = data

    with open("{}.json".format(eachfile), "w", encoding='utf-8') as json_file:
        json.dump(final_data,json_file,ensure_ascii=False,indent=4)