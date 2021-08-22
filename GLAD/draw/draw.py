import codecs
import re
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
from scipy.interpolate import spline
import matplotlib.ticker as ticker
from matplotlib import rc, font_manager

# log_list = ['train_cnn.log']
# log_list = ['train.log']
# log_list = ['train_gce.log']
# log_list = ['train_att_all.log','train_concat1535.log']
# log_list = ['同核1层800.log','异核1层800.log']
# log_list = ['没门控800.log']

def generate_data():
    log_list = ['glad_woz.log','IACNN_woz.log']
    for eachlog in log_list:
        data = []
        read_file = codecs.open(eachlog,'r','utf-8')
        lines = read_file.readlines()
        for line in lines:
            # print(line)
            # result = re.match('epoch',line)
            # result1 = line.find("\'epoch\':")
            result2 = line.find("\'eval_dev_joint_goal\': ")
            # result3 = line.find("\'epoch\':")
            # print(result)
            # if result1 is not -1 or result2 is not -1:
                # print(line)

            if result2 is not -1:
                # print(result2)
                # print(line[result2+len("\'eval_dev_joint_goal\': "):-2])
                with open ('data_jnt_{}.txt'.format(eachlog),'a+') as f:
                    f.write(line[result2+len("\'eval_dev_joint_goal\': "):-2])
                    f.write('\n')

        for line in lines:
            result2 = line.find("\'eval_train_turn_inform\': ")
            if result2 is not -1:
                with open ('data_inf_{}.txt'.format(eachlog),'a+') as f:
                    f.write(line[result2+len("\'eval_train_turn_inform\': "):-2])
                    f.write('\n')

        for line in lines:
            result2 = line.find("\'eval_train_turn_request\': ")
            if result2 is not -1:
                with open ('data_req_{}.txt'.format(eachlog),'a+') as f:
                    f.write(line[result2+len("\'eval_train_turn_request\': "):-2])
                    f.write('\n')




def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(float(previous) * factor + float(point) * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def draw():

    sizeOfFont = 10
    fontProperties = {'family':'serif', 'serif':['Euclid'], 'size' : sizeOfFont}
    ticks_font = font_manager.FontProperties(family='Euclid', style='normal',
    size=sizeOfFont, weight='normal', stretch='normal')
    rc('text', usetex=True)
    rc('font',**fontProperties)



    y = [line.strip()for line in open('data_异核1层800.log.txt', 'r').readlines()][0:300]
    y = [float(eachy) for eachy in y]
    x = list(range(len(y)))
    x = np.array(x)
    y = np.array(y)
    x_new = np.linspace(x.min(),x.max(),300)
    y_new = spline(x,y,x_new)

    x = x_new
    y = y_new
    # y = smooth_curve(y)
    y = ['%.2f%%' % (float(each_y) * 100) for each_y in y]

    pl.figure('123')
    pl.title('123')
    pl.xlabel("train epoch",fontsize=sizeOfFont,family='Euclid')
    pl.ylabel("joint goal",fontsize=sizeOfFont,family='Euclid')
    new_ticks = np.linspace(0, 100, 5)
    pl.yticks(new_ticks)
    # pl.scatter(x, y)              #散点图
    pl.plot(x,y)

    # pl.savefig('./png/所有用户的{feature}特征分布'.format(feature = feature))
    pl.savefig('./png/123')




if __name__ == '__main__':
    
    if not os.path.exists('./png/'):
        os.makedirs('./png/')
    # draw()
    generate_data()


