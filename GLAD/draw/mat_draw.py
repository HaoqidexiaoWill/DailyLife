'''
Created on Nov 3, 2016

draw a learning curve

@author: xiul
'''

import argparse, json
from scipy.interpolate import spline
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
def read_performance_records(path):
    """ load the performance score (.json) file """
    
    data = json.load(open(path, 'rb'))
    # for key in data['success_rate'].keys():
    #     if int(key) > -1:
            # print("%s\t%s\t%s\t%s" % (key, data['success_rate'][key], data['ave_turns'][key], data['ave_reward'][key]))
            

def load_performance_file(path):
    """ load the performance score (.json) file """
    
    data = json.load(open(path, 'rb'))
    # numbers = {'x': [], 'success_rate':[], 'ave_turns':[], 'ave_rewards':[]}
    keylist = [int(key) for key in data['success_rate'].keys()]
    keylist.sort()

    # for key in keylist:
    #     if int(key) > -1:
    #         numbers['x'].append(int(key))
    #         numbers['success_rate'].append(data['success_rate'][str(key)])
    #         numbers['ave_turns'].append(data['ave_turns'][str(key)])
    #         numbers['ave_rewards'].append(data['ave_reward'][str(key)])
    # numbers['x']=np.array(numbers['x'])
    # numbers['success_rate']=np.array(numbers['success_rate'])
    # return numbers
def average(numbers,area):

    new_x_list = []
    new_y_list = []
    for i in range(0, len(numbers["x"])-area, area):
        
        new_x = numbers["x"][i+math.floor(area/2)]
        new_y = np.mean([numbers["success_rate"][i+j] for j in range(0, area-1)])
        new_x_list.append(new_x)
        new_y_list.append(new_y)

    return new_x_list, new_y_list

def draw_learning_curve(numbers,numbers_2):
    """ draw the learning curve """
    
    pl.xlabel('Simulation Epoch')
    pl.ylabel('Success Rate')
    #plt.title('Learning Curve')
    pl.grid(True)


    new_y_list = smooth_curve(numbers['success_rate'])
    new_y_list_2 = smooth_curve(numbers_2['success_rate'])
    pl.plot(numbers['x'], new_y_list,  label='RL-AUS')
    pl.plot(numbers['x'], new_y_list_2, label='RL-world model')
    
    
    pl.legend()
    # plt.show()
    pl.savefig('pic')
            
    
            
def main(params):
    cmd = params['cmd']
    
    if cmd == 0:
        numbers = load_performance_file(params['result_file'])
        numbers_2 = load_performance_file(params['result_file_2'])
        draw_learning_curve(numbers,numbers_2)

    elif cmd == 1:
        read_performance_records(params['result_file'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cmd', dest='cmd', type=int, default=1, help='cmd')
    
    parser.add_argument('--result_file', dest='result_file', type=str, default='data_cnn.txt.json', help='path to the result file')
    parser.add_argument('--result_file_2', dest='result_file_2', type=str,
                        default='data_train_att_all.log.txt.json', help='path to the result file')

    args = parser.parse_args()
    params = vars(args)
    print (json.dumps(params, indent=2))

    main(params)



