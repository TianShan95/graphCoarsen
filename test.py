# import pandas as pd
#
# a = {'a': [1, 2, 3, 4, 5, 6], 'b': [7 ,8 ,9, 10, 11, 12]}
# a1 = {'a': [11, 12, 13, 14, 15, 16], 'b': [27 ,28 ,29, 30, 31, 32]}
#
# b = pd.DataFrame(a)
# b1 = pd.DataFrame(a1)
# frames = [b, b1]
# f = pd.concat(frames)
# print(f)
# print(type(f))
# print(f.get('a').values)
# print(len(f.get('a').values))
# print(type(f.get('a').values))

# import os
#
# print(os.path.basename('Pre_trained'))
# import time
# print(time.time)
# import networkx as nx
# print(float((nx.__version__)[:3]))

# import logging
# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
# logging.basicConfig(filename='log_test.log', level=logging.DEBUG, format=LOG_FORMAT)
# a = logging.info('sdfa')

# a = ['a']
# b = []
# print(b.append(len(a)))
# a.append('b')
# a.append('c')
# print(a)
# b.append(len(a))
# print(b)
# print(max(b))
#
# a = [1, 2, 3, 4, 5, 100]
# print(sum(a))

# from args import arg_parse
#
# args = arg_parse()
#
# out_dir = args.bmname+ '/tar_' + '_graphSize_' +str(args.gs) + str(args.train_ratio) + '_ter_' + str(args.test_ratio) + '/'   +  'num_shuffle' + str(args.num_shuffle)  + '/' +  'numconv_' + str(args.num_gc_layers) + '_dp_' + str(args.dropout) + '_wd_' + str(args.weight_decay) + '_b_' + str(args.batch_size) + '_hd_' + str(args.hidden_dim) + '_od_' + str(args.output_dim)  + '_ph_' + str(args.pred_hidden) + '_lr_' + str(args.lr)  + '_concat_' + str(args.concat)
# print(out_dir)
# out_dir = out_dir + '_ps_' + '_graphSize_' +str(args.gs) + args.pool_sizes  + '_np_' + str(args.num_pool_matrix) + '_nfp_' + str(args.num_pool_final_matrix) + '_norL_' + str(args.normalize)  + '_mask_' + str(args.mask) + '_ne_' + args.norm  + '_cf_' + str(args.con_final)
# print(out_dir)

# import pandas as pd
#
# a = pd.DataFrame()
# print(a)
# a1 = {'a': [1, 2, 3, 4, 5, 6], 'b': [7 ,8 ,9, 10, 11, 12]}
# a2 = {'a': [11, 12, 13, 14, 15, 16], 'b': [27 ,28 ,29, 30, 31, 32]}
#
# b1= pd.DataFrame(a1)
# print(b1)
# b2 = pd.DataFrame(a2)
# b = [a1, a2]
# # for i in b:
# #     a.append(pd.DataFrame(i))
# a.append(b1)
# a.append(b2)
# print(a)

# import time
# print(        f.write('train step consume total time: ' + str(time.time()-train_start_time) + 's -----------------------------\n'))

# import matplotlib.pyplot as plt
# axes1 = plt.gca()
# axes2 = axes1.twiny()
#
# # axes2.set_xticks([.33, .66, .99])
#
# axes1.set_xlabel("x-axis 1")
# axes2.set_xlabel("x-axis 2")
#
# x1 = [1, 2, 3, 4, 5, 6, 7, 8]
# x2 = [1, 1, 1, 1, 1, 2, 2, 2]
# y = []

import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates
from matplotlib.transforms import Transform
from matplotlib.ticker import (
    AutoLocator, AutoMinorLocator)

# fig, ax = plt.subplots(constrained_layout=True)
# x = np.arange(0, 360, 1)
# y = np.sin(2 * x * np.pi / 180)
# ax.plot(x, y)
#
# ax.set_xlabel('angle [degrees]')
# ax.set_ylabel('signal')
# ax.set_title('Sine wave')
#
#
# def deg2rad(x):
#     return x * np.pi / 180
#
#
# def rad2deg(x):
#     return x * 180 / np.pi
#
# secax = ax.secondary_xaxis('top', functions=(deg2rad, rad2deg))
# secax.set_xlabel('angle [rad]')
# plt.show()

# import argparse
#
#
# def arg_parse():
#     parser = argparse.ArgumentParser(description='Arguments.')
#     parser.add_argument('--bmname', dest='bmname',
#                         help='Name of the benchmark dataset')
#     parser.add_argument('--max-nodes', dest='max_nodes', type=int,
#                         help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
#     parser.add_argument('--lr', dest='lr', type=float,
#                         help='Learning rate.')
#
#
#     parser.set_defaults(max_nodes=100,
#                         feature_type='default',
#                         )
#     return parser.parse_args()
#
# a = arg_parse()
# print('max_mode: ', a.max_nodes)
# print('a: ', a.lr)


# import time
# # a = time.time()
# # print(time.time())
# #
# # time.sleep(5)
# # print(type(time.time()-a))
#
# print(type(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
#

# val = 12.3
#
# print(f'{val:.2f}')
# print(f'{val:.5f}')


import argparse

parser = argparse.ArgumentParser()

# By default it will fail with multiple arguments.
parser.add_argument('--default')

# Telling the type to be a list will also fail for multiple arguments,
# but give incorrect results for a single argument.
parser.add_argument('--list-type', type=list)

# This will allow you to provide multiple arguments, but you will get
# a list of lists which is not desired.
parser.add_argument('--list-type-nargs', type=list, nargs='+')

# This is the correct way to handle accepting multiple arguments.
# '+' == 1 or more.
# '*' == 0 or more.
# '?' == 0 or 1.
# An int is an explicit number of arguments to accept.
parser.add_argument('--nargs', nargs='+')

# To make the input integers
parser.add_argument('--nargs-int-type', nargs='+', type=int)

# An alternate way to accept multiple inputs, but you must
# provide the flag once per input. Of course, you can use
# type=int here if you want.
parser.add_argument('--append-action', action='append')

# To show the results of the given option to screen.
for _, value in parser.parse_args()._get_kwargs():
    if value is not None:
        print(value)
