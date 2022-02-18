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

import os

print(os.path.basename('Pre_trained'))
import time
print(time.time)
import networkx as nx
print(float((nx.__version__)[:3]))