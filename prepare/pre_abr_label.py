'''
制作数据集的属性标签
'''

import pandas as pd
import numpy as np

path = '../data/Animals_with_Attributes2/'

train_att = np.load(path+'train_attributelabel.npy',allow_pickle=True)
test_att = np.load(path+'AWA2_test_continuous_attributelabel.npy',allow_pickle=True)

train_att = pd.DataFrame(train_att,columns=['attribute'])
print(train_att['attribute'])
print(train_att.columns)

def transform(x):
    return list(map(float,test_att[0][0].split('  ')[1:]))

train_att['attribute'] = train_att['attribute'].map(transform)

print(len(list(map(float,test_att[0][0].split('  ')[1:]))))

print(len(train_att['attribute'][1]))

train_att.to_csv('train_attributelabel_list.csv')