'''
测试xgboost、SVM、KNeighborsClassifier的分类效果
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import xgboost as xgb

traindata = np.load('traindata_svc.npy')
trainlabel = np.load('trainlabel_svc_num.npy')
print(traindata.shape)
print(trainlabel.shape)

# 使用高斯核函数的svm
svm_model = SVC(kernel='rbf')
# 线性svm
svm_model_linear = LinearSVC()
# K近邻分类
knn_model = KNeighborsClassifier()
# xgboost
xgb_model = xgb()

X_shuffle,y_shuffle = shuffle(traindata,trainlabel,random_state=3247)
X_train,X_val,y_train,y_val = train_test_split(X_shuffle,y_shuffle,test_size=0.2)

cross_val_score(svm_model,X_train,y_train,scoring='accuracy',cv=3)
cross_val_score(svm_model_linear,X_train,y_train,scoring='accuracy',cv=3)
cross_val_score(knn_model,X_train,y_train,scoring='accuracy',cv=3)
cross_val_score(xgb_model,X_train,y_train,scoring='accuracy',cv=3)