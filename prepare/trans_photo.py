'''
将数据集的图片存储为(224,224)的numpy.array形式
'''
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm

image_size = 224    # 指定图片大小
path = '../data/Animals_with_Attributes2/'

classname = pd.read_csv(path+'classes.txt',header=None,sep='\t')
dic_class2name = {classname.index[i]:classname.iloc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.iloc[i][1]:classname.index[i] for i in range(classname.shape[0])}
# 两个字典，记录标签信息，分别是数字对应到文字，文字对应到数字

# 根据目录读取一类图像，read_num指定每一类读取多少图片，图片统一大小为image_size
def load_Img(imgDir,read_num='max'):
    imgs = os.listdir(imgDir)
    imgs = np.ravel(pd.DataFrame(imgs).sort_values(0).values)
    if read_num == 'max':
        imgNum = len(imgs)
    else:
        imgNum = read_num
    data = np.empty((imgNum,image_size,image_size,3),dtype="float32")
    print(imgNum)
    for i in range(imgNum):
        img = Image.open(imgDir+"/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        if arr.shape[1] > arr.shape[0]:
            arr = cv2.copyMakeBorder(arr,int((arr.shape[1]-arr.shape[0])/2),int((arr.shape[1]-arr.shape[0])/2),0,0,cv2.BORDER_CONSTANT,value=0)
        else:
            arr = cv2.copyMakeBorder(arr,0,0,int((arr.shape[0]-arr.shape[1])/2),int((arr.shape[0]-arr.shape[1])/2),cv2.BORDER_CONSTANT,value=0)
        arr = cv2.resize(arr,(image_size,image_size))
        if len(arr.shape) == 2:
            temp = np.empty((image_size,image_size,3))
            temp[:,:,0] = arr
            temp[:,:,1] = arr
            temp[:,:,2] = arr
            arr = temp
        data[i,:,:,:] = arr
    return data,imgNum

# 读取数据
def load_data(train_classes,test_classes,num):
    read_num = num

    traindata_list = []
    trainlabel_list = []
    testdata_list = []
    testlabel_list = []

    for item in tqdm(train_classes.iloc[:,0].values.tolist()):
        tup = load_Img(path+'JPEGImages/'+item,read_num=read_num)
        #traindata_list.append(tup[0])
        trainlabel_list += [dic_name2class[item]]*tup[1]

    for item in tqdm(test_classes.iloc[:,0].values.tolist()):
        tup = load_Img(path+'JPEGImages/'+item,read_num=read_num)
        #testdata_list.append(tup[0])
        testlabel_list += [dic_name2class[item]]*tup[1]

    return np.array(trainlabel_list),np.array(testlabel_list)
    #return np.row_stack(testdata_list),np.array(testlabel_list)
    #return np.row_stack(traindata_list),np.array(trainlabel_list),np.row_stack(testdata_list),np.array(testlabel_list)

train_classes = pd.read_csv(path+'trainclasses.txt',header=None)
test_classes = pd.read_csv(path+'testclasses.txt',header=None)

'''train_classes_1 = train_classes[:10]
train_classes_2 = train_classes[10:20]
train_classes_3 = train_classes[20:30]
train_classes_4 = train_classes[30:]'''

'''traindata,trainlabel = load_data(train_classes_1,test_classes,num='max')
np.save(path+'AW2_traindata_1.npy',traindata)
np.save(path+'AW2_trainlabel_1.npy',trainlabel)
print(traindata.shape,trainlabel.shape)

traindata,trainlabel = load_data(train_classes_2,test_classes,num='max')
np.save(path+'AW2_traindata_2.npy',traindata)
np.save(path+'AW2_trainlabel_2.npy',trainlabel)
print(traindata.shape,trainlabel.shape)

traindata,trainlabel = load_data(train_classes_3,test_classes,num='max')
np.save(path+'AW2_traindata_3.npy',traindata)
np.save(path+'AW2_trainlabel_3.npy',trainlabel)
print(traindata.shape,trainlabel.shape)

traindata,trainlabel = load_data(train_classes_4,test_classes,num='max')
np.save(path+'AW2_traindata_4.npy',traindata)
np.save(path+'AW2_trainlabel_4.npy',trainlabel)
print(traindata.shape,trainlabel.shape)'''
#traindata,trainlabel,testdata,testlabel = load_data(train_classes,test_classes,num='max')
trainlabel,testlabel = load_data(train_classes,test_classes,num='max')

np.save(path+'new_trainlabel.npy',trainlabel)
np.save(path+'new_testlabel.npy',testlabel)

#print(traindata.shape,trainlabel.shape)
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)

# 将图像和标签保存为numpy数组，下次可以直接读取
#np.save(path+'AW2_traindata.npy',traindata)
#np.save(path+'AW2_trainlabel.npy',trainlabel)
'''np.save(path+'AW2_testdata.npy',testdata)
np.save(path+'Aw2_testlabel.npy',testlabel)'''
