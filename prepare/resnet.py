'''
使用经过预训练的resnet101模型提取数据集的图像特征
'''
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torchvision
from torchvision import datasets,models,transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from torch import nn,optim
#import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression

path = '../data/Animals_with_Attributes2/'

classname = pd.read_csv(path+'classes.txt',header=None,sep='\t')
dic_class2name = {classname.index[i]:classname.iloc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.iloc[i][1]:classname.index[i] for i in range(classname.shape[0])}

# 制作测试10类的属性表
def make_test_attributetable():
    attribute_bmatrix = pd.read_csv(path+'predicate-matrix-binary.txt',header=None,sep=' ')
    test_classes = pd.read_csv(path+'testclasses.txt',header=None)
    test_classes_flag = []
    for item in test_classes.iloc[:,0].values.tolist():
        test_classes_flag.append(dic_name2class[item])
    return attribute_bmatrix.iloc[test_classes_flag,:]

class dataset(Dataset):
    def __init__(self,data,label,transform):
        super().__init__()
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index]),self.label[index]

    def __len__(self):
        return self.data.shape[0]

class FeatureExtractor(nn.Module):
    def __init__(self,submodule,extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self,x):
        outputs = []
        for name,model in self.submodule._modules.items():
            if name is "fc":
                x = x.view(x.size(0),-1)
            x = model(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

#traindata = np.load(path+'AW2_traindata_3.npy')
#trainlabel = np.load(path+'AW2_trainlabel_3.npy')
#train_attributelabel = np.load(path+'AWA2_trainlabel_attribute.npy')

testdata = np.load(path+'AW2_testdata.npy')
testlabel = np.load(path+'AW2_testlabel.npy')
#test_attributelabel = np.load(path+'AW2_testlabel_attribute.npy')

#print(traindata.shape,trainlabel.shape,train_attributelabel.shape)
#print(testdata.shape,testlabel.shape,test_attributelabel.shape)

data_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
#train_dataset = dataset(traindata,trainlabel,data_tf)
test_dataset = dataset(testdata,testlabel,data_tf)

#train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

model = models.resnet101(pretrained=True)

if torch.cuda.is_available():
    model=model.cuda()

model.eval()

exact_list = ['avgpool']    # 提取最后一层池化层的输出作为图像特征
myexactor = FeatureExtractor(model,exact_list)

train_feature_list = []

'''for data in tqdm(train_loader):
    img,label = data
    if torch.cuda.is_available():
        with torch.no_grad():
            img = Variable(img).cuda()
        with torch.no_grad():
            label = Variable(label).cuda()
    else:
        with torch.no_grad():
            img = Variable(img)
        with torch.no_grad():
            label = Variable(label)
    feature = myexactor(img)[0]
    feature = feature.resize(feature.shape[0],feature.shape[1])
    train_feature_list.append(feature.detach().cpu().numpy())

trainfeatures = np.row_stack(train_feature_list)
'''
test_feature_list = []

for data in tqdm(test_loader):
    img,label = data
    if torch.cuda.is_available():
        with torch.no_grad():
            img = Variable(img).cuda()
        with torch.no_grad():
            label = Variable(label).cuda()
    else:
        with torch.no_grad():
            img = Variable(img)
        with torch.no_grad():
            label = Variable(label)
    feature = myexactor(img)[0]
    feature = feature.resize(feature.shape[0],feature.shape[1])
    test_feature_list.append(feature.detach().cpu().numpy())

testfeatures = np.row_stack(test_feature_list)
np.save(path+'test_attributelabel.npy',testfeatures)

#print(trainfeatures.shape,testfeatures.shape)