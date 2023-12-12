'''
将由视觉特征生成的属性特征与所有类别的属性特征计算余弦相似度，选取最相似的一类作为预测类别
测试使用余弦相似度进行动物类别预测的效果
'''
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from model.model import CVAE,Discriminator,Regressor

path = '../data/Animals_with_Attributes2/'
device = ('cuda' if torch.cuda.is_available() else 'cpu')

testdata = np.load(path+'testdata')
testlabel = np.load(path+'testlabel')

# 定义模型
cvae_model = CVAE(x_dim=2048, s_dim=85).to(device)
discriminator_model = Discriminator(x_dim=2048, s_dim=85).to(device)
regressor_model = Regressor(x_dim=2048, s_dim=85).to(device)

# 加载预训练模型参数
checkpoint = torch.load('/kaggle/input/best-model/Generator_model.pth')
cvae_model.load_state_dict(checkpoint['cvae_state_dict'])
discriminator_model.load_state_dict(checkpoint['discriminator_state_dict'])
regressor_model.load_state_dict(checkpoint['regressor_state_dict'])

# 通过模型中的回归器模型regressor，输入图像特征，输出预测的图像属性标签
# 从图像特征生成属性标签
X_tensor = torch.tensor(testdata,dtype=torch.float32)
X_tensor = X_tensor.to(device)
predict_attr_label = regressor_model.forward(X_tensor)
print(predict_attr_label.shape)
# 建立属性特征和类别特征的映射，将得到的属性标签转换为类别标签
classes = pd.read_csv(path+'classes.txt',sep='\t',header=None)
attribute_classes = pd.read_csv(path+'predicate-matrix-binary.txt',header=None)
attribute_classes = attribute_classes[0].apply(lambda x:[int(num) for num in x.split(' ')])
attribute_classes_np = np.array(list(attribute_classes))
attribute_classes_tensor = torch.Tensor(attribute_classes_np).to(device)

# 根据向量的余弦相似度匹配动物种类
predict_result = []
a = 0
for predicted_attribute in predict_attr_label:
    a += 1
    similarities = [F.cosine_similarity(predicted_attribute, true_label, dim=0) for true_label in attribute_classes_tensor]
    i = similarities.index(max(similarities))
    predict_result.append(i)
    if a%1000 == 0:
        print(a)

# 计算预测准确率
print(accuracy_score(np.array(predict_result),testlabel))