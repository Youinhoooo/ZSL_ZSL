'''
模型训练
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from model.model import CVAE,Discriminator,Regressor

path = '../data/Animals_with_Attributes2/'

traindata = np.load(path+'traindata_svc.npy')
trainlabel = np.load(path+'trainlabel_svc_num.npy')
testdata = np.load(path+'testdata.npy')
testlabel = np.load(path+'testlabel.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义参数
learning_rate = 0.0001
num_epochs = 500

# 定义模型
cvae_model = CVAE(x_dim=2048, s_dim=85).to(device)
discriminator_model = Discriminator(x_dim=2048, s_dim=85).to(device)
regressor_model = Regressor(x_dim=2048, s_dim=85).to(device)

# 定义优化器
cvae_optimizer = torch.optim.Adam(cvae_model.parameters(), lr=learning_rate)
discriminator_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=learning_rate)
regressor_optimizer = torch.optim.Adam(regressor_model.parameters(), lr=learning_rate)

# 生成数据集
y = np.array(trainlabel['attribute'].to_list())
X = minmax_scale(traindata)
# 将numpy数组转化为pytorch张量
X_tensor = torch.tensor(X,dtype=torch.float32)
y_tensor = torch.tensor(y,dtype=torch.float32)
# 划分训练集和测试集
X_train,X_val,y_train,y_val = train_test_split(X_tensor,y_tensor,test_size=0.2,random_state=42)
X_train,y_train = X_train.to(device),y_train.to(device)
X_val,y_val = X_val.to(device),y_val.to(device)
train_dataset = TensorDataset(X_train,y_train)
val_dataset = TensorDataset(X_val,y_val)

# 创建DataLoader加载数据
batch_size = 64
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

for epoch in range(num_epochs):
    cvae_epoch_loss = 0.0
    discriminator_epoch_loss = 0.0
    regressor_epoch_loss = 0.0
    for batch_data in train_loader:
        X, S = batch_data  # 根据你的数据加载器得到 X 和 S

        # 清零梯度
        cvae_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        regressor_optimizer.zero_grad()

        # CVAE 前向传播
        Xp, mu, log_sigma = cvae_model(X, S)
        Sp = regressor_model(Xp)

        # 计算 CVAE 损失
        cvae_loss = cvae_model.vae_loss(X, Xp, mu, log_sigma)
        # print(cvae_loss.data)

        # 判别器前向传播和损失计算

        # print(nn.MSELoss(discriminator_model.forward(X,S),torch.tensor(1.0).cuda()))
        discriminator_loss = discriminator_model.dis_loss(X=X, Xp=Xp, S=S, Sp=Sp)

        # 回归器前向传播和损失计算
        regressor_loss = regressor_model.reg_loss(Sp, S, Xp, X)

        # 总损失
        total_loss = cvae_loss + discriminator_loss + regressor_loss

        # 反向传播和优化
        total_loss.backward()
        cvae_optimizer.step()
        discriminator_optimizer.step()
        regressor_optimizer.step()

        # 累积损失
        cvae_epoch_loss += cvae_loss.item()
        discriminator_epoch_loss += discriminator_loss.item()
        regressor_epoch_loss += regressor_loss.item()

    # 计算平均损失并输出
    num_batches = len(train_loader)
    avg_cvae_loss = cvae_epoch_loss / num_batches
    avg_discriminator_loss = discriminator_epoch_loss / num_batches
    avg_regressor_loss = regressor_epoch_loss / num_batches

    print(
        f'Epoch:{epoch + 1},Avg Cvae Loss:{avg_cvae_loss:.4f},Avg Discriminator Loss:{avg_discriminator_loss:.4f},Avg Regressor Loss:{avg_regressor_loss:.4f}')

    # 判断是否是最小损失，如果是则保存模型参数
    min_total_loss = float('inf')
    current_total_loss = avg_cvae_loss + avg_discriminator_loss + avg_regressor_loss
    if current_total_loss < min_total_loss:
        min_total_loss = current_total_loss
        best_model_dict = {
            'cvae_state_dict': cvae_model.state_dict(),
            'discriminator_state_dict': discriminator_model.state_dict(),
            'regressor_state_dict': regressor_model.state_dict(),
        }

#----------------------------------------------------------------------------------------------------------#
# 使用训练好的模型中的CVAE模型，输入测试集中的标签(属性标签)，生成预测的图像特征
# 从测试数据集的标签生成图像特征
S = np.array(testlabel['attribute'].to_list())
S_tensor = torch.tensor(S,dtype=torch.float32)
S_tensor = S_tensor.to(device)

# 得到预测的图像特征
Xp = cvae_model.sample(S_tensor)

# 生成新的数据集
traindata_new = np.append(minmax_scale(traindata),np.copy(Xp.cpu().detach().numpy()),axis=0)
# 用属性表示的标签
trainlabel_new = np.append(np.array(trainlabel['attribute'].to_list()),np.array(testlabel['attribute'].to_list()),axis=0)

# 生成新的训练数据集
y = trainlabel_new
X = traindata_new
# 将numpy数组转化为pytorch张量
X_tensor = torch.tensor(X,dtype=torch.float32)
y_tensor = torch.tensor(y,dtype=torch.float32)
# 划分训练集和测试集
X_train,X_val,y_train,y_val = train_test_split(X_tensor,y_tensor,test_size=0.2,random_state=42)
X_train,y_train = X_train.to(device),y_train.to(device)
X_val,y_val = X_val.to(device),y_val.to(device)
train_dataset = TensorDataset(X_train,y_train)
val_dataset = TensorDataset(X_val,y_val)

for epoch in range(num_epochs):
    cvae_epoch_loss = 0.0
    discriminator_epoch_loss = 0.0
    regressor_epoch_loss = 0.0
    for batch_data in train_loader:
        X, S = batch_data  # 根据你的数据加载器得到 X 和 S

        # 清零梯度
        cvae_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        regressor_optimizer.zero_grad()

        # CVAE 前向传播
        Xp, mu, log_sigma = cvae_model(X, S)
        Sp = regressor_model(Xp)

        # 计算 CVAE 损失
        cvae_loss = cvae_model.vae_loss(X, Xp, mu, log_sigma)
        # print(cvae_loss.data)

        # 判别器前向传播和损失计算

        # print(nn.MSELoss(discriminator_model.forward(X,S),torch.tensor(1.0).cuda()))
        discriminator_loss = discriminator_model.dis_loss(X=X, Xp=Xp, S=S, Sp=Sp)

        # 回归器前向传播和损失计算
        regressor_loss = regressor_model.reg_loss(Sp, S, Xp, X)

        # 总损失
        total_loss = cvae_loss + discriminator_loss + regressor_loss

        # 反向传播和优化
        total_loss.backward()
        cvae_optimizer.step()
        discriminator_optimizer.step()
        regressor_optimizer.step()

        # 累积损失
        cvae_epoch_loss += cvae_loss.item()
        discriminator_epoch_loss += discriminator_loss.item()
        regressor_epoch_loss += regressor_loss.item()

    # 计算平均损失并输出
    num_batches = len(train_loader)
    avg_cvae_loss = cvae_epoch_loss / num_batches
    avg_discriminator_loss = discriminator_epoch_loss / num_batches
    avg_regressor_loss = regressor_epoch_loss / num_batches

    print(
        f'Epoch:{epoch + 1},Avg Cvae Loss:{avg_cvae_loss:.4f},Avg Discriminator Loss:{avg_discriminator_loss:.4f},Avg Regressor Loss:{avg_regressor_loss:.4f}')

    # 判断是否是最小损失，如果是则保存模型参数
    min_total_loss = float('inf')
    current_total_loss = avg_cvae_loss + avg_discriminator_loss + avg_regressor_loss
    if current_total_loss < min_total_loss:
        min_total_loss = current_total_loss
        best_model_dict = {
            'cvae_state_dict': cvae_model.state_dict(),
            'discriminator_state_dict': discriminator_model.state_dict(),
            'regressor_state_dict': regressor_model.state_dict(),
        }

# 保存模型
torch.save(best_model_dict,'model/best_model.pth')