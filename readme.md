# ZSL

关键：输入给模型一张图像，先提取出图像的特征，然后转化为词嵌入，然后与字典中最接近的词嵌入进行比较，特出图像的标签。

特点：将每一个类别Yi都表示成语义向量ai的形式，这个语义向量的每一个维度都表示这个类别的一个属性，例如黑色、有羽毛等。当这个类别包含这种属性时，那在其维度上被设置为非零值。

对于对象分类问题，需要建立图像特征空间和语义空间两者之间的联系。

## 方法

1.基于嵌入的方法：提取出图像的特征向量后投影到语义向量上，在语义空间中寻找与此语义向量距离最近的向量对应的标签作为此图像的标签。

缺点：由于投影函数是在训练期间仅使用已知类学习得来的，因此它将偏向于将已知类标签作为输出进行预测。另外，也不能保证学习的投影函数会在测试时正确地将未知类图像特征正确地映射到相应的语义空间。这是由于这样一个事实，即深度网络仅在训练过程中学会了将已知类的图像特征映射到语义空间，并且可能无法在测试时对新颖的未知类正确地执行相同的操作。

2.基于生成模型的方法：为了克服上述缺点，**有必要让零样本分类器在已知类和未知类的所有图像上训练**。由此引出基于生成模型的方法。**生成方法旨在为未知类使用语义属性生成图像特征**。一般是用一个条件生成对抗网络来实现的，该条件生成对抗网络生成以特定类的语义属性为条件的图像特征。（从未知类的语义属性生成图像特征）

与基于嵌入的方法类似，我们使用特征提取网络获取 N 维特征向量。首先，将属性向量输入到生成模型中，生成器生成一个以属性向量为条件的 N 维输出向量。训练生成模型，使合成特征向量尽量逼近原始 N 维特征向量。

一旦训练好生成器模型，将冻结生成器模型的参数，并将未知类的属性向量输入生成器模型，以生成未知类的图像特征。现在，有了未知类的图像特征和已知类的图像特征，可以训练一个分类器，从图像特征向量得到标签。

鉴别器网络：识别属性向量对某个类的归属度。

零样本学习领域的挑战（存在的问题）

域漂移：从图像到图像特征向量的特征提取器只从训练集的样本训练得到，因此在测试时如果遇到来自训练分布以外的未知类的话，模型的性能可能会很差。

偏见Bias

跨域知识迁移

语义损失

Hubness