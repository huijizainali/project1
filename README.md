# project1
构建两层神经网络分类器
## 数据集介绍：
本次实验的数据集为MINIST数据集，该数据集是一个著名的手写训练集。其来自美国国家标准与技术研究所，National Institute of Standards and Technology (NIST)。训练集由来自250个不同人手写的数字构成, 其中50%是高中学生, 50%来自人口普查局的工作人员。测试集也是同样比例的手写数字数据。该数据集中一共包含60000个训练数据和10000个测试数据，每张图片的像素为28*28，等价于一个28*28的0-1矩阵。本次实验中，我们从keras库中直接导入该数据集，将每张图片转为一个784=28*28的向量，并且将每张图片的标签转为一个10维的向量，其中‘0’对应于（1,0,0,0,0,0,0,0,0,0,），‘1’对应于（0,1,0,0,0,0,0,0,0,0）……‘10’对应于（0,0,0,0,0,0,0,0,0,1）。
## 网络结构：
本次实验的网络结构一共是两层，训练数据经过隐藏层后直接到输出层。层与层是之间是全连接的形式，其中输入层到隐藏层的激活函数为Relu函数，隐藏层到输出层之间的激活函数为Softmax，损失函数为交叉熵损失函数。设隐藏层大小为n，则该网络结构的参数个数一共为（784+10+1）*n + 10。
在本次实验的训练过程中，我们采用的优化器为SGD，并设其学习率每隔10个epoch变为原来的0.8。为缓和过拟合，我们使用了L2正则化。
代码描述：
	我们首先定义了基础的函数，Relu和Softmax，我们定义了类‘Layer’来表示网络层，其包含了4个函数， 在正向传递时，每个层可以通过get_output函数计算该层的输出结果，这个结果将被下一层作为输入数据进行使用。在反向传递时，每一层的输入的梯度可以通过get_input_grad函数计算得到。如果是最后一层，那么梯度计算方程将利用目标结果进行计算。如果是中间的某一层，那么梯度就是梯度计算函数的输出结果。如果每个层有迭代参数的话，那么可以在get_params_iter函数中实现，并且在get_params_grad函数中按照原来的顺序实现参数的梯度。之后，我们便以此定义了线性层（LinearLayer）、Relu层（ReluLayer）和Softmax层（SoftmaxLayer）。
	之后在函数forward_step中计算输出结果，在backward_step中计算返回的梯度，最后再函数update_params中更新参数的值。
## 参数查找：
我们一共需要查找三个超参，分别是初始学习率、隐藏层大小和L2正则化强度，我们首先假设正则化强度为0，在隐藏层神经元个数分别为32、64、128、256、512时，对初始学习率分别取值0.01、0.1、0.2、0.3、0.4时，随机梯度下降的每一个batch取128，epoch取200，也就代表运算到最后学习率大约下降为原来的1/100，计算最后验证集上的准确度.得到隐藏层神经元个数为256。而后查找学习率和正则化强度，决定初始学习率为0.6，正则化强度为0
## 模型训练：
	本次实验的神经网络模型包含两层：输入层（784个神经元）到隐藏层（256个神经元），该层共754*256+256=193280个参数，激活函数为Relu，隐藏层（256个神经元）到输出层（10个神经元），共256*10+10=2570个参数，激活函数为Softmax，损失函数为交叉熵损失函数。随机梯度下降的batch为128，epoch为200次，初始学习率大小为0.6，学习率每隔10次epoch变为原来的0.8，正则化强度为0，即不加正则化强度。
