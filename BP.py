import numpy as np
import math
from matplotlib import pyplot as plt
#定义激活函数
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
#定义激活函数的导数
def d_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
#定义损失函数:平方差损失函数
def loss_function(predict,label):
    return 1/2*(label-predict)**2

class Forward_Backward:
    #定义网络结构，layers_dim表示每层神经元的个数
    def __init__(self,layers_dim,lr):
        self.parameter={}   #保存各层的权重和偏差
        self.grad={} #保存每一层权重和偏差的梯度
        self.lr=lr  #学习率
        for i in range(1,len(layers_dim)):
            self.parameter["w"+str(i)]=np.random.random([layers_dim[i],layers_dim[i-1]]) #初始化权重矩阵
            self.parameter["b"+str(i)]=np.zeros([layers_dim[i],1])
        self.L =len(self.parameter)//2
    #输出当前的权重和偏差
    def print_wb(self):
        pass
    def forward(self,x):
        a=[]    #保存每一层经过激活函数的输出值
        z=[]    #保存每一层不经过激活函数的输出值
        cache={}    #保存每一层激活和未激活的输出值
        a.append(x) #为了保证后边计算的方便性，将输入层和输出层都存储在a,z中
        z.append(x)
        #第一层为输入层，而最后一层是输出层
        for i in range(1,self.L):
            #_z=self.parameter["w"+str(i)].dot(x)+self.parameter['b'+str(i)]     #TODO:这里有错误吧，上一层的输出是当前层的输入，所以x应该替换为a[i-1],不然这种只适合三层的网络
            _z = self.parameter["w" + str(i)].dot(a[i-1]) + self.parameter['b' + str(i)]
            _a=sigmoid(_z)  #sigmo激活
            z.append(_z)
            a.append(_a)
        #输出层不需要经过激活函数
        _z=self.parameter["w"+str(self.L)].dot(a[self.L-1])+self.parameter['b'+str(self.L)]
        #为了保持维度一致
        _a=_z   #将输出层的值也添加到a,z中，保持a,z含义一致：每一层激活后和未激活后输出值
        z.append(_z)
        a.append(_a)
        cache["z"]=z
        cache["a"]=a
        return cache,a[self.L]
    # 定义反向传播求梯度
    # TODO:这里后续加上关于偏差b的梯度更新
    def backward(self, cache,predict,label):
        #先求输出层到上一层的倒数
        m=label.shape[1]
        self.grad["dz"+str(self.L)]=predict-label
        #最后一层没有激活函数
        self.grad["dw"+str(self.L)]=self.grad["dz"+str(self.L)].dot(cache['a'][self.L-1].T)/m
        self.grad["db" + str(self.L)] = np.sum(self.grad["dz"+str(self.L)],axis=1,keepdims=True)/m
        for i in reversed(range(1,self.L)):
            self.grad["dz"+str(i)]=self.parameter["w"+str(i+1)].T.dot(self.grad["dz"+str(i+1)])*d_sigmoid(cache['z'][i])
            self.grad["dw"+str(i)]=self.grad["dz"+str(i)].dot(cache['a'][i-1].T)/m
            self.grad["db" + str(i)] = np.sum(self.grad["dz" + str(i)], axis=1, keepdims=True) / m
    # 权重更新
    def undate_para(self):
        for i in range(1,self.L+1):
            self.parameter["w"+str(i)]-=self.lr*self.grad["dw"+str(i)]
            self.parameter["b" + str(i)] -= self.lr * self.grad["db" + str(i)]
    def compute_loss(self,predict,label):
        #return np.mean(np.square(predict-label))
        return np.mean(loss_function(predict,label))
#加载数据集
def dataloader():
    x=np.arange(0.0,1.0,0.01)
    y=20*np.sin(2*np.pi*x)
    return x,y
x,y=dataloader()
x=x.reshape(1,100)
y=y.reshape(1,100)

loss_list=[]
epoch=20000
#学习率设置为0.1
net=Forward_Backward((1,25,1),0.03)
#训练网络参数
for i in range(epoch):
    cache,output=net.forward(x)
    net.backward(cache,output,y)
    net.undate_para()
    loss=net.compute_loss(output,y)
    loss_list.append(loss)
    print("[ %i ] the loss is %f"%(i,loss))
#画图
plt.title("the loss of the train")
plt.plot(range(epoch),loss_list)
plt.show()
plt.title("y=sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x,output,label='predict')
plt.scatter(x,y,label='groundtrush')
plt.legend()
plt.show()
plt.title("y=sin(x)")
plt.xlabel("x")
plt.ylabel("y")
x=x.flatten()
y=y.flatten()
output=output.flatten()
plt.plot(x,output,label='predict')
plt.plot(x,y,label='groundtrush')
plt.legend()
plt.show()


