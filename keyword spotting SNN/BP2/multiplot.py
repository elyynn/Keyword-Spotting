import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

#读取csv中指定列的数据
data = pd.read_csv('E:\\gam001_test_acc.csv')
data2 = pd.read_csv('E:\\mel001_test_acc.csv')

#data_loss = data[['train_loss']] #class 'pandas.core.frame.DataFrame'
data_acc = data[['train Loss']]
data_acc2 = data2[['train Loss']]
x = np.arange(0,50,1)
y1 =np.array(data_acc)#将DataFrame类型转化为numpy数组
y2 = np.array(data_acc2)
#绘图
plt.plot(x,y1,label="Gammatone")
plt.plot(x,y2,label="Mel",fontsize=30)
#plt.title("loss & accuracy") 
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()   #显示标签
plt.show()
