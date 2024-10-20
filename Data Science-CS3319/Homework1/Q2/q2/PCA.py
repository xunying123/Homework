import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载训练和测试数据
train_data_list=[]
train_label_list=[]
test_data_list=[]
test_label_list=[]

for i in range(1, 3):
    train_data = np.load(f'./Data Science/Homework1/Q2/dataset/{i}/train_data.npy')
    train_label = np.load(f'./Data Science/Homework1/Q2/dataset/{i}/train_label.npy')
    test_data = np.load(f'./Data Science/Homework1/Q2/dataset/{i}/test_data.npy')
    test_label = np.load(f'./Data Science/Homework1/Q2/dataset/{i}/test_label.npy')
    train_data_list.append(train_data)
    train_label_list.append(train_label)
    test_data_list.append(test_data)
    test_label_list.append(test_label)


train_data=np.vstack(train_data_list)
train_label=np.hstack(train_label_list)
test_data=np.vstack(test_data_list)
test_label=np.hstack(test_label_list)

pca = PCA(n_components=2)

train_data_2d = pca.fit_transform(train_data)


test_data_2d = pca.transform(test_data)

# 可视化数据
plt.figure(figsize=(8, 6))
plt.scatter(train_data_2d[:, 0], train_data_2d[:, 1], c=train_label, cmap='viridis', marker='o', edgecolor='k')
plt.title('PCA on Train Data')
plt.colorbar(label='Labels')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(test_data_2d[:, 0], test_data_2d[:, 1], c=test_label, cmap='viridis', marker='o', edgecolor='k')
plt.title('PCA on Test Data')
plt.colorbar(label='Labels')
plt.show()
