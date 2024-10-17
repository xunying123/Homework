import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载训练和测试数据
train_data = np.load('./Data Science/Homework1/Q2/dataset/1/train_data.npy')
train_label = np.load('./Data Science/Homework1/Q2/dataset/1/train_label.npy')
test_data = np.load('./Data Science/Homework1/Q2/dataset/1/test_data.npy')
test_label = np.load('./Data Science/Homework1/Q2/dataset/1/test_label.npy')

tsne = TSNE(n_components=2, perplexity=50, learning_rate=300, max_iter=2000)

train_data_2d = tsne.fit_transform(train_data)
test_data_2d = tsne.fit_transform(test_data)

# 可视化降维后的数据
plt.figure(figsize=(8, 6))
plt.scatter(train_data_2d[:, 0], train_data_2d[:, 1], c=train_label, cmap='viridis', marker='o', edgecolor='k')
plt.title('TSNE on Train Data')
plt.colorbar(label='Labels')
plt.show()

# 可视化降维后的数据
plt.figure(figsize=(8, 6))
plt.scatter(test_data_2d[:, 0], test_data_2d[:, 1], c=test_label, cmap='viridis', marker='o', edgecolor='k')
plt.title('TSNE on test Data')
plt.colorbar(label='Labels')
plt.show()