import numpy as np
import matplotlib.pyplot as plt

# 定义维度
dimensions = [2, 100, 1000, 10000]
num_points = 100
num_bins = 100

# 生成随机数据并计算距离
distances = {}
for dim in dimensions:
    data = np.random.randn(num_points, dim)
    dist_matrix = np.linalg.norm(data[:, np.newaxis] - data, axis=2)
    distances[dim] = dist_matrix[np.triu_indices(num_points, k=1)] 

# 距离可视化
plt.figure(figsize=(12, 8))

for i, dim in enumerate(dimensions):
    plt.subplot(2, 2, i+1)
    plt.hist(distances[dim], bins=num_bins, density=True, alpha=0.75)
    plt.title(f'Distance Histogram for Dimension {dim}')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.xlim(left=0)

plt.tight_layout()
plt.show()
