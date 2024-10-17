import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

train_data = np.load('./Data Science/Homework1/Q2/dataset/3/train_data.npy')
train_label = np.load('./Data Science/Homework1/Q2/dataset/3/train_label.npy')
test_data = np.load('./Data Science/Homework1/Q2/dataset/3/test_data.npy')
test_label = np.load('./Data Science/Homework1/Q2/dataset/3/test_label.npy')

# 假设已经有合并后的数据 combined_data 和 combined_label
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2, random_state=42)

# KNN 模型
knn_model = KNeighborsClassifier(n_neighbors=7)  # k=5
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"KNN 模型的准确率: {knn_accuracy}")
print("KNN 模型的分类报告:")
print(classification_report(y_test, y_pred_knn))
