import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

train_data = np.load('./Data Science/Homework1/Q2/dataset/3/train_data.npy')
train_label = np.load('./Data Science/Homework1/Q2/dataset/3/train_label.npy')
test_data = np.load('./Data Science/Homework1/Q2/dataset/3/test_data.npy')
test_label = np.load('./Data Science/Homework1/Q2/dataset/3/test_label.npy')

# 假设已经有合并后的数据 combined_data 和 combined_label
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2, random_state=42)

# SVM 模型
svm_model = SVC(kernel='linear')  # 使用线性核，也可以尝试 'rbf'
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM 模型的准确率: {svm_accuracy}")
print("SVM 模型的分类报告:")
print(classification_report(y_test, y_pred_svm))