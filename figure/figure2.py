from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib as plt
from mlxtend.plotting import plot_decision_regions
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 划分测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 对特征值进行标准化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.svm import SVC

#%matplotlib inline  #用在jupyter
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

# 将标准化后的训练数据和测试数据重新整合到一起
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt.pyplot.figure(figsize=(12, 7))
plot_decision_regions(X_combined_std, y_combined, classifier=svm,test_idx=range(105, 150))
plt.title('支持向量机模型的决策区域', fontsize=19, color='w')
plt.xlabel('标准化处理后的花瓣长度', fontsize=15)
plt.ylabel('标准化处理后的花瓣宽度', fontsize=15)
plt.legend(loc=2, fontsize=15, scatterpoints=2)

print()
