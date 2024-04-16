import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy
from operator import itemgetter
from sklearn.model_selection import train_test_split

class PLWMKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distance(self, X1, X2):
        return scipy.spatial.distance.euclidean(X1, X2)

    def Density(self, xi, data):
        aveR = self.distance(xi,data)**0.02
        return 1 / (aveR+1e-6)

    def predict(self, X_test):
        final_output = []
        myclass = list(set(self.y_train))
        for i in range(len(X_test)):
            eucDist = []
            for j in range(len(self.X_train)):
                dist = scipy.spatial.distance.euclidean(self.X_train[j], X_test[i])
                eucDist.append([dist, j, self.y_train[j]])
            eucDist.sort()

            nearest_labels = []
            for fi in range(8):
                nearest_labels.append(eucDist[fi][2])
            # 确定最近的K个样本中包含哪些类别
            nearest_classes = np.unique(nearest_labels)
            # 检查 nearest_classes 是否只包含一个元素
            if len(nearest_classes) == 1:
                # 如果只包含一个元素，将其添加到 final_output 列表中
                final_output.append(nearest_classes[0])
                continue

            minimum_dist_per_class = []
            for c in nearest_classes:
                minimum_class = []
                for di in range(len(eucDist)):
                    if (len(minimum_class) != self.k):
                        if (eucDist[di][2] == c):
                            minimum_class.append(eucDist[di])
                    else:
                        break
                minimum_dist_per_class.append(minimum_class)

            indexData = []
            for a in range(len(minimum_dist_per_class)):
                temp_index = []
                for j in range(len(minimum_dist_per_class[a])):
                    temp_index.append(minimum_dist_per_class[a][j][1])
                indexData.append(temp_index)

            centroid = []
            for a in range(len(indexData)):
                UtransposeData = self.X_train[indexData[a]]
                Densitydata = []
                for mi in range(len(indexData[a])):
                    Densitydata.append(self.Density(UtransposeData[mi], X_test[i]))
                # 将 self.X_train[indexData[a]] 转换为 numpy 数组
                X_train_array = np.array(self.X_train[indexData[a]])
                # 将 Densitydata 转换为 numpy 数组
                Densitydata_array = np.array(Densitydata)
                # 将每个向量的所有维度值都乘以对应的比例值
                result = X_train_array * Densitydata_array[:, np.newaxis]

                transposeData = result.T
                tempCentroid = []
                for j in range(len(transposeData)):
                    tempCentroid.append(np.mean(transposeData[j]))
                centroid.append(tempCentroid)
            centroid = np.array(centroid)

            eucDist_final = []
            for b in range(len(centroid)):
                dist = scipy.spatial.distance.euclidean(centroid[b], X_test[i])
                eucDist_final.append([dist, nearest_classes[b]])
            sorted_eucDist_final = sorted(eucDist_final, key=itemgetter(0))
            final_output.append(sorted_eucDist_final[0][1])
        return final_output

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        value = 0
        for i in range(len(y_test)):
            if (predictions[i] == y_test[i]):
                value += 1
        return value / len(y_test)

# 导入数据集
df = pd.read_csv('.csv')

# 最大最小值归一化
scaler = MinMaxScaler()

# 获取特征列和标签列
X = df.iloc[:, :-1].values  # 提取除最后一列外的所有列作为特征
y = df.iloc[:, -1].values   # 提取最后一列作为标签

# 最大最小值归一化
X = scaler.fit_transform(X)

# 划分训练集和测试集
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)
x_train = X[:-50]
y_train = y[:-50]
x_test = X[-50:]
y_test = y[-50:]

# 初始化PLWMKNN分类器
K = 11  # 选择K值
classifier = PLWMKNN(k=K)

# 训练模型
classifier.fit(x_train, y_train)

# 输出精度
accuracy = classifier.score(x_test, y_test)
print("Accuracy:", accuracy)


