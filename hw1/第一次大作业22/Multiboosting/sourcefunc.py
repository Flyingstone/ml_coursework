import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score



def loadDataSet():
    '''
        用于载入数据，由于决策树的输入为数值型，这里将数据集中的字母转化为数字，数字的选取
        并不会影响分类结果。
    '''
    dataset = []
    fr = open('krkopt.data')
    for line in fr:
        line = line.strip().split(',')

        for k in range(3):
            i = k * 2
            if line[i] == 'a':
                line[i] = 1
            if line[i] == 'b':
                line[i] = 2
            if line[i] == 'c':
                line[i] = 3
            if line[i] == 'd':
                line[i] = 4
            if line[i] == 'e':
                line[i] = 5
            if line[i] == 'f':
                line[i] = 6
            if line[i] == 'g':
                line[i] = 7
            if line[i] == 'h':
                line[i] = 8

        dataset.append(line)
    x = np.array(dataset)[:, :6]
    y = np.array(dataset)[:, -1]
    return x, y


class MultiBoostClassifier():

    def __init__(self, n_estimators=30, max_depth=15):
        self.n_estimators = n_estimators # 设置基学习器个数，基学习器默认为决策树
        self.max_depth = max_depth # 设置决策树最大层数，防止过拟合，若为None则没有最大层数限制
        self.__CLFlist = []
        self.__beta=[]

    def __vecI(self, n, i):
        # 由于multiboost算法中 I 数组为无限维，这里用一个函数来代替
        if i < n:
            return int(i * self.n_estimators / n) + 1
        else:
            return self.n_estimators

    def fit(self, X, y):
        self.__CLFlist = []
        self.__beta = []

        n = np.sqrt(self.n_estimators)
        k = 1
        sampleSize = len(y)
        sampleWeights = np.ones(sampleSize)

        for t in range(self.n_estimators):

            if self.__vecI(n, k) == t + 1:
                sampleWeights = np.random.exponential(size=sampleSize)
                sampleWeights = sampleWeights * (n / sampleWeights.sum())
                k = k + 1

            while(True):
                dTree = DecisionTreeClassifier(max_depth=self.max_depth)
                dTree.fit(X=X, y=y, sample_weight=sampleWeights)

                prediction = dTree.predict(X)
                error_t = 0.0

                for i in range(sampleSize):
                    if prediction[i] != y[i]:
                        error_t += sampleWeights[i]
                error_t = error_t / sampleSize

                if error_t > 0.5:
                    sampleWeights = np.random.exponential(size=sampleSize)
                    k = k + 1
                    continue
                elif error_t == 0:
                    beta_t = 1e-10
                    sampleWeights = np.random.exponential(size=sampleSize)
                    sampleWeights = sampleWeights * (n / sampleWeights.sum())
                    k = k + 1
                    self.__beta.append(beta_t)
                    self.__CLFlist.append(dTree)
                    break
                else:
                    beta_t = error_t / (1 - error_t)
                    prediction = dTree.predict(X)
                    for i in range(sampleSize):
                        if prediction[i] != y[i]:
                            sampleWeights[i] = sampleWeights[i] / (2 * error_t)
                        else:
                            sampleWeights[i] = sampleWeights[i] / \
                                (2 - 2 * error_t)
                        if sampleWeights[i] < 1e-8:
                            sampleWeights[i] = 1e-8

                    self.__beta.append(beta_t)
                    self.__CLFlist.append(dTree)
                    break

    def score(self, X, y):
        '''
            用于测试分类准确率
            这个函数速度比较慢，有待优化
        '''

        result = []
        sampleSize = len(X)
        round = 0

        for item in X:
            '''
            用于计数，测试算法时比较直观
            if round%10==0:
                print('round = %d' % round)
            round += 1
            '''

            resDic=dict()    
            for t in range(self.n_estimators):
                prediction = (self.__CLFlist[t].predict(
                    item.reshape((1, -1))))[0]
                resDic[prediction]=resDic.get(prediction,0)+np.log10(1 / self.__beta[t])
            res = sorted(resDic.items(), key=lambda x: x[1], reverse=True)
            result.append(res[0][0])  


        score = np.sum(np.array(result) == y) / len(y)
        return score



