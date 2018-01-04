from sourcefunc import *


x, y = loadDataSet()
classifierNames = ['Decision Tree','Bagging', 'Random Forest', 'Multiboosting','Adaboost']


def testClassifier(n_estimators=30,max_depth=15):
    
    '''
        用来对比 DecisionTree, Bagging, Random Forest, Multiboost 四种分类器的性能
        用 n_estimators 和 max_depth 分别控制基学习器的数量以及决策树的最大层数，分别考察这两个因素
        对分类性能的影响，设置 max_depth 主要是为了防止过拟合。
    '''
    classifiers = [DecisionTreeClassifier(max_depth=max_depth),
               BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators),
               RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth),
               MultiBoostClassifier(n_estimators=n_estimators, max_depth=max_depth),
               AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth),n_estimators=n_estimators)]

    sampleSize = len(y)
    finalScores = []
    for i in range(3):
        
        # 在样本中随机选取80%作为训练基，剩下20%作为测试集，并且进行三轮测试
        # 三轮测试的结果取平均作为最终分类正确率
        # 这样随机选取训练测试集可以评估模型的泛化能力，取平均是为了让结果更可靠

        index = np.random.permutation(sampleSize)
        training = index[:int(0.8 * sampleSize)]
        test = index[int(0.8 * sampleSize):]

        tempScores = []
        for j in range(5):
            print('Round %d, running %s' % (i, classifierNames[j]))
            classifiers[j].fit(x[training], y[training])
            tempScores.append(classifiers[j].score(x[test], y[test]))
        finalScores.append(tempScores)
    return np.mean(np.array(finalScores),axis=0)



def testN():
    '''
        用来测试分类器和基分类器数量的关系
        取基分类器的数量分别为20，40，60，80，100
        将四种分类器的分类结果可视化
    '''

    N=[20,40,60,80,100]
    result=[]
    for n_estimators in N:
        print('n_estimators = %d' % n_estimators)
        result.append(testClassifier(n_estimators=n_estimators,max_depth=15))
    result=np.array(result)

    

    result = result.T

    '''
    # 将结果可视化，可选可不选
    colors=['red','blue','green','black','orange']
    for i in range(5):
        plt.plot(N,result[i],'--',c=colors[i])
        plt.scatter(N,result[i],c=colors[i],label=classifierNames[i])
    
    plt.xticks([20,40,60,80,100])
    plt.ylim([0.70,0.95])
    plt.yticks([0.70,0.75,0.80,0.85,0.90,0.95])
    plt.xlabel('numbers of base estimators')
    plt.ylabel('accuracy')

    plt.legend()  
    plt.show()
    '''

    return result
    

def testMax_depth():
    sampleSize=len(y)  
    finalScores=[]

    for max_depth in ([10,15,20,None]):
        dtree=DecisionTreeClassifier(max_depth=max_depth)
        mbc=MultiBoostClassifier(n_estimators=20,max_depth=max_depth)
        tempScores=[]
        print('max_depth = ',max_depth)
        for j in range(3):
            print ('round ',j)
            index = np.random.permutation(sampleSize)
            training = index[:int(0.8 * sampleSize)]
            test = index[int(0.8 * sampleSize):]

            dtree.fit(x[training],y[training])
            mbc.fit(x[training],y[training])
            tempScores.append([dtree.score(x[test],y[test]),mbc.score(x[test],y[test])])
        tempScores=np.array(tempScores)

        finalScores.append(np.mean(tempScores,axis=0))            
    return np.array(finalScores)

