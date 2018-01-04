
from sourcefunc import *
import time

def RFvsBG():
    '''
        用来测试Random Forest和Bagging的运行时间以及正确率
    '''
    
    bgc = BaggingClassifier(n_estimators=100)
    rfc = RandomForestClassifier(n_estimators=100)
    X, y = loadDataSet()
    cv = ShuffleSplit(n_splits=3, test_size=0.1)

    print('Running Bagging method...')
    start = time.time()
    result_1 = cross_val_score(bgc, X=X, y=y, cv=cv)
    stop = time.time()
    deltaTime_1 = stop - start
    print('The average accuracy is: %f, the std is %f'% (result_1.mean(),result_1.std()))
    print('Runtime: %f' % deltaTime_1)

    print('Running RandomForest method...')
    start = time.time()
    result_2 = cross_val_score(rfc, X=X, y=y, cv=cv)
    stop = time.time()
    deltaTime_2 = stop - start
    print('The average accuracy is: %f, the std is %f'% (result_2.mean(),result_2.std()))
    print('Runtime: %f' % deltaTime_2)


