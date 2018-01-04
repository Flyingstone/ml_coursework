import numpy as np
from pandas import Series, DataFrame
from nltk.corpus import stopwords
from fpGrowth import *

conferences = ['IJCAI', 'AAAI', 'CVPR', 'NIPS', 'SIGIR', 'KDD', 'COLT', 'KR']

years = [['2007', '2008', '2009'], ['2010', '2011'], [
    '2012', '2013'], ['2014', '2015'], ['2016', '2017']]

myStopWords = set(stopwords.words('english')) | {
    'via', 'using', 'based', 'data', 'towards', 'method', 'model', 'models', 'learning', 'approach', 'modeling', 'efficient', 'efficiency', 'problem', 'problems'}


def loadData():

    # 用于读取数据，将文章的作者，标题，会议以及年份信息储存到pandas.DataFrame格式的数据结构中，方便后续的查阅和读取
    # 在这里我们只读取了'IJCAI', 'AAAI', 'CVPR', 'NIPS', 'SIGIR', 'KDD', 'COLT', 'KR'这八个会议及其子会议的文章信息
    # 不属于这八个会议或者没有标明作者信息的文章被剔除掉

    fr = open('./FilteredDBLP.txt', encoding='utf-8')
    itemLists = []
    while(True):
        item = fr.readline()
        if item == '':
            break
        if '#' in item:
            a = {'author': ''}
            for item in fr:
                item = item.strip().split('\t')
                key = item[0]
                value = item[1]
                if key == 'author':
                    a['author'] += (value + ',')
                if key == 'title':
                    a['title'] = value.lower()
                if key == 'year':
                    a['year'] = value
                if key == 'Conference':
                    a['conference'] = value
                    break
        a['author'] = set(a['author'][:-1].split(','))

        if a['author'] == {''}:
            a['author'] = np.NaN
        itemLists.append(a)
    result = DataFrame(itemLists)

    def f(x):

        # 用于将数据中各项会议的子刊记录到其主会议中，配合pandas.apply()函数使用
        # 比如FCA4AI@IJCAI记录为IJCAI，MPREF@AAAI记录为AAAI，其他会议同理

        for item in conferences:
            if item in x:
                return item
        return x

    result.conference = result.conference.apply(f)
    result = result[result.conference.isin(
        ['IJCAI', 'AAAI', 'CVPR', 'NIPS', 'SIGIR', 'KDD', 'COLT', 'KR'])]
    result = result.dropna()
    result = result.sort_values(['year', 'conference'])
    result.index = np.arange(result.index.size)
    #result.to_csv('dataset.csv',index=False, header=True, encoding='utf-8')
    return result


dataset = loadData()


def authorList():
    authorlist = []
    for item in dataset.author:
        # authorlist.append(set(item.strip().split(',')))
        authorlist.append(item)
    return authorlist


def themeList(dataset):
    '''
    themeList = []
    sortedTheme = dict()

    for conference in conferences:
        conThemeList = []
        conferenceTheme = dict()
        for item in dataset[dataset.conference == conference].title:
            original = set(item[:-1].strip().split())
            conThemeList.append(original - myStopWords)
            themeList.append(original - myStopWords)
        for item in conThemeList:
            for theme in item:
                conferenceTheme[theme] = conferenceTheme.get(theme, 0) + 1
        conferenceTheme = sorted(
            conferenceTheme.items(), key=lambda d: d[1], reverse=True)
        sortedTheme[conference] = conferenceTheme
    return themeList, sortedTheme
    '''

    themeList = []
    themeDic = dict()
    for item in dataset.title:
        original = set(item[:-1].strip().split())
        themeList.append(original - myStopWords)
    for item in themeList:
        for theme in item:
            themeDic[theme] = themeDic.get(theme, 0) + 1
    sortedTheme = sorted(themeDic.items(), key=lambda d: d[1], reverse=True)
    return sortedTheme

def findTeamThemes(dataset):
    return themeList(dataset)

def findSuppoters():

    # 这个函数用来寻找会议的主要支持者，每个会议按照发文章数量取前十五名

    result = dict()
    for conference in conferences:
        authorFreqDic = dict()
        authors = dataset.author[dataset.conference == conference].values
        for item in authors:
            # item = item.strip().split(',')
            for name in item:
                authorFreqDic[name] = authorFreqDic.get(name, 0) + 1
        result[conference] = dict(
            sorted(authorFreqDic.items(), key=lambda d: d[1], reverse=True)[:20])
    return result


def findSuppotersbyYear():

    # 这个函数用来寻找每个会议在每个时间段内的主要支持者，用于后续比较支持者随时间变化的情况时使用

    result = dict()
    for conference in conferences:

        result[conference] = dict()
        for year in years:
            authorFreqDic = dict()
            authors = dataset.author[dataset.conference ==
                                     conference][dataset.year.isin(year)].values
            for item in authors:
                for name in item:
                    authorFreqDic[name] = authorFreqDic.get(name, 0) + 1
            result[conference][','.join(year)] = dict(
                sorted(authorFreqDic.items(), key=lambda d: d[1], reverse=True))
    return result


def findSupportersChange():

    # 这个函数用来发现各项会议在不同时间段支持者的变化情况，输出结果为嵌套字典，记录各个会议每位主要支持者在每个时间段发文章的数目

    result = dict()
    freqAuthors = findSuppoters()
    freqAuthorsbyYear = findSuppotersbyYear()
    for conference in conferences:
        result[conference] = dict()
        for author in freqAuthors[conference]:
            result[conference][author] = []
            for year in years:
                year = ','.join(year)
                result[conference][author].append(
                    freqAuthorsbyYear[conference][year].get(author, 0))
    return result


def findFreqTeam():
    # 这个函数用于发现经常合作且人数大于三的团队
    minSup = 5
    initAuthor = createInitSet(dataset.author)
    freqAuthorTree, AuthorHeader = createTree(initAuthor, minSup)  # 构造FPtree
    freqAuthorlist = []
    mineTree(freqAuthorTree, AuthorHeader, minSup,
             set([]), freqAuthorlist)  # 构造条件模式树挖掘频繁项
    freqAuthorlist = [item for item in freqAuthorlist if len(item) >= 3]
    team_3 = dict()
    compareSet = [key for key in initAuthor.keys() if len(key) >= 3]
    for item in freqAuthorlist:
        if len(item) < 3:
            continue
        else:
            for key in compareSet:
                if item <= key:
                    item = frozenset(item)
                    team_3[item] = team_3.get(item, 0) + initAuthor[key]
    return dict(sorted(team_3.items(), key=lambda d: d[1], reverse=True))






