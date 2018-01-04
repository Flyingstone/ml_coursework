# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:58:20 2017

@author: LPZ

"""

import fpGrowth

class readData:
    """读取文件
    Parameters
    ----------
    filename：文件名
    confname：会议名
    keyname：读取内容（如：'author','year'等）

    
    """
    def __init__(self,filename,confname,keyname):
        self.filename = filename
        self.confname = confname
        self.keyname = keyname
        
    def setKey(self,keyname):
        self.keyname = keyname
        
    def setConf(self,confname):
        self.confname = confname
    
    def openFile(self):
        #gbk读取出错，原因未知，改为utf-8正常读取
        fo = open(self.filename,'r',encoding= 'utf-8')
        self.fo = fo
        
    def closeFile(self):
        self.fo.close()
        
    def readFile(self):
        valueList = []
        temp = []
        #按行读取文件
        for line in self.fo.readlines():
            #对每行进行处理去除行尾'\n'，以'\t'为分隔符分开
            line = line[:-1]
            line = line.split('\t')
            #有两个以上数据进行判断
            if(len(line)>1):
                key = line[0]
                value = line[1]
#                print(key,value)
                #判断为需要数据则存入temp
                if(key==self.keyname):
                    temp.append(value)
#                    valueList.append(value)
                if(key=='Conference'):
                    #判断是否为要求会议
                    if(value==self.confname):
                        valueList.append(temp)
                    temp = []
        return valueList

#根据年份判断研究者是否活跃
def activeAuthor(authorList,yearList,lastYear):
    activeAuthorList = []
    activeTeamList = []
    #寻找活跃的研究者组
    for author,year in zip(authorList,yearList):
        if(year[0] >= lastYear):
            activeTeamList.append(author)
    #把活跃研究者取出来放入一个List中
    for authorTeam in activeTeamList:
        for author in authorTeam:
            activeAuthorList.append(author)
    #去重
    activeAuthorList = frozenset(activeAuthorList)
    return activeAuthorList

#对题目进行预处理，得到主题List
def titleInit(titleList,removeList=[]):
    subjectList = []
    #对题目进行处理，提取出主题词
    for title in titleList:
        #对题目进行处理，去掉最后的'.'，字符转为小写
        subject = title[0][:-1].lower()
        #把题目中词以空格分开
        subject = subject.split(' ')
        #排除removeList中的词
        subject = [x for x in subject if x not in removeList]
        #将主题词加入subjectList中
        subjectList.append(subject)
    return subjectList

#提取每个团队的主题词，使用fpGrowth进行频繁模式查询最常用主题
def subjectFreq(teamList,authorList,subjectList,minFreq):
    subjectFreqList = []
    for team in teamList:
        tempData = []
        #添加minFreq个'a'，保证FPTree不为空
        for i in range(minFreq):
            tempData.append(['a'])
        for author,subject in zip(authorList,subjectList):
            if(not [False for a in team if a not in author]):
                tempData.append(subject)
        tempSet = fpGrowth.createInitSet(tempData)
        #fpGrowth树建立
        subjectFPtree,subjectHeaderTab=fpGrowth.createTree(tempSet,minFreq)
        #挖掘主题
        subjectFreqItems = []
        fpGrowth.mineTree(subjectFPtree,subjectHeaderTab,minFreq,set([]),subjectFreqItems)
        #每个团队主题词添加到subjectFreqList
        subjectFreqList.append(subjectFreqItems)
    return subjectFreqList

#文件名
fileName = "test.txt"
#读取author数据
confName = "SIGIR"
authorRd = readData(fileName,confName,"author")
authorRd.openFile()
authorData = authorRd.readFile()

#读取年份数据
yearRd = readData(fileName,confName,"year")
yearRd.openFile()
yearData = yearRd.readFile()

#读取主题数据
titleRd = readData(fileName,confName,"title")
titleRd.openFile()
titleData = titleRd.readFile()

#查询活跃研究者
lastYear = "2015"
activeAuthorList = activeAuthor(authorData,yearData,lastYear)

#频繁模式查询团队，利用fpGrowth方法进行
#数据预处理
minMem = 3
authorSet = fpGrowth.createInitSet(authorData)
#fpGrowth树建立
authorFPtree,authoryHeaderTab=fpGrowth.createTree(authorSet,minMem)
#挖掘团队
teamList = []
fpGrowth.mineTree(authorFPtree,authoryHeaderTab,minMem,set([]),teamList)
teamList = fpGrowth.teamSelect(minMem,teamList)


#主题List查询
removeList = ['a','an','the','for','of','with','and','in','to']
subjectList = titleInit(titleData,removeList)

#团队主题
minFreq = 2
subjectFreqList = []
subjectFreqList = subjectFreq(teamList,authorData,subjectList,minFreq)


#关闭文件
yearRd.closeFile()
authorRd.closeFile()
