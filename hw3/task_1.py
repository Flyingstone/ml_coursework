from loadData import *
import os

def outputResult():
    
    #os.mkdir('./task_1')

    res = findSuppoters()
    fo = open('./task_1/freqAuthors.txt', 'w', encoding='utf-8')
    for conference in conferences:
        fo.write('#'*10 + '\n')
        fo.write(conference + '\n')
        for name in res[conference].keys():
            fo.write(name + '\t\t' + str(res[conference][name])+'\n')
    fo.close()


    res = findSupportersChange()
    fo = open('./task_1/authorsChange.txt', 'w', encoding='utf-8')
    for conference in conferences:
        fo.write('#' * 10 + '\n')
        fo.write(conference + '\n')
        for name in res[conference].keys():
            fo.write(name + '\t\t' +
                     '\t'.join(map(str, res[conference][name])) + '\n')
    fo.close()


    res = findFreqTeam()
    team_3 = [item for item in res.keys() if len(item) == 3]
    team_4 = [item for item in res.keys() if len(item) == 4]
    team_5 = [item for item in res.keys() if len(item) == 5]
    fo = open('./task_1/freqTeam.txt', 'w', encoding='utf-8')

    fo.write('#' * 10 + '\n')
    fo.write('FREQUENT TEAMS OF 3 MEMBERS\n')
    for name in team_3:
        fo.write(', '.join(name) + '\t\t' + str(res[name]) + '\n')
    
    fo.write('#' * 10 + '\n')
    fo.write('FREQUENT TEAMS OF 4 MEMBERS\n')
    for name in team_4:
        fo.write(', '.join(name) + '\t\t' + str(res[name]) + '\n')

    fo.write('#' * 10 + '\n')
    fo.write('FREQUENT TEAMS OF 5 MEMBERS\n')
    for name in team_5:
        fo.write(', '.join(name) + '\t\t' + str(res[name]) + '\n')
     
    fo.close()
