from loadData import *
import os

# os.mkdir('./task_2')



def outputResult():

    freqTeam = findFreqTeam()

    freqTeam = list(freqTeam.items())[0:10]
    for item in freqTeam:

        teamMembers = item[0]
        teamData = dataset[teamMembers <= dataset.author]
        #teamData.to_excel(excel_writer='./task_2/' + ', '.join(teamMembers) + '.xlsx', encoding='utf-8', index=False)
        teamData.to_csv('./task_2/' + ', '.join(teamMembers) + '.csv', encoding='utf-8', index=False)
        
        teamThemes = findTeamThemes(teamData)[:10]

        fo = open('./task_2/' + ', '.join(teamMembers) +
                  '.txt', 'w', encoding='utf-8')
        for theme in teamThemes:
            fo.write(theme[0] + '\t\t' + str(theme[1])+'\n')
        fo.close()
