import pandas as pd
import os
import matplotlib.pyplot as plt

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH = ROOT_PATH + "\\..\\datasets\\"
TRAIN_FILE = "CarRanking_train.csv"
TEST_FILE = "CarRanking_test.csv"

df   = pd.read_csv(PATH + TRAIN_FILE)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df)

def getSummary(df, attribute):
    averagesSeries = df.groupby([attribute])['Rank'].mean()
    attributeDict = {}
    min = averagesSeries.min()
    max = averagesSeries.max()
    attributeDict['attribute'] = attribute
    attributeDict['min']       = min
    attributeDict['max']       = max
    attributeDict['range']     = max - min

    averagesDf  = averagesSeries.to_frame()
    levels      = list(averagesDf.index)

    levelPartWorths = []
    for i in range(0, len(levels)):
        averagePartWorth = averagesSeries[i]
        levelName = levels[i]
        levelPartWorths.append({levelName:averagePartWorth})
    attributeDict['partWorths'] = levelPartWorths
    return attributeDict

def getImportances(attributeSummaries):
    ranges = []
    for i in range(0, len(attributeSummaries)):
        ranges.append(attributeSummaries[i]['range'])
    rangeSum = sum(ranges)

    for i in range(0, len(attributeSummaries)):
        importance = attributeSummaries[i]['range']/rangeSum
        attributeSummaries[i]['importance'] = importance
    return attributeSummaries
attributeNames = ['Safety','Fuel','Accessories']
attributeSummaries = []
for i in range(0, len(attributeNames)):
    attributeInfo = getSummary(df, attributeNames[i])
    attributeSummaries.append(attributeInfo)

attributeSummaries = getImportances(attributeSummaries)
print(attributeSummaries)

def plotImportances(attributeSummaries):
    X = []
    y = []
    for i in range(0, len(attributeSummaries)):
        X.append(attributeSummaries[i]['attribute'])
        y.append(attributeSummaries[i]['importance'])

    plt.bar(X, y)
    plt.title("Importances")
    plt.xticks(rotation=75)
    plt.show()

def plotLevels(attributeSummaries):
    X = []
    y = []
    for i in range(0, len(attributeSummaries)):
        attribute = attributeSummaries[i]['attribute']
        partWorths = attributeSummaries[i]['partWorths']
        for j in range(0, len(partWorths)):
            obj = partWorths[j]
            key = list(obj.keys())[0]
            val = list(obj.values())[0]
            label = attribute + "_" + key
            X.append(label)
            y.append(val)

    plt.bar(X, y)
    plt.title("Part-worths")
    plt.xticks(rotation=75)
    plt.show()

plotImportances(attributeSummaries)
plotLevels(attributeSummaries)

def getUtility(attributeSummaries, attribute, level):
    for attributeSummary in attributeSummaries:
        if(attributeSummary['attribute']==attribute):
            partWorths = attributeSummary['partWorths']
            for partWorth in partWorths:
                key = list(partWorth.keys())[0]
                if(key == level):
                    importance = attributeSummary['importance']
                    val = list(partWorth.values())[0]
                    return importance*val

dfTest    = pd.read_csv(PATH + TEST_FILE)
utilities = []
for i in range(0, len(dfTest)):
    utilitySum = 0
    for j in range(0, len(attributeNames)):
        attribute = attributeNames[j]
        level     = dfTest.iloc[i][attribute]
        utility   = getUtility(attributeSummaries, attribute, level)
        utilitySum += utility
    utilities.append(utilitySum)
dfTest['Utility'] = utilities
print(dfTest)
