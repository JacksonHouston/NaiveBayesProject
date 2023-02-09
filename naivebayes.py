import dataset

D = dataset.Dataset("mushroom-training.data")

totalMushrooms = len(D.instances) #get total number of mushrooms
totalEdible = len(D.selectSubset({"class":"e"})) #get total number of edible mushrooms
totalPoisonous = len(D.selectSubset({"class":"p"})) #get total number of poisonous mushrooms
percentEdible = totalEdible/totalMushrooms
percentPoisonous = totalPoisonous/totalMushrooms

m = 0 #virutal Sampling varible for testing
# 
#  Get all the attributes and their values and store them inside nested dictionaries
# 
trainData = {}
keys = D.attributes.keys() #get all the keys for the attributes
for key in keys:
    trainData[key] = {}
    for item in D.getAttributeValues(key):
        trainData[key][item] = {
            "Edible" : len(D.selectSubset({"class":"e", key:item})),#number of edible per item per key
            "Poisonous" : len(D.selectSubset({"class":"p", key:item})),#number of poisonous per item per key
            "p" : 1/len(D.getAttributeValues(key)) #p 1/number of types of attributes per key (Prior probability of particular value)
        }
#print(trainData['habitat']['l']['Edible']) #example of how to access data within dictionary

# 
#  Now we need the induction for the data we acquired in the section above.
# 
inductionData = {}

def getInductionTable(): # iterate through the trainData dictionary on each nested dictionary to access all attributes and their values
    for key in trainData:
        inductionData[key] = {}
        for item in trainData[key]:
            inductionData[key][item] = {
                "edible" : inductionTable(trainData[key][item]['Edible'],totalEdible, m, trainData[key][item]['p']), #sent data to function to compute the naive bayes fraction
                "poisonous" : inductionTable(trainData[key][item]['Poisonous'],totalPoisonous, m, trainData[key][item]['p'])
            }
            
def inductionTable(numItem, numTotal, m, p): # plug data into formula to calculate their probabilty
    top = numItem + (m * p)
    bottom = numTotal + m
    return top/bottom

getInductionTable() # call function to get induction data

def normalize(pos, neg):
    return pos / (pos + neg)

D2 =dataset.Dataset("mushroom-testing.data")

lines = len(D2.instances)
for line in range(lines):
    edibleProduct =1
    poisonousProduct=1
    for key in keys:
        value = D2.getInstanceValue(key, line)
        edibleProduct *= inductionData[key][value]['edible']
        poisonousProduct *= inductionData[key][value]['poisonous']
    edibleProduct *= percentEdible
    poisonousProduct *= percentPoisonous
    norm = normalize(edibleProduct, poisonousProduct)
    print("line no: ", line , "normalized: ", norm)    