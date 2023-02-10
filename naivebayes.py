import dataset

D1 = dataset.Dataset("mushroom-training.data")
D2 =dataset.Dataset("mushroom-testing.data")

totalMushrooms = len(D1.instances) #get total number of mushrooms
totalEdible = len(D1.selectSubset({"class":"e"})) #get total number of edible mushrooms
totalPoisonous = len(D1.selectSubset({"class":"p"})) #get total number of poisonous mushrooms
percentEdible = totalEdible/totalMushrooms
percentPoisonous = totalPoisonous/totalMushrooms

m = 0 #virutal Sampling varible for testing
# 
#  Get all the attributes and their values and store them inside nested dictionaries
# 
trainData = {}
keys = D1.attributes.keys() #get all the keys for the attributes
for key in keys:
    trainData[key] = {}
    for item in D1.getAttributeValues(key):
        trainData[key][item] = {
            "Edible" : len(D1.selectSubset({"class":"e", key:item})),#number of edible per item per key
            "Poisonous" : len(D1.selectSubset({"class":"p", key:item})),#number of poisonous per item per key
            "p" : 1/len(D1.getAttributeValues(key)) #p 1/number of types of attributes per key (Prior probability of particular value)
        }
#print(trainData['habitat']['l']['Edible']) #example of how to access data within dictionary

# 
#  Now we need the induction for the data we acquired in the section above.
# 
inductionData = {}

def getInductionTable(m): # iterate through the trainData dictionary on each nested dictionary to access all attributes and their values
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

getInductionTable(m) # call function to get induction data
## manually update these values to be correct...
inductionData["class"]["e"]["edible"] = percentEdible
inductionData["class"]["e"]["poisonous"] = percentPoisonous
inductionData["class"]["p"]["edible"] = percentEdible
inductionData["class"]["p"]["poisonous"] = percentPoisonous

#function to normalize data 
def normalize(pos, neg):
    return pos / (pos + neg)

def inference(data):
    lines = len(data.instances) #get length of testing data set
    accurateCase = 0
    for line in range(lines):
        edibleProduct =0        
        poisonousProduct=0
        for key in keys:    
            value = data.getInstanceValue(key, line) 
            if edibleProduct == 0:
                edibleProduct = inductionData[key][value]['edible']#if first time reading data set it equal
            else:
                edibleProduct *= inductionData[key][value]['edible']# multiply the all the probabilities of all the attributes together
            if poisonousProduct == 0:
                poisonousProduct = inductionData[key][value]['poisonous'] #if first time reading data set it equal
            else:
                poisonousProduct *= inductionData[key][value]['poisonous'] # multiply the all the probabilities of all the attributes together
            edibleProduct *= percentEdible # multiply by the probability of it being edible
            poisonousProduct *= percentPoisonous # multiply by the probability of it being poisonous
            norm = normalize(edibleProduct, poisonousProduct) #normalize for edible
            if norm > .50:
                result = "e"
            else:
                result = "p"
        if result == data.getInstanceValue("class", line):
                accurateCase += 1
    print("Number of Accuarte classifications: ", accurateCase)
    print("Number of total cases tested: ", lines)
    print("Accuracy level: ", accurateCase/lines)

print("Accuracy info for training data: ") 
inference(D1)
print("Accuracy info for testing data: ") 
inference(D2)