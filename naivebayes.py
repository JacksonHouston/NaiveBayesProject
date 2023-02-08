import dataset

D = dataset.Dataset("mushroom-training.data")

totalMushrooms = len(D.instances) #get total number of mushrooms
totalEdible = len(D.selectSubset({"class":"e"})) #get total number of edible mushrooms
totalPoisonous = len(D.selectSubset({"class":"p"})) #get total number of poisonous mushrooms

trainData = {} #initalize the dictionary for the training data
m = 0 #virutal Sampling varible for testing
keys = D.attributes.keys() #get all the keys for the attributes
for key in keys:
    for item in D.getAttributeValues(key):
        trainData.update({key: #nested dictionary to store the number of edible and poisonous for each item of each key
            {item: 
                {
                    "Edible":len(D.selectSubset({"class":"e", key:item})), #number of edible per item per key
                    "Poisonous":len(D.selectSubset({"class":"p", key:item})), #number of poisonous per item per key
                    "p": 1/len(D.getAttributeValues(key)) #p 1/number of types of attributes per key (Prior probability of particular value)
                }
            }
        })

induction = {}


print(trainData)