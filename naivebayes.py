import dataset
class Data:
    def __init__(self):
        self.self = self.Attribute()
    class Attribute:
        def __init__(self, name, value):
            self.name = ""
            self.value = self.Values()

        def show(self):
            print(self.name)
        
        class Values:
            def __init__(self, edible, poisonous, p):
                self.edible = 0
                self.poisonous = 0
                self.p = 0
            
            def display(self):
                print(self.edible)
                print(self.poisonous)
                print(self.p)


D = dataset.Dataset("mushroom-training.data")

totalMushrooms = len(D.instances) #get total number of mushrooms
totalEdible = len(D.selectSubset({"class":"e"})) #get total number of edible mushrooms
totalPoisonous = len(D.selectSubset({"class":"p"})) #get total number of poisonous mushrooms

trainData = {}
attributes = {}
items = {}
m = 0 #virutal Sampling varible for testing
keys = D.attributes.keys() #get all the keys for the attributes
for key in keys:
    for item in D.getAttributeValues(key):
        items.update({
            "Edible" : len(D.selectSubset({"class":"e", key:item})),
            "Poisionous" : len(D.selectSubset({"class":"p", key:item})),
            "p" : 1/len(D.getAttributeValues(key))
            
        })
        #print(items)
    attributes.update({item : items})
    trainData.update({key: attributes})
        # print(key, item)
        # print("Edible", len(D.selectSubset({"class":"e", key:item}))) #number of edible per item per key
        # print("Poisonous",len(D.selectSubset({"class":"p", key:item}))) #number of poisonous per item per key
        # print("p", 1/len(D.getAttributeValues(key))) #p 1/number of types of attributes per key (Prior probability of particular value)
print(trainData['habitat']['l'])

for i in attributeList:
    print(i)

def getFract(numItem, numTotal, m, p):
    top = numItem + (m * p)
    bottom = numTotal + m
    return top/bottom

#print(trainData['population'].keys())
#print(trainData)