import dataset

D1 = dataset.Dataset("mushroom-training.data")
D2 = dataset.Dataset("mushroom-testing.data")

totalMushrooms = len(D1.instances) #get total number of mushrooms
totalEdible = len(D1.selectSubset({"class":"e"})) #get total number of edible mushrooms
totalPoisonous = len(D1.selectSubset({"class":"p"})) #get total number of poisonous mushrooms
percentEdible = totalEdible/totalMushrooms
percentPoisonous = totalPoisonous/totalMushrooms

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
# 
#  Now we need the induction for the data we acquired in the section above.
# 
inductionData = {} #initalize the induction data dictionary
# iterate through the trainData dictionary on each nested dictionary to access all attributes and their values
def getInductionTable(m): # pass the m (virtual sample)
    for key in trainData:
        inductionData[key] = {}  # initalize a nested dictonary for each key
        for item in trainData[key]:
            inductionData[key][item] = { # for each item inside the key dictionaries call the function to get the induction data for each
                "edible" : inductionTable(trainData[key][item]['Edible'],totalEdible, m, trainData[key][item]['p']), #sent data to function to compute the naive bayes fraction
                "poisonous" : inductionTable(trainData[key][item]['Poisonous'],totalPoisonous, m, trainData[key][item]['p'])
            }
    ## manually update these values to be correct...
    inductionData["class"]["e"]["edible"] = inductionTable(totalEdible, totalMushrooms, m, .5)
    inductionData["class"]["e"]["poisonous"] = inductionTable(totalPoisonous, totalMushrooms, m, .5)
    inductionData["class"]["p"]["edible"] = inductionTable(totalEdible, totalMushrooms, m, .5)
    inductionData["class"]["p"]["poisonous"] = inductionTable(totalPoisonous, totalMushrooms, m, .5)
    print("Classification Accuracy for training data: ") # call the function and print 
    inference(D1)
    print("Classification Accuracy for testing data: ") #call the function and print
    inference(D2)            

def inductionTable(numItem, numTotal, m, p): # plug data into formula to calculate their probabilty
    top = (numItem + (m * p))
    bottom = (numTotal + m)
    return top/bottom

def normalize(x, y): #function to normalize data 
    return x / (x + y)

def inference(data): #function that makes and inference on every row of the provided dataset, normalizes them, checks their classification and outputs accuracy
    lines = len(data.instances) # get number of rows/lines in the current dataset
    accurateCase = 0 #track number of accurate cases
    for line in range(lines): #loop through the "length" of the lines variable 
        edibleProduct =1 #initalize to 1 so it doesnt effect the multiplication       
        poisonousProduct=1
        for key in keys:    # loop through all the key attributes in the dataset (see line 16)
            value = data.getInstanceValue(key, line) #get the value you want to pull the probabilities from
            edibleProduct *= inductionData[key][value]['edible']# multiply the all the probabilities of all the attributes together
            poisonousProduct *= inductionData[key][value]['poisonous'] # multiply the all the probabilities of all the attributes together
        norm = normalize(edibleProduct, poisonousProduct) #normalize for edible
        if norm > 0.5: # if normalize returns a value greater than 0.50 than we assume it is edible. else, assume poisonous
            result = "e"
        else:
            result = "p"
        if result == data.getInstanceValue("class", line): # compare the result computed in the if-statement above to the 'class' value at the line we're looking at in the loop
                accurateCase += 1 #if they match increment the number of accurate cases by one
    print(str(accurateCase/lines*100)+"%") # calculate the percent of accurate cases and print that
#
# output
#
print("+++ M = 0 +++")
getInductionTable(0) # call function to get induction data
print("\n\n\n+++ M = 1 +++")
getInductionTable(1) # call function to get induction data
print("\n\n\n+++ M = 5 +++")
getInductionTable(5) # call function to get induction data
print("\n\n\n+++ M = 1000 +++")
getInductionTable(1000) # call function to get induction data