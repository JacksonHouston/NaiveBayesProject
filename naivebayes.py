import dataset

D = dataset.Dataset("mushroom-training.data")
#print("Attributes in the data set are: ", D.attributes.keys())

#selectionCriteria = {"cap-shape":"b", "class":"p"}
#print("There are", len(D.instances), "instances in total")
#print("There are", len(D.selectSubset(selectionCriteria)), \  " poisonous examples with a bell-shaped cap")

totalEdible = {"class":"e"}
print("There are", len(D.selectSubset(totalEdible)), "edible mushrooms")
totalPoisonous = {"class":"p"}
print("There are", len(D.selectSubset(totalPoisonous)), "poisonous mushrooms")
selection = {"class":"e", "cap-shape":"b", "gill-size":"b"}
print("There are", len(D.selectSubset(selection)), "edible mushrooms with bell-shaped caps and narrow gills")