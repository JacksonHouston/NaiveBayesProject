import dataset

D = dataset.Dataset("mushroom-training.data")

totalMushrooms = len(D.instances)
totalEdible = len(D.selectSubset({"class":"e"}))
totalPoisonous = len(D.selectSubset({"class":"p"}))
#print((totalEdible/totalMushrooms)*100, "'%' are edilbe")
habitat = ["d","g","m","l","p","u","w"]
classes = ["p","e"]
totalHabEd=0
totalHadPo=0
for h in habitat:
    for c in classes:
        selectionCriteria = {"habitat":h, "class":c}
        print("There are", len(D.selectSubset(selectionCriteria)), c ," examples in habitat" , h)
        if c == "e":
            totalHabEd += len(D.selectSubset(selectionCriteria))
        else:
            totalHadPo += len(D.selectSubset(selectionCriteria))