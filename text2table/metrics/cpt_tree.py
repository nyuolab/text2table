import json
import pandas as pd

class TreeNode:
    def __init__(self, number, name):
        self.number = number
        self.name = name
        self.children = []
    
    def addChild(self, child):
        self.children.append(child)

    def serialize(self):
        s = {}
        for child in self.children:
            s[child.name] = child.serialize()
        return s

cpt_data=pd.read_csv('cpt_dict.csv')
dummy = TreeNode(None, None) # think of this as the root/table

car = TreeNode(1111, "car")
dummy.addChild(car)

engine = TreeNode(3333, "engine")
car.addChild(engine)

fan = TreeNode(4312, "fan")
engine.addChild(fan)

print(json.dumps(dummy.serialize(), indent=4))