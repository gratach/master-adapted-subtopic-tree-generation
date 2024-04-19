from pathlib import Path
from openai import OpenAI
from random import randint
from io import StringIO
from json import loads, dumps
rootpath = Path(__file__).parent
class SubtopicTreeBranch:
    def __init__(self, name, parent, isLeaf = False):
        self.name = name
        self.parent = parent
        self.isLeaf = isLeaf
        self.children = []
        self.leafCount = 1 if isLeaf else 0
        if parent:
            parent.addChild(self)
    def addChild(self, child):
        self.children.append(child)
        self.leafCount += child.leafCount
        if self.parent:
            self.parent.updateLeafCount(child.leafCount)
    def changeParent(self, newParent):
        if self.parent:
            self.parent.children.remove(self)
            self.parent.updateLeafCount(-self.leafCount)
        self.parent = newParent
        if newParent:
            newParent.addChild(self)
    def findLeaf(self, name):
        if self.isLeaf:
            return [name] if self.name == name else None
        for child in self.children:
            result = child.findLeaf(name)
            if result != None:
                return [self.name] + result
        return None
    def updateLeafCount(self, delta):
        self.leafCount += delta
        if self.parent:
            self.parent.updateLeafCount(delta)
    def addTopic(self, name):
        if self.isLeaf:
            raise Exception("Cannot add topic to leaf node")
        subtopics = [x for x in self.children if not x.isLeaf]
        topicAddedToDeaperLayer = False
        if len(subtopics) > 0:
            if len(subtopics) == 1:
                query = f'Is the topic {subtopics[0].name} including "{name}" as a subtopic? Return 1 if yes, 0 if no.'
            else:
                subtopicNameSelection = "{" + ", ".join([f'{i + 1} : "{x.name}"' for i, x in enumerate(subtopics)]) + "}"
                query = f'Which of the topics {subtopicNameSelection} is most likely to contain "{name}" as a subtopic? Return only the number without description. Return 0 if none.'
            subtopicChosen = False
            maxTries = 10
            tryNumber = 0
            while not subtopicChosen and tryNumber < maxTries:
                answer = gpt_3_5_turbo_completion(query)
                try:
                    index = int(answer)
                    assert 0 <= index <= len(subtopics)
                    subtopicChosen = True
                except:
                    pass
                tryNumber += 1
            if not subtopicChosen:
                print(f"Failed to choose subtopic of topic {self.name} for keyword {name}")
            # Add the topic to the chosen subtopic
            elif index > 0:
                subtopics[index - 1].addTopic(name)
                topicAddedToDeaperLayer = True
        if not topicAddedToDeaperLayer:
            SubtopicTreeBranch(name, self, True)
            # Perform a branch summary if the number of leaf nodes is greater than 15
            if len(self.children) > 15:
                self.performSubBranchSummary()
    def performSubBranchSummary(self):
        subBranches = self.children.copy()
        for subBranch in subBranches:
            subBranch.changeParent(None)
        self.sortIntoSubCategories(subBranches)
    def sortIntoSubCategories(self, subBranches):
        termsList = "{" + ", ".join([f'{i}: "{x.name}"' for i, x in enumerate(subBranches)])
        query = f'Sort the following terms after topics and return the result formatted as {{"Name of first topic": [1, 2, ...], "Name of second topic" : [3, 4, ...], ...}} without explanation where 1, 2, 3 and 4 are indices of the terms. All topics should be subtopics of "{self.name}". There should be approximately 5 topics. The terms are: {termsList}'
        resultValid = False
        maxTries = 5
        tryNumber = 0
        while not resultValid and tryNumber < maxTries:
            answer = gpt_4_turbo_completion(query).strip()
            if answer.startswith("json"):
                answer = answer[4:]
            try:
                categories = loads(answer)
                assert type(categories) == dict
                for key, value in categories.items():
                    assert type(value) == list
                    for x in value:
                        assert type(x) == int
                        assert 0 <= x < len(subBranches)
                resultValid = True
            except:
                pass
            tryNumber += 1
        if not resultValid:
            print(f"Failed to sort terms into subcategories for topic {self.name}")
            for subBranch in subBranches:
                subBranch.changeParent(self)
            return
        choosenSubBranchIndices = set()
        for key, value in categories.items():
            subBranch = SubtopicTreeBranch(key, self)
            for index in value:
                if index not in choosenSubBranchIndices:
                    subBranches[index].changeParent(subBranch)
                    choosenSubBranchIndices.add(index)
        for i in range(len(subBranches)):
            if i not in choosenSubBranchIndices:
                subBranches[i].changeParent(self)
    def writeAsText(self, stream = None, indentString = ""):
        filestream = stream if stream else StringIO()
        filestream.write(f"{indentString}{self.name}\n")
        for i, child in enumerate(self.children):
            if i + 1 == len(self.children):
                child.writeAsText(filestream, indentString + " ")
            else:
                child.writeAsText(filestream, indentString + "│")
        if not stream:
            return filestream.getvalue()
    def safeAsJsonDict(self):
        return {
            self.name: self.safeBranchesAsJsonDict()
        }
    def loadFromJsonDict(jsonDict):
        key = list(jsonDict.keys())[0]
        branch = SubtopicTreeBranch(key, None, jsonDict[key] is None)
        branch.loadBranchesFromJsonDict(jsonDict[key])
        return branch
    def safeBranchesAsJsonDict(self):
        return {
            x.name: (x.safeBranchesAsJsonDict() if not x.isLeaf else None) for x in self.children
        }
    def loadBranchesFromJsonDict(self, jsonDict):
        for key, value in jsonDict.items():
            child = SubtopicTreeBranch(key, self, value is None)
            if value:
                child.loadBranchesFromJsonDict(value)




openaiClient = OpenAI()
def gpt_3_5_turbo_completion(query):
    answer = openaiClient.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": query
            }
        ],
        seed = randint(0, 1000000)
    )
    return answer.choices[0].message.content

def gpt_4_turbo_completion(query):
    answer = openaiClient.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": query
            }
        ],
        seed = randint(0, 1000000)
    )
    return answer.choices[0].message.content

treepath = rootpath / "subtopic_tree.json"
if treepath.exists():
    with treepath.open("r") as f:
        rootBranch = SubtopicTreeBranch.loadFromJsonDict(loads(f.read()))
else:
    rootBranch = SubtopicTreeBranch("Physics", None)

with (rootpath / "technical_terms.txt").open("r") as f:
    technical_terms = f.read().split("\n")
for technical_term in technical_terms:
    if not rootBranch.findLeaf(technical_term):
        rootBranch.addTopic(technical_term)
        #with (rootpath / "subtopic_tree_view.txt").open("w") as f:
        #    f.write(rootBranch.writeAsText())

with treepath.open("w") as f:
    f.write(dumps(rootBranch.safeAsJsonDict()))

