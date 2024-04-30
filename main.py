from pathlib import Path
from openai import OpenAI
from random import randint
from io import StringIO
from json import loads, dumps
from scipy.special import comb # pip install scipy
import matplotlib.pyplot as plt
from math import sqrt
from random import shuffle
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
    def addTopic_simpleFreeAssociationAlgorithm(self, name):
        if self.isLeaf:
            raise Exception("Cannot add topic to leaf node")
        subtopics = [x for x in self.children if not x.isLeaf]
        shuffle(subtopics) # To ensure that the evaluation and the creation query differes
        if len(subtopics) > 0:
            subtopicNameSelection = "{" + ", ".join([f'{i}: "{x.name}"' for i, x in enumerate(subtopics)]) + "}"
            query = f'Which of the topics {subtopicNameSelection} is most likely to contain "{name}" as a subtopic? Return only the number without description.'
            def answerConversion(answer):
                index = int(answer)
                assert 0 <= index < len(subtopics)
                return index
            index, subtopicChosen = tryRecieveAnswer(query, gpt_3_5_turbo_completion, answerConversion)
            if not subtopicChosen:
                print(f"Failed to choose subtopic of topic {self.name} for keyword {name}")
                SubtopicTreeBranch(name, self, True)
                return
            subtopics[index].addTopic_simpleFreeAssociationAlgorithm(name)
        else:
            SubtopicTreeBranch(name, self, True)
            leafs = [x for x in self.children if x.isLeaf]
            if len(leafs) > 5:
                query = f'In which subtopics can the topic "{self.name}" be divided? Return nothing but the list of subtopics formatted as: ["subtopic1", "subtopic2", ...]'
                def answerConversion(answer):
                    subtopics = loads(answer)
                    assert isinstance(subtopics, list)
                    for subtopic in subtopics:
                        assert isinstance(subtopic, str)
                    return subtopics
                subtopics, subtopicsFound = tryRecieveAnswer(query, gpt_3_5_turbo_completion, answerConversion)
                if not subtopicsFound:
                    print(f"Failed to find subtopics of topic {self.name}")
                    return
                for leaf in leafs:
                    leaf.changeParent(None)
                for subtopic in subtopics:
                    SubtopicTreeBranch(subtopic, self, False)
                for leaf in leafs:
                    self.addTopic_simpleFreeAssociationAlgorithm(leaf.name)
    def addTopic_freeAssociationAlgorithm(self, name):
        if self.isLeaf:
            raise Exception("Cannot add topic to leaf node")
        subtopics = [x for x in self.children if not x.isLeaf]
        shuffle(subtopics) # To ensure that the evaluation and the creation query differes
        if len(subtopics) > 0:
            # If there are to many subtopics, ask the user to choose the most likely one
            if len(subtopics) > 14:
                subtopicNameSelection = "{" + ", ".join([f'{i}: "{x.name}"' for i, x in enumerate(subtopics)]) + "}"
                query = f'Which of the topics {subtopicNameSelection} is most likely to contain "{name}" as a subtopic? Return only the number without description.'
                def answerConversion(answer):
                    index = int(answer)
                    assert 0 <= index < len(subtopics)
                    return index
                index, subtopicChosen = tryRecieveAnswer(query, gpt_3_5_turbo_completion, answerConversion)
                if not subtopicChosen:
                    print(f"Failed to choose subtopic of topic {self.name} for keyword {name}")
                    SubtopicTreeBranch(name, self, True)
                    return
                subtopics[index].addTopic_freeAssociationAlgorithm(name)
            # If there are few subtopics, ask the user if the keyword belongs to one of them and leave the option to create a new subtopic
            else:
                if len(subtopics) == 1:
                    query = f'Is the topic {subtopics[0].name} including "{name}" as a subtopic? Return 1 if yes, 0 if no.'
                else:
                    subtopicNameSelection = "{" + ", ".join([f'{i + 1} : "{x.name}"' for i, x in enumerate(subtopics)]) + "}"
                    query = f'Which of the topics {subtopicNameSelection} contains "{name}" as a subtopic? Return only the number without description. Return 0 if none.'
                def answerConversion(answer):
                    index = int(answer)
                    assert 0 <= index <= len(subtopics)
                    return index
                index, subtopicChosen = tryRecieveAnswer(query, gpt_3_5_turbo_completion, answerConversion)
                if not subtopicChosen:
                    print(f"Failed to choose subtopic of topic {self.name} for keyword {name}")
                    index = 0
                # Add the topic to the chosen subtopic
                if index > 0:
                    subtopics[index - 1].addTopic_freeAssociationAlgorithm(name)
                else:
                    subtopicNameSelection = "{" + ", ".join([f'"{x.name}"' for x in subtopics]) + "}"
                    query = f'The topic "{self.name}" has the subtopic categories "{subtopicNameSelection}. In which category does the term "{name}" belong? Return the name of one of the existing categories or the name of a new subtopic category of "{self.name}". Return nothing but the name without description. Make sure that the returned term is surrounded by quotation marks.'
                    def answerConversion(answer):
                        assert answer.startswith('"') and answer.endswith('"')
                        answer =  answer[1:-1]
                        assert not '"' in answer
                        return answer
                    newSubtopicName, newSubtopicAdded = tryRecieveAnswer(query, gpt_4_turbo_completion, answerConversion)
                    if not newSubtopicAdded:
                        print(f"Failed to add new subtopic of topic {self.name} for keyword {name}")
                        SubtopicTreeBranch(name, self, True)
                    else:
                        newbranch = None
                        for subtopic in subtopics:
                            if subtopic.name == newSubtopicName:
                                newbranch = subtopic
                                break
                        if not newbranch:
                            newbranch = SubtopicTreeBranch(newSubtopicName, self, False)
                        newbranch.addTopic_freeAssociationAlgorithm(name)
        else:
            SubtopicTreeBranch(name, self, True)
            leafs = [x for x in self.children if x.isLeaf]
            if len(leafs) > 5:
                query = f'In which subtopics can the topic "{self.name}" be divided? Return nothing but the list of subtopics formatted as: ["subtopic1", "subtopic2", ...]'
                def answerConversion(answer):
                    subtopics = loads(answer)
                    assert isinstance(subtopics, list)
                    for subtopic in subtopics:
                        assert isinstance(subtopic, str)
                    return subtopics
                subtopics, subtopicsFound = tryRecieveAnswer(query, gpt_4_turbo_completion, answerConversion)
                if not subtopicsFound:
                    print(f"Failed to find subtopics of topic {self.name}")
                    return
                for leaf in leafs:
                    leaf.changeParent(None)
                for subtopic in subtopics:
                    SubtopicTreeBranch(subtopic, self, False)
                for leaf in leafs:
                    self.addTopic_freeAssociationAlgorithm(leaf.name)
    def addTopic_subdevisionAlgorithm(self, name):
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
    def interactiveNavigation(self):
        print("\n\n")
        print(self.name + (" + " if self.isLeaf else " - " + f"({self.leafCount} Terms)"))
        for i, child in enumerate(self.children):
            print(f"    {i + 1}: {child.name} {'+' if child.isLeaf else '-'} ({child.leafCount} Terms)")
        print(f"0: Go back to {self.parent.name}" if self.parent else "0: Exit")
        choice = input("Choose a subtopic: ")
        if choice == "exit" or choice == "":
            return None
        try:
            choice = int(choice)
            if choice == 0:
                return self.parent.interactiveNavigation if self.parent else None
            if choice < 1 or choice > len(self.children):
                print("Invalid choice")
                return self.interactiveNavigation
            return self.children[choice - 1].interactiveNavigation
        except:
            print("Invalid choice")
            return self.interactiveNavigation
    def searchSubbranchThatMightContain(self, name):
        children = self.children.copy()
        if len(children) == 0:
            return None
        shuffle(children)
        for child in children:
            if child.isLeaf and name == child.name:
                return child
        subtopicNameSelection = "{" + ", ".join([f'{i + 1} : "{x.name}"' for i, x in enumerate(children)]) + "}"
        query = f'Which of the topics {subtopicNameSelection} is most likely to contain "{name}" as a subtopic? Return only the number without description. Return 0 if none.'
        def answerConversion(answer):
            index = int(answer)
            assert 0 <= index <= len(children)
            return children[index - 1] if index > 0 else None
        answer, subtopicChosen = tryRecieveAnswer(query, gpt_3_5_turbo_completion, answerConversion)
        return answer
    def iterateLeaves(self):
        if self.isLeaf:
            yield self
        else:
            for child in self.children:
                yield from child.iterateLeaves()


def tryRecieveAnswer(query, completionFunction, answerConversion = lambda x: True, maxTries = 10):
    tryNumber = 0
    while tryNumber < maxTries:
        answer = completionFunction(query)
        try:
            answer = answerConversion(answer)
            return (answer, True)
        except:
            pass
        tryNumber += 1
    print(f"Failed to recieve answer for query: {query}")
    return (None, False)

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

def loadSubtopicTree():
    treepath = rootpath / "subtopic_tree.json"
    if treepath.exists():
        with treepath.open("r") as f:
            rootBranch = SubtopicTreeBranch.loadFromJsonDict(loads(f.read()))
    else:
        rootBranch = SubtopicTreeBranch("Physics", None)
    return rootBranch

def saveSubtopicTree(rootBranch):
    treepath = rootpath / "subtopic_tree.json"
    with treepath.open("w") as f:
        f.write(dumps(rootBranch.safeAsJsonDict()))

def main_addTechnicalTermsToSubtopicTree_subdevisionAlgorithm():
    rootBranch = loadSubtopicTree()
    with (rootpath / "technical_terms.txt").open("r") as f:
        technical_terms = f.read().split("\n")
    for technical_term in technical_terms:
        if not rootBranch.findLeaf(technical_term):
            rootBranch.addTopic_subdevisionAlgorithm(technical_term)
    saveSubtopicTree(rootBranch)

def main_addTechnicalTermsToSubtopicTree_freeAssociationAlgorithm():
    rootBranch = loadSubtopicTree()
    with (rootpath / "technical_terms.txt").open("r") as f:
        technical_terms = f.read().split("\n")
    for technical_term in technical_terms:
        if not rootBranch.findLeaf(technical_term):
            rootBranch.addTopic_freeAssociationAlgorithm(technical_term)
    saveSubtopicTree(rootBranch)

def main_addTechnicalTermsToSubtopicTree_simpleFreeAssociationAlgorithm():
    rootBranch = loadSubtopicTree()
    with (rootpath / "technical_terms.txt").open("r") as f:
        technical_terms = f.read().split("\n")
    for technical_term in technical_terms:
        if not rootBranch.findLeaf(technical_term):
            rootBranch.addTopic_simpleFreeAssociationAlgorithm(technical_term)
    saveSubtopicTree(rootBranch)

def main_navigateSubtopicTree():
    rootBranch = loadSubtopicTree()
    navigationFunction = rootBranch.interactiveNavigation
    while navigationFunction:
        navigationFunction = navigationFunction()

def main_writeTermPaths():
    rootBranch = loadSubtopicTree()
    with (rootpath / "technical_terms.txt").open("r") as f:
        technical_terms = f.read().split("\n")
    with (rootpath / "term_paths.txt").open("w") as f:
        for term in technical_terms:
            path = rootBranch.findLeaf(term)
            if path:
                f.write(f"{term}: {path}\n\n")
            else:
                print(f"{term}: Not found\n")

def main_plotBranchLengths():
    rootBranch = loadSubtopicTree()
    with (rootpath / "technical_terms.txt").open("r") as f:
        technical_terms = f.read().split("\n")
    lengths = []
    for term in technical_terms:
        path = rootBranch.findLeaf(term)
        if path:
            length = len(path) - 1
            while length >= len(lengths):
                lengths.append(0)
            lengths[length] += 1
        else:
            print(f"{term}: Not found\n")
    lengths = lengths[1:]
    plt.bar(range(1, len(lengths) + 1), lengths)
    plt.xlabel("Depth of term in subtopic tree")
    plt.ylabel("Number of terms")
    plt.show()

def main_plotLeafDepthDependentOnOrder():
    rootBranch = loadSubtopicTree()
    with (rootpath / "technical_terms.txt").open("r") as f:
        technical_terms = f.read().split("\n")
    depthArray = []
    for term in technical_terms:
        path = rootBranch.findLeaf(term)
        if path:
            depthArray.append(len(path) - 1)
        else:
            print(f"{term}: Not found\n")
    plt.scatter(range(1, len(depthArray) + 1), depthArray)
    plt.xlabel("Addition order of term")
    plt.ylabel("Depth of term in subtopic tree")
    plt.show()

def main_plotNumberOfSubBranchesPerJunction():
    rootBranch = loadSubtopicTree()
    numberOfSubBranchesBins = []
    def traverseBranch(branch):
        if not branch.isLeaf:
            numberOfSubBranches = len(branch.children)
            while numberOfSubBranches >= len(numberOfSubBranchesBins):
                numberOfSubBranchesBins.append(0)
            numberOfSubBranchesBins[numberOfSubBranches] += 1
            for child in branch.children:
                traverseBranch(child)
    traverseBranch(rootBranch)
    plt.bar(range(len(numberOfSubBranchesBins)), numberOfSubBranchesBins)
    plt.xlabel("Number of subbranches")
    plt.ylabel("Number of junction nodes")
    plt.show()

def main_subtopicTreeStatistics():
    rootBranch = loadSubtopicTree()
    numberLeafNodes = 0
    leafDepthSum = 0
    numberJunctionNodes = 0
    def traverseBranch(branch, depth):
        nonlocal numberLeafNodes
        nonlocal leafDepthSum
        nonlocal numberJunctionNodes
        if branch.isLeaf:
            numberLeafNodes += 1
            leafDepthSum += depth
        else:
            if branch.leafCount > 0:
                numberJunctionNodes += 1
            for child in branch.children:
                traverseBranch(child, depth + 1)
    traverseBranch(rootBranch, 0)
    print(f"Number of leaf nodes: {numberLeafNodes}")
    print(f"Average depth of leaf nodes: {leafDepthSum / numberLeafNodes}")
    print(f"Number of supporting junction nodes: {numberJunctionNodes}")

def main_searchForTerms():
    numberOfSearches = 100
    rootBranch = loadSubtopicTree()
    with (rootpath / "technical_terms.txt").open("r") as f:
        search_terms = f.read().split("\n")
    search_results = [] # Format: (name_of_term, is_found, serch_path, number_of_viewed_nodes, number_of_viewed_bytes)
    with (rootpath / "search_results.txt").open("w") as f:
        f.write("[\n")
        for i in range(numberOfSearches):
            search_term = search_terms[randint(0, len(search_terms) - 1)]
            branch = rootBranch
            serchPath = [branch.name]
            viewedNodes = 0
            viewedBytes = 0
            while branch and not branch.isLeaf:
                viewedNodes += len(branch.children)
                viewedBytes += sum([len(x.name) for x in branch.children])
                branch = branch.searchSubbranchThatMightContain(search_term)
                if branch:
                    serchPath.append(branch.name)
            search_results.append((search_term, branch != None and branch.isLeaf and branch.name == search_term, serchPath, viewedNodes, viewedBytes))
            f.write("    " + dumps(search_results[-1]) + ("," if i + 1 < numberOfSearches else "") + "\n")
            f.flush()
        f.write("]\n")

def main_calculateSearchStatistics():
    rootBranch = loadSubtopicTree()
    numberOfLeafNodes = rootBranch.leafCount
    numberOfLeafBytes = sum([len(x.name) for x in rootBranch.iterateLeaves()])
    with (rootpath / "search_results.txt").open("r") as f:
        search_results = loads(f.read())
    numberOfSearches = len(search_results)
    numberFound = 0
    numberNotFound = 0
    numberViewedNodesOfFoundSearches = 0
    numberViewedNodesOfNotFoundSearches = 0
    numberViewedBytesOfFoundSearches = 0
    numberViewedBytesOfNotFoundSearches = 0
    for result in search_results:
        if result[1]:
            numberFound += 1
            numberViewedNodesOfFoundSearches += result[3]
            numberViewedBytesOfFoundSearches += result[4]
        else:
            numberNotFound += 1
            numberViewedNodesOfNotFoundSearches += result[3]
            numberViewedBytesOfNotFoundSearches += result[4]
    foundFraction = numberFound / numberOfSearches if numberOfSearches > 0 else None
    averageViewedNodesOfFoundSearches = numberViewedNodesOfFoundSearches / numberFound if numberFound > 0 else None
    averageViewedNodesOfNotFoundSearches = numberViewedNodesOfNotFoundSearches / numberNotFound if numberNotFound > 0 else None
    averageViewedNodesPerSearch = ((1 / foundFraction) - 1) * averageViewedNodesOfNotFoundSearches + averageViewedNodesOfFoundSearches if foundFraction else None
    viewedNodesFraction = averageViewedNodesPerSearch / numberOfLeafNodes if numberOfLeafNodes > 0 else None
    averageViewedBytesOfFoundSearches = numberViewedBytesOfFoundSearches / numberFound if numberFound > 0 else None
    averageViewedBytesOfNotFoundSearches = numberViewedBytesOfNotFoundSearches / numberNotFound if numberNotFound > 0 else None
    averageViewedBytesPerSearch = ((1 / foundFraction) - 1) * averageViewedBytesOfNotFoundSearches + averageViewedBytesOfFoundSearches if foundFraction else None
    viewedBytesFraction = averageViewedBytesPerSearch / numberOfLeafBytes if numberOfLeafBytes > 0 else None
    # Calculate the errors
    numberFoundVariance = 0
    for k in range(1, numberOfSearches + 1):
        numberFoundVariance += ((numberFound - k) ** 2) * comb(numberOfSearches, k, exact=True) * (foundFraction ** k) * ((1 - foundFraction) ** (numberOfSearches - k))
    numberFoundError = sqrt(numberFoundVariance)
    foundFractionError = numberFoundError / numberOfSearches

    averageViewedNodesOfFoundSearchesVariance = 0
    averageViewedNodesOfNotFoundSearchesVariance = 0
    for result in search_results:
        if result[1]:
            averageViewedNodesOfFoundSearchesVariance += (result[3] - averageViewedNodesOfFoundSearches) ** 2
        else:
            averageViewedNodesOfNotFoundSearchesVariance += (result[3] - averageViewedNodesOfNotFoundSearches) ** 2
    averageViewedNodesOfFoundSearchesError = sqrt(averageViewedNodesOfFoundSearchesVariance / numberFound / (numberFound - 1)) if numberFound > 1 else None
    averageViewedNodesOfNotFoundSearchesError = sqrt(averageViewedNodesOfNotFoundSearchesVariance / numberNotFound / (numberNotFound - 1)) if numberNotFound > 1 else None
    averageViewedNodesPerSearchError = sqrt(
        (averageViewedNodesOfFoundSearchesError ** 2) +
        (((1 / foundFraction - 1) * averageViewedNodesOfNotFoundSearchesError) ** 2) +
        ((averageViewedNodesOfNotFoundSearches / (foundFraction ** 2) * foundFractionError) ** 2)
    ) if foundFraction > 0 else None
    viewedNodesFractionError = averageViewedNodesPerSearchError / numberOfLeafNodes if numberOfLeafNodes > 0 else None
    
    averageViewedBytesOfFoundSearchesVariance = 0
    averageViewedBytesOfNotFoundSearchesVariance = 0
    for result in search_results:
        if result[1]:
            averageViewedBytesOfFoundSearchesVariance += (result[4] - averageViewedBytesOfFoundSearches) ** 2
        else:
            averageViewedBytesOfNotFoundSearchesVariance += (result[4] - averageViewedBytesOfNotFoundSearches) ** 2
    averageViewedBytesOfFoundSearchesError = sqrt(averageViewedBytesOfFoundSearchesVariance / numberFound / (numberFound - 1)) if numberFound > 1 else None
    averageViewedBytesOfNotFoundSearchesError = sqrt(averageViewedBytesOfNotFoundSearchesVariance / numberNotFound / (numberNotFound - 1)) if numberNotFound > 1 else None
    averageViewedBytesPerSearchError = sqrt(
        (averageViewedBytesOfFoundSearchesError ** 2) +
        (((1 / foundFraction - 1) * averageViewedBytesOfNotFoundSearchesError) ** 2) +
        ((averageViewedBytesOfNotFoundSearches / (foundFraction ** 2) * foundFractionError) ** 2)
    ) if foundFraction > 0 else None
    viewedBytesFractionError = averageViewedBytesPerSearchError / numberOfLeafBytes if numberOfLeafBytes > 0 else None
    # Print the results
    print(f"Number of leaf nodes: {numberOfLeafNodes}")
    print(f"Number of searches: {numberOfSearches}")
    print(f"Number of found searches: {numberFound}")
    print(f"Number of not found searches: {numberNotFound}")
    print(f"Found fraction: {foundFraction} ± {foundFractionError}")
    print(f"Average viewed nodes of found searches: {averageViewedNodesOfFoundSearches} ± {averageViewedNodesOfFoundSearchesError}")
    print(f"Average viewed nodes of not found searches: {averageViewedNodesOfNotFoundSearches} ± {averageViewedNodesOfNotFoundSearchesError}")
    print(f"Average viewed nodes per search: {averageViewedNodesPerSearch} ± {averageViewedNodesPerSearchError}")
    print(f"Viewed nodes fraction: {viewedNodesFraction} ± {viewedNodesFractionError}")
    print(f"Average viewed bytes of found searches: {averageViewedBytesOfFoundSearches} ± {averageViewedBytesOfFoundSearchesError}")
    print(f"Average viewed bytes of not found searches: {averageViewedBytesOfNotFoundSearches} ± {averageViewedBytesOfNotFoundSearchesError}")
    print(f"Average viewed bytes per search: {averageViewedBytesPerSearch} ± {averageViewedBytesPerSearchError}")
    print(f"Viewed bytes fraction: {viewedBytesFraction} ± {viewedBytesFractionError}")

#main_addTechnicalTermsToSubtopicTree_subdevisionAlgorithm()
#main_addTechnicalTermsToSubtopicTree_freeAssociationAlgorithm()
#main_navigateSubtopicTree()
#main_writeTermPaths()
#main_plotBranchLengths()
#main_subtopicTreeStatistics()
#main_plotLeafDepthDependentOnOrder()
#main_plotNumberOfSubBranchesPerJunction()
main_searchForTerms()
main_calculateSearchStatistics()
