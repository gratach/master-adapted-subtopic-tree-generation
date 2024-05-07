from json import loads, dumps
from pathlib import Path
from random import randint, shuffle
from mastg_lib import gpt_4_turbo_completion, tryRecieveAnswer, gpt_3_5_turbo_completion
rootpath = Path(__file__).parent

def createCorrelationGraphFromSubtopicTreeNeighborhoods(corelationsName):
    """
    Create a graph of subtopics and their correlations based on the subtopic tree neighborhoods.
    """
    graph = []
    correlationsPath = rootpath / "correlations" / corelationsName / "graph.json"
    if correlationsPath.exists():
        raise Exception("Correlation graph already exists.")
    correlationsPath.parent.mkdir(parents=True, exist_ok=True)
    treeNames = [x.name for  x in (rootpath / "trees").iterdir() if x.is_dir()]
    for treeName in treeNames:
        tree = loads((rootpath / "trees" / treeName / "subtopic_tree.json").read_text())
        def iterateLeafChuncks(node):
            leafs = []
            for name, children in node.items():
                if children == None:
                    leafs.append(name)
                else:
                    yield from iterateLeafChuncks(children)
            yield leafs
        if graph == []:
            for leaf in [y for x in iterateLeafChuncks(tree) for y in x]:
                graph.append([leaf, []])
        for leafChunck in iterateLeafChuncks(tree):
            connectedNodes = []
            for i, node in enumerate(graph):
                if node[0] in leafChunck:
                    connectedNodes.append((i, node))
            for i, node in connectedNodes:
                for j, otherNode in connectedNodes:
                    if i != j:
                        if j not in node[1]:
                            node[1].append(j)
    correlationsPath.write_text(dumps(graph, indent=4))

def navigateCorrelationGraph(correlationsName, startNodeIndex = None):
    """
    Navigate the correlation graph starting from a node.
    """
    graph = loads((rootpath / "correlations" / correlationsName / "graph.json").read_text())
    index = startNodeIndex if startNodeIndex != None else randint(0, len(graph) - 1)
    while True:
        print(graph[index][0])
        print("Connected nodes:")
        for i, node in enumerate(graph[index][1]):
            print(f"{i}: {graph[node][0]}")
        print("Choose a connected node to navigate to or type 'exit' to exit.")
        choice = input()
        if choice == "exit":
            break
        try:
            index = graph[index][1][int(choice)]
        except:
            if choice.startswith("goto "):
                termName = choice[5:]
                indexes = [i for i, node in enumerate(graph) if termName in node[0]]
                if len(indexes) == 0:
                    print("Term not found.")
                elif len(indexes) == 1:
                    index = indexes[0]
                else:
                    print("Multiple terms found.")
            elif choice.startswith("connect "):
                termName = choice[8:]
                indexes = [i for i, node in enumerate(graph) if node[0] == termName]
                if len(indexes) == 0:
                    print("Term not found.")
                elif len(indexes) == 1:
                    if indexes[0] not in graph[index][1]:
                        graph[index][1].append(indexes[0])
                        graph[indexes[0]][1].append(index)
                (rootpath / "correlations" / correlationsName / "graph.json").write_text(dumps(graph, indent=4))
            else:
                print("Invalid choice.")
            continue

def countConnectedSubGraphs(graph):
    """
    Count the number of connected subgraphs in the correlation graph.
    """
    visited = [False] * len(graph)
    def dfs(nodeIndex):
        visited[nodeIndex] = True
        for connectedNodeIndex in graph[nodeIndex][1]:
            if not visited[connectedNodeIndex]:
                dfs(connectedNodeIndex)
    count = 0
    for i in range(len(graph)):
        if not visited[i]:
            count += 1
            dfs(i)
    return count

def calculateAveragePathLength(graph):
    """
    Calculate the average path length between two random nodes in the correlation graph.
    """
    import random
    totalPathLength = 0
    totalNumberOfPaths = 0
    for node in range(len(graph)):
        distances = [-1] * len(graph)
        distances[node] = 0
        farthestDistance = 0
        newNodesAdded = True
        while newNodesAdded:
            newNodesAdded = False
            for i, connectedNodes in enumerate(graph):
                if distances[i] == farthestDistance:
                    for connectedNode in connectedNodes[1]:
                        if distances[connectedNode] == -1:
                            distances[connectedNode] = farthestDistance + 1
                            totalPathLength += farthestDistance + 1
                            totalNumberOfPaths += 1
                            newNodesAdded = True
            farthestDistance += 1
    return totalPathLength / totalNumberOfPaths


def generateCorrelationGraphStatistic(correlationsName):
    """
    Generate statistics for the correlation graph.
    """
    graph = loads((rootpath / "correlations" / correlationsName / "graph.json").read_text())
    statisticsPath = rootpath / "correlations" / correlationsName / "statistics.json"
    if statisticsPath.exists():
        statisticsJson = loads(statisticsPath.read_text())
    else:
        statisticsJson = {}
    statisticsJson["nodeCount"] = len(graph)
    statisticsJson["edgeCount"] = sum([len(node[1]) for node in graph]) // 2
    statisticsJson["averageDegree"] = statisticsJson["edgeCount"] * 2 / statisticsJson["nodeCount"]
    statisticsJson["maxDegree"] = max([len(node[1]) for node in graph])
    statisticsJson["minDegree"] = min([len(node[1]) for node in graph])
    statisticsJson["connectedSubGraphCount"] = countConnectedSubGraphs(graph)
    statisticsJson["averagePathLength"] = calculateAveragePathLength(graph)
    statisticsJson["hasError"] = len([j for i, node in enumerate(graph) for j in node[1] if i not in graph[j][1]]) > 0
    statisticsPath.write_text(dumps(statisticsJson, indent=4))
    print(dumps(statisticsJson, indent=4))


def createMostImportantCorrelations(targetCorrelationsName, sourceCorrelationsName, numberOfCorrelations, maxSelection = 0):
    """
    Create a correlation graph of the most important correlations of another correlation graph.
    """
    sourceGraph = loads((rootpath / "correlations" / sourceCorrelationsName / "graph.json").read_text())
    targetGraphPath = rootpath / "correlations" / targetCorrelationsName / "graph.json"
    if targetGraphPath.exists():
        raise Exception("Correlation graph already exists.")
    targetGraph = [[node[0], []] for node in sourceGraph]
    maxSelection = max([*[len(node[1]) for node in sourceGraph], maxSelection])
    order = list(range(len(sourceGraph)))
    shuffle(order)
    for i in order:
        missingNodeCount = max(numberOfCorrelations - len(targetGraph[i][1]), 0)
        if missingNodeCount == 0:
            continue
        selection = [x for x in sourceGraph[i][1] if not x in targetGraph[i][1]]
        # If the selection count is smaller than the maximum selection count, fill it up with random nodes.
        notSelectedNodes = [j for j in range(len(sourceGraph)) if j not in selection]
        while len(selection) < maxSelection:
            selection.append(notSelectedNodes.pop(randint(0, len(notSelectedNodes) - 1)))
        # Find the most important nodes
        selectionString = "{" + ", ".join([f"{i}: {sourceGraph[j][0]}" for i, j in enumerate(selection)]) + "}"
        if missingNodeCount > 1:
            query = f"Which {missingNodeCount} terms from the following selection have the strongest connection to {sourceGraph[i][0]}? Return nothing but a list of the indexes of the terms formated as [index1, index2, ...]. The selection is: {selectionString}"
            def answerConversion(answer):
                answer = answer.strip()
                assert answer[0] == "[" and answer[-1] == "]"
                choosenTerms = set([int(x) for x in answer[1:-1].split(", ")])
                assert len(choosenTerms) == missingNodeCount
                return [selection[x] for x in choosenTerms]
        else:
            query = f"Which term from the following selection has the strongest connection to {sourceGraph[i][0]}? Return only the index of the term without explanation. The selection is: {selectionString}"
            def answerConversion(answer):
                return [selection[int(answer)]]
        answer, success = tryRecieveAnswer(query, gpt_3_5_turbo_completion, answerConversion)
        if not success:
            continue
        for j in answer:
            if j not in targetGraph[i][1]:
                targetGraph[i][1].append(j)
                targetGraph[j][1].append(i)
    targetGraphPath.parent.mkdir(parents=True, exist_ok=True)
    targetGraphPath.write_text(dumps(targetGraph, indent=4))

def printMostConnectedTerms(correlationsName, numberOfTerms):
    """
    Print the most connected terms of a correlation graph.
    """
    graph = loads((rootpath / "correlations" / correlationsName / "graph.json").read_text())
    print("Most connected terms:")
    for i in sorted(range(len(graph)), key = lambda x: len(graph[x][1]), reverse = True)[:numberOfTerms]:
        print(f"{graph[i][0]}: {len(graph[i][1])} connections")

#createCorrelationGraphFromSubtopicTreeNeighborhoods("tree_cor")
navigateCorrelationGraph("tree_cor_mi_man")
#generateCorrelationGraphStatistic("tree_cor_mi_man")
#createMostImportantCorrelations("tree_cor_mi", "tree_cor", 5, 20)
#printMostConnectedTerms("tree_cor_mi_man", 10)
        
        