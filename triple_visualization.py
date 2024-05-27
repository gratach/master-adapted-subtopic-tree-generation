from json import dumps, loads
from random import choice
from pathlib import Path
rootPath = Path(__file__).parent
def safeVisNetworkJSONToHTMLFile(jsonData, htmlFilePath):
    with open(htmlFilePath, "w") as htmlFile:
        htmlFile.write(
            f"""
            <!DOCTYPE html>
            <html lang="en-US">
            
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>Inline vis</title>
            </head>
            
            <body>
                <div id="mynetwork" style="width:100vw; height:100vh;"></div>
                <script src=" https://cdn.jsdelivr.net/npm/vis-data@7.1.9/peer/umd/vis-data.min.js "></script>
                <script src=" https://cdn.jsdelivr.net/npm/vis-network@9.1.9/peer/umd/vis-network.min.js "></script>
                <link href=" https://cdn.jsdelivr.net/npm/vis-network@9.1.9/styles/vis-network.min.css " rel="stylesheet">
                <script>
                    var jsonData = {dumps(jsonData)};
                    // create a network
                    var container = document.getElementById("mynetwork");
                    var options = {{}};
                    var network = new vis.Network(container, jsonData, options);
                </script>
            </body>
            
            </html>
            """
        )

def convertSemanticTriplesToVisNetworkJSON(semanticTriples):
    nodes = []
    edges = []
    indexPerTerm = {}
    indexCounter = 1
    for subj, pred, obj in semanticTriples:
        for term in subj, obj:
            if term not in indexPerTerm:
                indexPerTerm[term] = indexCounter
                indexCounter += 1
                nodes.append({
                    "id": indexPerTerm[term], 
                    "label": term, 
                    "font": {"size": 20}, 
                    "color" : {"background": "rgba(140, 220, 255, 255)", 
                                "border": "rgba(0, 0, 255, 255)"}})
        nodes.append({
            "id": indexCounter, 
            "label": pred,
            "font": {"size": 16},
            "color" : {"background": "rgba(160, 255, 100, 255)", 
                        "border": "darkgreen"},
            "shape" : "box"})
        edges.append({"from": indexPerTerm[subj], "to": indexCounter, "color": "rgba(100, 200, 60, 255)", "width": 3})
        edges.append({"from": indexCounter, "to": indexPerTerm[obj], "arrows": "to", "color": "rgba(100, 200, 60, 255)", "width": 3})
        indexCounter += 1
    return {"nodes": nodes, "edges": edges}

def selectNonOverlapingTripleSubset(semanticTriples, maxsteps = -1):
    hexfield = {} # {hexcoord: concept} Where hexcoord is a tuple of 2 coordinates that represent two directions in a hexagonal grid with an 60 degree angle between them
    hexDirections = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    startConcept = choice(semanticTriples)[0]
    hexfield[(0, 0)] = startConcept
    usedConcepts = {startConcept}
    unusedConcepts = set([*[triple[0] for triple in semanticTriples], *[triple[2] for triple in semanticTriples]])
    unusedConcepts.remove(startConcept)
    currentHexDirectionIndex = 2
    currentHexpos = (1, 0)
    selectedTriples = []
    step = 0
    while step != maxsteps and unusedConcepts:
        step += 1
        # Select the concept with the most connections to the neighbors
        neighbors = [hexfield[(currentHexpos[0] + hexDirections[i][0], currentHexpos[1] + hexDirections[i][1])] for i in range(6) if (currentHexpos[0] + hexDirections[i][0], currentHexpos[1] + hexDirections[i][1]) in hexfield]
        triplesWithNewSubject = [triple for triple in semanticTriples if triple[2] in neighbors and triple[0] in unusedConcepts]
        triplesWithNewObject = [triple for triple in semanticTriples if triple[0] in neighbors and triple[2] in unusedConcepts]
        numberOfConnectionsPerConcept = {}
        for triple in triplesWithNewSubject:
            numberOfConnectionsPerConcept[triple[0]] = numberOfConnectionsPerConcept.get(triple[0], 0) + 1
        for triple in triplesWithNewObject:
            if len([t for t in triplesWithNewSubject if t[0] == triple[2] and t[2] == triple[0]]) == 0:
                numberOfConnectionsPerConcept[triple[2]] = numberOfConnectionsPerConcept.get(triple[2], 0) + 1
        maxConnections = max(numberOfConnectionsPerConcept.values())
        bestConcepts = [concept for concept, connections in numberOfConnectionsPerConcept.items() if connections == maxConnections]
        selectedConcept = choice(bestConcepts) if bestConcepts else choice(list(unusedConcepts))
        # Add the selected concept to the hexfield
        hexfield[currentHexpos] = selectedConcept
        usedConcepts.add(selectedConcept)
        unusedConcepts.remove(selectedConcept)
        # Add the triples to the selected triples
        for neighbor in neighbors:
            tripleSelection = [*[t for t in triplesWithNewSubject if t[0] == selectedConcept and t[2] == neighbor], *[t for t in triplesWithNewObject if t[2] == selectedConcept and t[0] == neighbor]]
            if tripleSelection:
                selectedTriples.append(choice(tripleSelection))
        # Navigate to a new hexagon in a spiral pattern
        nextHexDirectionIndex = (currentHexDirectionIndex + 1) % 6
        nextHexposOnCurvedPath = (currentHexpos[0] + hexDirections[nextHexDirectionIndex][0], currentHexpos[1] + hexDirections[nextHexDirectionIndex][1])
        # If it is possible to move in a curved path, do so
        if nextHexposOnCurvedPath not in hexfield:
            currentHexDirectionIndex = nextHexDirectionIndex
        currentHexpos = (currentHexpos[0] + hexDirections[currentHexDirectionIndex][0], currentHexpos[1] + hexDirections[currentHexDirectionIndex][1])
    return selectedTriples

def visualizeSemanticTriplesFlat(triplesName, maxsteps = -1):
    triplePath = rootPath / "triples" / triplesName / "triples.json"
    with triplePath.open("r") as tripleFile:
        semanticTriples = loads(tripleFile.read())
    flatTriples = selectNonOverlapingTripleSubset(semanticTriples, maxsteps)
    visualizationPath = rootPath / "triples" / triplesName / "visualization_flat.html"
    safeVisNetworkJSONToHTMLFile(convertSemanticTriplesToVisNetworkJSON(flatTriples), str(visualizationPath))

visualizeSemanticTriplesFlat("tri_1", 80)

