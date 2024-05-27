from pathlib import Path
from json import loads, dumps
from mastg_lib import gpt_4_turbo_completion, tryRecieveAnswer, gpt_3_5_turbo_completion
from random import choice, shuffle
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb 
rootpath = Path(__file__).parent
def generateTriples(correlationsName, triplesName):
    correlationGraph = loads((rootpath / "correlations" / correlationsName / "graph.json").read_text())
    triplePath = rootpath / "triples" / triplesName / "triples.json"
    if triplePath.exists():
        triples = loads(triplePath.read_text())
    else:
        triples = []
        triplePath.parent.mkdir(parents=True, exist_ok=True)
    totalNumberOfTriples = sum([len(node[1]) for node in correlationGraph])
    with triplePath.open("w") as f:
        f.write("[")
        writtenTriples = 0
        for triple in triples:
            writtenTriples += 1
            f.write("    " + dumps(triple) + (",\n" if writtenTriples < totalNumberOfTriples else ""))
            f.flush()
        for i in range(len(correlationGraph)):
            subj = correlationGraph[i][0]
            for j in correlationGraph[i][1]:
                obj = correlationGraph[j][0]
                if len([triple for triple in triples if triple[0] == subj and triple[2] == obj]) > 0:
                    continue
                # Generate a predicate
                query = 'Semantic triples such as ["Star", "emits", "Light"] and ["Rocket", "can bring cargo to", "Space"] consists of a subject, a predicate, and an object. What is the predicate for the triple ["' + subj + '", ??? , "' + obj + '"]? Return only the predicate quoted by "" without explanation.'
                def answerConversion(answer):
                    answer = answer.strip()
                    assert answer[0] == '"' and answer[-1] == '"'
                    answer = answer[1:-1]
                    assert not '"' in answer
                    return answer
                predicate, predicateFound = tryRecieveAnswer(query, gpt_3_5_turbo_completion, answerConversion)
                if not predicateFound:
                    predicate = "is related to"
                triples.append([subj, predicate, obj])
                writtenTriples += 1
                f.write("    " + dumps([subj, predicate, obj]) + (",\n" if writtenTriples < totalNumberOfTriples else ""))
                f.flush()
        f.write("\n]")

def navigateThroughTriples(triplesName, startNode = None):
    triples = loads((rootpath / "triples" / triplesName / "triples.json").read_text())
    currentNode = startNode if startNode is not None else choice(triples)[0]
    while True:
        print(currentNode)
        connections = [(triple[1], triple[2]) for triple in triples if triple[0] == currentNode]
        for i, connection in enumerate(connections):
            print(f'    "{connection[0]}" "{connection[1]}" ({i})')
        selection = input("Choose a connection (e to exit): ")
        if selection == "e":
            break
        else:
            try:
                currentNode = connections[int(selection)][1]
            except:
                print("Invalid selection")

def createPredicateQuantityChart(triplesName, numberOfPredicates = 10):
    triples = loads((rootpath / "triples" / triplesName / "triples.json").read_text())
    predicates = [triple[1] for triple in triples]
    predicateQuantity = {}
    for predicate in predicates:
        if predicate in predicateQuantity:
            predicateQuantity[predicate] += 1
        else:
            predicateQuantity[predicate] = 1
    predicateQuantity = sorted(predicateQuantity.items(), key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots()
    predicateNames = [predicate[0] for predicate in predicateQuantity[:numberOfPredicates]]
    y_pos = np.arange(len(predicateNames))
    quantity = [predicate[1] for predicate in predicateQuantity[:numberOfPredicates]]

    ax.barh(y_pos, quantity, align='center')
    ax.set_yticks(y_pos, labels=predicateNames)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Predicate Quantity')
    ax.set_title('Top ' + str(numberOfPredicates) + ' Predicates in ' + triplesName)

    plt.savefig(rootpath / "triples" / triplesName / "predicate_quantity_chart.svg")
    plt.savefig(rootpath / "triples" / triplesName / "predicate_quantity_chart.pdf")

def testTripleTruthValue(triplesName):
    triples = loads((rootpath / "triples" / triplesName / "triples.json").read_text())
    trueTriplePath = rootpath / "triples" / (triplesName + "_true") / "triples.json"
    if trueTriplePath.exists():
        trueTriples = loads(trueTriplePath.read_text())
    else:
        trueTriples = []
        trueTriplePath.parent.mkdir(parents=True, exist_ok=True)
    falseTriplePath = rootpath / "triples" / (triplesName + "_false") / "triples.json"
    if falseTriplePath.exists():
        falseTriples = loads(falseTriplePath.read_text())
    else:
        falseTriples = []
        falseTriplePath.parent.mkdir(parents=True, exist_ok=True)
    rubbishTriplePath = rootpath / "triples" / (triplesName + "_rubbish") / "triples.json"
    if rubbishTriplePath.exists():
        rubbishTriples = loads(rubbishTriplePath.read_text())
    else:
        rubbishTriples = []
        rubbishTriplePath.parent.mkdir(parents=True, exist_ok=True)
    with trueTriplePath.open("w") as fTrue:
        fTrue.write("[")
        fTrue.write(",".join(["\n    " + dumps(triple) for triple in trueTriples]))
        fTrue.flush()
        with falseTriplePath.open("w") as fFalse:
            fFalse.write("[")
            fFalse.write(",".join(["\n    " + dumps(triple) for triple in falseTriples]))
            fFalse.flush()
            with rubbishTriplePath.open("w") as fRubbish:
                fRubbish.write("[")
                fRubbish.write(",".join(["\n    " + dumps(triple) for triple in rubbishTriples]))
                fRubbish.flush()
                for triple in triples:
                    if triple in trueTriples or triple in falseTriples or triple in rubbishTriples:
                        continue
                    query = 'Is the semantic triple ["' + triple[0] + '", "' + triple[1] + '" , "' + triple[2] + '"] either true (1), false (2), or not understandable (3)? Return the number as a single digit without explanation.'
                    def answerConversion(answer):
                        answer = answer.strip()
                        assert answer in ["1", "2", "3"]
                        return ["true", "false", "rubbish"][int(answer) - 1]
                    truthValue, truthValueFound = tryRecieveAnswer(query, gpt_3_5_turbo_completion, answerConversion)
                    if not truthValueFound:
                        truthValue = "rubbish"
                    if truthValue == "true":
                        trueTriples.append(triple)
                        fTrue.write((",\n" if len(trueTriples) > 1 else "") + "    " + dumps(triple))
                        fTrue.flush()
                    elif truthValue == "false":
                        falseTriples.append(triple)
                        fFalse.write((",\n" if len(falseTriples) > 1 else "") + "    " + dumps(triple))
                        fFalse.flush()
                    else:
                        rubbishTriples.append(triple)
                        fRubbish.write((",\n" if len(rubbishTriples) > 1 else "") + "    " + dumps(triple))
                        fRubbish.flush()
                fTrue.write("\n]")
                fFalse.write("\n]")
                fRubbish.write("\n]")
    with (rootpath / "triples" / triplesName / "truth_statistics.json").open("w") as f:
        f.write(dumps({
            "true": len(trueTriples),
            "truePercentage": len(trueTriples) / len(triples) * 100,
            "false": len(falseTriples),
            "falsePercentage": len(falseTriples) / len(triples) * 100,
            "rubbish": len(rubbishTriples),
            "rubbishPercentage": len(rubbishTriples) / len(triples) * 100
        }))

def testObjectIdentifingCapabilities(triplesName, correlationsName, numberOfSamples, numberOfChoices):
    triples = loads((rootpath / "triples" / triplesName / "triples.json").read_text())
    correlationGraph = loads((rootpath / "correlations" / correlationsName / "graph.json").read_text())
    objectIdentifingTestPath = rootpath / "triples" / triplesName / ("object_identifing_test_with_" + str(numberOfChoices) + "_choices.json")
    if objectIdentifingTestPath.exists():
        objectIdentifingTest = loads(objectIdentifingTestPath.read_text())
    else:
        objectIdentifingTest = []
    # Format of objectIdentifingTest: [subject, predicate, object, [alternative objects], selected object]
    with objectIdentifingTestPath.open("w") as f:
        f.write("[")
        f.write(",".join(["\n    " + dumps(test) for test in objectIdentifingTest]))
        f.flush()
        i = numberOfSamples - len(objectIdentifingTest)
        while i > 0:    
            randomTriple = choice(triples)
            subjectNode = [node for node in correlationGraph if node[0] == randomTriple[0]][0]
            allPossibleObjects = [correlationGraph[j][0] for j in subjectNode[1]]
            otherObjects = [obj for obj in allPossibleObjects if obj != randomTriple[2]]
            shuffle(otherObjects)
            if len(otherObjects) < numberOfChoices - 1:
                continue
            otherObjects = otherObjects[:numberOfChoices - 1]
            objectChoices = otherObjects + [randomTriple[2]]
            shuffle(objectChoices)
            query = 'What is the missing object for the semantic triple ["' + randomTriple[0] + '", "' + randomTriple[1] + '" , ???]? Choose the correct object from the following choices: {' + ', '.join([f"{i + 1}: \"{objectChoices[i]}\"" for i in range(len(objectChoices))]) + '}. Return the number of the correct object as a single digit without explanation.'
            def answerConversion(answer):
                answer = answer.strip()
                assert answer in [str(i + 1) for i in range(len(objectChoices))]
                return objectChoices[int(answer) - 1]
            selectedObject, selectedObjectFound = tryRecieveAnswer(query, gpt_3_5_turbo_completion, answerConversion)
            if not selectedObjectFound:
                selectedObject = choice(objectChoices)
            objectIdentifingTest.append([randomTriple[0], randomTriple[1], randomTriple[2], otherObjects, selectedObject])
            f.write((",\n" if len(objectIdentifingTest) > 1 else "") + "    " + dumps(objectIdentifingTest[-1]))
            f.flush()
            i -= 1
        f.write("\n]")
        # Calculate statistics
        falseObjectSelections = 0
        trueObjectSelections = 0
        numberOfSamples = len(objectIdentifingTest)
        for test in objectIdentifingTest:
            if test[4] == test[2]:
                trueObjectSelections += 1
            else:
                falseObjectSelections += 1
        falseObjectSelectionPropotion = falseObjectSelections / numberOfSamples
        falseObjectSelectionVariance = 0
        for k in range(0, numberOfSamples + 1):
            falseObjectSelectionVariance += ((k - falseObjectSelections) ** 2) * comb(numberOfSamples, k, exact=True) * (falseObjectSelectionPropotion ** k) * ((1 - falseObjectSelectionPropotion) ** (numberOfSamples - k))
        falseObjectSelectionPropotionError = sqrt(falseObjectSelectionVariance) / numberOfSamples
        falseObjectSelectionPerCorrectObjectSelection = falseObjectSelections / trueObjectSelections if trueObjectSelections > 0 else None
        falseObjectSelectionPerCorrectObjectSelectionComparedToRandom = falseObjectSelectionPerCorrectObjectSelection / (numberOfChoices - 1)
        # See https://github.com/gratach/thoughts/blob/master/topics/master-thesis/evaluation/false-choices-per-correct-choice-compared-to-random.md
        falseObjectSelectionPerCorrectObjectSelectionComparedToRandomError = falseObjectSelectionPropotionError / (numberOfChoices - 1) / ((1 - falseObjectSelectionPropotion) ** 2)
        with (rootpath / "triples" / triplesName / "object_identifing_test_statistics.json").open("w") as f:
            statistics = {
                "falseObjectSelections": falseObjectSelections,
                "trueObjectSelections": trueObjectSelections,
                "falseObjectSelectionPropotion": falseObjectSelectionPropotion,
                "falseObjectSelectionVariance": falseObjectSelectionVariance,
                "falseObjectSelectionPropotionError": falseObjectSelectionPropotionError,
                "falseObjectSelectionPerCorrectObjectSelection": falseObjectSelectionPerCorrectObjectSelection,
                "falseObjectSelectionPerCorrectObjectSelectionComparedToRandom": falseObjectSelectionPerCorrectObjectSelectionComparedToRandom,
                "falseObjectSelectionPerCorrectObjectSelectionComparedToRandomError": falseObjectSelectionPerCorrectObjectSelectionComparedToRandomError
            }
            f.write(dumps(statistics, indent=4))
            print(dumps(statistics, indent=4))
#generateTriples("tree_cor_mi_man", "tri_1")
#navigateThroughTriples("tri_1")
#createPredicateQuantityChart("tri_1", 20)
#testTripleTruthValue("tri_1")
testObjectIdentifingCapabilities("tri_1", "tree_cor_mi_man", 500, 5)