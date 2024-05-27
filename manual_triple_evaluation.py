from pathlib import Path
from json import loads, dumps
from random import choice, shuffle
from mastg_lib import gpt_4_turbo_completion, tryRecieveAnswer, gpt_3_5_turbo_completion
from math import sqrt
from matplotlib import pyplot as plt

rootpath = Path(__file__).parent

def generateAdditionalFreeAssociatedTriples(triplesName, number, completionFunction = gpt_3_5_turbo_completion):
    triplePath = rootpath / "triples" / triplesName / "triples.json"
    triplePath.parent.mkdir(parents=True, exist_ok=True)
    termsPath = rootpath / "technical_terms_shuffled.txt"
    terms = termsPath.read_text().split("\n")
    if triplePath.exists():
        triples = loads(triplePath.read_text())
    else:
        triples = []
    j = 0
    for i in range(number):
        while True:
            nextTerm = terms[j]
            j += 1
            if len([triple for triple in triples if triple[0] == nextTerm]) == 0:
                break
        querry = f'Semantic triples such as ["Star", "emits", "Light"] and ["Rocket", "can bring cargo to", "Space"] consists of a subject, a predicate, and an object. Give me five examples of semantic triples that contain "{nextTerm}" as subject and return them in an array formatted like [["sub1", "pred1", "obj1"], ["sub2", "pred2", "obj2"], ...]. Return nothing but the array without explanation.'
        def answerConversion(answer):
            result = loads(answer)
            assert isinstance(result, list)
            assert all(isinstance(triple, list) for triple in result)
            assert all(len(triple) == 3 for triple in result)
            assert all(isinstance(term, str) for triple in result for term in triple)
            return result
        answer, success = tryRecieveAnswer(querry, completionFunction, answerConversion)
        if success:
            triples.extend(answer)
    with triplePath.open("w") as tripleFile:
        tripleFile.write("[")
        for i, triple in enumerate(triples):
            if i > 0:
                tripleFile.write(",\n    ")
            tripleFile.write(dumps(triple))
        tripleFile.write("\n]")

def manuallyEvaluateTriples(triplesName = None):
    if triplesName is None:
        triplesName = input("Enter the name of the triples to evaluate: ")
    triplePath = rootpath / "triples" / triplesName / "triples.json"
    triples = loads(triplePath.read_text())
    tripleEvaluationPath = rootpath / "triples" / triplesName / "manual_evaluation.json"
    # Triple evaluation file format: [[["sub1", "pred1", "obj1"], is valid triple (true/false), truth value (true/false/null), context dependent score (0-10), triviality score (0-10)], ...]"
    if tripleEvaluationPath.exists():
        tripleEvaluations = loads(tripleEvaluationPath.read_text())
    else:
        tripleEvaluations = []
    with tripleEvaluationPath.open("w") as tripleEvaluationFile:
        tripleEvaluationFile.write("[")
        for i, tripleEvaluation in enumerate(tripleEvaluations):
            if i > 0:
                tripleEvaluationFile.write(",")
            tripleEvaluationFile.write("\n    ")
            tripleEvaluationFile.write(dumps(tripleEvaluation))
        tripleEvaluationFile.flush()
        for triple in triples:
            if any(triple == tripleEvaluation[0] for tripleEvaluation in tripleEvaluations):
                continue
            print(triple)
            print("Is this a valid triple? (y/n/exit)")
            valid = input()
            if valid == "exit":
                break
            if valid == "y":
                print("Is this triple true? (y/n/u/exit)")
                truth = input()
                if truth == "exit":
                    break
                print("How context dependent is this triple? (0-10/exit)")
                contextDependent = int(input())
                if contextDependent == "exit":
                    break
                print("How trivial is this triple? (0-10/exit)")
                trivial = int(input())
                if trivial == "exit":
                    break
                tripleEvaluations.append([triple, True, None if truth == "u" else truth == "y", contextDependent, trivial])
            else:
                tripleEvaluations.append([triple, False, None, None, None])
            if len(tripleEvaluations) > 1:
                tripleEvaluationFile.write(",")
            tripleEvaluationFile.write("\n    ")
            tripleEvaluationFile.write(dumps(tripleEvaluations[-1]))
            tripleEvaluationFile.flush()
        tripleEvaluationFile.write("\n]")

def sortTriplesInOrderOfSubjectListAndShortenThemTo5PerSubject(sourceTriplesName, targetTriplesName):
    triplePath = rootpath / "triples" / sourceTriplesName / "triples.json"
    triples = loads(triplePath.read_text())
    termsPath = rootpath / "technical_terms_shuffled.txt"
    terms = termsPath.read_text().split("\n")
    triples.sort(key=lambda triple: terms.index(triple[0]))
    shortenedTriples = []
    currentSubject = None
    currentSubjectCount = 0
    for triple in triples:
        if triple[0] != currentSubject:
            currentSubject = triple[0]
            currentSubjectCount = 0
        if currentSubjectCount < 5:
            shortenedTriples.append(triple)
            currentSubjectCount += 1
    targetTriplePath = rootpath / "triples" / targetTriplesName / "triples.json"
    targetTriplePath.parent.mkdir(parents=True, exist_ok=True)
    with targetTriplePath.open("w") as targetTripleFile:
        targetTripleFile.write("[")
        for i, triple in enumerate(shortenedTriples):
            if i > 0:
                targetTripleFile.write(",")
            targetTripleFile.write("\n    " + dumps(triple))
        targetTripleFile.write("\n]")

def createShuffledTechnicalTermsList():
    termsPath = rootpath / "technical_terms.txt"
    terms = termsPath.read_text().split("\n")
    shuffledTermsPath = rootpath / "technical_terms_shuffled.txt"
    shuffledTermsPath.touch()
    shuffledTerms = shuffledTermsPath.read_text().split("\n")
    shuffle(terms)
    for term in shuffledTerms:
        terms.remove(term)
    shuffledTerms.extend(terms)
    shuffledTermsPath.write_text("\n".join(shuffledTerms))

def generateSubjectObjectClone(oldTriplesName, newTriplesName, numberOfAdditionalTriples = 10, completionFunction = gpt_3_5_turbo_completion):
    oldTriplePath = rootpath / "triples" / oldTriplesName / "triples.json"
    oldTriples = loads(oldTriplePath.read_text())
    newTriplePath = rootpath / "triples" / newTriplesName / "triples.json"
    newTriplePath.parent.mkdir(parents=True, exist_ok=True)
    if newTriplePath.exists():
        newTriples = loads(newTriplePath.read_text())
    else:
        newTriples = []
    assert len(oldTriples) >= len(newTriples)
    for oldTriple, newTriple in zip(oldTriples, newTriples):
        assert oldTriple[0] == newTriple[0]
        assert oldTriple[2] == newTriple[2]
    for i in range(numberOfAdditionalTriples):
        if len(newTriples) == len(oldTriples):
            break
        oldTriple = oldTriples[len(newTriples)]
        query = 'Semantic triples such as ["Star", "emits", "Light"] and ["Rocket", "can bring cargo to", "Space"] consists of a subject, a predicate, and an object. What is the predicate for the triple ["' + oldTriple[0] + '", ??? , "' + oldTriple[2] + '"]? Return only the predicate quoted by "" without explanation.'
        def answerConversion(answer):
            answer = answer.strip()
            assert answer[0] == '"' and answer[-1] == '"'
            answer = answer[1:-1]
            assert not '"' in answer
            return answer
        answer, success = tryRecieveAnswer(query, completionFunction, answerConversion)
        if success:
            newTriple = [oldTriple[0], answer, oldTriple[2]]
        else :
            newTriple = [oldTriple[0], "is related to", oldTriple[2]]
        newTriples.append(newTriple)
    with newTriplePath.open("w") as newTripleFile:
        newTripleFile.write("[")
        for i, triple in enumerate(newTriples):
            if i > 0:
                newTripleFile.write(",\n    ")
            newTripleFile.write(dumps(triple))
        newTripleFile.write("\n]")

def calculateProportionWithErrorOfBinaryEvent(numberOfEvents, numberOfPositiveEvents):
    assert numberOfEvents >= 0
    assert numberOfPositiveEvents >= 0
    assert numberOfPositiveEvents <= numberOfEvents
    p = numberOfPositiveEvents / numberOfEvents
    numberOfPositiveEventsVariance = 0
    error = sqrt(p * (1 - p) / numberOfEvents) if numberOfEvents > 0 and p > 0 and p < 1 else None
    return p, error

def calculateAverageWithError(valueList):
    assert len(valueList) > 0
    average = sum(valueList) / len(valueList)
    varianceSum = sum((value - average) ** 2 for value in valueList)
    error = sqrt(varianceSum / len(valueList) / (len(valueList) - 1)) if len(valueList) > 1 else None
    return average, error

def createEvaluationStatistics(triplesName):
    tripleEvaluationPath = rootpath / "triples" / triplesName / "manual_evaluation.json"
    evaluationStatisticsPath = rootpath / "triples" / triplesName / "evaluation_statistics.json"
    tripleEvaluations = loads(tripleEvaluationPath.read_text())
    totalTripleCount = len(tripleEvaluations)
    validTripleCount = len([triple for triple, valid, _, _, _ in tripleEvaluations if valid])
    trueTriplesCount = len([triple for triple, valid, truth, _, _ in tripleEvaluations if valid and truth == True])
    falseTriplesCount = len([triple for triple, valid, truth, _, _ in tripleEvaluations if valid and truth == False])
    unknownTriplesCount = len([triple for triple, valid, truth, _, _ in tripleEvaluations if valid and truth == None])
    averageTrivialityScore, trivialityError = calculateAverageWithError([triviality for triple, valid, _, _, triviality in tripleEvaluations if valid])
    averageContextDependenceScore, contextDependenceError = calculateAverageWithError([contextDependence for triple, valid, _, contextDependence, _ in tripleEvaluations if valid])
    validTripleProportion, validTripleError = calculateProportionWithErrorOfBinaryEvent(totalTripleCount, validTripleCount)
    trueTripleProportion, trueTripleError = calculateProportionWithErrorOfBinaryEvent(trueTriplesCount + falseTriplesCount, trueTriplesCount)
    unknownTripleProportion, unknownTripleError = calculateProportionWithErrorOfBinaryEvent(validTripleCount, unknownTriplesCount)
    outputJson = {
        "totalTripleCount": totalTripleCount,
        "validTripleCount": validTripleCount,
        "trueTriplesCount": trueTriplesCount,
        "falseTriplesCount": falseTriplesCount,
        "unknownTriplesCount": unknownTriplesCount,
        "averageTrivialityScore": averageTrivialityScore,
        "trivialityError": trivialityError,
        "averageContextDependenceScore": averageContextDependenceScore,
        "contextDependenceError": contextDependenceError,
        "validTripleProportion": validTripleProportion,
        "validTripleError": validTripleError,
        "trueTripleProportion": trueTripleProportion,
        "trueTripleError": trueTripleError,
        "unknownTripleProportion": unknownTripleProportion,
        "unknownTripleError": unknownTripleError
    }
    print(dumps(outputJson, indent=4))
    with evaluationStatisticsPath.open("w") as evaluationStatisticsFile:
        evaluationStatisticsFile.write(dumps(outputJson, indent=4))

def plotHeatmapOfTrivialityAndContextDependence(triplesName):
    triplesEvaluationPath = rootpath / "triples" / triplesName / "manual_evaluation.json"
    tripleEvaluations = loads(triplesEvaluationPath.read_text())
    # The triviality and context dependence scores are in the range of 0-10
    field = [[0 for _ in range(11)] for _ in range(11)]
    for triple, valid, _, contextDependence, triviality in tripleEvaluations:
        if valid:
            field[contextDependence][triviality] += 1
    # Plot the heatmap with matplotlib
    fig, ax = plt.subplots()
    im = ax.imshow(field)
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    # Write the text of the values in the heatmap
    for i in range(11):
        for j in range(11):
            text = ax.text(j, i, field[i][j], ha="center", va="center", color="w")
    ax.set_xlabel("Triviality")
    ax.set_ylabel("Context Dependence")
    ax.set_title("Heatmap of Context Dependence and Triviality Scores")
    fig.tight_layout()
    plt.show()
    

#generateAdditionalFreeAssociatedTriples("free_ass_gpt4", 90, gpt_4_turbo_completion)
#generateAdditionalFreeAssociatedTriples("free_ass_gpt3_5", 1, gpt_3_5_turbo_completion)
manuallyEvaluateTriples()
#createShuffledTechnicalTermsList()
#generateSubjectObjectClone("tri_1_curated", "tri_1_curated_gpt4", 480, gpt_4_turbo_completion)
#sortTriplesInOrderOfSubjectListAndShortenThemTo5PerSubject("tri_1", "tri_1_curated")
#createEvaluationStatistics("tri_1_curated")
#createEvaluationStatistics("tri_1_curated_gpt4")
#createEvaluationStatistics("free_ass_gpt3_5")
#createEvaluationStatistics("free_ass_gpt4")
#plotHeatmapOfTrivialityAndContextDependence("tri_1_curated")
#plotHeatmapOfTrivialityAndContextDependence("tri_1_curated_gpt4")
#plotHeatmapOfTrivialityAndContextDependence("free_ass_gpt3_5")
#plotHeatmapOfTrivialityAndContextDependence("free_ass_gpt4")
